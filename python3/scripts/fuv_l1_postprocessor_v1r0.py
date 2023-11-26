# Ulas Kamaci - 2022-03-24
# ICON FUV L1 Post-processing code.

import numpy as np
from scipy.ndimage import median_filter, convolve1d
from scipy.signal import convolve2d
from keras.models import load_model
import os, sys, torch, netCDF4, datetime, fnmatch
import torch.nn as nn
import torch.nn.functional as F
from shutil import copyfile

################ Version Control ################
postprocessor_version = 1
postprocessor_revision = 0

artifact_removal_version = 2
artifact_removal_revision = 2
# Artifact Removal History:
# v2r2 [2022-03-24]: neural network based star removal is implemented. - Ulas Kamaci

################ Neural Network ################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mp_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(mp_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, scale, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, start_filters, bilinear=True, residual=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.residual = residual
        sf = start_filters

        self.inc = DoubleConv(in_channels, sf)
        self.down1 = Down(sf, sf*2, (2,1))
        self.down2 = Down(sf*2, sf*4, (2,1))
        self.down3 = Down(sf*4, sf*8, (2,2))
        factor = 2 if bilinear else 1
        self.down4 = Down(sf*8, sf*16 // factor, (2,3))
        self.up1 = Up(sf*16, sf*8 // factor, (2,3), bilinear)
        self.up2 = Up(sf*8, sf*4 // factor, (2,2), bilinear)
        self.up3 = Up(sf*4, sf*2 // factor, (2,1), bilinear)
        self.up4 = Up(sf*2, sf, (2,1), bilinear)
        self.outc = OutConv(sf, 1)

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.residual:
            return x0 - logits
        else:
            return logits
################  ################  ################

def predictor(br, network, mode):
    '''
    Function to run the neural network on the data to remove stars.

    Args:
        br (ndarray): (6,orbit_ind,256) array of brightness profiles in
            Rayleighs for a particular orbit
        network : Keras or Pytorch structure for the neural network
        mode (int): integer specifying the operation mode. {1:day, 2:night}

    Returns:
        out (ndarray) - star removed brightness profiles

    '''
    br = br.transpose(1,2,0)
    if mode==1: # use the pytorch model
        with torch.no_grad():
            out = network(torch.from_numpy(br[:,None,:,:]).float())[:,0].numpy()
    elif mode==2: # use the keras model
        # Project the brightnesses <-50 R to -50 R for the nighttime network
        br[br<-50]=-50
        out = network.predict(br[:,:,:,None])[:,:,:,0]
    return out.transpose(2,0,1)

def medfilt3d(br, threshold=None, win_size=(5,10,10), mode=2, full=False):
    '''
    Function that performs star removal on an orbit of l1 brightness profiles.
    Assumes the array units are in Rayleighs.

    It operates on 3d images (stripe, epoch, altitude) rather than individual
    1d (altitude) profiles to exploit the information to its best. The star
    detection works based on sliding a 3d window in the data and comparing the
    center element's value with the median of the window. If the value is above
    a threshold, the center pixel is marked as a star and its value is replaced
    with the median of that window. Window size and threshold are determined
    empirically.

    If full=True is selected, it works as an outlier detector + correction.

    Args:
        br (ndarray): (6,orbit_ind,256) array of brightness profiles in
            Rayleighs for a particular orbit
        threshold (float): the threshold value above which deviation from the
            median results in classifying the pixel as a star
        win_size (tuple): 3d tuple specifying the window size which is centered
            at the pixel of interest
        mode (int): integer specifying the operation mode. {1:day, 2:night}
        full (bool): if True, rather than star removal, it works as an outlier
            correction function.

    Returns:
        br_corrected - star removed brightness profiles
    '''
    br = br.copy()
    br_med = median_filter(br, size=win_size)
    br_diff = br - br_med
    brmed = np.nanmedian(br, axis=0)
    brmean = np.nanmean(br, axis=0)
    if full==True:
        filter = (abs(br_med) >= 5) & (abs(br_diff) > 4*abs(br_med))
        filter2 = (abs(br_med) < 5) & (abs(br_diff) > 20)
        filter = filter | filter2
        br_filt = br.copy()
        ind = (filter==1)
        br_filt[ind] = br_med[ind]
        return filter, br_filt

    # daytime mode
    if mode == 1:
        filter = br_diff > threshold
        # only operate on pixel 175 and above
        filter[:,:,:175] = 0
        filter[np.where((filter==1) & (br_med>1000))] = 0
        br_filt = br.copy()
        ind = (filter==1)
        br_filt[ind] = br_med[ind]
        return filter, br_filt
    # nighttime mode
    if mode == 2:
        filter = br_diff > threshold
        br_filt = br.copy()
        ind = (filter==1)
        indsig = (brmed > 40) & (brmean - brmed < 20)
        ind = ind * (1 - np.repeat(indsig[np.newaxis], 6, axis=0))
        ind = ind.astype(bool)
        br_filt[ind] = br_med[ind]
        return filter, br_filt

def hot_pixel_correction(br):
    '''
    Function that performs hot pixel correction. It operates on 3d
    (stripe,epoch,altitude) orbit images. Operates on stripes independently.

    The method exploits the fact that hot pixels consistently have high values
    through time along the orbit. Therefore, it takes the average of the 3d
    (stripe,epoch,altitude) profile along the epoch dimension. Idea is that
    the averaged profile will reveal the hot pixels. Then a moving minimum
    filter is applied on the averaged profile to estimate the underlying true
    signal. The difference between the averaged profile and the minimum filter
    is subtracted from the profiles to finalize the hot pixel correction.

    Args:
        br (ndarray): (6,orbit_ind,256) array of brightness profiles in
            Rayleighs for a particular orbit

    Returns:
        br_corrected (ndarray): artifact removed profiles
    '''
    brx = np.nanmean(br, axis=1)
    brx_lp = convolve2d(brx, 0.1*np.ones((1,10)), mode='same', boundary='symm')
    br_cor = br.copy()
    diff = np.zeros_like(brx)
    for i in range(6):
        diff[i] = brx[i] - brx_lp[i]
        br_cor[i] -= diff[i]
    return diff, br_cor

def daynight_index(br, threshold=30):
    index = br.shape[1]
    y = np.mean(br, axis=(0,2))
    y = convolve1d(y, 0.05*np.ones(20), mode='reflect')
    y1 = np.diff(y)
    ind = np.sort(np.where(y>threshold)[0])
    for i in ind:
        if y1[i] > 5:
            index = i
            break
    return index

def artifact_removal_orbit(br, mode, network):
    '''
    Function that performs star removal and additionally hot pixel correction
    for nighttime. It operates on 3d (stripe,epoch,altitude) orbit images
    individually.

    Args:
        br (ndarray): (6,orbit_ind,256) array of brightness profiles in
            Rayleighs for a particular orbit
        mode (int): integer specifying the mode. {1: day, 2: night}
        network (str): the neural network function for star removal

    Returns:
        br_corrected (ndarray): artifact removed profiles
    '''
    if mode == 1:
        # perform outlier correction
        # _, br_corrected = medfilt3d(br, win_size=(6,1,10), full=True)
        # remove stars
        br_corrected = predictor(br, network=network, mode=mode)
        # reset the bottom 50 pixels due to the neural network artifacts
        br_corrected[:,:,:50] = br[:,:,:50]
    elif mode == 2:
        try:
            # find out the dayside part of the orbit if there's any
            index = daynight_index(br, threshold=30)
        except:
            # in case there is an error
            index = br.shape[1]
        # apply star removal to remove the bright stars which can effect the
        # hot pixel correction algotihm
        _, br_corrected = medfilt3d(br[:,:index,:], threshold=50, mode=mode)
        # apply hot pixel correction to find the hot pixel offsets
        diff, _ = hot_pixel_correction(br_corrected)

        # apply the hot pixel correction only to the starry raw image so that
        # the stars are not removed
        br2 = br[:,:index,:] - diff[:,np.newaxis,:]

        # # perform outlier correction
        # _, br2 = medfilt3d(br2, win_size=(6,1,10), full=True)
        # remove stars
        br2 = predictor(br2, network=network, mode=mode)

        # add back the untouched dayside part back to the orbit
        br_corrected = np.append(br2, br[:,index:,:], axis=1)

    return br_corrected

def br_nan_filler(br, mode='median'):
    '''
    Fill the NaN values at the top and bottom portions with the median
    value of the 20x6 [alt x stripe] windows at the edges. Repeat for each
    Epoch.

    Args:
        br (ndarray): 1 day of all stripe profiles in Rayleighs with dimension
            [6,epoch,256]
        mode (str): 'median' or 'zero' filling
    '''
    if mode=='median':
        br_filled = br.copy()
        filltop = np.nanmedian(br[:,:,:20], axis=(0,2))
        fillbot = np.nanmedian(br[:,:,-20:], axis=(0,2))
        br_filled[:,:,:30] = br[:,:,:30].filled(fill_value=filltop[None,:,None])
        br_filled[:,:,-30:] = br[:,:,-30:].filled(fill_value=fillbot[None,:,None])
        br_filled[:,:,30:-30] = br[:,:,30:-30].filled(fill_value=0)
    elif mode=='zero':
        br_filled = br.filled(fill_value=0)

    return br_filled

def profiler(l1, channel):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    if channel==1:
        profname = 'ICON_L1_FUVA_SWP_PROF_'
    else:
        profname = 'ICON_L1_FUVB_LWP_PROF_'

    br = l1.variables[profname+'{}'.format(mirror_dir[0])][:]
    profiles = np.ma.zeros((6, br.shape[0], br.shape[1]))
    for i in range(6):
        profiles[i] = l1.variables[profname+'{}'.format(mirror_dir[i])][:]

    return profiles

def artifact_removal(profiles, channel, fuv_mode, path_to_networks):
    '''
    Performs hot pixel correction on one day of profiles.
    It first parses the orbits using the fuv_mode variable, then processes
    each orbit individually. Assumes the array units are in Rayleighs.

    Args:
        profiles (ndarray): 1 day of all stripe profiles in Rayleighs with
            dimension [6,epoch,256]
        channel (int): integer specifying the FUV channel (1:SW, 2:LW)
        fuv_mode (ndarray): ICON_L1_FUV_Mode variable where 1:day, 2:night
        path_to_networks (str): full path to the folder containing the neural
            networks for star removal

    Returns:
        profiles_cleaned (ndarray): artifact removed profiles, same dimension
            as profiles
    '''
    # initialize the neural networks
    device = torch.device('cpu')
    day_network_file = os.path.join(path_to_networks,'day_network_v1r0.pth')
    day_network = UNet(in_channels=1,
                 start_filters=32,
                 bilinear=True,
                 residual=True).to(device)
    night_network_file = os.path.join(path_to_networks,'night_network_v1r0')

    day_network.load_state_dict(torch.load(day_network_file, map_location=device))
    night_network = load_model(night_network_file)
    day_network.eval()

    # make sure input is a masked array; if not, create a masked array from it
    if not np.ma.is_masked(profiles):
        profiles = np.ma.array(profiles, mask=np.isnan(profiles))

    # initialize the artifact removed array
    profiles_cleaned = profiles.copy()

    if channel == 1: # SW channel has both day and night profiles
        modes = [1,2]
    elif channel == 2: # LW channel has only day profiles
        modes = [1]
    for mode in modes:
        if mode==1:
            network = day_network
        elif mode==2:
            network = night_network
        mode_orbit = (fuv_mode == mode).astype(np.int)
        orbits = np.diff(mode_orbit, prepend=0)
        orbits[orbits==-1] = 0
        idxs = np.where(fuv_mode==mode)[0][:]
        orbits = np.cumsum(orbits)[idxs]

        profiles_filled = br_nan_filler(profiles, mode='median')
        for orbit in np.unique(orbits):
            orbit_ind = np.where(orbits==orbit)[0]
            br_corrected = artifact_removal_orbit(
                br=profiles_filled[:,idxs[orbit_ind],:],
                mode=mode, network=network
            )
            profiles_cleaned[:, idxs[orbit_ind],:] = br_corrected

    profiles_cleaned.mask = profiles.mask
    profiles_cleaned = profiles_cleaned.filled(fill_value=np.nan)

    return profiles_cleaned

def create_var(
    nc,
    var_name,
    ref_var,
    data,
    **kwargs
):
    '''
    Given a netCDF4 dataset, this function creates a new variable in the dataset
    with the name given by `var_name`. The new variable is initially assigned
    the same attributes as the `ref_var` which is a netCDF4 variable. The data
    of the new variable is assigned the values of the numpy array `data`. The
    kwargs hold any specific attributes that will be assigned to (or changed in)
    the variable.

    Args:
        nc (netCDF4.Dataset): the netCDF dataset to be operated on
        var_name (str): string specifying the new variable name
        ref_var (netCDF4.Variable): the reference variable whose attributes are
            copied to the new variable
        data (numpy.ndarray): array whose values are copied into the new variable
        kwargs (dict): optional keyword arguments with attribute keys and values
            to be assigned to the new variable.
    Returns:
        nc (netCDF4.Dataset): the modified netCDF dataset with the new variable
    '''
    nc.createVariable(var_name, ref_var.datatype, ref_var.dimensions, fill_value=ref_var.FILLVAL)
    nc.variables[var_name].setncatts(ref_var.__dict__)
    nc.variables[var_name][:] = data
    for key, value in kwargs.items():
        nc.variables[var_name].setncattr(key, value)
    return nc

def create_vars(
    l1,
    channel,
    profiles,
    task='artifact_removal'
):
    '''
    Create new variables in the given l1 netCDF dataset with the appropriate
    attributes and values based on the provided task and the profiles.

    Args:
        l1 (netCDF4.Dataset): the netCDF dataset to be operated on
        channel (int): 1: SWP channel, 2: LWP channel
        profiles (numpy.ndarray): array whose values are copied into the new variable
        task (str): string specifying the task, which determines the names and
            the attributes of the created variables. 'artifact_removal' is the
            only implemented task for now.

    Returns:
        nc (netCDF4.Dataset): the modified netCDF dataset with the new variable
    '''
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    profname = 'ICON_L1_FUVA_SWP_PROF_' if channel==1 else 'ICON_L1_FUVB_LWP_PROF_'
    if task=='artifact_removal':
        num = len(fnmatch.filter(list(l1.variables.keys()), '*_P0_CLEAN*'))
        if num > 0:
            print('WARNING: Input file already has CLEAN variables. The new '+
                'variables will take the suffix CLEAN{}.'.format(num+1))
        # if there are already CLEAN variables in the file, name the new ones
        # with an incremented number
        suffix = '_CLEAN' if num==0 else '_CLEAN{}'.format(num+1)

    for i,j in enumerate(mirror_dir):
        if task=='artifact_removal':
            kwargs = {
                'CATDESC': l1[profname+j].CATDESC + ' without stars',
                'CONTENT': l1[profname+j].CONTENT + ' without stars',
                'FIELDNAM': l1[profname+j].FIELDNAM + '_clean',
                'VAR_NOTES': (l1[profname+j].VAR_NOTES +
                    ' without stars using artifact_removal_v{}r{}'.format(
                        artifact_removal_version, artifact_removal_revision
                    )
                )
            }
        l1 = create_var(l1, var_name=profname+j+suffix, ref_var=l1[profname+j],
            data=profiles[i], **kwargs)
    return l1

def postprocessor(
    file_in=None,
    file_out=None,
    path_to_networks=None
):
    '''
    Copies the L1 file at `file_in` into `file_out` and opens `file_out` in
    writing mode. Reads the unprocessed brightness profiles in the L1 dataset
    and generates artifact removed brightness profiles. Generates new netCDF
    variables using the artifact removed profiles with updated attributes and
    names, and writes them into `file_out`. The input `file_in` remains
    untouched, and the differences between `file_in` and `file_out` are 1. their
    name, 2. the additional variables in `file_out`, and 3. the following
    updated global attributes: 'File_Date', 'Generation_Date', and
    'Software_Version'.

    Args:
        file_in (str): string specifying the path to the input l1 file
            (including its name). It can either be a SWP or a LWP file.
        file_out (str): string specifying the path to the output l1 file
            (including its name) to be generated.
        path_to_networks (str): string specifying the path to the folder
            containing the neural networks.

    Returns:
        Returns 0 on success.
    '''
    assert path_to_networks is not None, "Please specify path_to_networks"
    assert ((file_in is not None) and (file_out is not None)), "Please provide both input and output paths"
    copyfile(file_in, file_out)
    l1 = netCDF4.Dataset(file_out, 'r+')
    assert (l1.Instrument=='FUV-A' or l1.Instrument=='FUV-B'), "Unknown file type"
    channel = 1 if l1.Instrument=='FUV-A' else 2
    # Extract the uncprocessed brightness profiles
    profiles = profiler(l1, channel=channel)
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    # Perform artifact removal and get the artifact removed profiles
    profiles_clean = artifact_removal(profiles=profiles, channel=channel,
        fuv_mode=mode, path_to_networks=path_to_networks)
    l1 = create_vars(l1, channel, profiles_clean, task='artifact_removal')

    # Modify the Global Attributes of the L1 file
    timenow = datetime.datetime.utcnow()
    l1.File_Date = timenow.strftime('%a, %d %b %Y, %Y-%m-%dT%H:%M:%S.%f')[:-3] + ' UTC'
    l1.Generation_Date = timenow.strftime('%Y%m%d')
    l1.Software_Version += ' > UIUC FUV L1 Post-Processor v{}.{}'.format(
        postprocessor_version, postprocessor_revision)
    l1.close()
    return 0

if __name__== "__main__":
    postprocessor(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
