import numpy as np
import sys, glob, netCDF4
from scipy.ndimage import median_filter
from shutil import copyfile

def sliding_min(x, winsize=5, mode='reflect'):
    '''
    Applies moving minimum filter with the window size of `winsize`.
    INPUTS:
        x - 1d or 2d ndarray. if 2d, function is applied recursively on the
            second dimension
        winsize - integer specifying the window size
        mode - string specifying how to treat the array boundary (goes into np.pad)
    OUTPUTS:
        out - filtered array with the same dimension as the input
    '''
    if len(x.shape) == 1:
        out = np.zeros_like(x)
        padfront = int((winsize-1)/2)
        padafter = winsize - 1 - padfront
        padded = np.pad(x, (padfront, padafter), mode=mode)
        for i in range(len(x)):
            out[i] = np.min(padded[i: i + winsize])
        return out
    elif len(x.shape) == 2:
        out = np.zeros_like(x)
        for i in range(x.shape[0]):
            out[i] = sliding_min(x[i], winsize=winsize)
        return out

def medfilt3d(br, threshold=50, win_size=(5,10,10), mode=2):
    '''
    Function that performs star removal on an orbit of l1 brightness profiles.

    It operates on 3d images (stripe, epoch, altitude) rather than individual
    1d (altitude) profiles to exploit the information to its best. The star
    detection works based on sliding a 3d window in the data and comparing the
    center element's value with the median of the window. If the value is above
    a threshold, the center pixel is marked as a star and its value is replaced
    with the median of that window. Window size and threshold are determined
    empirically.
    INPUTS:
        br - (6,orbit_ind,256) array of brightness profile for a particular orbit
    OUTPUTS:
        br_corrected - star removed brightness profiles
    '''
    br = br.copy()
    br_med = median_filter(br, size=win_size)
    br_diff = br - br_med
    if mode == 1:
        filter = br_diff > threshold
        filter[:,:,:175] = 0
        # filter[np.where(br_med[filter==1]>1000)] = 0
        filter[np.where((filter==1) & (br_med>1000))] = 0
        br_filt = br.copy()
        ind = (filter==1)
        br_filt[ind] = br_med[ind]
        return filter, br_filt
    if mode == 2:
        filter = br_diff > threshold
        br_filt = br.copy()
        ind = (filter==1)
        br_filt[ind] = br_med[ind]
        return filter, br_filt

def hot_pixel_correction(br):
    '''
    Function that performs hot pixel correction. It operates on 3d
    (stripe,epoch,altitude) orbit images individually. Operates on stripes
    independently.

    The method exploits the fact that hot pixels consistently have high values
    through time along the orbit. Therefore, it takes the average of the 3d
    (stripe,epoch,altitude) profile along the epoch dimension. Idea is that
    the averaged profile will reveal the hot pixels. Then a moving minimum
    filter is applied on the averaged profile to estimate the underlying true
    signal. The difference between the averaged profile and the minimum filter
    is subtracted from the profiles to finalize the hot pixel correction.
    INPUTS:
        br - (6,night_ind,256) array of brightness profile for a particular orbit
    OUTPUTS:
        br_corrected - hot pixel corrected brightness profile
    '''
    brx = np.mean(br, axis=1)
    brx_min = sliding_min(brx, winsize=5)
    br_cor = br.copy()
    for i in range(6):
        diff = brx[i] - brx_min[i]
        br_cor[i] -= diff
    br_cor[br_cor < 0] = 1e-4
    return br_cor

def artifact_removal_orbit(br, mode, br_err=None, Nmc=100):
    '''
    Function that performs pre-processing on the l1 brightness profiles together
    with their uncertainties. It performs star removal and additionally hot
    pixel correction for nighttime. It operates on 3d (stripe,epoch,altitude)
    orbit images individually.
    INPUTS:
        br - (6,orbit_ind,256) array of brightness profile for a particular orbit
        br_err - (6,orbit_ind,256) array of uncertainty profile for a particular orbit
        mode - 1: day, 2: night
        Nmc - the number of Monte Carlo trials for uncertainty propagation
    OUTPUTS:
        br_corrected - star removed and hot pixel corrected brightness profile
        br_err_modified - modified uncertainty profile according to the correction
    '''
    if mode == 1:
        # apply star removal
        _, br_corrected = medfilt3d(br, threshold=50)
    elif mode == 2:
        # apply star removal
        _, br_corrected = medfilt3d(br, threshold=50)
        # apply hot pixel correction
        br_corrected = hot_pixel_correction(br_corrected)
        # apply star removal again since it can detect the stars better now
        _, br_corrected = medfilt3d(br_corrected, threshold=10)

    if br_err is not None:
    # perform monte carlo uncertainty propagation
        br_corr_n = np.zeros((Nmc, br.shape[0], br.shape[1], br.shape[2]))
        for i in range(Nmc):
            print('Trial {}'.format(i))
            br_n = np.random.normal(loc=br, scale=br_err)
            br_corr_n[i] = artifact_removal_orbit(br_n, mode)
        br_corrected_err = np.std(br_corr_n, axis=0)
        return br_corrected, br_corrected_err

    return br_corrected

def artifact_removal(l1, mode):
    '''
    Code that performs pre-processing on the l1 brightness profiles together
    with their uncertainties. It performs star removal and hot pixel correction.
    INPUTS:
        l1 - the l1 netCDF dataset
        mode - 1: day, 2: night
    OUTPUTS:
        l1 - modified l1 netCDF dataset
    '''
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    mode_l1 = l1.variables['ICON_L1_FUV_Mode'][:]
    mode_night = (mode_l1 == mode).astype(np.int)
    nights = np.diff(mode_night, prepend=0)
    nights[nights==-1] = 0
    idxs = np.where(mode_l1==2)[0][:]
    nights = np.cumsum(nights)[idxs]
    for night in np.unique(nights):
        night_ind = np.where(nights==night)[0]
        br = np.zeros((6, len(night_ind), 256))
        br_err = np.zeros((6, len(night_ind), 256))
        mask = np.zeros_like(br, dtype=np.bool)
        for i in range(6):
            tmp = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]][idxs[night_ind],:]
            br_err[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]][idxs[night_ind],:].filled(fill_value=0)
            mask[i] = tmp.mask
            br[i] = tmp.filled(fill_value=0)
        br_corrected, br_err_modified = artifact_removal_orbit(br, mode, br_err)
        for i in range(6):
            l1.variables[
                'ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]
            ][idxs[night_ind],:] = np.ma.array(
                br_corrected[i], mask=mask[i]
            ).filled(fill_value=np.nan)
            l1.variables[
                'ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]
            ][idxs[night_ind],:] = np.ma.array(
                br_err_modified[i], mask=mask[i]
            ).filled(fill_value=np.nan)
    return l1


def destarrer(date):
    path_dir = '/home/kamo/resources/iconfuv/nc_files/'
    file_input = path_dir + 'l1/ICON_L1_FUV_SWP_{}_v03r*'.format(date)
    file_input = glob.glob(file_input)
    file_input.sort()
    file_input = file_input[-1]

    file_input_c = file_input.split('_v')[0] + '_v78r000.NC'
    copyfile(file_input, file_input_c)

    data = netCDF4.Dataset(file_input_c, mode='r+')

    import time
    t0 = time.time()
    data = artifact_removal(data, mode=2)
    print('Elapsed time: {} minutes'.format((time.time()-t0)/60.))
    data.close()

if __name__== "__main__":
    destarrer(str(sys.argv[1]))
