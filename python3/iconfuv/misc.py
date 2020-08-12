import os, glob, netCDF4, pyglow
from airglow.FUV_L2 import get_msisGPI, FUV_Level_2_Density_Calculation, find_hm_Nm_F2, nan_checker
import airglow.ICON_FUV_fwd_model as FUV_F # for simulating data
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from scipy.signal import correlate, convolve2d
from skimage.transform import radon
from itertools import product
import pandas as pd
from dateutil import parser

path_dir = '/home/kamo/resources/iconfuv/nc_files/'

def profiler(l1, err=False, clean=False):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    foo = '_CLEAN' if clean else ''
    if 'ICON_L1_FUVB_CCD_TEMP' in list(l1.variables.keys()):
        profname = 'ICON_L1_FUVB_LWP_PROF_'
    else:
        profname = 'ICON_L1_FUVA_SWP_PROF_'

    br = l1.variables[profname+'{}{}'.format(mirror_dir[0], foo)][:]
    profiles = np.zeros((6, br.shape[0], br.shape[1]))
    for i in range(6):
        profiles[i] = l1.variables[profname+'{}{}'.format(mirror_dir[i], foo)][:]
    if err is True:
        profiles_err = np.zeros((6, br.shape[0], br.shape[1]))
        for i in range(6):
            profiles_err[i] = l1.variables[profname+'%s_Error' % mirror_dir[i]][:]
        profiles = np.swapaxes(profiles, 1, 2)
        profiles_err = np.swapaxes(profiles_err, 1, 2)
        return profiles, profiles_err

    profiles = np.swapaxes(profiles, 1, 2)
    return profiles

def get_oplus(l1=None, anc=None, epoch=100, stripe=3, bkgcor=False):
    limb = 150.
    contribution ='RRMN'
    reg_method = 'Tikhonov'
    weight_resid = False
    Spherical = True
    regu_order = 2
    reg_param = 2500.

    file_GPI = path_dir + 'ICON_Ancillary_GPI_2015-001-to-2020-187_v01r000.NC'
    gpi = netCDF4.Dataset(file_GPI, mode='r')

    mirror_dir = ['M9','M6','M3','P0','P3','P6']

    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    idx = np.where(mode==2)[0][epoch]
    dn = parser.parse(anc.variables['ICON_ANCILLARY_FUV_TIME_UTC'][idx])

    # Read the geophysical indeces
    ap3 = gpi['ap3'][:]
    ap = gpi['ap'][:]
    year_day = gpi['year_day'][:]
    f107 = gpi['f107d'][:]
    # Make sure this GPI has the average f107 in it
    if 'f107a' in gpi.variables.keys():
        f107a = gpi['f107a'][:]
    else:
        f107a = gpi['f107d'][:]
    gpi.close()

    my_f107, my_f107a, my_f107p, my_apmsis = get_msisGPI(dn, year_day, f107, f107a, ap, ap3)

    tanalts = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,2]
    tanlons = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,1]
    tanlats = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,0]
    satlatlonalt = [
        anc.variables['ICON_ANCILLARY_FUV_LATITUDE'][idx],
        anc.variables['ICON_ANCILLARY_FUV_LONGITUDE'][idx],
        anc.variables['ICON_ANCILLARY_FUV_ALTITUDE'][idx]
    ]
    local_time = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LST'][idx, -1, 2]

    limb_i = np.where(np.squeeze(tanalts)>=limb)[0]
    br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[stripe]][idx,:]
    if bkgcor is True:
        # br -= np.nanmedian(br[tanalts>400])
        med = np.median(br[10:50])
        med=0 if med>0 else med
        br -= med
    br = br[limb_i]
    unmasked_ind = nan_checker(br)
    limb_i0 = limb_i[unmasked_ind]
    br = br[::-1]
    unmasked_ind_f = nan_checker(br)
    limb_i = limb_i[unmasked_ind_f]
    br = br[unmasked_ind_f]
    err = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idx,limb_i0]
    err = err[::-1]
    err[np.where(err<1e-1)] = 1 # set the error to 1 if it is 0
    h = np.squeeze(tanalts[limb_i0])
    h = h[::-1]
    az = np.squeeze(anc.variables['ICON_ANCILLARY_FUVA_FOV_AZIMUTH_ANGLE'][idx,limb_i0,stripe])
    az = az[::-1]
    ze = np.squeeze(anc.variables['ICON_ANCILLARY_FUVA_FOV_ZENITH_ANGLE'][idx,limb_i0,stripe])
    ze = ze[::-1]
    tanlons = tanlons[limb_i0]
    tanlats = tanlats[limb_i0]
    tanlons = tanlons[::-1]
    tanlats = tanlats[::-1]
    br[br<0] = 0

    ver,Ne,h_centered,Sig_ver,Sig_Ne = FUV_Level_2_Density_Calculation(
        br,h,satlatlonalt,az,ze,
        Sig_Bright = np.diag(err**2), weight_resid=False,
        limb = limb,Spherical = Spherical, reg_method = reg_method,
        regu_order = regu_order, contribution =contribution,dn = dn,
        f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis,
        reg_param=reg_param
    )

    return Ne, h_centered



def index_finder(data, utc, type='l2'):
    str = 'ICON_L25_UTC_Time' if type=='l2' else 'ICON_L1_FUVA_SWP_Center_Times'
    dn = parser.parse(utc)
    dns = []
    for x in data.variables[str][:]:
        dns.append(parser.parse(x).replace(tzinfo=None))
    dns = np.array(dns)
    return np.argmin(abs(dns-dn))


def embed_epoch(file, date):
    df=pd.read_csv(file)
    df = df[df['date']==date].copy()
    df = df.reset_index(drop=True)
    file_l2 = lastfile(path_dir + 'l2/ICON_L2-5_FUV_Night_{}_v01r*'.format(date))
    l2 = netCDF4.Dataset(file_l2, mode='r')
    dates = [df['date'][i]+'/'+df['utc'][i] for i in range(len(df))]
    df['utc'] = dates
    df=df.drop(['date'], axis=1)
    dn = []
    for d in l2.variables['ICON_L25_UTC_Time']:
        dn.append(parser.parse(d))
    dn = np.array(dn)
    eps = [np.argmin(abs(parser.parse(df['utc'][i])-dn)) for i in range(len(df))]
    lats = [l2.variables['ICON_L25_O_Plus_Profile_Latitude'][ee,-1,ss] for ee,ss in zip(eps,df['stripe'])]
    lons = [l2.variables['ICON_L25_O_Plus_Profile_Longitude'][ee,-1,ss] for ee,ss in zip(eps,df['stripe'])]
    o_lats = l2.variables['ICON_L25_Observatory_Position_Latitude'][eps]
    o_lons = l2.variables['ICON_L25_Observatory_Position_Longitude'][eps]
    orbs = l2.variables['ICON_L25_Orbit_Number'][eps]
    df['l25_epoch'] = eps
    df['tangent150_lats'] = lats
    df['tangent150_lons'] = lons
    df['iconsc_lats'] = o_lats
    df['iconsc_lons'] = o_lons
    df['icon_orbits'] = orbs
    l2.close()
    return df

def lastfile(x):
    """
    Sort all the files complying with `x` alphabetically and return the last.
    """
    y = glob.glob(x)
    y.sort()
    assert len(y) > 0, 'No file found with the given name'
    return y[-1]

def loncorrect(lon):
    if lon.size==1:
        if lon > 180:
            lon -= 360
    else:
        lon[lon>180] -= 360
    return lon

def size_equalizer(x, ref_size, mode='center', val=0):
    """
    Crop or zero-pad a 2D array so that it has the size `ref_size`.
    Both cropping and zero-padding are done such that the symmetry of the
    input signal is preserved.
    Args:
        x (ndarray): array which will be cropped/zero-padded
        ref_size (list): list containing the desired size of the array [r1,r2]
        mode (str): ('center', 'topleft') where x should be placed when zero padding
    Returns:
        ndarray that is the cropper/zero-padded version of the input
    """
    if len(x.shape) == 2:
        if x.shape[0] > ref_size[0]:
            pad_left, pad_right = 0, 0
            crop_left = 0 if mode == 'topleft' else (x.shape[0] - ref_size[0] + 1) // 2
            crop_right = crop_left + ref_size[0]
        else:
            crop_left, crop_right = 0, x.shape[0]
            pad_right = ref_size[0] - x.shape[0] if mode == 'topleft' else (ref_size[0] - x.shape[0]) // 2
            pad_left = ref_size[0] - pad_right - x.shape[0]
        if x.shape[1] > ref_size[1]:
            pad_top, pad_bottom = 0, 0
            crop_top = 0 if mode == 'topleft' else (x.shape[1] - ref_size[1] + 1) // 2
            crop_bottom = crop_top + ref_size[1]
        else:
            crop_top, crop_bottom = 0, x.shape[1]
            pad_bottom = ref_size[1] - x.shape[1] if mode == 'topleft' else (ref_size[1] - x.shape[1]) // 2
            pad_top = ref_size[1] - pad_bottom - x.shape[1]

        # crop x
        cropped = x[crop_left:crop_right, crop_top:crop_bottom]
        # pad x
        padded = np.pad(
            cropped,
            ((pad_left, pad_right), (pad_top, pad_bottom)),
            mode='constant',
            constant_values=val
        )

        return padded

    elif len(x.shape) == 3:
        padded = np.zeros((x.shape[0], ref_size[0], ref_size[1]))
        for i in range(x.shape[0]):
            padded[i] = size_equalizer(x[i], ref_size)
        return padded

def liner(angle=np.arctan(15./8.), length=10):
    x1, y1 = length/2 * np.cos(angle), length/2 * np.sin(angle)
    x2, y2 = -x1, -y1
    c1, c2 = x1 + abs(x1), x2 + abs(x2)
    r1, r2 = -y1 + abs(y1), -y2 + abs(y2)
    r1, r2, c1, c2 = np.int(r1), np.int(r2), np.int(c1), np.int(c2)
    img = np.zeros((max(r1,r2)+1, max(c1,c2)+1))
    rr, cc, val = line_aa(r1, c1, r2, c2)
    img[rr, cc] = val
    # img = convolve2d(img, np.reshape(0.25*np.ones(4), (2,2)))
    return img

def orientation_finder_old(br):
    angle_list = np.linspace(0, np.pi, 180)
    angle_max = angle_list[0]
    energy_max = 0
    diff_ker = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    brd = correlate(br, diff_ker, mode='full')
    for angle in angle_list:
        feature = liner(angle=angle, length=20)
        feature_d = correlate(feature, diff_ker, mode='full')
        energy = np.linalg.norm(correlate(brd, feature_d, mode='full'))
        if energy > energy_max:
            angle_max = angle
            energy_max = energy
    return angle_max

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def orientation_finder(br):
    rad = min(br.shape) // 2
    brf = np.fft.fftn(br)
    brf[0,:] = 0
    brf[:,0] = 0
    brfsa = abs(np.fft.fftshift(brf))
    theta = np.linspace(0., 180., 180, endpoint=False)
    sinogram = radon(brfsa, theta=theta, circle=True)
    sum = np.sum(sinogram[rad-10:rad+10], axis=0)
    return np.argmax(sum) * np.pi / 180.

def sliding_min(x, winsize=5, mode='reflect'):
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


def get_br(date='2020-01-03', epoch=300, stripe=None, v=3, r=0, size='full', swapaxes=False, mode=2):
    i0 = 256 if size=='full' else 156
    path_dir = '/home/kamo/resources/iconfuv/nc_files/'
    file_l1 = path_dir + 'l1/ICON_L1_FUV_SWP_{}_v{:02d}r{:03d}.NC'.format(date,v,r)
    # file_anc = path_dir + 'l0/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)
    l1 = netCDF4.Dataset(file_l1, mode='r')
    # anc = netCDF4.Dataset(file_anc, mode='r')
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    mode_l1 = l1.variables['ICON_L1_FUV_Mode'][:]
    mode_night = (mode_l1 == mode).astype(np.int)
    nights = np.diff(mode_night, prepend=0)
    nights[nights==-1] = 0
    idxs = np.where(mode_l1==mode)[0][:]
    nights = np.cumsum(nights)[idxs]
    # orbits = anc.variables['ICON_ANCILLARY_FUV_ORBIT_NUMBER'][idxs]
    # orbit = orbits[epoch]
    night = nights[epoch]
    # orbit_ind = np.where(orbits==orbit)[0]
    night_ind = np.where(nights==night)[0]
    # print('Epoch:{}\nOrbit:{}\nOrbit Inds:[{}-{}]'.format(epoch, orbit, orbit_ind[0], orbit_ind[-1]))
    print('Epoch:{}\nNight:{}\nNight Inds:[{}-{}]'.format(epoch, night, night_ind[0], night_ind[-1]))
    if stripe is not None:
        br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][idxs[night_ind],256-i0:]
        br_err = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idxs[night_ind],256-i0:].filled(fill_value=0)
        br.fill_value = br.min()
        br = br.filled()
        if swapaxes is True:
            br = br.swapaxes(0,1)
            br_err = br_err.swapaxes(0,1)
        return br, br_err
    br = np.zeros((6, len(night_ind), i0))
    br_err = np.zeros((6, len(night_ind), i0))
    for i in range(6):
        br[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]][idxs[night_ind],256-i0:].filled(fill_value=0)
        br_err[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]][idxs[night_ind],256-i0:].filled(fill_value=0)
    if swapaxes is True:
        br = br.swapaxes(1,2)
        br_err = br_err.swapaxes(1,2)
    l1.close()
    return br, br_err


def get_br_nights(l1):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    mode_night = (mode == 2).astype(np.int)
    nights = np.diff(mode_night, prepend=0)
    nights[nights==-1] = 0
    idxs = np.where(mode==2)[0][:]
    nights = np.cumsum(nights)[idxs]
    brs = []
    brsc = []
    brs_err = []
    mask_arr = []
    for night in np.unique(nights):
        night_ind = np.where(nights==night)[0]
        br = np.zeros((6, len(night_ind), 256))
        brc = np.zeros((6, len(night_ind), 256))
        br_err = np.zeros((6, len(night_ind), 256))
        mask = np.zeros_like(br, dtype=np.bool)
        for i in range(6):
            tmp = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]][idxs[night_ind],:]
            tmp2 = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[i]][idxs[night_ind],:]
            br_err[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]][idxs[night_ind],:].filled(fill_value=0)
            mask[i] = tmp.mask
            br[i] = tmp.filled(fill_value=0)
            brc[i] = tmp2.filled(fill_value=0)
        brs.append(br)
        brsc.append(brc)
        brs_err.append(br_err)
        mask_arr.append(mask)
    return brs, brsc, brs_err, mask_arr, nights, idxs


def shiftappend(im, w=[5,5]):
    ss, aa, bb = im.shape
    r, c = w
    im2 = np.zeros((r*c*ss, aa+r-1, bb+c-1))
    im_large = size_equalizer(im, [aa+r-1, bb+c-1])
    sr = np.arange(r) - int((r-1)/2)
    sc = np.arange(c) - int((c-1)/2)
    sh = list(product(sr,sc))
    for i in range(r*c):
        im3 = np.roll(im_large, sh[i][0], axis=1)
        im2[ss*i: ss*(i+1)] = np.roll(im3, sh[i][1], axis=2)
    return size_equalizer(im2, [aa,bb])


def boxer(pt):
    pt2 = np.array([
        list(
            product(
                [pt[i, 0] - 1, pt[i, 0]], #pt[i, 0] + 1],
                [pt[i, 1] - 1, pt[i, 1]] #, pt[i, 1] + 1]
            )
        ) for i in range(pt.shape[0])
    ])
    return pt2.reshape((-1,2))


def labeler(br):
    br_masked = br.copy()
    mask = np.zeros_like(br)
    plt.figure(figsize=(8,8))
    plt.imshow(br_masked, vmax=300)
    plt.tight_layout()
    while True:
        pt = np.rint(np.asarray(plt.ginput(10, timeout=0))).astype(np.int)
        if pt.size is not 0:
            # pt = boxer(pt)
            br_masked[pt[:,1], pt[:,0]] = np.max(br_masked)
            mask[pt[:,1], pt[:,0]] = 1
        plt.imshow(br_masked, vmax=300)
        if plt.waitforbuttonpress():
            break
    return br_masked, mask

def labeler_init(br):
    br_masked = br.copy()
    plt.figure(figsize=(8,8))
    plt.imshow(br_masked, vmax=min(500, br.max()))
    plt.tight_layout()
    while True:
        pt = np.rint(np.asarray(plt.ginput(-1, timeout=1000))).astype(np.int)
        if len(pt) == 0:
            break
        if plt.waitforbuttonpress():
            break
    pt[:, 0] -= pt[:, 0].min()
    pt[:, 1] -= pt[:, 1].min()
    pt = np.flip(pt, axis=1)
    return pt

def feature_labeler(br, feature, mask=None, num=None, box=False):
    br_masked = br.copy()
    if box is True:
        feature = np.array([[0,0],[0,1],[1,0],[1,1]])
    if mask is not None:
        br_masked[mask==1] = min(np.max(br_masked), 500)
    else:
        mask = np.zeros_like(br)
    left = True
    if np.sum(feature, axis=1).min() > 0:
        left = False
    plt.figure(figsize=(8,8))
    plt.imshow(br_masked, vmax=min(500, br.max()))
    plt.tight_layout()
    while True:
        pt = np.rint(np.asarray(plt.ginput(2, timeout=1e3))).astype(np.int)
        plt.cla()
        plt.imshow(br_masked, vmax=min(500, br.max()))
        plt.show()
        if len(pt) == 0:
            break
        if left is False:
            pt[:, 0] -= feature[:, 1].max()
        pt = np.flip(pt, axis=1)
        ind = []
        for i in range(len(pt)):
            ind.append(feature + pt[i])
        ind = np.array(ind).reshape((-1,2))
        for i in ind:
            if (i[0] < br_masked.shape[0]) and (i[1] < br_masked.shape[1]):
                br_masked[i[0], i[1]] = np.max(br_masked)
                mask[i[0], i[1]] = 1
    # if num is not None:
    #     np.save('destarring/dataset/mask{}.npy'.format(num), mask)
    #     print('succesfully wrote mask{}.npy'.format(num))
    #     return 0
    # else:
    #     if len(os.listdir('./destarring/dataset')) == 0:
    #         num = 1
    #     else:
    #         num = 1
    #         for f in os.listdir('./destarring/dataset'):
    #             if f[0] == 'm':
    #                 fnum = int(f.split('.')[0].split('k')[-1])
    #                 num = fnum + 1 if fnum > num else num
    np.save('destarring/dataset/mask{}.npy'.format(num), mask)
