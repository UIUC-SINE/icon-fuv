# Ulas Kamaci - 2020-06-04
# artifact_removal v1.3
# given a day of brightness profiles in Rayleighs, performs star removal and hot
# pixel correction on the profiles.

import numpy as np
from scipy.ndimage import median_filter

def sliding_min(x, winsize=5, mode='reflect'):
    '''
    Applies moving minimum filter with the window size of `winsize`.

    Args:
        x (ndarray): 1d or 2d ndarray. if 2d, function is applied recursively on
            the second dimension
        winsize (ndarray): integer specifying the window size
        mode (ndarray): string specifying how to treat the array boundary
            (goes into np.pad)

    Returns:
        out (ndarray): filtered array with the same dimension as the input
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

def medfilt3d(br, threshold, win_size=(5,10,10), mode=2):
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

    Args:
        br (ndarray): (6,orbit_ind,256) array of brightness profiles in
            Rayleighs for a particular orbit
        threshold (float): the threshold value above which deviation from the
            median results in classifying the pixel as a star
        win_size (tuple): 3d tuple specifying the window size which is centered
            at the pixel of interest
        mode (int): integer specifying the operation mode. {1:day, 2:night}

    Returns:
        br_corrected - star removed brightness profiles
    '''
    br = br.copy()
    br_med = median_filter(br, size=win_size)
    br_diff = br - br_med
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
    brx = np.mean(br, axis=1)
    brx_min = sliding_min(brx, winsize=5)
    br_cor = br.copy()
    for i in range(6):
        diff = brx[i] - brx_min[i]
        br_cor[i] -= diff
    return br_cor

def artifact_removal_orbit(br, mode):
    '''
    Function that performs star removal and additionally hot pixel correction
    for nighttime. It operates on 3d (stripe,epoch,altitude) orbit images
    individually.

    Args:
        br (ndarray): (6,orbit_ind,256) array of brightness profiles in
            Rayleighs for a particular orbit
        mode (int): integer specifying the mode. {1: day, 2: night}

    Returns:
        br_corrected (ndarray): artifact removed profiles
    '''
    if mode == 1:
        # apply star removal
        _, br_corrected = medfilt3d(br, threshold=100, win_size=(3,3,10), mode=mode)
    elif mode == 2:
        # apply star removal
        _, br_corrected = medfilt3d(br, threshold=50, mode=mode)
        # apply hot pixel correction
        br_corrected = hot_pixel_correction(br_corrected)
        # apply star removal again since it can detect the stars better now
        _, br_corrected = medfilt3d(br_corrected, threshold=10, mode=mode)

    return br_corrected

def artifact_removal(profiles, channel, fuv_mode):
    '''
    Performs star removal and hot pixel correction. Assumes the array units are
    in Rayleighs.

    Args:
        profiles (ndarray): 1 day of all stripe profiles in Rayleighs with dimension
            [6,256,epoch]
        channel (int): integer specifying the FUV channel (1:SW, 2:LW)
        fuv_mode (ndarray): ICON_L1_FUV_Mode variable where 1:day, 2:night

    Returns:
        profiles_cleaned (ndarray): artifact removed profiles, same dimension
            as profiles
    '''
    # swap the epoch and altitude axes of the profiles
    profiles = np.swapaxes(profiles, 1, 2)

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
        mode_orbit = (fuv_mode == mode).astype(np.int)
        orbits = np.diff(mode_orbit, prepend=0)
        orbits[orbits==-1] = 0
        idxs = np.where(fuv_mode==mode)[0][:]
        orbits = np.cumsum(orbits)[idxs]
        for orbit in np.unique(orbits):
            orbit_ind = np.where(orbits==orbit)[0]
            br_corrected = artifact_removal_orbit(
                br=profiles[:,idxs[orbit_ind],:].filled(fill_value=0),
                mode=mode
            )
            profiles_cleaned[:, idxs[orbit_ind],:] = br_corrected

    profiles_cleaned.mask = profiles.mask
    profiles_cleaned = profiles_cleaned.filled(fill_value=np.nan)

    # swap the epoch and altitude axes back again to match the input dimension
    profiles_cleaned = np.swapaxes(profiles_cleaned, 1, 2)
    return profiles_cleaned
