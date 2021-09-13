# Ulas Kamaci - 2021-05-03
# artifact_removal_v2.0
# given a day of brightness profiles in Rayleighs, performs star removal and hot
# pixel correction on the profiles. The star removal module is a neural network.

import numpy as np
from scipy.ndimage import median_filter, convolve1d
from scipy.signal import convolve2d
from keras.models import load_model
from keras import backend

def remove_stars(b, path_to_model):
	'''
	Function that performs star removal for nighttime data using a neural network.
	This function operates on 3d (stripe, epoch, altitude) orbit images
	individually.

	Args:
		b (ndarray)				: (stripe, epoch, altitude) array of brightness profiles in
									Rayleighs for a particular orbit.
									Hot and cold pixel correctiond have to be applied to the profiles
									before to call this function for a good performance.
		path_to_model (string)	: path to the neural network model trained to remove stars from b.
								The path can be a hdf5 file or a directory.

	Returns:
		b_corrected (ndarray): star removed profiles
	'''
	model = load_model(path_to_model)
	# swap epoch and altitude axes to match model input format
	b = np.transpose(b, (1,2,0))

	# keras model needs an additional dimension channel
	# (epoch, altitude, stripe, color)
	b_corrected = model.predict(b[:,:,:,None])

	#Get rid of the extra dimension channel
	b_corrected = b_corrected[:,:,:,0]

	#Return dimension to the original format
	b_corrected = np.transpose(b_corrected, (2,0,1) )

	return(b_corrected)

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
    brmed = np.nanmedian(br, axis=0)
    brmean = np.nanmean(br, axis=0)
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
    brx = np.mean(br, axis=1)
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

def artifact_removal_orbit(br, mode, path_to_model):
    '''
    Function that performs star removal and additionally hot pixel correction
    for nighttime. It operates on 3d (stripe,epoch,altitude) orbit images
    individually.

    Args:
        br (ndarray): (6,orbit_ind,256) array of brightness profiles in
            Rayleighs for a particular orbit
        mode (int): integer specifying the mode. {1: day, 2: night}
        path_to_model (str): path to the neural network for star removal

    Returns:
        br_corrected (ndarray): artifact removed profiles
    '''
    if mode == 1:
        # apply star removal
        _, br_corrected = medfilt3d(br, threshold=100, win_size=(3,3,10), mode=mode)
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
        br2 = remove_stars(br2, path_to_model)

        # add back the untouched dayside part back to the orbit
        br_corrected = np.append(br2, br[:,index:,:], axis=1)

    return br_corrected

def artifact_removal(profiles, channel, fuv_mode, path_to_model):
    '''
    Performs hot pixel correction on one day of profiles.
    It first parses the orbits using the fuv_mode variable, then processes
    each orbit individually. Assumes the array units are in Rayleighs.

    Args:
        profiles (ndarray): 1 day of all stripe profiles in Rayleighs with dimension
            [6,256,epoch]
        channel (int): integer specifying the FUV channel (1:SW, 2:LW)
        fuv_mode (ndarray): ICON_L1_FUV_Mode variable where 1:day, 2:night
        path_to_model (str): path to the neural network for star removal

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
    profiles_filled = profiles.copy()

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
        # Fill the NaN values at the top and bottom portions with the median
        # value of the 5x6 [alt x stripe] windows at the edges. Repeat for each
        # Epoch.
        filltop = np.nanmedian(profiles[:,:,:5], axis=(0,2))
        fillbot = np.nanmedian(profiles[:,:,-5:], axis=(0,2))
        profiles_filled[:,:,:5] = profiles[:,:,:5].filled(fill_value=filltop[None,:,None])
        profiles_filled[:,:,-5:] = profiles[:,:,-5:].filled(fill_value=fillbot[None,:,None])
        profiles_filled[:,:,5:-5] = profiles[:,:,5:-5].filled(fill_value=0)
        for orbit in np.unique(orbits):
            orbit_ind = np.where(orbits==orbit)[0]
            br_corrected = artifact_removal_orbit(
                br=profiles_filled[:,idxs[orbit_ind],:],
                mode=mode, path_to_model=path_to_model
            )
            profiles_cleaned[:, idxs[orbit_ind],:] = br_corrected

    profiles_cleaned.mask = profiles.mask
    profiles_cleaned = profiles_cleaned.filled(fill_value=np.nan)

    # swap the epoch and altitude axes back again to match the input dimension
    profiles_cleaned = np.swapaxes(profiles_cleaned, 1, 2)
    return profiles_cleaned
