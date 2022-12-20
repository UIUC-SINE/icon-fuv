# Ulas Kamaci - 2022-04-02
# uncertainty_calculation v3.0
# uncertainty_calculator is the main function where gain_calculator_day and
# night are called within that.
# History:
# v2.0 - 2020-06-01
# v3.0 - 2022-04-02: In gain_day calculation, use background std calculated from
# low intensity day indices instead of saa indices. Set the minimum flatfield
# counts to 100. Correct the outlier pixels in the calculated background
# standard deviations from saa data. Add new outlier elimination steps in the
# gain calculations. Ulas Kamaci.

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, convolve1d

def gain_calculator_day(signalday,profile_flatfield):
    """
    Computes the average daytime gain from one day of daytime data.

    Args:
        signalday (ndarray): array of daytime flatfielded profiles
        profile_flatfield (ndarray): array of flatfielding coefficients for
            the science pixels

    Returns:
        gain_day (float): computed gain value
    """
    # eliminate pixels and epochs containing values < -200 for >50% of time
    # along the other dimension
    indbl = np.where(signalday < -200)
    ind0i, ct0 = np.unique(indbl[0], return_counts=True)
    ind1i, ct1 = np.unique(indbl[1], return_counts=True)
    ind0o = ind0i[ct0>signalday.shape[1]*0.5]
    ind1o = ind1i[ct1>signalday.shape[0]*0.5]
    indx = np.array(list(set(np.arange(signalday.shape[0])) - set(ind0o)))
    indy = np.array(list(set(np.arange(signalday.shape[1])) - set(ind1o)))
    signalday1 = signalday[indx[:,np.newaxis], indy[np.newaxis,:]]
    profile_flatfield = profile_flatfield[indx[:,np.newaxis], indy[np.newaxis,:]]
    # low pass filter dayside along the epoch dimension
    # this will be used as the mean of the signal
    signalday_lp = convolve2d(signalday1, 0.1*np.ones((10,1)), mode='same', boundary='symm')

    # transform all the 2d arrays into 1d arrays
    signalday = signalday1.flatten()
    signalday_lp = signalday_lp.flatten()
    profile_flatfield = profile_flatfield.flatten()
    # make sure using signal larger than -200
    index_p2 = np.where(signalday > -200)[0]
    signalday = signalday[index_p2]
    signalday_lp = signalday_lp[index_p2]
    profile_flatfield = profile_flatfield[index_p2]

    # find the indices corresponding to the top 10 percent of the low pass signal
    # (excluding the top 1 percent since there might be outliers)
    index1 = np.argsort(signalday_lp)[
        int(0.9 * len(signalday_lp)) : int(0.99 * len(signalday_lp))
    ]

    # find the indices corresponding to the smallest 10 percent of the low pass signal
    # to characterize the background variance (more reliable than the saa data)
    index_bkg = np.argsort(signalday_lp)[ : int(0.1 * len(signalday_lp))]

    # make sure signal indices are larger than some small positive number, say 2
    index_p = np.where(signalday_lp[index1] > 2)[0]

    # update the indices
    index = index1[index_p]

    # compute the background fluctuations by detrending the corresponding data
    background_data = signalday[index_bkg] - signalday_lp[index_bkg]
    background_data_var = np.std(background_data)**2

    # detrend the signal and normalize (not a complete normalization) such that
    # its sample variance is equal to the gain
    detrended_normalized = (
        (signalday[index] - signalday_lp[index]) /
        np.sqrt(abs(signalday_lp[index]) / profile_flatfield[index])
    )

    # compute the background noise contribution to the sample variance of the
    # detrended_normalized array
    background_contributor = np.mean(
        background_data_var /
        (abs(signalday_lp[index]) / profile_flatfield[index])
    )

    gain_day = np.std(detrended_normalized)**2 - background_contributor

    return gain_day

def gain_calculator_night(signalnight,profile_flatfield):
    """
    Computes the average nighttime gain from one day of nighttime data.

    Args:
        signalnight (ndarray): array of nighttime flatfielded profiles
        profile_flatfield (ndarray): array of flatfielding coefficients for
            the science pixels

    Returns:
        gain_night (float): computed gain value
    """
    sigind = np.argsort(np.sum(signalnight,axis=1))[int(0.1*len(signalnight)):]
    ind_nonzero = np.where(profile_flatfield[sigind[0]]!=1)[0]
    br = signalnight[sigind[:,np.newaxis], ind_nonzero[np.newaxis,:]]

    # eliminate pixels and epochs containing values < -200 for >50% of time
    # along the other dimension
    indbl = np.where(br < -200)
    ind0i, ct0 = np.unique(indbl[0], return_counts=True)
    ind1i, ct1 = np.unique(indbl[1], return_counts=True)
    ind0o = ind0i[ct0>br.shape[1]*0.5]
    ind1o = ind1i[ct1>br.shape[0]*0.5]
    indx = np.array(list(set(np.arange(br.shape[0])) - set(ind0o)))
    indy = np.array(list(set(np.arange(br.shape[1])) - set(ind1o)))
    br = br[indx[:,np.newaxis], indy[np.newaxis,:]]

    # run outlier elimination to get rid of the bad data
    br_med = median_filter(br, size=(1,15))
    br_diff = br - br_med
    filt = abs(br_diff) > 30
    br_filt = br.copy()
    br_filt[filt==1] = br_med[filt==1]

    # run a more extreme outlier elimination with larger threshold and bigger window
    br_med0 = median_filter(br, size=(1,40))
    br_diff0 = br - br_med0
    filt0 = abs(br_diff0) > 200
    br_filt[filt0==1] = br_med0[filt0==1]

    # run hot pixel correction
    brfiltmean = np.mean(br_filt, axis=0)
    diff = brfiltmean - convolve1d(brfiltmean, 0.05*np.ones(20), mode='reflect')
    br_filt = br_filt - diff

    # low pass filter nightside along the altitude dimension
    # this will be used as the mean of the signal
    signalnight_lp = convolve2d(br_filt, 0.1*np.ones((1,10)), mode='same', boundary='symm')

    # transform all the 2d arrays into 1d arrays
    signalnight = br_filt.flatten()
    signalnight_lp = signalnight_lp.flatten()
    profile_flatfield = profile_flatfield[
        sigind[:,np.newaxis], ind_nonzero[np.newaxis,:]
    ][indx[:,np.newaxis], indy[np.newaxis,:]].flatten()
    # make sure using signal larger than -200
    index_p2 = np.where(signalnight > -200)[0]
    signalnight = signalnight[index_p2]
    signalnight_lp = signalnight_lp[index_p2]
    profile_flatfield = profile_flatfield[index_p2]

    # find the indices corresponding to the smallest 10 percent of the low pass signal
    # to characterize the background variance (more reliable than the saa data)
    index_bkg = np.argsort(signalnight_lp)[ : int(0.1 * len(signalnight_lp))]

    # compute the background fluctuations by detrending the corresponding data
    background_data = signalnight[index_bkg] - signalnight_lp[index_bkg]
    background_data_var = np.std(background_data)**2

    # find the indices corresponding to the lowest 95 percent of the low pass signal
    # (excluding the top 5 percent since there might be stars)
    index1 = np.argsort(signalnight_lp)[ : int(0.95 * len(signalnight_lp))]

    # make sure these indices are larger than some small positive number, say 2
    # so that we get rid of the negative values
    index_p = np.where(signalnight_lp[index1] > 2)[0]

    # update the indices with the positivity restriction
    index = index1[index_p]

    # detrend the signal and normalize (not a complete normalization) such that
    # its sample variance is equal to the gain
    detrended_normalized = (
        (signalnight[index] - signalnight_lp[index]) /
        np.sqrt(abs(signalnight_lp[index]) / profile_flatfield[index])
    )

    # compute the background noise contribution to the sample variance of the
    # detrended_normalized array
    background_contributor = np.mean(
        background_data_var /
        (abs(signalnight_lp[index]) / profile_flatfield[index])
    )

    gain_night = np.std(detrended_normalized)**2 - background_contributor

    return gain_night

def uncertainty_calculator(
        profile_raw,
        index_day,
        profile_flatfield,
        background_mean=None,
        background_std=None,
        index_saa=None,
        index_night=None
):
    """
    Computes the 1-sigma uncertainties of the profiles given 1 day of data for
    one stripe. Requires the day indices to be provided, night indices are
    optional. A separate gain calculation is performed for night data.
    Computed gain(s) are also returned together with the uncertainty profiles.

    Args:
        profile_raw (ndarray): array of raw profiles
        index_day (ndarray): array of day indices
        profile_flatfield (ndarray): array of flatfielding coefficients for
            the science pixels
        background_mean (ndarray): mean values of the flatfielded backgrounds
            If a single altitude profile is given, it will be assumed constant
            along all the day. optional
        background_std (ndarray): standard deviation values of the flatfielded
            backgrounds. If a single altitude profile is given, it will be
            assumed constant along all the day. optional
        index_saa (ndarray): array of saa indices. optional
        index_night (ndarray): array of night indices. optional

    Returns:
        uncertainty_profile (ndarray): array of uncertainties, same shape as
            profile_raw
        gain_day (float): computed gain value from day profiles
        gain_night (float): computed gain value from night profiles. optional
        background_mean (ndarray): computed background mean from saa data.
            optional
        background_std (ndarray): computed background standard deviation from
            saa data. optional
    """
    # copy the arrays first so that modifications inside the function doesnt
    # affect the original arrays
    profile_flatfield = profile_flatfield.copy()
    profile_raw = profile_raw.copy()

    # if there are NaN values in the profiles, set them to zero
    profile_flatfield[np.isnan(profile_flatfield)] = 0
    profile_raw[np.isnan(profile_raw)] = 0

    # set the small values to 100 to prevent division errors
    profile_flatfield[profile_flatfield < 100] = 100

    # perform flatfielding
    profile_ff = profile_raw / profile_flatfield

    if background_mean is not None:
        calculated_background = False
        # if a single altitude profile is provided, copy it along the time indices
        if len(np.squeeze(background_mean).shape) == 1:
            background_mean = np.repeat(
                background_mean[np.newaxis,:], len(profile_ff), axis=0
            )
            background_std = np.repeat(
                background_std[np.newaxis,:], len(profile_ff), axis=0
            )

    else:
        calculated_background = True
        # take the saa indices to calculate the background parameters
        background_saa = profile_ff[index_saa]

        # calculate the mean and standard deviation of the background using saa data
        background_mean_altitude = np.median(background_saa, axis=0)
        background_std_altitude = np.std(background_saa, axis=0)

        # perform median filter based outlier detection to correct the standard
        # deviation for constant pixels
        bkgmed_mean = median_filter(background_mean_altitude, size=50)
        bkgmed_std = median_filter(background_std_altitude, size=50)
        indzero = np.where((bkgmed_mean-background_mean_altitude)/abs(bkgmed_mean) > 0.8)[0]
        indnonzero = np.where((bkgmed_mean-background_mean_altitude)/abs(bkgmed_mean) <= 0.8)[0]

        background_std_altitude[indzero] = bkgmed_std[indzero]

        # replicate the mean and the deviation along the time axis (since we assume
        # they don't change over time)
        background_mean = np.repeat(
            background_mean_altitude[np.newaxis,:], len(profile_ff), axis=0
        )
        background_std = np.repeat(
            background_std_altitude[np.newaxis,:], len(profile_ff), axis=0
        )

    # subtract the mean background from the total signal to estimate the actual signal
    signal = profile_ff - background_mean

    # compute the gain for the day
    gain_day = gain_calculator_day(
        signal[index_day[:,np.newaxis], indnonzero[np.newaxis,:]],
        profile_flatfield[index_day[:,np.newaxis], indnonzero[np.newaxis,:]]
    )

    signal_variance = background_std**2
    signal_variance[index_day] = (
        (gain_day / profile_flatfield[index_day]) *
        abs(signal[index_day])
    )

    if index_night is not None:
        # compute the gain for the night
        if len(indnonzero)>200:
            # eliminate the top and bottom 50 pixels since sometimes cause problems
            indnonzero = indnonzero[50:-50]
        gain_night = gain_calculator_night(
            signal[index_night[:,np.newaxis], indnonzero[np.newaxis,:]],
            profile_flatfield[index_night[:,np.newaxis], indnonzero[np.newaxis,:]]
        )

        # make sure that the calculated night gain is larger than day (because
        # of higher voltage) . otherwise, equate it to the day gain
        gain_night = max(gain_day, gain_night)

        signal_variance[index_night] = (
            (gain_night / profile_flatfield[index_night]) *
            abs(signal[index_night])
        )

        uncertainty_profile = np.sqrt(signal_variance + background_std**2)
        if calculated_background is True:
            return uncertainty_profile, gain_day, gain_night, background_mean, background_std
        else:
            return uncertainty_profile, gain_day, gain_night

    else:
        uncertainty_profile = np.sqrt(signal_variance + background_std**2)
        if calculated_background is True:
            return uncertainty_profile, gain_day, background_mean, background_std
        else:
            return uncertainty_profile, gain_day
