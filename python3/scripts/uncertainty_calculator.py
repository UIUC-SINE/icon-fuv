# Ulas Kamaci - 2020-05-23
# uncertainty_calculation v1.1
# uncertainty_calculator is the main function where gain_calculator_day and
# night are called within that.

import numpy as np
from scipy.signal import fftconvolve

def gain_calculator_day(signalday,background_std,profile_flatfield):
    """
    Computes the average daytime gain from one day of daytime data.

    Args:
        signalday (ndarray): array of daytime flatfielded profiles
        background_std (ndarray): array of background standard deviation
            profiles (same dimension as signalday)
        profile_flatfield (ndarray): array of flatfielding coefficients for
            the science pixels

    Returns:
        gain_day (float): computed gain value
    """
    # low pass filter dayside along the epoch dimension
    # this will be used as the mean of the signal
    signalday_lp = fftconvolve(signalday, 0.1*np.ones((10,1)), mode='same')

    # transform all the 2d arrays into 1d arrays
    signalday = signalday.flatten()
    signalday_lp = signalday_lp.flatten()
    background_std = background_std.flatten()
    profile_flatfield = profile_flatfield.flatten()

    # find the indices corresponding to the top 10 percent of the low pass signal
    # (excluding the top 1 percent since there might be outliers)
    index1 = np.argsort(signalday_lp)[
        int(0.9 * len(signalday_lp)) : int(0.99 * len(signalday_lp))
    ]

    # make sure these indices are larger than some small positive number, say 2
    index_p = np.where(signalday_lp[index1] > 2)[0]

    # update the indices with the positivity restriction
    index = index1[index_p]

    # detrend the signal and normalize (not a complete normalization) such that
    # its sample variance is equal to the gain
    detrended_normalized = (
        (signalday[index] - signalday_lp[index]) /
        np.sqrt(abs(signalday_lp[index]) / profile_flatfield[index])
    )

    # compute the background noise contribution to the sample variance of the
    # detrended_normalized array
    background_contributor = np.mean(
        (background_std**2)[index] /
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
    # low pass filter nightside along the altitude dimension
    # this will be used as the mean of the signal
    signalnight_lp = fftconvolve(signalnight, 0.1*np.ones((1,10)), mode='same')

    # transform all the 2d arrays into 1d arrays
    signalnight = signalnight.flatten()
    signalnight_lp = signalnight_lp.flatten()
    profile_flatfield = profile_flatfield.flatten()

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
    """
    # perform flatfielding
    profile_ff = profile_raw / profile_flatfield

    if background_mean is not None:
        # if a single altitude profile is provided, copy it along the time indices
        if len(np.squeeze(background_mean).shape) == 1:
            background_mean = np.repeat(
                background_mean[np.newaxis,:], len(profile_ff), axis=0
            )
            background_std = np.repeat(
                background_std[np.newaxis,:], len(profile_ff), axis=0
            )

    else:
        # take the saa indices to calculate the background parameters
        background_saa = profile_ff[index_saa]

        # calculate the mean and standard deviation of the background using saa data
        background_mean_altitude = np.median(background_saa, axis=0)
        background_std_altitude = np.std(background_saa, axis=0)

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
        signal[index_day],
        background_std[index_day],
        profile_flatfield[index_day]
    )

    signal_variance = background_std**2
    signal_variance[index_day] = (
        (gain_day / profile_flatfield[index_day]) *
        abs(signal[index_day])
    )

    if index_night is not None:
        # compute the gain for the night
        gain_night = gain_calculator_night(
            signal[index_night],
            profile_flatfield[index_night]
        )

        # make sure that the calculated night gain is larger than day (because
        # of higher voltage) . otherwise, equate it to the day gain
        gain_night = max(gain_day, gain_night)

        signal_variance[index_night] = (
            (gain_night / profile_flatfield[index_night]) *
            abs(signal[index_night])
        )

        uncertainty_profile = np.sqrt(signal_variance + background_std**2)
        return uncertainty_profile, gain_day, gain_night

    else:
        uncertainty_profile = np.sqrt(signal_variance + background_std**2)
        return uncertainty_profile, gain_day
