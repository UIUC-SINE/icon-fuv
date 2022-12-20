import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from iconfuv.uncertainty_calculator_debugging import uncertainty_calculator
from scipy.signal import fftconvolve

df1 = readsav('/home/kamo/resources/icon-fuv/ncfiles/l1/input_2022_013_swp.sav')
df2 = readsav('/home/kamo/resources/icon-fuv/ncfiles/l1/input_2022_014_swp.sav')
# sw_bkg_std = df['sw_bkg_std'] / 100.
# sw_bkg = df['sw_bkg'] / 100.
my_raw1 = df1['my_raw']
my_flatfield1 = df1['my_flatfield']
index_day1 = df1['index_day']
index_night1 = df1['index_night']
index_saa1 = df1['index_saa']
my_raw2 = df2['my_raw']
my_flatfield2 = df2['my_flatfield']
index_day2 = df2['index_day']
index_night2 = df2['index_night']
index_saa2 = df2['index_saa']

i = 1
# (unc1, gain_day1, gain_night1, bkg_mean1, bkg_std1, profile_flatfield1, profile_ff1,
#     signal1, signal_variance1) = uncertainty_calculator(
# o1 = uncertainty_calculator(
#     index_day=index_day1,
#     index_night=index_night1,
#     index_saa=index_saa1,
#     # background_mean=sw_bkg[:,i],
#     # background_std=sw_bkg_std[:,i],
#     profile_raw=my_raw1[:,50:200,i],
#     profile_flatfield=my_flatfield1[:,50:200,i]
# )

# (unc2, gain_day2, gain_night2, bkg_mean2, bkg_std2, profile_flatfield2, profile_ff2,
#     signal2, signal_variance2) = uncertainty_calculator(
o2 = uncertainty_calculator(
    index_day=index_day2,
    index_night=index_night2,
    index_saa=index_saa2,
    # background_mean=sw_bkg[:,i],
    # background_std=sw_bkg_std[:,i],
    profile_raw=my_raw2[:,50:200,i],
    profile_flatfield=my_flatfield2[:,50:200,i]
)

signalday = o2[-2][index_day2]
background_std = o2[4][index_day2]
profile_flatfield = o2[5][index_day2]

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
