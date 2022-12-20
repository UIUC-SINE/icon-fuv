import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from iconfuv.uncertainty_calculator_v2r0 import uncertainty_calculator as uncertainty_calculator_v2r0
from iconfuv.uncertainty_calculator_v3r2 import uncertainty_calculator as uncertainty_calculator_v3r2
from iconfuv.uncertainty_calculator_v3r1 import uncertainty_calculator as uncertainty_calculator_v3r1
from iconfuv.uncertainty_calculator_v3r0 import uncertainty_calculator as uncertainty_calculator_v3r0
from scipy.signal import fftconvolve
from scipy.signal import convolve2d
from scipy.ndimage import median_filter, convolve1d

# df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/input_2022_013_swp.sav')
# df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/input_2022_014_swp.sav')
df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/uncertainty_results_sw_2022-244.sav')
# df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/uncertainty_results_lw_2022-089.sav')
# df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/uncertainty_results_lw_2022-060.sav')
# df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/uncertainty_input_2021-235.sav')
# df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/uncertainty_results_cronus_sw_2022-001.sav')
# df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/uncertainty_results_cronus_lw_2022-001.sav')
# df2 = readsav('/home/kamo/resources/iconfuv/nc_files/l1/example_sw_2020-01-30_v2.sav')
# df1 = df2
# sw_bkg_std = df['sw_bkg_std'] / 100.
# sw_bkg = df['sw_bkg'] / 100.
# my_raw1 = df1['my_raw']
# my_flatfield1 = df1['my_flatfield']
# index_day1 = df1['index_day']
# index_night1 = df1['index_night']
# index_saa1 = df1['index_saa']
my_raw2 = df2['my_raw']
my_flatfield2 = df2['my_flatfield']
# my_raw2 = df2['profile_raw']
# my_flatfield2 = df2['profile_flatfield']
index_day2 = df2['index_day']
index_night2 = df2['index_night']
index_saa2 = df2['index_saa']
# background_mean = df2['lw_bkg'].copy()
# background_std = df2['lw_bkg_std'].copy()

o1=[]; o2=[]; o3=[]
# o3=[]
for i in range(6):
    print(i)
    # # (unc1, gain_day1, gain_night1, bkg_mean1, bkg_std1, profile_flatfield1, profile_ff1,
    # #     signal1, signal_variance1) = uncertainty_calculator(
    # o1.append(uncertainty_calculator_v2r0(
    #     index_day=index_day1,
    #     index_night=index_night1,
    #     index_saa=index_saa1,
    #     profile_raw=my_raw1[:,:,i],
    #     profile_flatfield=my_flatfield1[:,:,i]
    #     # profile_raw=my_raw1[:,50:200,i],
    #     # profile_flatfield=my_flatfield1[:,50:200,i]
    # ))
    #
    # # (unc2, gain_day2, gain_night2, bkg_mean2, bkg_std2, profile_flatfield2, profile_ff2,
    # #     signal2, signal_variance2) = uncertainty_calculator(
    # o2.append(uncertainty_calculator_v3r1(
    #     index_day=index_day2,
    #     # index_night=index_night2,
    #     # index_saa=index_saa2,
    #     background_mean=background_mean[:,:,i]/100.,
    #     background_std=background_std[:,i]/100.,
    #     profile_raw=my_raw2[:,:,i],
    #     profile_flatfield=my_flatfield2[:,:,i]
    #     # profile_raw=my_raw2[:,50:200,i],
    #     # profile_flatfield=my_flatfield2[:,50:200,i]
    # ))

    o3.append(uncertainty_calculator_v3r2(
        index_day=index_day2,
        index_night=index_night2,
        index_saa=index_saa2,
        # background_mean=background_mean[:,:,i]/100.,
        # background_std=background_std[:,i]/100.,
        profile_raw=my_raw2[:,:,i],
        profile_flatfield=my_flatfield2[:,:,i]
        # profile_raw=my_raw2[:,50:200,i],
        # profile_flatfield=my_flatfield2[:,50:200,i]
    ))

# str = 5
#
# background_saa = o3[str][6][index_saa2]
#
# # calculate the mean and standard deviation of the background using saa data
# background_mean_altitude = np.median(background_saa, axis=0)
# background_std_altitude = np.std(background_saa, axis=0)
#
# # perform median filter based outlier detection to correct the standard
# # deviation for constant pixels
# bkgmed_mean = median_filter(background_mean_altitude, size=50)
# bkgmed_std = median_filter(background_std_altitude, size=50)
# indzero = np.where((bkgmed_mean-background_mean_altitude)/abs(bkgmed_mean) > 0.8)[0]
# indnonzero = np.where((bkgmed_mean-background_mean_altitude)/abs(bkgmed_mean) <= 0.8)[0]
# # indnonzero = indnonzero[50:-50]
#
# signalday0 = o3[str][-2][index_day2[:,np.newaxis], indnonzero[np.newaxis,:]]
# profile_flatfield0 = o3[str][5][index_day2[:,np.newaxis], indnonzero[np.newaxis,:]]
#
# # eliminate pixels and epochs containing values < -200 for >50% of time
# # along the other dimension
# indbl = np.where(signalday0 < -200)
# ind0i, ct0 = np.unique(indbl[0], return_counts=True)
# ind1i, ct1 = np.unique(indbl[1], return_counts=True)
# ind0o = ind0i[ct0>signalday0.shape[1]*0.5]
# ind1o = ind1i[ct1>signalday0.shape[0]*0.5]
# indx = np.array(list(set(np.arange(signalday0.shape[0])) - set(ind0o)))
# indy = np.array(list(set(np.arange(signalday0.shape[1])) - set(ind1o)))
# signalday11 = signalday0[indx[:,np.newaxis], indy[np.newaxis,:]]
# signalday1 = signalday11.copy()
# ind200 = signalday1 < -200
# signalday1[ind200] = np.nan
# signanmed = np.repeat(np.nanmedian(signalday1, axis=1)[:,np.newaxis],
#     signalday1.shape[1], axis=1)
# signalday1[ind200] = signanmed[ind200]
#
# profile_flatfield0 = profile_flatfield0[indx[:,np.newaxis], indy[np.newaxis,:]]
# # low pass filter dayside along the epoch dimension
# # this will be used as the mean of the signal
# signalday_lpsig0 = convolve2d(signalday1, 0.1*np.ones((10,1)), mode='same', boundary='symm')
# signalday_lpbkg0 = convolve2d(signalday1, 0.1*np.ones((1,10)), mode='same', boundary='symm')
#
# # transform all the 2d arrays into 1d arrays
# signalday = signalday1.flatten()
# signalday_lpsig = signalday_lpsig0.flatten()
# signalday_lpbkg = signalday_lpbkg0.flatten()
# profile_flatfield = profile_flatfield0.flatten()
# # make sure using signal larger than -200
# index_p2 = np.where(signalday11.flatten() > -200)[0]
# signalday = signalday[index_p2]
# signalday_lpsig = signalday_lpsig[index_p2]
# signalday_lpbkg = signalday_lpbkg[index_p2]
# profile_flatfield = profile_flatfield[index_p2]
#
# # find the indices corresponding to the top 10 percent of the low pass signal
# # (excluding the top 1 percent since there might be outliers)
# index1 = np.argsort(signalday_lpsig)[
#     int(0.9 * len(signalday_lpsig)) : int(0.99 * len(signalday_lpsig))
# ]
#
# # find the indices corresponding to the smallest 10 percent of the low pass signal
# # to characterize the background variance (more reliable than the saa data)
# index_bkg = np.argsort(signalday_lpbkg)[ : int(0.1 * len(signalday_lpbkg))]
#
# # make sure signal indices are larger than some small positive number, say 2
# index_p = np.where(signalday_lpsig[index1] > 2)[0]
#
# # update the indices
# index = index1[index_p]
#
# # compute the background fluctuations by detrending the corresponding data
# background_data = signalday[index_bkg] - signalday_lpbkg[index_bkg]
# background_data_var = np.std(background_data)**2
#
# # detrend the signal and normalize (not a complete normalization) such that
# # its sample variance is equal to the gain
# detrended_normalized = (
#     (signalday[index] - signalday_lpsig[index]) /
#     np.sqrt(abs(signalday_lpsig[index]) / profile_flatfield[index])
# )
#
# # compute the background noise contribution to the sample variance of the
# # detrended_normalized array
# background_contributor = np.mean(
#     background_data_var /
#     (abs(signalday_lpsig[index]) / profile_flatfield[index])
# )
#
# gain_day = np.std(detrended_normalized)**2 - background_contributor
