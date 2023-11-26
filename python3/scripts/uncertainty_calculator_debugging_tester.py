import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from iconfuv.uncertainty_calculator_debugging import uncertainty_calculator

df = readsav('/home/kamo/resources/icon-fuv/ncfiles/l1/uncertainty_results_cronus_sw_2022-001.sav')
df2 = readsav('/home/kamo/resources/icon-fuv/ncfiles/l1/uncertainty_results_Shrike_sw_2022-001.sav')
# sw_bkg_std = df['sw_bkg_std'] / 100.
# sw_bkg = df['sw_bkg'] / 100.
my_raw_c = df['my_raw']
my_flatfield_c = df['my_flatfield']
index_day_c = df['index_day']
index_night_c = df['index_night']
index_saa_c = df['index_saa']
my_raw_s = df2['my_raw_shrike']
my_flatfield_s = df2['my_flatfield_shrike']
index_day_s = df2['index_day_shrike']
index_night_s = df2['index_night_shrike']
index_saa_s = df2['index_saa_shrike']

i = 5
(unc_uc, gain_day_uc, gain_night_uc, bkg_mean_uc, bkg_std_uc, profile_flatfield_uc, profile_ff_uc,
    signal_uc, signal_variance_uc) = uncertainty_calculator(
    index_day=index_day_c,
    index_night=index_night_c,
    index_saa=index_saa_c,
    # background_mean=sw_bkg[:,i],
    # background_std=sw_bkg_std[:,i],
    profile_raw=my_raw_c[:,:,i],
    profile_flatfield=my_flatfield_c[:,:,i]
)
(unc_us, gain_day_us, gain_night_us, bkg_mean_us, bkg_std_us, profile_flatfield_us, profile_ff_us,
    signal_us, signal_variance_us) = uncertainty_calculator(
    index_day=index_day_s,
    index_night=index_night_s,
    index_saa=index_saa_s,
    # background_mean=sw_bkg[:,i],
    # background_std=sw_bkg_std[:,i],
    profile_raw=my_raw_s[:,:,i],
    profile_flatfield=my_flatfield_s[:,:,i]
)

(unc_c, gain_day_c, gain_night_c, bkg_mean_c, bkg_std_c, profile_flatfield_c, profile_ff_c,
    signal_c, signal_variance_c) = df[f'result_stripe_cronus_{i}'].transpose(2,0,1)

(unc_s, gain_day_s, gain_night_s, bkg_mean_s, bkg_std_s, profile_flatfield_s, profile_ff_s,
    signal_s, signal_variance_s) = df2[f'result_stripe_shrike_{i}'].transpose(2,0,1)
