import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

df = readsav('/home/kamo/resources/iconfuv/documents/my_results.sav')
df2 = readsav('/home/kamo/resources/iconfuv/documents/bkg_example.sav')

calc_unc = df['uncertainties'] # true uncertainty
unc_clean = df['uncertainty_clean'] # wrong uncertainty after star removal
lw_bkg_std = df['lw_bkg_std']/100.
rayleigh_conv = df['rayleigh_conv']
gain_lw = df['gain_lw']
profile_clean = df['result2']
profile_flatfield = df['my_flatfield']

sigvar = gain_lw/profile_flatfield*(abs(profile_clean)/rayleigh_conv[:,np.newaxis,np.newaxis])#/100.)
uncer = np.sqrt(sigvar + lw_bkg_std**2)

calc_sig = (calc_unc**2 -lw_bkg_std**2) / gain_lw * profile_flatfield
cleaned_signal = profile_clean/rayleigh_conv[:,np.newaxis,np.newaxis]/100.

# calc_unc_sig = np.sqrt(calc_unc**2 - (lw_bkg_std)**2)

# %% plot
plt.figure()
# plt.plot(calc_unc_sig[0,:,3], label='True Uncertainty Signal')
plt.plot(calc_unc[0,:,3], label='True Uncertainty')
plt.plot(lw_bkg_std[:,3], label='Background Std')
plt.plot(uncer[0,:,3], label='Uncer Std')
plt.legend()
plt.show()

# %% plot_diff
plt.figure()
plt.plot(calc_sig[150,:,3], label='True Signal')
plt.plot(cleaned_signal[150,:,3], label='Cleaned Signal')
# plt.plot((calc_sig-cleaned_signal)[150,:,3], label='Diff')
plt.legend()
plt.show()
