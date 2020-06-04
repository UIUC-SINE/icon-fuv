import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from iconfuv.uncertainty_calculator import uncertainty_calculator

df = readsav('/home/kamo/resources/iconfuv/documents/example_2020-001.sav')
sw_bkg_std = df['sw_bkg_std'] / 100.
sw_bkg = df['sw_bkg'] / 100.
my_raw = df['my_raw']
my_flatfield = df['my_flatfield']
index_day = df['index_day']
index_night = df['index_night']

for i in range(6):
    unc, gain_day, gain_night = uncertainty_calculator(
        index_day=index_day,
        index_night=index_night,
        background_mean=sw_bkg[:,i],
        background_std=sw_bkg_std[:,i],
        profile_raw=my_raw[:,:,i],
        profile_flatfield=my_flatfield[:,:,i]
    )

    print('gain_day: {}\ngain_night: {}'.format(gain_day,gain_night))
