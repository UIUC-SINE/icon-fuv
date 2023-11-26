# 2023-03-29
# Ulas Kamaci

import numpy as np
from iconfuv.misc import lastfile
import netCDF4
from shutil import copyfile
from dateutil import parser
import datetime
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

date = '2022-02-26'
orbit = 11
stripe = 3
ep = 20

file_l1 = lastfile(path_dir+'l1/ICON_L1_FUV_SWP_{}_v0*'.format(date))
l1 = netCDF4.Dataset(file_l1, mode='r')

mirror_dir = ['M9','M6','M3','P0','P3','P6']
mode_l1 = l1.variables['ICON_L1_FUV_Mode'][:]
mode_night = (mode_l1 == 2).astype(np.int)
nights = np.diff(mode_night, prepend=0)
nights[nights==-1] = 0
idxs = np.where(mode_l1==2)[0][:]
nights = np.cumsum(nights)[idxs]
idx = np.where(nights==orbit)[0]
br = np.zeros((6,len(idx),256))

for i in range(6):
    br[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_{}_CLEAN'.format(mirror_dir[i])][idxs[idx],:]

# %% plot
# fig, ax = plt.subplots(6,1, figsize=[6.4,8.6])
# for i in range(6):
#     ax[i].imshow(br[i].T, aspect='auto', origin='lower', cmap='jet')
#     ax[i].set_title('Stripe {}'.format(i))
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(3,1, sharex=True)
ax[0].imshow(br[2].T, aspect='auto', origin='lower', cmap='jet')
ax[0].grid(which='both', axis='both')
ax[0].set_title('FUV L1 Night - 2022-02-26 - Stripe:3 - Orbit:13000')
ax[0].set_ylabel('Brightness [R]')
ax[1].plot(br[2].mean(axis=1), label='Avg')
ax[1].grid(which='both', axis='both')
ax[1].set_ylabel('Brightness [R]')
ax[1].legend()
ax[2].plot(br[2].max(axis=1), color='r', label='Max')
ax[2].grid(which='both', axis='both')
ax[2].set_ylabel('Brightness [R]')
ax[2].set_xlabel('Profile Number')
ax[2].legend()
plt.show()
