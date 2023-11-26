# 2023-03-27
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

def averager(date, winsize):
    file_l2 = lastfile(path_dir+'l2/2021/ICON_L2-5_FUV_Night_{}_v0*'.format(date))
    l2 = netCDF4.Dataset(file_l2, mode='r')

    op = l2.variables['ICON_L25_O_Plus_Density'][:]
    alt = l2.variables['ICON_L25_O_Plus_Profile_Altitude'][:]
    on = l2.variables['ICON_L25_Orbit_Number'][:]
    op2 = op.copy()

    # perform running averager on each orbit separately
    for i in np.unique(on):
        op2[on==i] = convolve(op[on==i], np.ones((winsize,1,1))/winsize, mode='nearest')

    return op, op2, op2.mean(axis=2), on, alt

winsize = [5,15,25]
# date = '2022-03-31'
date = '2021-10-06'
orbit = 13
stripe = 3
ep = 20

op2l = [] ; op2ml = []

for i in range(len(winsize)):
    op, op2, op2m, on, alt = averager(date=date, winsize=winsize[i])
    op2l.append(op2)
    op2ml.append(op2m)

ons = np.unique(on)
idx = on==ons[orbit]
altt = alt[idx][0][:,stripe]

# %% plot
fig, ax = plt.subplots(6,1, figsize=[6.4,8.6])
for i in range(6):
    ax[i].imshow(op[idx,:,i].T, aspect='auto')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(len(winsize)+1,1, figsize=[6.4,6.6])
ax[0].imshow(op[idx,:,stripe].T, aspect='auto')
for i in range(len(winsize)):
    ax[i+1].imshow(op2l[i][idx,:,stripe].T, aspect='auto')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(len(winsize)+1,1, figsize=[6.4,6.6])
ax[0].imshow(op[idx].mean(axis=2).T, aspect='auto')
for i in range(len(winsize)):
    ax[i+1].imshow(op2ml[i][idx,:].T, aspect='auto')
plt.tight_layout()
plt.show()

plt.figure(figsize=[4.6,5.6])
plt.plot(op[idx,:,stripe][ep], altt, label='T=0')
plt.plot(op2l[0][idx,:,stripe][ep], altt, label='T=5')
plt.plot(op2l[1][idx,:,stripe][ep], altt, label='T=15')
plt.plot(op2l[2][idx,:,stripe][ep], altt, label='T=25')
plt.title('Marked Profile for Different T'.format(ep))
plt.xlabel('O+ density [1/cm^3]')
plt.ylabel('Altitude [km]')
plt.grid(which='both', axis='both')
plt.legend()
plt.tight_layout()
plt.show()

# plt.figure(figsize=[4.6,5.6])
# plt.plot(op[idx,:,stripe][ep], altt, color='salmon', label='T=0')
# plt.plot(op[idx][ep].mean(axis=1), altt, color='red', label='T=0 stripe avg')
# plt.plot(op2l[2][idx,:,stripe][ep], altt, color='skyblue', label='T=25')
# plt.plot(op2ml[2][idx][ep], altt, color='darkblue', label='T=25 stripe avg')
# plt.title('Comparing Time Avg vs Time & Stripe Avg'.format(ep))
# plt.xlabel('O+ density [1/cm^3]')
# plt.ylabel('Altitude [km]')
# plt.grid(which='both', axis='both')
# plt.legend()
# plt.tight_layout()
# plt.show()
