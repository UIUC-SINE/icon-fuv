from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil import parser
import netCDF4
from iconfuv.plotting import tohban

def loncorrect(lon):
    if lon.size==1:
        if lon > 180:
            lon -= 360
    else:
        lon[lon>180] -= 360
    return lon

# %% files ----------------------------------
date = '2020-01-04'
file_anc='/home/kamo/icon/nc_files/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)
file_l1='/home/kamo/icon/nc_files/ICON_L1_FUV_SWP_{}_v02r000.NC'.format(date)
file_l2='/home/kamo/icon/nc_files/ICON_L2_FUV_Oxygen-Profile-Night_{}_v01r000.NC'.format(date)
# file_l1='/home/kamo/icon/nc_files/ICON_L1_FUV_SWP_{}_v99r000.NC'.format(date)
# file_l2='/home/kamo/icon/nc_files/ICON_L2_FUV_Oxygen-Profile-Night_{}_v99r000.NC'.format(date)

anc = netCDF4.Dataset(file_anc, mode='r')
l1 = netCDF4.Dataset(file_l1, mode='r')
l2 = netCDF4.Dataset(file_l2, mode='r')

mirror_dir = ['M9','M6','M3','P0','P3','P6']

# %% params ---------------------------------
epoch=310
stripes=[0,1,2,3,4,5]

mode = anc.variables['ICON_ANCILLARY_FUV_ACTIVITY'][:]
orb = anc.variables['ICON_ANCILLARY_FUV_ORBIT_NUMBER'][:]
idx = np.where(mode==258)[0][epoch]

# take the `epoch`^th night index of the `stripe`^th brightness profiles
br = np.zeros((256,6))
for stripe in stripes:
    br[:,stripe] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][idx,:]
alt_l2 = l2.variables['ICON_L25_O_Plus_Profile_Altitude'][epoch, :, :]

# plt.figure(1); plt.plot(alt_l1[-ap:]); plt.title('Tangent Altitudes for 6 Stripes')
plt.figure()
for i in range(br.shape[1]):
    plt.plot(br[:,i], label='{}'.format(i))
plt.title('Brightness - Stripe {}'.format(stripes))
plt.xlabel('Brightness [R]')
plt.ylabel('Tang. Altitudes [km]')
plt.ticklabel_format(scilimits=(0,3))
plt.legend()
plt.show()
