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
epoch=300
stripe=2

mode = anc.variables['ICON_ANCILLARY_FUV_ACTIVITY'][:]
orb = anc.variables['ICON_ANCILLARY_FUV_ORBIT_NUMBER'][:]
idx = np.where(mode==258)[0][epoch]

# take the `epoch`^th night index of the `stripe`^th brightness profiles
br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][idx,:]
oplus = l2.variables['ICON_L25_O_Plus_Density'][epoch, :, stripe]
alt_l2 = l2.variables['ICON_L25_O_Plus_Profile_Altitude'][epoch, :, :]
lon_l2 = l2.variables['ICON_L25_O_Plus_Profile_Longitude'][epoch, :, :]
lat_l2 = l2.variables['ICON_L25_O_Plus_Profile_Latitude'][epoch, :, :]
alt_l1 = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx, :, :, 2]
err = l2.variables['ICON_L25_Error_Code'][epoch, stripe]
lt = l2.variables['ICON_L25_Solar_Local_Time'][epoch, stripe]
satlon = l2.variables['ICON_L25_Observatory_Position_Longitude'][epoch]
satlat = l2.variables['ICON_L25_Observatory_Position_Latitude'][epoch]
t = parser.parse(l2.variables['ICON_L25_UTC_Time'][epoch])

ap = sum(alt_l2[:, stripe].mask==False) #num of active pixels (>150)

# plt.figure(1); plt.plot(alt_l1[-ap:]); plt.title('Tangent Altitudes for 6 Stripes')
plt.figure()
plt.plot(br[-ap:], alt_l1[-ap:, stripe])
plt.title('Brightness - Stripe {}'.format(stripe))
plt.xlabel('Brightness [R]')
plt.ylabel('Tang. Altitudes [km]')
plt.ticklabel_format(scilimits=(0,3))
plt.show()
plt.figure()
plt.plot(oplus, alt_l2[:, stripe])
plt.title('$O^+$ Profile - Stripe {}'.format(stripe))
plt.xlabel('$O^+$ Density [$cm^{-3}$]')
plt.ylabel('Tang. Altitudes [km]')
plt.ticklabel_format(scilimits=(0,3))
plt.show()

fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(
    llcrnrlon=-180.,llcrnrlat=-50.,urcrnrlon=180.,urcrnrlat=50.,
    resolution='l', projection='merc'
)
m.drawcoastlines()
m.fillcontinents(zorder=0)
m.drawparallels(np.arange(-30,31,20),labels=[1,1,0,1])
m.drawmeridians(np.arange(-180,180,60),labels=[1,1,0,1])
m.nightshade(t, alpha=0.3)
x, y = m(loncorrect(satlon),satlat)
m.scatter(x,y,15,marker='o',color='r')
x, y = m(loncorrect(lon_l2[:,0].compressed()[0]), lat_l2[:,0].compressed()[0])
m.scatter(x,y,15,marker='o',color='b')
x, y = m(loncorrect(lon_l2[:,-1].compressed()[0]), lat_l2[:,-1].compressed()[0])
m.scatter(x,y,15,marker='o',color='b')
x, y = m(loncorrect(lon_l2[:,0].compressed()[-1]), lat_l2[:,0].compressed()[-1])
m.scatter(x,y,15,marker='o',color='b')
x, y = m(loncorrect(lon_l2[:,-1].compressed()[-1]), lat_l2[:,-1].compressed()[-1])
m.scatter(x,y,15,marker='o',color='b')
ax.set_title('SLT:{} - orbit:{}'.format(str(timedelta(seconds=lt*3600))[:-7], orb[idx]))
plt.show()

tohban(l2=l2, l1=l1, anc=anc, epoch=epoch, stripe=stripe)

# %% close ---------------------------------
anc.close()
l1.close()
l2.close()
