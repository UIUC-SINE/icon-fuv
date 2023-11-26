import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from iconfuv.misc import lastfile
import netCDF4
from dateutil import parser

path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'
orbit = 0
pixnum = -100
date = '2020-11-05'

def loncorrect(lon):
    lon = np.array(lon)
    if lon.size==1:
        if lon > 180:
            lon -= 360
    else:
        lon[lon>180] -= 360
    return lon

file_anc = lastfile(path_dir+'l0/ICON_L0P_FUV_Ancillary_{}_v0*'.format(date))
anc = netCDF4.Dataset(file_anc, mode='r')

orbits = anc.variables['ICON_ANCILLARY_FUV_ORBIT_NUMBER'][:]
stat = anc.variables['ICON_ANCILLARY_FUV_STATUS'][:]
ons = np.unique(orbits)
idx = np.where(orbits==ons[orbit])[0]
idx = idx[stat[idx]==258]

tlats = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx, pixnum, :, 0]
talts = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx, pixnum, :, 2]
tlons = loncorrect(
    anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx, pixnum, :, 1]
)
satlons = loncorrect(
    anc.variables['ICON_ANCILLARY_FUV_LONGITUDE'][idx]
)
satlats = anc.variables['ICON_ANCILLARY_FUV_LATITUDE'][idx]

t0 = parser.parse(anc.variables['ICON_ANCILLARY_FUV_TIME_UTC'][idx[0]])
t1 = parser.parse(anc.variables['ICON_ANCILLARY_FUV_TIME_UTC'][idx[-1]])

# Create a Basemap instance
# m = Basemap(projection='ortho', lon_0=satlons[6], lat_0=satlats[6], resolution='l')

m = Basemap(
    llcrnrlon=-40.,llcrnrlat=-20.,urcrnrlon=0.,urcrnrlat=30.,
    resolution='l', projection='merc'
)

# Draw the coastlines and country boundaries
m.drawcoastlines()
m.fillcontinents(color='coral', lake_color='aqua')

m.drawparallels(np.arange(-90,120,15), labels=[1,1,0,1])
m.drawmeridians(np.arange(0,420,30), labels=[1,1,0,1])
m.drawmapboundary(fill_color='aqua')

# Convert the latitudes and longitudes to the map projection coordinates
x1, y1 = m(satlons, satlats)
x2, y2 = m(tlons, tlats)

# Plot the two points
m.plot(x1, y1, 'bo', markersize=5, label='ICON')
m.plot(x2, y2, 'ro', markersize=2, label='250km Tan Alt')

# Compute the great circle path between the two points
for i in np.arange(60,76):
    for j in range(6):
        m.drawgreatcircle(satlons[i], satlats[i], tlons[i,j], tlats[i,j], linewidth=1, color='g')

# Show the map
# plt.title('LoS of 6 stripes for the 100. pixel from top')
# plt.legend()
plt.show()


# m2 = Basemap(
#     llcrnrlon=-130.,llcrnrlat=-50.,urcrnrlon=130.,urcrnrlat=50.,
#     resolution='l', projection='merc'
# )
# m2.drawcoastlines()
# m2.fillcontinents(zorder=0)
# m2.drawparallels(np.arange(-30,31,20),labels=[1,1,0,1])
# m2.drawmeridians(np.arange(-180,180,60),labels=[1,1,0,1])
# # m2.nightshade(t0, alpha=0.3)
# m2.nightshade(t1, alpha=0.3)
# x, y = m2(satlons,satlats)
# m2.scatter(x,y,15,marker='o',color='b', label='ICON')
# x, y = m2(tlons[:,2],tlats[:,2])
# m2.scatter(x,y,15,marker='o',color='r', label='250km TanAlt')
# plt.legend()
# plt.show()