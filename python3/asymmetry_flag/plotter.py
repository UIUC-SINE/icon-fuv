import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from mpl_toolkits.basemap import Basemap
from ionex import get_tecmap
import datetime, netCDF4
from dateutil import parser
from iconfuv.misc import loncorrect, lastfile, index_finder

path_l2s = '/home/kamo/resources/icon-fuv/ncfiles/l2/'

utc_time = '2022-10-04 01:17:40'
date = utc_time.split(' ')[0]
utc = parser.parse(utc_time)
year = utc.year
day = utc.timetuple().tm_yday
utc_hours = utc.hour + utc.minute/60

file_l2 = lastfile(path_l2s + 'ICON_L2-5_FUV_Night_{}_v05*'.format(date))
l2 = netCDF4.Dataset(file_l2, 'r')
dn = [] # UTC
for d in l2.variables['ICON_L25_UTC_Time']:
    dn.append(parser.parse(d).replace(tzinfo=None))
dn = np.array(dn)
ind = np.argmin(abs(parser.parse(utc_time)-dn))
orbs = l2.variables['ICON_L25_Orbit_Number'][:]
inds = orbs==orbs[ind]

ds = np.array([i.total_seconds() for i in dn-dn[inds][0]])
inds = np.squeeze(np.where(abs(ds) < 2000.))

tlons = loncorrect(
    l2.variables['ICON_L25_O_Plus_Profile_Longitude'][inds, -1, 2]
)
tlats = l2.variables['ICON_L25_O_Plus_Profile_Latitude'][inds, -1, 2]
satlons = loncorrect(
    l2.variables['ICON_L25_Observatory_Position_Longitude'][inds]
)
satlats = l2.variables['ICON_L25_Observatory_Position_Latitude'][inds]
l2.close()

tecmap = get_tecmap(year, day, utc_hours)

fig, ax = plt.subplots()
m = Basemap(
    llcrnrlon=-180.,llcrnrlat=-87.5,urcrnrlon=180.,urcrnrlat=87.5,
    resolution='l', projection='cyl', ax=ax
)
m.drawcoastlines(linewidth=0.5, color='gray')
# m.fillcontinents(zorder=2)
m.drawparallels(np.arange(-80,81,20),labels=np.ones(9))
m.drawmeridians(np.arange(-180,180,30),labels=np.ones(13))
m.nightshade(utc, alpha=0.3)
x, y = m(satlons,satlats)
m.scatter(x,y,15,marker='o',color='r')
x, y = m(tlons,tlats)
m.scatter(x,y,15,marker='o',color='b')
h = m.imshow(tecmap, origin='upper', cmap='inferno', aspect='auto')

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size='5%', pad=0.5, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb = plt.colorbar(h, cax=ax_cb)
# plt.rc('text', usetex=True)
cb.set_label('TECU ($10^{16} \\mathrm{l}/\\mathrm{m}^2$)')

plt.show()
