import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from mpl_toolkits.basemap import Basemap
from iconfuv.ionex import get_tecmap
from matplotlib.lines import Line2D
import datetime
from skimage.draw import line

def xy_to_rc(x,y):
    col = ((x+180)/5).astype(int)
    row = 70-((y+87.5)/2.5).astype(int)
    return row, col

min_dn = datetime.datetime(2022,10,5,0,5,0)
tecmap, tecutc = get_tecmap(utc=min_dn)

x1 = np.load('x1.npy')
x2 = np.load('x2.npy')
y1 = np.load('y1.npy')
y2 = np.load('y2.npy')
x3 = 2*x2 - x1
y3 = 2*y2 - y1
r1,c1 = xy_to_rc(x1,y1)
r3,c3 = xy_to_rc(x3,y3)
rows = [] ; cols = []
stds = []
for i in range(len(x1)):
    rr,cc = line(r1[i], c1[i], r3[i], c3[i])
    rows.append(rr)
    cols.append(cc)
    stds.append(np.std(tecmap[rr,cc]))

fig, ax = plt.subplots(2,1, figsize=(10,7.2))
m = Basemap(
    llcrnrlon=-180.,llcrnrlat=-87.5,urcrnrlon=180.,urcrnrlat=87.5,
    resolution='l', projection='cyl', ax=ax[0]
)
m.drawcoastlines(linewidth=0.5, color='gray')
# m.fillcontinents(zorder=2)
m.drawparallels(np.arange(-80,81,20),labels=[1,0,0,1])
m.drawmeridians(np.arange(-180,180,30),labels=[1,0,0,1])
m.nightshade(min_dn, alpha=0.2)
# x1, y1 = m(satlons,satlats)
m.scatter(x1,y1,15,marker='o',color='r')
# x2, y2 = m(tlons,tlats)
m.scatter(x2,y2,15,marker='o',color='b')
m.scatter(x3,y3,15,marker='o',color='yellow')
h = m.imshow(tecmap, origin='upper', cmap='inferno', aspect='auto')
divider = make_axes_locatable(ax[0])
ax_cb = divider.new_horizontal(size='5%', pad=0.5, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb = plt.colorbar(h, cax=ax_cb)
# plt.rc('text', usetex=True)
cb.set_label('TECU ($10^{16} \\mathrm{l}/\\mathrm{m}^2$)')
ax[0].set_title('GNSS-IGS TEC MAP - {:02d}:00 UTC'.format(tecutc))
legend_elements = [Line2D([0], [0], marker='o', color='r', label='ICON',
                        markersize=5), 
                    Line2D([0], [0], marker='o', color='b', label='150km Tan-Alt',
                        markersize=5)]
ax[0].legend(handles=legend_elements)

ax[1].plot(stds, '-o')
ax[1].set_title('Standard Deviation of TEC along LOS vs Epoch')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('TEC Std. Dev.')
ax[1].grid(which='both', axis='both')

plt.show()