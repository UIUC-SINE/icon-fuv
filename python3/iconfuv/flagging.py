# 2023-11-26 - Ulas Kamaci
# This code includes the core parts of the various flags for the L2.5 data.

import numpy as np
from iconfuv.ionex import get_tecmap
from iconfuv.plotting import xy_to_rc
from skimage.draw import line
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.draw import line
from matplotlib.lines import Line2D

def asymmetry_tec_flagger(min_dn, satlats, satlons, tlats, tlons):

    tecmap, tecutc = get_tecmap(utc=min_dn)

    x1, y1 = (satlons,satlats)
    x2, y2 = (tlons,tlats)

    thrs = 0.5
    x3 = (2*x2 - x1)%360
    x3 = x3 - 360*(x3//180)
    y3 = (2*y2 - y1)%180
    y3 = y3 - 180*(y3//90)
    r1,c1 = xy_to_rc(x1,y1)
    r3,c3 = xy_to_rc(x3,y3)
    stds = np.zeros_like(r3)
    means = np.zeros_like(r3)
    locs = np.where((r3.mask==False) & (c3.mask==False))[0]
    tcs = tecmap.shape[1]
    for i in locs:
        rr,cc = line(r1[i], c1[i], r3[i], c3[i])
        if len(np.unique(cc))>tcs/2:
            if c1[i] > c3[i]:
                c1h = c1[i]
                c3h = c3[i] + tcs
            else:
                c1h = c1[i] + tcs
                c3h = c3[i]
            rr,cc = line(r1[i], c1h, r3[i], c3h)
        stds[i] = np.std(tecmap[rr,cc%tcs])
        means[i] = np.mean(tecmap[rr,cc%tcs])
    std_norm = np.divide(stds,means) 

    flag = std_norm > thrs
    return flag, x3,y3, stds, std_norm

def asymmetry_tec_plotter(fig, ax, min_dn, satlats, satlons, tlats, tlons,x3, y3):

    tecmap, tecutc = get_tecmap(utc=min_dn)

    m = Basemap(
        llcrnrlon=-180.,llcrnrlat=-87.5,urcrnrlon=180.,urcrnrlat=87.5,
        resolution='l', projection='cyl', ax=ax
    )
    m.drawcoastlines(linewidth=0.5, color='gray')
    # m.fillcontinents(zorder=2)
    m.drawparallels(np.arange(-80,81,20),labels=[1,0,0,1])
    m.drawmeridians(np.arange(-180,180,30),labels=[1,0,0,1])
    m.nightshade(min_dn, alpha=0.2)
    x1, y1 = m(satlons,satlats)
    m.scatter(x1,y1,15,marker='o',color='r')
    x2, y2 = m(tlons,tlats)
    m.scatter(x2,y2,15,marker='o',color='b')
    m.scatter(x3,y3,15,marker='o',color='yellow')
    h = m.imshow(tecmap, origin='upper', cmap='inferno', aspect='auto')
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size='5%', pad=0.5, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cb = plt.colorbar(h, cax=ax_cb)
    # plt.rc('text', usetex=True)
    cb.set_label('TECU ($10^{16} \\mathrm{l}/\\mathrm{m}^2$)')
    ax.set_title('GNSS-IGS TEC MAP - {:02d}:00 UTC'.format(tecutc))
    legend_elements = [Line2D([0], [0], marker='o', color='r', label='ICON',
                            markersize=5), 
                        Line2D([0], [0], marker='o', color='b', label='{} km Tan-Alt'.format(300),
                            markersize=5),
                        Line2D([0], [0], marker='o', color='yellow', label='End of LoS',
                            markersize=5)
                        ]
    ax.legend(handles=legend_elements)

    return ax


def asymmetry_flagger(brmax, brmax_alt):
    asymm_flag = np.zeros(len(brmax), dtype=bool)

    asymm_center = (brmax>100) & (brmax_alt<200)
    asymm_center = asymm_center.data
    asymm_center_int = asymm_center.astype(int)
    asymm_check = np.convolve(asymm_center_int, np.ones(7), 'valid')

    if asymm_check.max() >= 5:
        asymm_center = np.argmax( np.convolve(asymm_center_int, np.hamming(13), 'valid') )

        asymm_flag = np.zeros(len(brmax), dtype=bool)
        asymm_flag[max(0,asymm_center-25):asymm_center+25] = True

        asymm_flag = asymm_flag & (brmax_alt < 250)
    
    return asymm_flag


def aurora_flagger(brmax_alt, brmax_tlat):
    aurora_flag = (brmax_tlat>30) & (brmax_alt<220)
    return aurora_flag