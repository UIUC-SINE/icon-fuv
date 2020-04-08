from airglow.FUV_L2 import CreateSummaryPlot
from iconfuv.plotting import tohban2
import numpy as np
import os, sys
path_dir = '/home/kamo/resources/iconfuv/nc_files/'

def plotter(layout, date):
    if layout=='toh':
        weighted = True
        rewrite = False
        max_ne = None
        med = False
        orbit = True
        if max_ne is not None:
            maxstr = '{:.0e}'.format(max_ne)
            maxstr = '_' + maxstr.split('+0')[0] + maxstr.split('+0')[1]
        else:
            maxstr = ''
        w = '_W' if weighted else ''
        r = '_R' if rewrite else ''
        rnum = '99' if rewrite else '01'
        rnum = '77' if med else rnum
        m = '_med' if med else ''
        orb = '_orbit' if orbit else ''
        path = path_dir + 'figures/tohban/{}{}{}{}{}{}'.format(date, r, w, m, maxstr, orb)
        os.mkdir(path)
        for i in range(6):
            os.mkdir('{}/stripe{}'.format(path,i))
            print('stripe: {}'.format(i+1))
            CreateSummaryPlot(
                file_netcdf=path_dir + 'ICON_L2-5_FUV_Night{}_orbit_{}_v{}r000.NC'.format(w, date, rnum),
                png_stub='{}/stripe{}/ICON_L2-5_FUV_Night_{}_v01r000.png'.format(path, i, date),
                stripe=i,
                max_ne=max_ne
            )
    elif layout=='tohx':
        path_fig = path_dir + 'figures/tohban_x/{}_orbit/'.format(date)
        os.mkdir(path_fig)
        for i in range(6):
            os.mkdir('{}/stripe{}'.format(path_fig,i))
            print('stripe: {}'.format(i+1))
            tohban2(
                file_l1=path_dir+'ICON_L1_FUV_SWP_{}_v77r000.NC'.format(date),
                file_l2=path_dir+'ICON_L2-5_FUV_Night_orbit_{}_v01r000.NC'.format(date),
                file_anc=path_dir+'ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date),
                png_stub=path_fig+'stripe{}/ICON_L2-5_FUV_Night_{}_v01r000.png'.format(i, date),
                stripe=i
            )


if __name__== "__main__":
    plotter(str(sys.argv[1]), str(sys.argv[2]))
