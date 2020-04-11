from airglow.FUV_L2 import CreateSummaryPlot
from iconfuv.plotting import tohban2, tohban_l1
import numpy as np
import os, sys
path_dir = '/home/kamo/resources/iconfuv/nc_files/'

def plotter(layout, date):
    if layout=='toh':
        path = path_dir + 'figures/tohban/{}/v2'.format(date)
        os.mkdir(path)
        for i in range(6):
            os.mkdir('{}/stripe{}'.format(path,i))
            print('stripe: {}'.format(i+1))
            CreateSummaryPlot(
                file_netcdf=path_dir + 'l2/ICON_L2-5_FUV_Night_{}_v01r000.NC'.format(date),
                png_stub='{}/stripe{}/ICON_L2-5_FUV_Night_{}_v01r000.png'.format(path, i, date),
                stripe=i,
                max_ne=None
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
    elif layout=='tohl1':
        path_fig = path_dir + 'figures/tohban_l1/{}/'.format(date)
        if not os.path.isdir(path_fig):
            os.mkdir(path_fig)
        i=0
        while i<10:
            i += 1
            if not os.path.isdir(path_fig+'v{}/'.format(i)):
                path_fig += 'v{}/'.format(i)
                os.mkdir(path_fig)
                break
        tohban_l1(
            file_l1=path_dir+'l1/ICON_L1_FUV_SWP_{}_v03r000.NC'.format(date),
            png_dir=path_fig
        )


if __name__== "__main__":
    plotter(str(sys.argv[1]), str(sys.argv[2]))
