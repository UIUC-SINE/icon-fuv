from airglow.FUV_L2 import CreateSummaryPlot
from iconfuv.plotting import tohban2
import numpy as np
import os

# date = '2020-01-03'
# weighted = False
# rewrite = True
# max_ne = None
# med = False
# if max_ne is not None:
#     maxstr = '{:.0e}'.format(max_ne)
#     maxstr = '_' + maxstr.split('+0')[0] + maxstr.split('+0')[1]
# else:
#     maxstr = ''
# w = '_W' if weighted else ''
# r = '_R' if rewrite else ''
# rnum = '99' if rewrite else '01'
# rnum = '88' if med else rnum
# m = '_med' if med else ''
# path = './nc_files/figures/tohban/{}{}{}{}{}'.format(date, r, w, m, maxstr)
# os.mkdir(path)
# for i in range(6):
#     os.mkdir('{}/stripe{}'.format(path,i))
#     print('stripe: {}'.format(i+1))
#     CreateSummaryPlot(
#         file_netcdf='nc_files/ICON_L2_FUV_Oxygen-Profile-Night{}_{}_v{}r000.NC'.format(w, date, rnum),
#         png_stub='{}/stripe{}/ICON_L2-5_FUV_Night_{}_v01r000.png'.format(path, i, date),
#         stripe=i,
#         max_ne=max_ne
#     )

date = '2020-01-05' #.format(j)
path = './nc_files/figures/tohban_x/{}_W'.format(date)
os.mkdir(path)
for i in range(6):
    os.mkdir('{}/stripe{}'.format(path,i))
    print('stripe: {}'.format(i+1))
    tohban2(
        file_l1='nc_files/ICON_L1_FUV_SWP_{}_v02r000.NC'.format(date),
        file_l2='nc_files/ICON_L2_FUV_Oxygen-Profile-Night_W_{}_v01r000.NC'.format(date),
        file_anc='nc_files/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date),
        png_stub='{}/stripe{}/ICON_L2-5_FUV_Night_{}_v01r000.png'.format(path, i, date),
        stripe=i
    )
