import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from airglow.FUV_L2_test import Get_lvl2_5_product
from shutil import copyfile
import iconfuv
from iconfuv.misc import loncorrect, lastfile
from datetime import datetime, timedelta
from dateutil import parser

path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'
date = '2022-03-31'
utctime = '19:41:20Z'
reg = 100000

file_l2 = path_dir + 'l2/ICON_L2-5_FUV_Night_orb_{}_v05r000.NC'.format(date)
file_anc = lastfile(path_dir + 'l0/ICON_L0P_FUV_Ancillary_{}_v*'.format(date))
file_l1 = lastfile(path_dir + 'l1/ICON_L1_FUV_SWP_{}_v*'.format(date))
# file_anc = path_dir + 'l0/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)
file_GPI = path_dir + 'ICON_Ancillary_GPI_2015-001-to-2023-011_v01r000.NC'

file_l1_mod = file_l1.split('SWP')[0] + 'SWP_orbit' + file_l1.split('SWP')[1]
copyfile(file_l1, file_l1_mod)
l1 = netCDF4.Dataset(file_l1_mod, mode='r+')
anc = netCDF4.Dataset(file_anc, mode='r+')

mode = l1.variables['ICON_L1_FUV_Mode'][:]
mode_night = (mode == 2).astype(np.int)
nights = np.diff(mode_night, prepend=0)
nights[nights==-1] = 0
idxs = np.where(mode==2)[0][:]
nights = np.cumsum(nights)[idxs]

utc = '{} {}'.format(date, utctime)
dn = []
for d in anc.variables['ICON_ANCILLARY_FUV_TIME_UTC'][:][idxs]:
    dn.append(parser.parse(d))
dn = np.array(dn)
epoch_l1 = np.argmin(abs(parser.parse(utc)-dn))

orbit = nights[epoch_l1]

not_ind = np.where(nights!=orbit)[0]
l1.variables['ICON_L1_FUVA_SWP_Quality_Flag'][idxs[not_ind]] = 99
l1.close()

Get_lvl2_5_product(
    file_input = file_l1_mod,
    file_ancillary = file_anc,
    file_output = file_l2,
    file_GPI = file_GPI,
    reg_param = reg
)

os.rename(file_l2, file_l2[:-3]+'_reg_{}'.format(reg)+'.NC')