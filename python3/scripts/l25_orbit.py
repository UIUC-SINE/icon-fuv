import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from airglow.FUV_L2 import Get_lvl2_5_product
from shutil import copyfile
import iconfuv

path_dir = '/home/kamo/resources/iconfuv/nc_files/'
date = '2020-03-20'
orbit = 5
stripe = 3 # 0-5
stripe = None

file_l1 = path_dir + 'l1/ICON_L1_FUV_SWP_{}_v01r000.NC'.format(date)
file_l2 = path_dir + 'l2/ICON_L2-5_FUV_Night_{}_v01r000.NC'.format(date)
file_anc = path_dir + 'l0/ICON_L0P_FUV_Ancillary_{}_v99r019.NC'.format(date)
# file_anc = path_dir + 'l0/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)
file_GPI = path_dir + 'ICON_Ancillary_GPI_2015-001-to-2020-132_v01r000.NC'

file_l1_mod = file_l1.split('SWP')[0] + 'SWP_orbit' + file_l1.split('SWP')[1]
copyfile(file_l1, file_l1_mod)
l1 = netCDF4.Dataset(file_l1_mod, mode='r+')
mode = l1.variables['ICON_L1_FUV_Mode'][:]
mode_night = (mode == 2).astype(np.int)
nights = np.diff(mode_night, prepend=0)
nights[nights==-1] = 0
idxs = np.where(mode==2)[0][:]
nights = np.cumsum(nights)[idxs]

not_ind = np.where(nights!=orbit)[0]
l1.variables['ICON_L1_FUVA_SWP_Quality_Flag'][idxs[not_ind]] = 99
l1.close()

Get_lvl2_5_product(
    file_input = file_l1_mod,
    file_ancillary = file_anc,
    file_output = file_l2,
    file_GPI = file_GPI,
)
