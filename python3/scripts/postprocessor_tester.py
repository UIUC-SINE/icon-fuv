import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from fuv_l1_postprocessor_v1r0 import postprocessor
from iconfuv.misc import lastfile

pathdir = '/home/kamo/resources/icon-fuv/ncfiles/l1/'
nndir = '/home/kamo/resources/icon-fuv/python3/iconfuv/neural_networks/'

dates = [
    '2020-09-26',
    '2021-03-21',
    '2021-05-20'
]

l1s_sw = [] # original l1 files
l1ps_sw = [] # modified l1 files
l1s_lw = []
l1ps_lw = []

for date in dates:
    file_sw = lastfile(pathdir+'**/'+'ICON_L1_FUV_SWP_{}*'.format(date))
    file_lw = lastfile(pathdir+'**/'+'ICON_L1_FUV_LWP_{}*'.format(date))
    file_sw_out = file_sw
    file_lw_out = file_lw
    file_sw_out = file_sw[:-9] + '99' + file_sw[-7:]
    file_lw_out = file_lw[:-9] + '99' + file_lw[-7:]
    postprocessor(file_in=file_sw, file_out=file_sw_out, path_to_networks=nndir)
    postprocessor(file_in=file_lw, file_out=file_lw_out, path_to_networks=nndir)
    l1s_sw.append(netCDF4.Dataset(file_sw, 'r'))
    l1s_lw.append(netCDF4.Dataset(file_lw, 'r'))
    l1ps_sw.append(netCDF4.Dataset(file_sw_out, 'r'))
    l1ps_lw.append(netCDF4.Dataset(file_lw_out, 'r'))
