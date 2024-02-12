# 2023-12-12 Ulas Kamaci
# This script is for the final testing of the L2.5 software update v4r3, which 
# includes the aurora and asymmetry flags. This code compares the 
# 1. v4r2 output
# 2. v4r3 output, and
# 3. post-processed (flagged) v4r2 

import matplotlib.pyplot as plt
import numpy as np
import netCDF4, datetime
from dateutil import parser
from copy import deepcopy
import glob, os
from iconfuv.misc import lastfile
from all_flagger_py2 import all_flagger
path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

def aurora_finder(flag):
    if np.ma.isMaskedArray(flag):
        flag = flag.data
    flag[flag>127] -= 128
    return flag>63

def asymm_finder(flag):
    if np.ma.isMaskedArray(flag):
        flag = flag.data
    return flag>127


# date_str = '2020-01-01'
# date_str = '2021-01-01'
# date_str = '2022-03-20'
# date_str = '2022-04-05'
date_str = '2022-08-19'
file_l20 = path_dir + 'l2/ICON_L2-5_FUV_Night_{}_v05r090.NC'.format(date_str)
file_l21 = path_dir + 'l2/ICON_L2-5_FUV_Night_{}_v05r091.NC'.format(date_str)
file_l1 = lastfile(path_dir + 'l1/ICON_L1_FUV_SWP_{}_v05*'.format(date_str))

l20 = netCDF4.Dataset(file_l20, mode='r')
l21 = netCDF4.Dataset(file_l21, mode='r')
l1 = netCDF4.Dataset(file_l1, mode='r')


quals0 = l20.variables['ICON_L25_Quality'][:]
flags0 = l20.variables['ICON_L25_Quality_Flags'][:]
quals1 = l21.variables['ICON_L25_Quality'][:]
flags1 = l21.variables['ICON_L25_Quality_Flags'][:]

l20.close()
l21.close()
l1.close()

(nighttime_counter, 
flag_counter, 
asymm_flag_counter, 
aurora_flag_counter, 
qual1_counter, 
qual1flag_counter,
asymm_qual1flag_counter, 
aurora_qual1flag_counter, 
hmf2s_all, 
flagqual1_locs,
asymm_flagqual1_locs,
aurora_flagqual1_locs,
quals2,
flags2,
flag_locs,
asymm_flag_locs,
aurora_flag_locs) = all_flagger(file_l1=file_l1, file_l2=file_l20)


aurora0 = aurora_finder(flags0.copy())
aurora1 = aurora_finder(flags1.copy())
aurora2 = aurora_finder(flags2.copy())

asymm0 = asymm_finder(flags0.copy())
asymm1 = asymm_finder(flags1.copy())
asymm2 = asymm_finder(flags2.copy())

print('Aurora 1 - 2: {} - {}'.format(np.sum(aurora1), np.sum(aurora2)))
print('Asymm 1 - 2: {} - {}'.format(np.sum(asymm1), np.sum(asymm2)))
print('Aurora1 == Aurora2: {}'.format(np.allclose(aurora1,aurora2)))
print('Asymm1 == Asymm2: {}'.format(np.allclose(asymm1,asymm2)))
print('Flags1 == Flags2: {}'.format(np.allclose(flags1,flags2)))
print('Quals1 == Quals2: {}'.format(np.allclose(quals1,quals2)))
print('Aurora1 - Stripe Consistency: {}'.format(np.allclose(aurora1, np.repeat(aurora1[:,[0]],6,axis=1))))
print('Aurora2 - Stripe Consistency: {}'.format(np.allclose(aurora2, np.repeat(aurora2[:,[0]],6,axis=1))))
print('Asymm1 - Stripe Consistency: {}'.format(np.allclose(asymm1, np.repeat(asymm1[:,[0]],6,axis=1))))
print('Asymm2 - Stripe Consistency: {}'.format(np.allclose(asymm2, np.repeat(asymm2[:,[0]],6,axis=1))))

flags0c = flags0.copy()
flags0c[np.where(aurora1==1)] += 64
flags0c[np.where(asymm1==1)] += 128

print('Flag0c == Flag1: {}'.format(np.allclose(flags0c,flags1)))