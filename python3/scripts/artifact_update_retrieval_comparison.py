import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from iconfuv.misc import lastfile

path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

dates = [
    '2020-01-01',
    '2020-02-02',
    '2020-03-15',
    '2020-04-02',
    '2020-06-18',
    '2020-10-17',
    '2020-11-27'
]

quals_old = []
nmf2s_old = []
nmf2s_q1_old = []
quals_new = []
nmf2s_new = []
nmf2s_q1_new = []

for date in dates:
    file_l2_old = lastfile(path_dir+'l2/ICON_L2-5_FUV_Night_{}_v04r*'.format(date))
    file_l2_new = lastfile(path_dir+'l2/ICON_L2-5_FUV_Night_{}_v90r*'.format(date))
    l2_old = netCDF4.Dataset(file_l2_old, mode='r')
    l2_new = netCDF4.Dataset(file_l2_new, mode='r')

    qual_old = l2_old.variables['ICON_L25_Quality'][:]
    qual_new = l2_new.variables['ICON_L25_Quality'][:]
    quals_old.append(qual_old)
    quals_new.append(qual_new)

    nmf2_old = l2_old.variables['ICON_L25_NMF2'][:]
    nmf2_new = l2_new.variables['ICON_L25_NMF2'][:]
    nmf2s_old.append(nmf2_old)
    nmf2s_new.append(nmf2_new)

    nmf2_q1_old = nmf2_old[qual_old==1]
    nmf2_q1_new = nmf2_new[qual_new==1]
    nmf2s_q1_old.append(nmf2_q1_old)
    nmf2s_q1_new.append(nmf2_q1_new)

    l2_old.close()
    l2_new.close()
