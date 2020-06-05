import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from iconfuv.misc import  lastfile

dates = [
    '2019-11-16',
    '2019-11-17',
    '2019-11-18',
    '2019-11-19',
    '2019-11-20',
    '2019-11-21',
    '2019-11-22',
    '2019-11-23',
    '2019-11-24',
    '2019-11-25',
    '2019-11-26'
]
stripe=2

for date in dates:
    file = lastfile('/home/ukamaci2/icon/nc_files/l0/ICON_L0P_FUV_Ancillary_{}_v03r*'.format(date))
    anc = netCDF4.Dataset(file, mode='r')
    plt.figure()
    plt.plot(anc.variables['ICON_ANCILLARY_FUV_LATITUDE'][:])
    plt.savefig('./lat_{}.png'.format(date))
    plt.close()
    anc.close()
