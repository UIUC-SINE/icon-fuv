import numpy as np
from iconfuv.artifact_removal2 import artifact_removal
from iconfuv.misc import lastfile
import netCDF4
from shutil import copyfile
from dateutil import parser
import datetime

path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'
path = '/home/kamo/resources/icon-fuv/python3/iconfuv/neural_networks'

def rewriter(date):
    file_l1 = lastfile(path_dir+'l1/ICON_L1_FUV_SWP_{}_v0*'.format(date))
    l1 = netCDF4.Dataset(file_l1, mode='r')

    file_l1_new = file_l1[:-9] + '90' + file_l1[-7:]
    copyfile(file_l1, file_l1_new)

    l1_new = netCDF4.Dataset(file_l1_new, mode='r+')
    mode = l1.variables['ICON_L1_FUV_Mode'][:]

    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    profiles = np.zeros((6, len(mode), 256))
    for i in range(6):
        profiles[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]][:]

    profiles = np.swapaxes(profiles, 1, 2)
    profiles_new = artifact_removal(profiles, channel=1, fuv_mode=mode,
        # path_to_model=path_dir + '/../python3/scripts/residual_network_v3.71.h5')
        path_to_networks=path)

    for i in range(6):
        l1_new.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[i]][:] = (
            profiles_new[i].swapaxes(0,1)
        )

    l1.close()
    l1_new.close()
