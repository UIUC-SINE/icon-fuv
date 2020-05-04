#!/usr/bin/env python
# coding: utf-8

# Code to run IRI/MSIS and the RRMN model to simulate ICON FUV observations and replace the data in a L1 file. Used on the Memorial Day simulation to replace noise measurements with simulated data so we can get Scott a multi-day L2.5 file.
#
# Note that our current FUV forward model does not use the updated/correct sensitivity value (the one published in Mende et al.). So, the simulation is not expected to be as representative of reality as if we had asked Scott et al to run the instrument model. However, for what Scott is looking to do, this is not a problem (it is realistic enough).
import sys, netCDF4
import numpy as np
from shutil import copyfile
import glob
from iconfuv.destarring.star_remover import star_removal

path_dir = '/home/kamo/resources/iconfuv/nc_files/'

def destarrer(date):
    file_input = path_dir + 'l1/ICON_L1_FUV_SWP_{}_v03r*'.format(date)
    file_input = glob.glob(file_input)
    file_input.sort()
    file_input = file_input[-1]

    file_input_c = file_input.split('_v')[0] + '_v77r000.NC'
    copyfile(file_input, file_input_c)

    data = netCDF4.Dataset(file_input_c, mode='r+')

    data = star_removal(data)
    # mirror_dir = ['M9','M6','M3','P0','P3','P6']
    #
    # for stripe in range(6):
    #     data.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][:] = medfilt(
    #         data.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][:],
    #         kernel_size=[1,15]
    #     )
    #     data.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][:] = np.sqrt(
    #         data.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][:],
    #     )

    data.close()

if __name__== "__main__":
    destarrer(str(sys.argv[1]))
