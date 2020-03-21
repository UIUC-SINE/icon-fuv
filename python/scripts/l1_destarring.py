#!/usr/bin/env python
# coding: utf-8

# Code to run IRI/MSIS and the RRMN model to simulate ICON FUV observations and replace the data in a L1 file. Used on the Memorial Day simulation to replace noise measurements with simulated data so we can get Scott a multi-day L2.5 file.
#
# Note that our current FUV forward model does not use the updated/correct sensitivity value (the one published in Mende et al.). So, the simulation is not expected to be as representative of reality as if we had asked Scott et al to run the instrument model. However, for what Scott is looking to do, this is not a problem (it is realistic enough).
import netCDF4
import numpy as np
from shutil import copyfile
from scipy.signal import medfilt

date = '2020-01-01'
file_input = 'nc_files/ICON_L1_FUV_SWP_{}_v02r001.NC'.format(date)

file_input_c = file_input.split('v')[:-1][0] + 'v88r' + file_input.split('r')[-1]
copyfile(file_input, file_input_c)

data = netCDF4.Dataset(file_input_c, mode='r+')
mirror_dir = ['M9','M6','M3','P0','P3','P6']

for stripe in range(6):
    data.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][:] = medfilt(
        data.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][:],
        kernel_size=[1,15]
    )
    data.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][:] = np.sqrt(
        data.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][:],
    )

data.close()
