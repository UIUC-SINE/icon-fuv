#!/usr/bin/env python
# coding: utf-8

# Code to run IRI/MSIS and the RRMN model to simulate ICON FUV observations and replace the data in a L1 file. Used on the Memorial Day simulation to replace noise measurements with simulated data so we can get Scott a multi-day L2.5 file.
#
# Note that our current FUV forward model does not use the updated/correct sensitivity value (the one published in Mende et al.). So, the simulation is not expected to be as representative of reality as if we had asked Scott et al to run the instrument model. However, for what Scott is looking to do, this is not a problem (it is realistic enough).
import airglow.FUV_L2
import airglow.ICON_FUV_fwd_model as FUV_F # for simulating data
import netCDF4
import numpy as np
from dateutil import parser
from shutil import copyfile

date = '2019-11-22'
file_input = 'nc_files/ICON_L1_FUV_SWP_{}_v02r001.NC'.format(date)
file_ancillary = 'nc_files/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)
file_GPI = 'nc_files/ICON_Ancillary_GPI_2015-001-to-2019-362_v01r000.NC'

file_input_c = file_input.split('v')[:-1][0] + 'v99r' + file_input.split('r')[-1]
copyfile(file_input, file_input_c)

# Open input Level 1 and ancillary NetCDF files
data = netCDF4.Dataset(file_input_c, mode='r+')
ancillary = netCDF4.Dataset(file_ancillary, mode='r')
gpi = netCDF4.Dataset(file_GPI, mode='r')

if file_GPI is not None:
    gpi = netCDF4.Dataset(file_GPI, mode='r')

    # Read the geophysical indeces
    ap3 = gpi['ap3'][:]
    ap = gpi['ap'][:]
    year_day = gpi['year_day'][:]
    f107 = gpi['f107d'][:]
    f107a = gpi['f107d'][:]  ### replaced f107a with f107d because the provided GPI does not have f107a
else:
    ap3 = None
    ap = None
    year_day = None
    f107 = None
    f107a = None

# The tangent point WGS-84 coordinates at the center of the integration time
FUV_TANGENT_LATITUDES = ancillary.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][:,:,:,0]
FUV_TANGENT_LONGITUDES = ancillary.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][:,:,:,1]
FUV_TANGENT_ALTITUDES = ancillary.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][:,:,:,2]

# The az/el of the look vector
FUV_AZ = ancillary.variables['ICON_ANCILLARY_FUVA_FOV_AZIMUTH_ANGLE'][:,:,:]
FUV_ZE = ancillary.variables['ICON_ANCILLARY_FUVA_FOV_ZENITH_ANGLE'][:,:,:]

# The ICON WGS-84 location at the center of the integration time
ICON_WGS84_LATITUDE = ancillary.variables['ICON_ANCILLARY_FUV_LATITUDE'][:]
ICON_WGS84_LONGITUDE = ancillary.variables['ICON_ANCILLARY_FUV_LONGITUDE'][:]
ICON_WGS84_ALTITUDE = ancillary.variables['ICON_ANCILLARY_FUV_ALTITUDE'][:]

# Read the UTC of all measurements and store in a datetime variable
temp = ancillary.variables['ICON_ANCILLARY_FUV_TIME_UTC']
ANC_dn = []
for d in temp:
    ANC_dn.append(parser.parse(d))
ANC_dn = np.array(ANC_dn)

# Get Data from file.
# L1 data stores the individual stripes in different variables. Read them
# all stripes and combine into a single variable
mirror_dir = ['M9','M6','M3','P0','P3','P6']
FUV_1356_IMAGE = np.zeros(np.shape(FUV_AZ))
FUV_1356_ERROR = np.zeros(np.shape(FUV_AZ))
for ind, d in enumerate(mirror_dir):
    FUV_1356_IMAGE[:,:,ind] = data.variables['ICON_L1_FUVA_SWP_PROF_%s' % d][:]
    FUV_1356_ERROR[:,:,ind] = data.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % d][:]

# Get observation times from file and store in a datetime variable
temp = ancillary.variables['ICON_ANCILLARY_FUV_TIME_UTC']
FUV_dn = []
for d in temp:
    FUV_dn.append(parser.parse(d))
FUV_dn = np.array(FUV_dn)

# Get science mode
FUV_mode = ancillary.variables['ICON_ANCILLARY_FUV_ACTIVITY'][:]


# In[ ]:


for stripe in range(6):
    I_nn = np.zeros([256,6])
    I_n = np.zeros([256,6])
    phot_nn = np.zeros([256,6])

    for sample in range(len(FUV_AZ)):
        if FUV_mode[sample] == 258:
            print('{}/{} - {}/{}'.format(stripe+1, 6, sample+1, len(FUV_mode)))
            ze = np.squeeze(FUV_ZE[sample,:,stripe])
            az = np.squeeze(FUV_AZ[sample,:,stripe])
            satlat = ICON_WGS84_LATITUDE[sample]
            satlon = ICON_WGS84_LONGITUDE[sample]
            satalt = ICON_WGS84_ALTITUDE[sample]
            dn = ANC_dn[sample]
            my_f107, my_f107a, my_f107p, my_apmsis = FUV_L2.get_msisGPI(dn, year_day, f107, f107a, ap, ap3)
            sym_sph = 1

            I_nn[:,stripe], phot_nn[:,stripe] = FUV_F.get_Photons_from_Brightness_Profile_1356_nighttime(
                ze,az,satlat,satlon,satalt,dn,
                cont=1,
                symmetry=1, # 0 = spherical symmetry
                shperical=1, # 0 = spherical earth
                step = 100., # step size for line-of-sight integral. Larger --> runs faster
                f107=my_f107,
                f107a=my_f107a,
                f107p=my_f107p,
                apmsis=my_apmsis,
                stripes_used=1
            )

            temp, xx = FUV_F.add_noise_to_photon_and_brightness(phot_nn[:,stripe],stripes_used=1)
            I_n[:,stripe] = temp[0,:] # only use first realization

            # replace values
            data.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][sample,:] = I_n[:,stripe]
            data.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][sample,:] = np.sqrt(I_nn[:,stripe])

data.close()
