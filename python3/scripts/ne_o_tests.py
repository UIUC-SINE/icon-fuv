# Ulas Kamaci - 2020-06-01
import numpy as np
import matplotlib.pyplot as plt
import netCDF4, pyglow
from iconfuv.misc import lastfile
from dateutil import parser
from airglow.FUV_L2 import get_msisGPI, create_cells_Matrix_spherical_symmetry

path_dir = '/home/kamo/resources/iconfuv/nc_files/'

# determine the parameters
date = '2020-01-01'
epoch = 550
stripe = 2

# read the files
file_GPI = path_dir + 'ICON_Ancillary_GPI_2015-001-to-2020-132_v01r000.NC'
file_anc = lastfile(path_dir+'l0/ICON_L0P_FUV_Ancillary_{}_v01r*'.format(date))
gpi = netCDF4.Dataset(file_GPI, mode='r')
anc = netCDF4.Dataset(file_anc, mode='r')

# set the variables
local_time = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LST'][epoch, -1, stripe]
print('Local Time: {}'.format(local_time))
dn = parser.parse(anc.variables['ICON_ANCILLARY_FUV_TIME_UTC'][epoch])
ap3 = gpi['ap3'][:]
ap = gpi['ap'][:]
year_day = gpi['year_day'][:]
f107 = gpi['f107d'][:]
# Make sure this GPI has the average f107 in it
if 'f107a' in gpi.variables.keys():
    f107a = gpi['f107a'][:]
else:
    print('Cannot find f107a in provided GPI file. Using daily f107 instead')
    f107a = gpi['f107d'][:]


satlat = anc.variables['ICON_ANCILLARY_FUV_LATITUDE'][epoch]
satlon = anc.variables['ICON_ANCILLARY_FUV_LONGITUDE'][epoch]
satalt = anc.variables['ICON_ANCILLARY_FUV_ALTITUDE'][epoch]
lat_arr = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][epoch, :, stripe, 0]
lon_arr = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][epoch, :, stripe, 1]
alt_arr = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][epoch, :, stripe, 2]
ze = anc.variables['ICON_ANCILLARY_FUVA_FOV_ZENITH_ANGLE'][epoch, :, stripe]
ze = ze[alt_arr>150]
lat_arr = lat_arr[alt_arr>150]
lon_arr = lon_arr[alt_arr>150]
alt_arr = alt_arr[alt_arr>150]
O_p_arr = np.zeros_like(alt_arr)
O_arr = np.zeros_like(alt_arr)
Ne_arr = np.zeros_like(alt_arr)
RR1_arr = np.zeros_like(alt_arr)
RR2_arr = np.zeros_like(alt_arr)

D = create_cells_Matrix_spherical_symmetry(ze[::-1],satalt)

for i, alt in enumerate(alt_arr):
    my_f107, my_f107a, my_f107p, my_apmsis = get_msisGPI(dn, year_day, f107, f107a, ap, ap3)
    pt = pyglow.Point(dn, lat_arr[i], lon_arr[i], alt, user_ind=True)
    # pt = pyglow.Point(dn, satlat, satlon, alt, user_ind=True)
    pt.f107 = my_f107
    pt.f107a = my_f107a
    pt.f107p = my_f107p
    pt.apmsis = my_apmsis
    pt.run_iri()
    pt.run_msis()

    # Pull necessary constitutents
    O_p_arr[i] = pt.ni['O+']  # O+ density (1/cm^3)
    O_arr[i] = pt.nn['O']     # O density (1/cm^3)
    Ne_arr[i] = pt.ne         # electron density (1/cm^3)

    # Calcualte radiative recombination (equation 17) and mutual neutralization (equation 18)
    a1356 = 7.3e-13 # radiative recombination rate (cm^3/s)
    RR1_arr[i] = a1356*Ne_arr[i]*O_p_arr[i]  # radiative recombination (1/cm^3/s)
    RR2_arr[i] = a1356*Ne_arr[i]**2  # radiative recombination (1/cm^3/s)

br_calc = np.dot(D, RR1_arr[::-1])
br_calc = br_calc[::-1]
# %% plot
plt.figure()
plt.plot(Ne_arr, alt_arr, label='Ne profile')
plt.plot(O_p_arr, alt_arr, label='O+ profile')
plt.title('O+ vs Ne Comparison')
plt.xlabel('Density [$cm^{-1}$]')
plt.ylabel('Altitude [km]')
plt.grid(which='both', axis='both')
plt.legend()
plt.show()

plt.figure()
plt.plot(RR1_arr, alt_arr, label='RR=(Ne)*(O+)')
plt.plot(RR2_arr, alt_arr, label='RR=(Ne)$^2$')
plt.title('Computed VER Comparison')
plt.xlabel('Volume Emission Rate')
plt.ylabel('Altitude [km]')
plt.grid(which='both', axis='both')
plt.legend()
plt.show()

plt.figure()
plt.plot(br_calc, alt_arr, label='IRI Brightness')
plt.title('IRI Brightness')
plt.xlabel('Brightness [R]')
plt.ylabel('Tangent Altitude [km]')
plt.grid(which='both', axis='both')
plt.legend()
plt.show()
