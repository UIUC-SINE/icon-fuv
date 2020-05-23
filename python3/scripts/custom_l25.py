import matplotlib.pyplot as plt
import numpy as np
from airglow.FUV_L2 import get_msisGPI, FUV_Level_2_Density_Calculation, find_hm_Nm_F2, nan_checker
import airglow.ICON_FUV_fwd_model as FUV_F # for simulating data
import netCDF4, pyglow
from dateutil import parser
from datetime import timedelta
from scipy.signal import medfilt

date = '2020-01-20'
data = True # True: use data, False: use simulation for brightness
test = False # if True, artificially creates a hot pixel with high uncertainty
median = False
epoch = 58
stripe = 3
limb = 150.
contribution ='RRMN'
reg_method = 'Tikhonov'
weight_resid = False
Spherical = True
regu_order = 2

path_dir = '/home/kamo/resources/iconfuv/nc_files/'

file_l1='../../nc_files/ICON_L1_FUV_SWP_{}_v02r000.NC'.format(date)
# file_l2='nc_files/ICON_L3_FUV_Oxygen-Profile-Night_{}_v01r000.NC'.format(date)
file_anc='../../nc_files/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)
file_GPI = '../../nc_files/ICON_Ancillary_GPI_2015-001-to-2020-044_v01r000.NC'

anc = netCDF4.Dataset(file_anc, mode='r')
l1 = netCDF4.Dataset(file_l1, mode='r')
# l2 = netCDF4.Dataset(file_l2, mode='r')
gpi = netCDF4.Dataset(file_GPI, mode='r')

mirror_dir = ['M9','M6','M3','P0','P3','P6']

mode = anc.variables['ICON_ANCILLARY_FUV_ACTIVITY'][:]
idx = np.where(mode==258)[0][epoch]
dn = parser.parse(anc.variables['ICON_ANCILLARY_FUV_TIME_UTC'][idx])

# Read the geophysical indeces
ap3 = gpi['ap3'][:]
ap = gpi['ap'][:]
year_day = gpi['year_day'][:]
f107 = gpi['f107d'][:]
# Make sure this GPI has the average f107 in it
if 'f107a' in gpi.variables.keys():
    f107a = gpi['f107a'][:]
else:
    f107a = gpi['f107d'][:]
gpi.close()

my_f107, my_f107a, my_f107p, my_apmsis = get_msisGPI(dn, year_day, f107, f107a, ap, ap3)

# FUV_AZ = anc.variables['ICON_ANCILLARY_FUV_FOV_AZIMUTH_ANGLE'][:,:,:]
# FUV_ZE = anc.variables['ICON_ANCILLARY_FUV_FOV_ZENITH_ANGLE'][:,:,:]
tanalts = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,2]
tanlons = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,1]
tanlats = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,0]
satlatlonalt = [
    anc.variables['ICON_ANCILLARY_FUV_LATITUDE'][idx],
    anc.variables['ICON_ANCILLARY_FUV_LONGITUDE'][idx],
    anc.variables['ICON_ANCILLARY_FUV_ALTITUDE'][idx]
]
local_time = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LST'][idx, -1, 2]
print('Local Time:{}'.format(str(timedelta(seconds=local_time*3600))[:-7]))
print('Orbit:{}'.format(anc.variables['ICON_ANCILLARY_FUV_ORBIT_NUMBER'][idx]))

# Only consider values above the limb
limb_i = np.where(np.squeeze(tanalts)>=limb)[0]
br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][idx,limb_i]
unmasked_ind = nan_checker(br)
limb_i0 = limb_i[unmasked_ind]
br = br[::-1]
unmasked_ind_f = nan_checker(br)
limb_i = limb_i[unmasked_ind_f]
br = br[unmasked_ind_f]
err = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idx,limb_i0]
err = err[::-1]
err[np.where(err<1e-1)] = 1 # set the error to 1 if it is 0
h = np.squeeze(tanalts[limb_i0])
h = h[::-1]
az = np.squeeze(anc.variables['ICON_ANCILLARY_FUVA_FOV_AZIMUTH_ANGLE'][idx,limb_i0,stripe])
az = az[::-1]
ze = np.squeeze(anc.variables['ICON_ANCILLARY_FUVA_FOV_ZENITH_ANGLE'][idx,limb_i0,stripe])
ze = ze[::-1]
tanlons = tanlons[limb_i0]
tanlats = tanlats[limb_i0]
tanlons = tanlons[::-1]
tanlats = tanlats[::-1]

anc.close()
l1.close()

if median is True:
    br = medfilt(br, kernel_size=1)
    # err = medfilt(err, kernel_size=3)
    # err = np.sqrt(br)

if data is False:
    # Simulate brightness profile at the given space-time using MSIS/IRI
    br_nn, phot_nn = FUV_F.get_Photons_from_Brightness_Profile_1356_nighttime(
        ze,az,satlatlonalt[0],satlatlonalt[1],satlatlonalt[2],dn,
        cont=1,
        symmetry=0, # 0 = spherical symmetry
        shperical=0, # 0 = spherical earth
        step = 100., # step size for line-of-sight integral. Larger --> runs faster
        f107=my_f107,
        f107a=my_f107a,
        f107p=my_f107p,
        apmsis=my_apmsis,
        stripes_used=1
    )

    temp, xx = FUV_F.add_noise_to_photon_and_brightness(phot_nn,stripes_used=1)
    br = temp[0,:] # only use first realization
    err = np.sqrt(br_nn)
    if test is True: # artificially create a hot pixel with high uncertainty
        br[-10] *= 20
        err[-10] *= 1e3

ver,Ne,h_centered,Sig_ver,Sig_Ne = FUV_Level_2_Density_Calculation(
    br,h,satlatlonalt,az,ze,
    Sig_Bright = np.diag(err**2), weight_resid=False,
    limb = limb,Spherical = Spherical, reg_method = reg_method,
    regu_order = regu_order, contribution =contribution,dn = dn,
    f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis
)

ver_w,Ne_w,h_centered,Sig_ver_w,Sig_Ne_w = FUV_Level_2_Density_Calculation(
    br,h,satlatlonalt,az,ze,
    Sig_Bright = np.diag(err**2), weight_resid=True,
    limb = limb,Spherical = Spherical, reg_method = reg_method,
    regu_order = regu_order, contribution =contribution,dn = dn,
    f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis
)

ne_true = np.zeros_like(Ne)
for m in range(len(ne_true)):
    pt = pyglow.Point(dn, satlatlonalt[0], satlatlonalt[1], h_centered[m], user_ind=True)
    # pt = pyglow.Point(dn, tanlats[m], tanlons[m], h_centered[m], user_ind=True)
    pt.f107 = my_f107
    pt.f107a = my_f107a
    pt.f107p = my_f107p
    pt.apmsis = my_apmsis
    pt.run_iri()
    ne_true[m] = pt.ne

hm,Nm,sig_hm,sig_Nm = find_hm_Nm_F2(Ne,h_centered,Sig_NE=Sig_Ne)
hm_w,Nm_w,sig_hm_w,sig_Nm_w = find_hm_Nm_F2(Ne_w,h_centered,Sig_NE=Sig_Ne_w)

plt.figure()
plt.plot(Ne, h_centered, label='Unwhitened')
plt.plot(Nm, hm, 'bo')
plt.plot(Ne_w, h_centered, label='Whitened')
plt.plot(Nm_w, hm_w, 'ro')
plt.plot(ne_true, h_centered, label='True')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
plt.title('Data Retrieved O$^+$ Whitening Comparison')
plt.xlabel('$O^+$ Density [$cm^{-3}$]')
plt.ylabel('Altitude [km]')
plt.legend()

plt.figure()
plt.plot(np.diag(Sig_Ne), h_centered, label='Unwhitened')
plt.plot(np.diag(Sig_Ne_w), h_centered, label='Whitened')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
plt.title('Data Retrieved Ne Uncertainty Whitening Comparison')
plt.xlabel('VER Uncertainty [$cm^{-3}$]')
plt.ylabel('Altitude [km]')
plt.legend()

plt.figure()
plt.plot(br, h, label='Brightness')
plt.plot(err, h, label='Error')
# plt.plot(br_nn, h, label='No Noise')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
plt.title('Simulated Brightness' if data is False else 'Brightness')
plt.xlabel('135.6 nm Brightness [R]')
plt.ylabel('Tangent Altitudes [km]')
plt.show()
