import os, glob, netCDF4, pyglow
from airglow.FUV_L2 import get_msisGPI, FUV_Level_2_Density_Calculation, find_hm_Nm_F2, nan_checker
import airglow.ICON_FUV_fwd_model as FUV_F # for simulating data
import numpy as np


def get_oplus(l1=None, anc=None, epoch=100, stripe=3, bkgcor=False):
    limb = 150.
    contribution ='RRMN'
    reg_method = 'Tikhonov'
    weight_resid = False
    Spherical = True
    regu_order = 2
    reg_param = 2500.

    file_GPI = path_dir + 'ICON_Ancillary_GPI_2015-001-to-2020-187_v01r000.NC'
    gpi = netCDF4.Dataset(file_GPI, mode='r')

    mirror_dir = ['M9','M6','M3','P0','P3','P6']

    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    idx = np.where(mode==2)[0][epoch]
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

    tanalts = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,2]
    tanlons = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,1]
    tanlats = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][idx,:,stripe,0]
    satlatlonalt = [
        anc.variables['ICON_ANCILLARY_FUV_LATITUDE'][idx],
        anc.variables['ICON_ANCILLARY_FUV_LONGITUDE'][idx],
        anc.variables['ICON_ANCILLARY_FUV_ALTITUDE'][idx]
    ]
    local_time = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LST'][idx, -1, 2]

    limb_i = np.where(np.squeeze(tanalts)>=limb)[0]
    br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[stripe]][idx,:]
    if bkgcor is True:
        # br -= np.nanmedian(br[tanalts>400])
        med = np.median(br[10:50])
        med=0 if med>0 else med
        br -= med
    br = br[limb_i]
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
    br[br<0] = 0

    ver,Ne,h_centered,Sig_ver,Sig_Ne = FUV_Level_2_Density_Calculation(
        br,h,satlatlonalt,az,ze,
        Sig_Bright = np.diag(err**2), weight_resid=False,
        limb = limb,Spherical = Spherical, reg_method = reg_method,
        regu_order = regu_order, contribution =contribution,dn = dn,
        f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis,
        reg_param=reg_param
    )

    return Ne, h_centered
