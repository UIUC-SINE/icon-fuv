import numpy as np
from airglow.FUV_L2 import *
import netCDF4, pyglow
from dateutil import parser
from datetime import timedelta
import airglow.ICON_FUV_fwd_model as FUV_F # for simulating data

path_dir = '/home/kamo/resources/iconfuv/nc_files/'

def calc_solution_gd(A,b,nu,numiter,h,plot=False):
    ver = np.zeros_like(b)
    Ata = A.T.dot(A)
    Atb = A.T.dot(b)
    vers = []
    for i in range(numiter):
        ver = ver - nu * (Ata.dot(ver) - Atb)
        ver[ver<0] = 1e-5
        vers.append(ver)
        if plot:
            plt.plot(ver, h)
            plt.pause(0.1)
            plt.clf()
    return vers

def custom_destarring_orbit(l1, epoch, stripe):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    mode_night = (mode == 2).astype(np.int)
    nights = np.diff(mode_night, prepend=0)
    nights[nights==-1] = 0
    idxs = np.where(mode==2)[0][:]
    nights = np.cumsum(nights)[idxs]
    night = nights[epoch]
    night_ind = np.where(nights==night)[0]
    idx_new = epoch - night_ind[0]
    br = np.zeros((6, len(night_ind), 256))
    br_err = np.zeros((6, len(night_ind), 256))
    mask = np.zeros_like(br, dtype=np.bool)
    for i in range(6):
        tmp = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]][idxs[night_ind],:]
        br_err[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]][idxs[night_ind],:].filled(fill_value=0)
        mask[i] = tmp.mask
        br[i] = tmp.filled(fill_value=0)
    br_corrected, br_err_modified = l1_correction_orbit(br, br_err)
    br_corrected = np.ma.array(br_corrected, mask=mask).filled(fill_value=np.nan)
    br_err_modified = np.ma.array(br_err_modified, mask=mask).filled(fill_value=np.nan)
    return br_corrected[stripe, idx_new], br_err_modified[stripe, idx_new]

def custom_l25(date='2020-01-02', epoch=100, stripe=3, limb=150.,
    contribution='RRMN', l1_rev='v03r000', anc_rev='v01r000', method='derivative',
    weight_resid=False, reg_order=2, reg_param=0, br_fact=None, iri_comp=False,
    nonnegative=False):

    file_l1 = path_dir + 'l1/ICON_L1_FUV_SWP_{}_{}.NC'.format(date, l1_rev)
    file_anc = path_dir + 'l0/ICON_L0P_FUV_Ancillary_{}_{}.NC'.format(date, anc_rev)
    file_GPI = path_dir + 'ICON_Ancillary_GPI_2015-001-to-2020-132_v01r000.NC'

    anc = netCDF4.Dataset(file_anc, mode='r')
    l1 = netCDF4.Dataset(file_l1, mode='r')
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
    print('Local Time:{}'.format(str(timedelta(seconds=local_time*3600))[:-7]))
    print('Orbit:{}'.format(anc.variables['ICON_ANCILLARY_FUV_ORBIT_NUMBER'][idx]))

    # Only consider values above the limb
    limb_i = np.where(np.squeeze(tanalts)>=limb)[0]
    # br, err = custom_destarring_orbit(l1, epoch, stripe)
    br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[stripe]][idx,:]
    err = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idx,:]
    if nonnegative is True:
        br[br<0] = 0
        # br+=25
    br = br[limb_i]
    if br_fact is not None:
        br /= br_fact
    unmasked_ind = nan_checker(br)
    limb_i0 = limb_i[unmasked_ind]
    br_orig = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][idx,limb_i0]
    br_orig = br_orig[::-1]
    br = br[::-1]
    unmasked_ind_f = nan_checker(br)
    limb_i = limb_i[unmasked_ind_f]
    br = br[unmasked_ind_f]
    err_orig = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idx,limb_i0]
    err_orig = err_orig[::-1]
    err = err[limb_i0]
    err = err[::-1]
    # err[np.where(err<1e-1)] = 1 # set the error to 1 if it is 0
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
    if iri_comp:
        br, phot_nn = FUV_F.get_Photons_from_Brightness_Profile_1356_nighttime(
            ze,az,satlatlonalt[0],satlatlonalt[1],satlatlonalt[2],dn,
            cont=2,
            symmetry=0, # 0 = spherical symmetry
            shperical=0, # 0 = spherical earth
            step = 100., # step size for line-of-sight integral. Larger --> runs faster
            f107=my_f107,
            f107a=my_f107a,
            f107p=my_f107p,
            apmsis=my_apmsis,
            stripes_used=1
        )
        err = np.sqrt(br) + 1e-1

    anc.close()
    l1.close()

    h_centered = h + ( np.roll(h,1) - h )/2
    h_centered[0] = h[0] + ( h_centered[1] - h[1] )
    A = create_cells_Matrix_spherical_symmetry(ze[:],satlatlonalt[2])

    A = A.copy()
    Bright = br.copy()
    Sig_Bright = np.diag(err**2)

    # If we want to use a weighted least-squares formulation, we can "whiten" the errors and use
    # the exact same code.
    if weight_resid:
        W = sqrtm(np.linalg.inv(Sig_Bright)) # whitening matrix
        A = W.dot(A)
        Bright = W.dot(Bright)
        Sig_Bright = np.eye(len(Bright)) # whitening diagonalizes the covariance matrix

    # Check if reg_param is a vector or not
    if np.size(reg_param) == 1:
        # if its not a vector only if zero it will calculate the vector. Otherwise it will use the single value given.
        if reg_param == 0:
            reg_param = create_alpha_values(A)

    residual = np.zeros(len(reg_param))
    seminorm = np.zeros(len(reg_param))
    sols = np.zeros((len(reg_param), len(Bright)))
    nes = np.zeros((len(reg_param), len(Bright)))

    # create the matrix that defines the order of the regularization.
    L = get_rough_matrix(len(Bright),reg_order)

    # For every regularization parameter, estimate solution
    for i in range(0,len(reg_param)):
        sols[i] = calc_solution(A,Bright,reg_param[i],L) # for speed, we can omit the uncertainty prop here
        r = A.dot(sols[i]) - Bright
        residual[i] = np.linalg.norm(r)
        seminorm[i] = np.linalg.norm(L.dot(sols[i]))
        # FIXME - commenting out for speed up
        nes[i] = calculate_electron_density(sols[i], satlatlonalt, h_centered, dn, contribution=contribution,f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis, az=az, ze=ze)
        # nes[i] = np.zeros_like(sols[i])

    # Find the optimal regularization parameter using the maximum second derivative method
    reg_corner = Maximum_Curvature_gradiens(residual,seminorm,reg_param,method=method)
    # reg_corner = 2500.

    # Calculate the solution with the optimal parameter (and, if desired, also the uncertainty)
    if Sig_Bright is None:
        ver = calc_solution(A,Bright,reg_corner,L)
    else:
        # Note that if weight_resid was True, we already whitened the errors so don't need to weight
        # again in the sub-function
        ver,Sig_ver = calc_solution(A,Bright,reg_corner,L,Sig_Bright=Sig_Bright)

    ne, Sig_Ne = calculate_electron_density(ver, satlatlonalt, h_centered, dn, Sig_VER=Sig_ver, contribution=contribution,f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis, az=az, ze=ze)
    return (br,err,ver,np.sqrt(np.diagonal(Sig_ver)),ne,
        # np.sqrt(np.diagonal(Sig_Ne)),residual,seminorm,reg_corner,sols,
        np.sqrt(np.diagonal(Sig_Ne)),residual,seminorm,reg_corner,nes,sols,
        br_orig,err_orig,h_centered, A)
