# Ulas Kamaci - 01/20/2020
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from airlow.FUV_L2 import get_msisGPI, FUV_Level_2_Density_Calculation, find_hm_Nm_F2, create_cells_Matrix_spherical_symmetry, calculate_electron_density
import airglow.ICON_FUV_fwd_model as FUV_F # for simulating data
import netCDF4, pyglow, sys
from dateutil import parser
from datetime import timedelta

def whitening_tester(epoch, numreal):
    # set the simulation parameters
    test = False # if True, artificially creates a hot pixel with high uncertainty
    normalize = False # if True, normalize the uncertainties with the brightness
    stripe = 4
    limb = 150.
    contribution ='RRMN'
    reg_method = 'Tikhonov'
    Spherical = True
    regu_order = 2
    path = './whitening_tests/ep{}_st{}'.format(epoch,stripe)
    if os.path.isdir(path) is True:
        path = '{}_2'.format(path)
    os.mkdir(path)

    # Pick the date&time(epoch) for the simulation
    date = '2020-01-01'

    # Retrieve the geometry from the files
    file_anc='nc_files/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)
    file_GPI = 'nc_files/ICON_Ancillary_GPI_2015-001-to-2020-005_v01r000.NC'

    anc = netCDF4.Dataset(file_anc, mode='r')
    gpi = netCDF4.Dataset(file_GPI, mode='r')

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

    tanalts = anc.variables['ICON_ANCILLARY_FUV_TANGENTPOINTS_LATLONALT'][idx,:,stripe,2]
    tanlons = anc.variables['ICON_ANCILLARY_FUV_TANGENTPOINTS_LATLONALT'][idx,:,stripe,1]
    tanlats = anc.variables['ICON_ANCILLARY_FUV_TANGENTPOINTS_LATLONALT'][idx,:,stripe,0]
    satlatlonalt = [
        anc.variables['ICON_ANCILLARY_FUV_LATITUDE'][idx],
        anc.variables['ICON_ANCILLARY_FUV_LONGITUDE'][idx],
        anc.variables['ICON_ANCILLARY_FUV_ALTITUDE'][idx]
    ]
    local_time = anc.variables['ICON_ANCILLARY_FUV_TANGENTPOINTS_LST'][idx, -1, 2]
    orbit = anc.variables['ICON_ANCILLARY_FUV_ORBIT_NUMBER'][idx]
    print('Local Time:{}'.format(str(timedelta(seconds=local_time*3600))[:-7]))
    print('Orbit:{}'.format(orbit))

    # Only consider values above the limb
    limb_i = np.where(np.squeeze(tanalts)>=limb)[0]
    h = np.squeeze(tanalts[limb_i])
    h = h[::-1]
    az = np.squeeze(anc.variables['ICON_ANCILLARY_FUV_FOV_AZIMUTH_ANGLE'][idx,limb_i,stripe])
    az = az[::-1]
    ze = np.squeeze(anc.variables['ICON_ANCILLARY_FUV_FOV_ZENITH_ANGLE'][idx,limb_i,stripe])
    ze = ze[::-1]
    tanlons = tanlons[limb_i]
    tanlats = tanlats[limb_i]
    tanlons = tanlons[::-1]
    tanlats = tanlats[::-1]

    anc.close()

    # Simulate the brightness profile using MSIS/IRI at the given time/geometry
    br_nn, phot_nn = FUV_F.get_Photons_from_Brightness_Profile_1356_nighttime(
        ze,az,satlatlonalt[0],satlatlonalt[1],satlatlonalt[2],dn,
        cont=1,
        symmetry=0, # 0 = spherical symmetry
        shperical=0, # 0 = spherical earth
        step = 50., # step size for line-of-sight integral. Larger --> runs faster
        f107=my_f107,
        f107a=my_f107a,
        f107p=my_f107p,
        apmsis=my_apmsis,
        stripes_used=1
    )

    nm_err = np.zeros(numreal)
    hm_err = np.zeros(numreal)
    nm_err_w = np.zeros(numreal)
    hm_err_w = np.zeros(numreal)
    ne_array = np.zeros((numreal, br_nn.shape[0]))
    ne_w_array = np.zeros((numreal, br_nn.shape[0]))
    for iter in range(numreal):
        print('{}/{}'.format(iter+1, numreal))
        temp, xx, err = FUV_F.add_noise_to_photon_and_brightness(phot_nn,stripes_used=1,ret_cov=True)
        br = temp[0,:] # only use first realization
        err = np.sqrt(np.diag(err))
        if normalize is True:
            err /= br_nn
        if test is True: # artificially create a hot pixel with high uncertainty
            br[-10] *= 20
            err[-10] *= 1e3

        # Run the inversion code with different specifications
        ver,Ne,h_centered,Sig_ver,Sig_Ne = FUV_Level_2_Density_Calculation(
            br,h,satlatlonalt,az,ze,
            Sig_Bright = np.diag(err**2), weight_resid=False,
            limb = limb,Spherical = Spherical, reg_method = reg_method,
            regu_order = regu_order, contribution =contribution,dn = dn,
            f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis
        )

        ne_array[iter] = Ne

        ver_w,Ne_w,h_centered,Sig_ver_w,Sig_Ne_w = FUV_Level_2_Density_Calculation(
            br,h,satlatlonalt,az,ze,
            Sig_Bright = np.diag(err**2), weight_resid=True,
            limb = limb,Spherical = Spherical, reg_method = reg_method,
            regu_order = regu_order, contribution =contribution,dn = dn,
            f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis
        )

        ne_w_array[iter] = Ne_w

        # # Simulate the O plus profile at the given time/geometry
        # ne_true = np.zeros_like(Ne)
        # for m in range(len(ne_true)):
        #     pt = pyglow.Point(dn, satlatlonalt[0], satlatlonalt[1], h_centered[m], user_ind=True)
        #     # pt = pyglow.Point(dn, tanlats[m], tanlons[m], h_centered[m], user_ind=True)
        #     pt.f107 = my_f107
        #     pt.f107a = my_f107a
        #     pt.f107p = my_f107p
        #     pt.apmsis = my_apmsis
        #     pt.run_iri()
        #     ne_true[m] = pt.ne

        hm,Nm,sig_hm,sig_Nm = find_hm_Nm_F2(Ne,h_centered,Sig_NE=Sig_Ne)
        hm_w,Nm_w,sig_hm_w,sig_Nm_w = find_hm_Nm_F2(Ne_w,h_centered,Sig_NE=Sig_Ne_w)
        if iter == 0:
            D = create_cells_Matrix_spherical_symmetry(ze[:],satlatlonalt[2])
            Dinv = np.linalg.inv(D)
            verinv = np.dot(Dinv, br_nn)
            Ne_true = calculate_electron_density(verinv, satlatlonalt, h_centered, dn, Sig_VER=None, contribution=contribution,f107=my_f107, f107a=my_f107a, f107p=my_f107p, apmsis=my_apmsis, az=az, ze=ze)
            hm_t,Nm_t = find_hm_Nm_F2(Ne_true,h_centered)

        nm_err[iter] = Nm_t - Nm
        hm_err[iter] = hm_t - hm
        nm_err_w[iter] = Nm_t - Nm_w
        hm_err_w[iter] = hm_t - hm_w

        fig = plt.figure()
        plt.plot(Ne, h_centered, label='Unwhitened')
        plt.plot(Ne_w, h_centered, label='Whitened')
        plt.plot(Ne_true, h_centered, label='True Inv')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
        plt.title('Simulation Retrieved O$^+$ Whitening Comparison')
        plt.xlabel('$O^+$ Density [$cm^{-3}$]')
        plt.ylabel('Altitude [km]')
        plt.legend()
        fig.savefig('{}/profile_{}.png'.format(path,iter))

    np.save('{}/nm_err.npy'.format(path), nm_err)
    np.save('{}/nm_err_w.npy'.format(path), nm_err_w)
    np.save('{}/hm_err.npy'.format(path), hm_err)
    np.save('{}/hm_err_w.npy'.format(path), hm_err_w)
    np.save('{}/hm_nm_true.npy'.format(path), np.array([hm_t,Nm_t]))

    nm_std = np.sqrt(np.mean(nm_err**2))
    nm_std_perc = nm_std / Nm_t * 100
    nm_std_w = np.sqrt(np.mean(nm_err_w**2))
    nm_std_perc_w = nm_std_w / Nm_t * 100
    hm_std = np.sqrt(np.mean(hm_err**2))
    hm_std_perc = hm_std / hm_t * 100
    hm_std_w = np.sqrt(np.mean(hm_err_w**2))
    hm_std_perc_w = hm_std_w / hm_t * 100

    nm_std_perc_arr = np.zeros(numreal-1)
    nm_std_perc_arr_w = np.zeros(numreal-1)
    hm_std_perc_arr = np.zeros(numreal-1)
    hm_std_perc_arr_w = np.zeros(numreal-1)
    for k in range(numreal-1):
        nm_std = np.sqrt(np.mean(nm_err[:k+1]**2))
        nm_std_perc_arr[k] = nm_std / Nm_t * 100
        nm_std_w = np.sqrt(np.mean(nm_err_w[:k+1]**2))
        nm_std_perc_arr_w[k] = nm_std_w / Nm_t * 100
        hm_std = np.sqrt(np.mean(hm_err[:k+1]**2))
        hm_std_perc_arr[k] = hm_std / hm_t * 100
        hm_std_w = np.sqrt(np.mean(hm_err_w[:k+1]**2))
        hm_std_perc_arr_w[k] = hm_std_w / hm_t * 100

    fig = plt.figure()
    plt.plot(nm_std_perc_arr, label='Unwhitened')
    plt.plot(nm_std_perc_arr_w, label='Whitened')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
    plt.title('Nm Empirical Uncertainty vs Sample Number')
    plt.xlabel('Samples')
    plt.ylabel('Uncertainty [$cm^{-3}$]')
    plt.legend()
    fig.savefig('{}/nm_std_progress.png'.format(path))

    fig = plt.figure()
    plt.plot(hm_std_perc_arr, label='Unwhitened')
    plt.plot(hm_std_perc_arr_w, label='Whitened')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
    plt.title('hm Empirical Uncertainty vs Sample Number')
    plt.xlabel('Samples')
    plt.ylabel('Uncertainty [km]')
    plt.legend()
    fig.savefig('{}/hm_std_progress.png'.format(path))

    ne_avg = np.mean(ne_array, axis=0)
    ne_w_avg = np.mean(ne_w_array, axis=0)

    fig = plt.figure()
    plt.plot(ne_avg, h_centered, label='Unwhitened Avg')
    plt.plot(ne_w_avg, h_centered, label='Whitened Avg')
    plt.plot(Ne_true, h_centered, label='True Inv')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
    plt.title('AVERAGE Retrieved O$^+$ Whitening Comparison')
    plt.xlabel('$O^+$ Density [$cm^{-3}$]')
    plt.ylabel('Altitude [km]')
    plt.legend()
    fig.savefig('{}/profile_avg.png'.format(path))

    f = open('{}/info.txt'.format(path), 'w+')
    f.write('Date:{} \n'.format(date))
    f.write('Local Time:{} \n'.format(str(timedelta(seconds=local_time*3600))[:-7]))
    f.write('Orbit:{} \n'.format(orbit))
    f.write('Percentage NmF2 error for whitening: {}% \n'.format(nm_std_perc_w))
    f.write('Percentage NmF2 error for unwhitening: {}% \n'.format(nm_std_perc))
    f.write('Percentage hmF2 error for whitening: {}% \n'.format(hm_std_perc_w))
    f.write('Percentage hmF2 error for unwhitening: {}% \n'.format(hm_std_perc))
    f.close()

if __name__== "__main__":
    whitening_tester(int(sys.argv[1]), int(sys.argv[2]))
    # plt.figure()
    # plt.plot(np.diag(Sig_ver), h_centered, label='Unwhitened')
    # plt.plot(np.diag(Sig_ver_w), h_centered, label='Whitened')
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
    # plt.title('Data Retrieved VER Uncertainty Whitening Comparison')
    # plt.xlabel('VER Uncertainty [$ph/cm^{3}$]')
    # plt.ylabel('Altitude [km]')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(br, h, label='Noisy')
    # plt.plot(br_nn, h, label='No Noise')
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,2))
    # plt.title('Simulated Brightness')
    # plt.xlabel('135.6 nm Brightness [R]')
    # plt.ylabel('Tangent Altitudes [km]')
    # plt.show()
