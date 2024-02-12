# 2023-12-11 - Ulas Kamaci
# This code combines the two types of asymmetry flags, one with purely using 
# FUV data and the other with external TEC maps, as well as the aurora flag.

import matplotlib.pyplot as plt
import numpy as np
import netCDF4, datetime
from dateutil import parser
from copy import deepcopy
import glob, os
from tqdm import tqdm
path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

def asymmetry_flagger(brmax, brmax_alt):
    asymm_flag = np.zeros(len(brmax), dtype=bool)

    asymm_center = (brmax>100) & (brmax_alt<200)
    asymm_center = asymm_center.data
    asymm_center_int = asymm_center.astype(int)
    asymm_check = np.convolve(asymm_center_int, np.ones(7), 'valid')

    if np.nanmax(asymm_check) >= 5:
        asymm_center = np.nanargmax( np.convolve(asymm_center_int, np.hamming(13), 'valid') )

        asymm_flag = np.zeros(len(brmax), dtype=bool)
        asymm_flag[max(0,asymm_center-25):asymm_center+25] = True

        asymm_flag = asymm_flag & (brmax_alt < 250)
    
    return asymm_flag


def aurora_flagger(brmax_alt, brmax_tlat):
    aurora_flag = (brmax_tlat>30) & (brmax_alt<220)
    return aurora_flag

def all_flagger(file_l1=None, file_l2=None, stripe=2):
    nighttime_counter = 0
    flag_counter = 0
    asymm_flag_counter = 0
    aurora_flag_counter = 0
    qual1_counter = 0
    asymm_qual1flag_counter = 0
    qual1flag_counter = 0
    aurora_qual1flag_counter = 0
    hmf2s_all = []
    flagqual1_locs = []
    asymm_flagqual1_locs = []
    aurora_flagqual1_locs = []
    flag_locs = []
    asymm_flag_locs = []
    aurora_flag_locs = []

    l1 = netCDF4.Dataset(file_l1, mode='r')
    l2 = netCDF4.Dataset(file_l2, mode='r')
    mirror_dir = ['M9','M6','M3','P0','P3','P6']


    mode = l1.variables['ICON_L1_FUV_Mode'][:]

    # Get variables from netCDF file
    dn = [] # UTC
    for d in l2.variables['ICON_L25_UTC_Time']:
        dn.append(parser.parse(d))
    dn = np.array(dn)

    orbits = l2.variables['ICON_L25_Orbit_Number'][:]
    quals = l2.variables['ICON_L25_Quality'][:]
    flags = l2.variables['ICON_L25_Quality_Flags'][:]

    orbit_list = orbits

    for orbit in np.unique(orbit_list):
        try:
            orbit_ind = np.squeeze(np.where(orbits == orbit))

            if orbit_ind.size < 2:
                continue

            ds = np.array([i.total_seconds() for i in dn-dn[orbit_ind][0]])
            orbit_ind = np.squeeze(np.where(abs(ds) < 2000.))

            hmf2s = l2.variables['ICON_L25_HMF2'][orbit_ind, stripe]

            idx = np.where(mode==2)[0][orbit_ind]

            # X = np.transpose([dn,]*l2.dimensions['Altitude'].size)[orbit_ind]
            Y = l2.variables['ICON_L25_O_Plus_Profile_Altitude'][orbit_ind,:,stripe]
            X = np.transpose([dn,]*Y.shape[1])[orbit_ind]
            Y = np.ma.filled(Y, fill_value = np.nanmax(Y))
            T = l2.variables['ICON_L25_O_Plus_Profile_Latitude'][orbit_ind,:,stripe]
            T = np.ma.filled(T, fill_value = np.nanmax(T))
            Z = l2.variables['ICON_L25_O_Plus_Density'][orbit_ind,:,stripe]
            brc = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[stripe]][idx,:]

            out = np.diff(X,axis=0)
            mask = np.vstack([out > datetime.timedelta(seconds=24),np.ones([1,np.size(out,1)],dtype=bool)])
            Zm = np.ma.MaskedArray(Z,mask)
            Bmc = np.ma.ones(Zm.shape)*Zm # brightness counterpart
            maskcounts = np.sum(Zm.mask==True, axis=1)
            for i in range(Bmc.shape[0]):
                Bmc[i,maskcounts[i]:] = brc[i][::-1][:Bmc.shape[1]-maskcounts[i]]

            # Peak brightness
            brmax = Bmc.max(axis=1)
            brmax_alt = Y[range(len(brmax)), Bmc.argmax(axis=1)]
            brmax_tlat = T[range(len(brmax)), Bmc.argmax(axis=1)]


            aurora_flag = aurora_flagger(brmax_alt, brmax_tlat)
            asymm_flag = asymmetry_flagger(brmax, brmax_alt) 

            nighttime_counter += orbit_ind.size
            qual = quals[orbit_ind, stripe]

            if orbit_ind.size < 2:
                continue

            flag_locations = asymm_flag | aurora_flag
            asymm_flag[aurora_flag] = 0

            for st in [0,1,2,3,4,5]:
                locs = flag_locations & (quals[orbit_ind, st]==1)
                quals[orbit_ind[locs], st] = 0.5

                flags[orbit_ind, st] += 64*aurora_flag + 128*asymm_flag


            asymm_flag_counter += sum(asymm_flag)
            aurora_flag_counter += sum(aurora_flag)
            flag_counter += sum(flag_locations)
            qual1_counter += sum(qual==1)
            asymm_flagqual1_locations = asymm_flag & (qual==1)
            aurora_flagqual1_locations = aurora_flag & (qual==1)
            flagqual1_locations = flag_locations & (qual==1)
            asymm_qual1flag_counter += sum(asymm_flagqual1_locations)
            aurora_qual1flag_counter += sum(aurora_flagqual1_locations)
            qual1flag_counter += sum(flagqual1_locations)

            hmf2s_all.extend(hmf2s[qual==1])
            flagqual1_locs.extend(flagqual1_locations[qual==1])
            flag_locs.extend(flag_locations)
            asymm_flag_locs.extend(asymm_flag)
            aurora_flag_locs.extend(aurora_flag)
            asymm_flagqual1_locs.extend(asymm_flagqual1_locations[qual==1])
            aurora_flagqual1_locs.extend(aurora_flagqual1_locations[qual==1])

        except Exception as e:
            raise

    l1.close()
    l2.close()

    return (
        nighttime_counter, 
        flag_counter, 
        asymm_flag_counter, 
        aurora_flag_counter, 
        qual1_counter, 
        qual1flag_counter,
        asymm_qual1flag_counter, 
        aurora_qual1flag_counter, 
        hmf2s_all, 
        flagqual1_locs,
        asymm_flagqual1_locs,
        aurora_flagqual1_locs,
        quals,
        flags,
        flag_locs,
        asymm_flag_locs,
        aurora_flag_locs
        )


if __name__=='__main__':
    # %% cell0
    files_l1 = glob.glob(path_dir + 'l1/*SWP_2022*')
    files_l2 = glob.glob(path_dir + 'l2/2022/*')
    files_l1.sort()
    files_l2.sort()
    stripe=2
    path_fig = path_dir + 'figures/aurora_flagging/all_flagger_py2_test'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)

    nighttime_counter = 0
    flag_counter = 0
    asymm_flag_counter = 0 
    aurora_flag_counter = 0
    qual1_counter = 0
    qual1flag_counter = 0
    asymm_qual1flag_counter = 0 
    aurora_qual1flag_counter = 0
    hmf2s_all = []
    qual1flag_locs = []
    asymm_qual1flag_locs = []
    aurora_qual1flag_locs = []

    for file_l1,file_l2 in tqdm(zip(files_l1[9:10], files_l2[9:10])):
        (nighttime_ctr, 
        flag_ctr, 
        asymm_flag_ctr,
        aurora_flag_ctr,
        qual1_ctr,
        qual1flag_ctr,
        asymm_qual1flag_ctr,
        aurora_qual1flag_ctr,
        hmf2_all,
        qual1flag_loc,
        asymm_qual1flag_loc,
        aurora_qual1flag_loc,
        quals,
        flags,
        flag_locs,
        asymm_flag_locs,
        aurora_flag_locs
        ) = all_flagger(file_l1=file_l1, file_l2=file_l2, stripe=stripe)

        nighttime_counter += nighttime_ctr
        flag_counter += flag_ctr
        asymm_flag_counter += asymm_flag_ctr 
        aurora_flag_counter += aurora_flag_ctr
        qual1_counter += qual1_ctr
        qual1flag_counter += qual1flag_ctr
        asymm_qual1flag_counter += asymm_qual1flag_ctr 
        aurora_qual1flag_counter += aurora_qual1flag_ctr
        hmf2s_all.extend(hmf2_all)
        qual1flag_locs.extend(qual1flag_loc)
        asymm_qual1flag_locs.extend(asymm_qual1flag_loc)
        aurora_qual1flag_locs.extend(aurora_qual1flag_loc)

    hmf2s_all = np.array(hmf2s_all)
    qual1flag_locs = np.array(qual1flag_locs)
    asymm_qual1flag_locs = np.array(asymm_qual1flag_locs)
    aurora_qual1flag_locs = np.array(aurora_qual1flag_locs)

    np.save(path_fig+'/hmf2s_all.npy', hmf2s_all)
    np.save(path_fig+'/qual1flag_locs.npy', qual1flag_locs)
    np.save(path_fig+'/asymm_qual1flag_locs.npy', asymm_qual1flag_locs)
    np.save(path_fig+'/aurora_qual1flag_locs.npy', aurora_qual1flag_locs)

    print('Nighttime Counter: {}'.format(nighttime_counter))
    print('All Flag Counter: {} ({:.1f}%)'.format(flag_counter, 100*flag_counter/nighttime_counter))
    print('Asymm Flag Counter: {} ({:.1f}%)'.format(asymm_flag_counter, 100*asymm_flag_counter/nighttime_counter))
    print('Aurora Flag Counter: {} ({:.1f}%)'.format(aurora_flag_counter, 100*aurora_flag_counter/nighttime_counter))
    print('Qual=1 Counter: {}'.format(qual1_counter))
    print('Flag (Qual=1) Counter: {} ({:.1f}%)'.format(qual1flag_counter, 100*qual1flag_counter/qual1_counter))
    print('Asymm Flag (Qual=1) Counter: {} ({:.1f}%)'.format(asymm_qual1flag_counter, 100*asymm_qual1flag_counter/qual1_counter))
    print('Aurora Flag (Qual=1) Counter: {} ({:.1f}%)'.format(aurora_qual1flag_counter, 100*aurora_qual1flag_counter/qual1_counter))

    # hmf2s_all = np.load(path_fig+'/hmf2s_all.npy')
    # qual1flag_locs = np.load(path_fig+'/qual1flag_locs.npy')
    # asymm_qual1flag_locs = np.load(path_fig+'/asymm_qual1flag_locs.npy')
    # asymm_tec_qual1flag_locs = np.load(path_fig+'/asymm_tec_qual1flag_locs.npy')
    # aurora_qual1flag_locs = np.load(path_fig+'/aurora_qual1flag_locs.npy')

    plt.figure()
    (cts1, bins, bar) = plt.hist(hmf2s_all, 40, edgecolor='black', label='All')
    (cts2, bins, bar) = plt.hist(hmf2s_all[~qual1flag_locs], bins=bins, edgecolor='black', label='Flagged')
    plt.title('Histogram of All vs All-Flagged HmF2s of 2022')
    plt.xlabel('L2.5 HmF2 (km)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig(path_fig+'/hmf2_hist_all_vs_all_flagged.png', dpi=300)

    plt.figure()
    (cts1, bins, bar) = plt.hist(hmf2s_all, 40, edgecolor='black', label='All')
    (cts2, bins, bar) = plt.hist(hmf2s_all[~aurora_qual1flag_locs], bins=bins, edgecolor='black', label='Flagged')
    plt.title('Histogram of All vs Aurora-Flagged HmF2s of 2022')
    plt.xlabel('L2.5 HmF2 (km)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig(path_fig+'/hmf2_hist_all_vs_aurora_flagged.png', dpi=300)

    plt.figure()
    (cts1, bins, bar) = plt.hist(hmf2s_all, 40, edgecolor='black', label='All')
    (cts2, bins, bar) = plt.hist(hmf2s_all[~asymm_qual1flag_locs], bins=bins, edgecolor='black', label='Flagged')
    plt.title('Histogram of All vs Asymm-Flagged HmF2s of 2022')
    plt.xlabel('L2.5 HmF2 (km)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig(path_fig+'/hmf2_hist_all_vs_asymm_flagged.png', dpi=300)
