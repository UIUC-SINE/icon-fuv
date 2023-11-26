## Ulas Kamaci - 2022-02-05

import numpy as np
import matplotlib.pyplot as plt
import netCDF4, glob, datetime
import pandas as pd
from dateutil import parser
from iconfuv.misc import profiler
from scipy.ndimage import convolve1d

def clusterer(vals, times, gap=datetime.timedelta(days=7)):
    diff = np.diff(times)
    inds = np.append(np.append(0, np.where(diff > gap)[0] + 1), len(times))
    medvals = []
    medtimes = []
    for i in range(len(inds)-1):
        medvals.append(np.median(vals[inds[i]:inds[i+1]]))
        medtimes.append(timemedian(times[inds[i]:inds[i+1]]))
    return np.array(medvals), medtimes

def timemedian(times):
    return np.min(times) + np.median(times - np.min(times))

def l1_qual(profiles):
    sigmeans = np.zeros(len(profiles))
    noisestds = np.zeros(len(profiles))
    for i in range(len(profiles)):
        br_lp = convolve1d(profiles[i], 0.05*np.ones((20)), mode='reflect')
        br_hp = profiles[i] - br_lp
        mi = max(min(np.nanargmax(br_lp),len(br_lp)-20), 20)
        sigmeans[i] = np.nanmean(br_lp[mi-20:mi+20])
        noisestds[i] = max(5, np.nanstd(br_hp))
    return np.invert(((sigmeans < 3 * noisestds) | (sigmeans < 1)))

def modify_condition(condition, ind):
    ar = np.zeros_like(condition)
    ind2 = [i[ind] for i in np.where(condition)]
    ar[tuple(ind2)] = True
    return ar

def run0():
    file_dir_l1 = '/home/kamo/resources/icon-fuv/ncfiles/l1/2020/'
    file_dir_l2 = '/home/kamo/resources/icon-fuv/ncfiles/l2/2020/'

    files_l1 = glob.glob(file_dir_l1+'*')
    files_l1.sort()
    files_l2 = glob.glob(file_dir_l2+'*')
    files_l2.sort()

    # dim0:[1:n, 0:s], dim1:[1:l1, 0:no_l1], dim2:[1:med, 0:no_med]
    hmf2s = [[[[], []], [[], []]], [[[], []], [[], []]]]
    nmf2s = [[[[], []], [[], []]], [[[], []], [[], []]]]
    times = [[[[], []], [[], []]], [[[], []], [[], []]]]

    for file_l1,file_l2 in zip(files_l1,files_l2):
        print(file_l2)
        l1 = netCDF4.Dataset(file_l1, 'r')
        l2 = netCDF4.Dataset(file_l2, 'r')
        hmf2 = l2.variables['ICON_L25_HMF2'][:]
        nmf2 = l2.variables['ICON_L25_NMF2'][:]
        mlat = l2.variables['ICON_L25_Magnetic_Latitude'][:]
        qual = l2.variables['ICON_L25_Quality'][:]
        time = l2.variables['ICON_L25_UTC_Time'][:]
        time = np.array([parser.parse(i) for i in time])
        slt = l2.variables['ICON_L25_Local_Solar_Time'][:]
        alt_dim = l2.dimensions['Altitude'].size
        l2.close()
        FUV_mode = l1.variables['ICON_L1_FUV_Mode'][:]
        profiles = profiler(l1, clean=True)[:,-alt_dim:,FUV_mode==2].transpose(2,0,1) #dim:[ep,6,256]
        l1.close()

        condition_n = (
            ((slt > 20) & (slt < 22)) &
            (qual==1) &
            ((mlat > 12) & (mlat < 14))
        )
        condition_s = (
            ((slt > 20) & (slt < 22)) &
            (qual==1) &
            ((mlat > -14) & (mlat < -12))
        )

        l1ind_n = l1_qual(profiles[condition_n])
        l1ind_s = l1_qual(profiles[condition_s])

        conditionl1_n = modify_condition(condition_n, l1ind_n)
        conditionl1_s = modify_condition(condition_s, l1ind_s)

        for l1 in [0,1]:
            cond_n = conditionl1_n if l1==1 else condition_n
            cond_s = conditionl1_s if l1==1 else condition_s
            for med in [0,1]:
                if med == 0:
                    hmf2s[1][l1][med].extend(hmf2[cond_n])
                    hmf2s[0][l1][med].extend(hmf2[cond_s])
                    nmf2s[1][l1][med].extend(nmf2[cond_n])
                    nmf2s[0][l1][med].extend(nmf2[cond_s])
                    times[1][l1][med].extend(time[np.where(cond_n)[0]])
                    times[0][l1][med].extend(time[np.where(cond_s)[0]])
                elif med == 1:
                    hmf2_n = np.nanmedian(np.where(cond_n, hmf2, np.nan), axis=1)
                    hmf2_s = np.nanmedian(np.where(cond_s, hmf2, np.nan), axis=1)
                    nmf2_n = np.nanmedian(np.where(cond_n, nmf2, np.nan), axis=1)
                    nmf2_s = np.nanmedian(np.where(cond_s, nmf2, np.nan), axis=1)
                    ind_n = np.where(np.invert(np.isnan(hmf2_n)))[0]
                    ind_s = np.where(np.invert(np.isnan(hmf2_s)))[0]
                    hmf2s[1][l1][med].extend(hmf2_n[ind_n])
                    hmf2s[0][l1][med].extend(hmf2_s[ind_s])
                    nmf2s[1][l1][med].extend(nmf2_n[ind_n])
                    nmf2s[0][l1][med].extend(nmf2_s[ind_s])
                    times[1][l1][med].extend(time[ind_n])
                    times[0][l1][med].extend(time[ind_s])
    return hmf2s, nmf2s, times

def run01():
    file_dir_l1 = '/home/kamo/resources/icon-fuv/ncfiles/l1/2020/'
    file_dir_l2 = '/home/kamo/resources/icon-fuv/ncfiles/l2/2020/'

    files_l1 = glob.glob(file_dir_l1+'*')
    files_l1.sort()
    files_l2 = glob.glob(file_dir_l2+'*')
    files_l2.sort()

    # dim0:[1:n, 0:s], dim1:[1:l1, 0:no_l1], dim2:[1:med, 0:no_med]
    hmf2s = [[],[]]
    nmf2s = [[],[]]
    times = [[],[]]

    for file_l1,file_l2 in zip(files_l1,files_l2):
        print(file_l2)
        l1 = netCDF4.Dataset(file_l1, 'r')
        l2 = netCDF4.Dataset(file_l2, 'r')
        hmf2 = l2.variables['ICON_L25_HMF2'][:]
        nmf2 = l2.variables['ICON_L25_NMF2'][:]
        mlat = l2.variables['ICON_L25_Magnetic_Latitude'][:]
        qual = l2.variables['ICON_L25_Quality'][:]
        time = l2.variables['ICON_L25_UTC_Time'][:]
        time = np.array([parser.parse(i) for i in time])
        slt = l2.variables['ICON_L25_Local_Solar_Time'][:]
        alt_dim = l2.dimensions['Altitude'].size
        l2.close()
        FUV_mode = l1.variables['ICON_L1_FUV_Mode'][:]
        profiles = profiler(l1, clean=True)[:,-alt_dim:,FUV_mode==2].transpose(2,0,1) #dim:[ep,6,256]
        l1.close()

        condition_n = (
            ((slt > 20) & (slt < 22)) &
            (qual==1) &
            ((mlat > 12) & (mlat < 14))
        )
        condition_s = (
            ((slt > 20) & (slt < 22)) &
            (qual==1) &
            ((mlat > -14) & (mlat < -12))
        )

        l1ind_n = l1_qual(profiles[condition_n])
        l1ind_s = l1_qual(profiles[condition_s])

        conditionl1_n = modify_condition(condition_n, l1ind_n)
        conditionl1_s = modify_condition(condition_s, l1ind_s)
        hmf2_n = np.nanmedian(np.where(conditionl1_n, hmf2, np.nan), axis=1)
        hmf2_s = np.nanmedian(np.where(conditionl1_s, hmf2, np.nan), axis=1)
        nmf2_n = np.nanmedian(np.where(conditionl1_n, nmf2, np.nan), axis=1)
        nmf2_s = np.nanmedian(np.where(conditionl1_s, nmf2, np.nan), axis=1)
        ind_n = np.where(np.invert(np.isnan(hmf2_n)))[0]
        ind_s = np.where(np.invert(np.isnan(hmf2_s)))[0]
        hmf2s[1].extend(hmf2_n[ind_n])
        hmf2s[0].extend(hmf2_s[ind_s])
        nmf2s[1].extend(nmf2_n[ind_n])
        nmf2s[0].extend(nmf2_s[ind_s])
        times[1].extend(time[ind_n])
        times[0].extend(time[ind_s])
    return hmf2s, nmf2s, times

def plot0(hmf2s, nmf2s, times):
    # %% plot
    gaps = [datetime.timedelta(days=12),datetime.timedelta(days=7)]
    colors = ['b', 'r']
    # times_n = np.array([parser.parse(i) for i in times_n])
    # times_s = np.array([parser.parse(i) for i in times_s])
    # hmf2s_n = np.array(hmf2s_n)
    # hmf2s_s = np.array(hmf2s_s)
    # nmf2s_n = np.array(nmf2s_n)
    # nmf2s_s = np.array(nmf2s_s)
    #
    # med_hmf2s_n, medtimes_n = clusterer(hmf2s_n, times_n, gap=7)
    # med_hmf2s_s, medtimes_s = clusterer(hmf2s_s, times_s, gap=12)
    # med_nmf2s_n, _ = clusterer(nmf2s_n, times_n, gap=7)
    # med_nmf2s_s, _ = clusterer(nmf2s_s, times_s, gap=12)

    save_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

    for med in [0,1]:
        for l1 in [0,1]:
            plt.figure()
            plt.scatter(times[0][l1][med], hmf2s[0][l1][med], c=colors[0], s=3)
            meds, tim = clusterer(hmf2s[0][l1][med], np.array(times[0][l1][med]), gap=gaps[0])
            plt.plot(tim, meds, linewidth=3, c=colors[0], label='Mlat=-13')
            plt.scatter(tim, meds, c='k', s=40)
            plt.scatter(times[1][l1][med], hmf2s[1][l1][med], c=colors[1], s=3)
            meds, tim = clusterer(hmf2s[1][l1][med], np.array(times[1][l1][med]), gap=gaps[1])
            plt.plot(tim, meds, linewidth=3, c=colors[1], label='Mlat=+13')
            plt.scatter(tim, meds, c='k', s=40)
            plt.ylabel('HmF2 (km)')
            plt.xlabel('Date/Time')
            # plt.title('HmF2   Median={}   L1-Reject={}'.format(np.array(med).astype(bool), np.array(l1).astype(bool)))
            plt.title('HmF2 with Updated Quality Flags')
            plt.legend()
            # plt.savefig(save_dir+'hmf2_med_{}_l1_{}'.format(np.array(med).astype(bool), np.array(l1).astype(bool)), dpi=300)

            f, ax = plt.subplots()
            plt.scatter(times[0][l1][med], nmf2s[0][l1][med], c=colors[0], s=3)
            meds, tim = clusterer(nmf2s[0][l1][med], np.array(times[0][l1][med]), gap=gaps[0])
            plt.plot(tim, meds, linewidth=3, c=colors[0], label='Mlat=-13')
            plt.scatter(tim, meds, c='k', s=40)
            plt.scatter(times[1][l1][med], nmf2s[1][l1][med], c=colors[1], s=3)
            meds, tim = clusterer(nmf2s[1][l1][med], np.array(times[1][l1][med]), gap=gaps[1])
            plt.plot(tim, meds, linewidth=3, c=colors[1], label='Mlat=+13')
            plt.scatter(tim, meds, c='k', s=40)
            plt.ylabel('NmF2 (cm-3)')
            plt.xlabel('Date/Time')
            plt.title('NmF2   Median={}   L1-Reject={}'.format(np.array(med).astype(bool), np.array(l1).astype(bool)))
            ax.set_yscale('log')
            plt.legend()
            # plt.savefig(save_dir+'nmf2_med_{}_l1_{}'.format(np.array(med).astype(bool), np.array(l1).astype(bool)), dpi=300)
            plt.show()

def run1():
    file_dir_l2 = '/home/kamo/resources/icon-fuv/ncfiles/l2/2022/'

    files_l2 = glob.glob(file_dir_l2+'*')
    files_l2.sort()

    # dim0:[1:n, 0:s]
    hmf2s = [[],[]]
    nmf2s = [[],[]]
    times = [[],[]]

    for file in files_l2:
        print(file)
        l2 = netCDF4.Dataset(file, 'r')
        hmf2 = l2.variables['ICON_L25_HMF2'][:]
        nmf2 = l2.variables['ICON_L25_NMF2'][:]
        mlat = l2.variables['ICON_L25_Magnetic_Latitude'][:]
        qual = l2.variables['ICON_L25_Quality'][:]
        time = l2.variables['ICON_L25_UTC_Time'][:]
        time = np.array([parser.parse(i) for i in time])
        slt = l2.variables['ICON_L25_Local_Solar_Time'][:]
        alt_dim = l2.dimensions['Altitude'].size
        l2.close()

        condition_n = (
            ((slt > 20) & (slt < 22)) &
            (qual==1) &
            ((mlat > 12) & (mlat < 14))
        )
        condition_s = (
            ((slt > 20) & (slt < 22)) &
            (qual==1) &
            ((mlat > -14) & (mlat < -12))
        )

        hmf2_n = np.nanmedian(np.where(condition_n, hmf2, np.nan), axis=1)
        hmf2_s = np.nanmedian(np.where(condition_s, hmf2, np.nan), axis=1)
        nmf2_n = np.nanmedian(np.where(condition_n, nmf2, np.nan), axis=1)
        nmf2_s = np.nanmedian(np.where(condition_s, nmf2, np.nan), axis=1)
        ind_n = np.where(np.invert(np.isnan(hmf2_n)))[0]
        ind_s = np.where(np.invert(np.isnan(hmf2_s)))[0]
        hmf2s[1].extend(hmf2_n[ind_n])
        hmf2s[0].extend(hmf2_s[ind_s])
        nmf2s[1].extend(nmf2_n[ind_n])
        nmf2s[0].extend(nmf2_s[ind_s])
        times[1].extend(time[ind_n])
        times[0].extend(time[ind_s])

    return hmf2s, nmf2s, times

def plot1(hmf2s, nmf2s, times, title):
    import matplotlib
    # %% plot
    gaps = [datetime.timedelta(days=12),datetime.timedelta(days=7)]
    colors = ['b', 'r']
    # times_n = np.array([parser.parse(i) for i in times_n])
    # times_s = np.array([parser.parse(i) for i in times_s])
    # hmf2s_n = np.array(hmf2s_n)
    # hmf2s_s = np.array(hmf2s_s)
    # nmf2s_n = np.array(nmf2s_n)
    # nmf2s_s = np.array(nmf2s_s)
    #
    # med_hmf2s_n, medtimes_n = clusterer(hmf2s_n, times_n, gap=7)
    # med_hmf2s_s, medtimes_s = clusterer(hmf2s_s, times_s, gap=12)
    # med_nmf2s_n, _ = clusterer(nmf2s_n, times_n, gap=7)
    # med_nmf2s_s, _ = clusterer(nmf2s_s, times_s, gap=12)

    save_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

    f, ax = plt.subplots()
    plt.scatter(times[0], hmf2s[0], c=colors[0], s=3)
    meds, tim = clusterer(hmf2s[0], np.array(times[0]), gap=gaps[0])
    plt.plot(tim, meds, linewidth=3, c=colors[0], label='Mlat=-13')
    plt.scatter(tim, meds, c='k', s=40)
    plt.scatter(times[1], hmf2s[1], c=colors[1], s=3)
    meds, tim = clusterer(hmf2s[1], np.array(times[1]), gap=gaps[1])
    plt.plot(tim, meds, linewidth=3, c=colors[1], label='Mlat=+13')
    plt.scatter(tim, meds, c='k', s=40)
    plt.ylabel('HmF2 (km)')
    plt.xlabel('Date/Time')
    # plt.title('HmF2   Median={}   L1-Reject={}'.format(np.array(med).astype(bool), np.array(l1).astype(bool)))
    # plt.title('HmF2 {}'.format(title))
    plt.legend()
    plt.grid(which='both', axis='both')
    # ax.set_ylim((177.4258674621582, 410.46935043334963))
    plt.savefig(save_dir+'hmf2_{}'.format(title), dpi=300, transparent=True)
    print(f'{title} - hm = {ax.get_ylim()}')


    f, ax = plt.subplots()
    plt.scatter(times[0], nmf2s[0], c=colors[0], s=3)
    meds, tim = clusterer(nmf2s[0], np.array(times[0]), gap=gaps[0])
    ax.set_yscale('log')
    amo=np.array([3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6, 2e6, 3e6, 4e6])
    plt.yticks(amo,amo)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
    plt.plot(tim, meds, linewidth=3, c=colors[0], label='Mlat=-13')
    plt.scatter(tim, meds, c='k', s=40)
    plt.scatter(times[1], nmf2s[1], c=colors[1], s=3)
    meds, tim = clusterer(nmf2s[1], np.array(times[1]), gap=gaps[1])
    plt.plot(tim, meds, linewidth=3, c=colors[1], label='Mlat=+13')
    plt.scatter(tim, meds, c='k', s=40)
    plt.ylabel('NmF2 (cm-3)')
    plt.xlabel('Date/Time')
    # plt.title('NmF2 {}'.format(title))
    plt.legend()
    plt.grid(which='both', axis='both')
    # ax.set_ylim((202973.08451631025, 2137463.51362193))
    plt.savefig(save_dir+'nmf2_{}'.format(title), dpi=300, transparent=True)
    print(f'{title} - nm = {ax.get_ylim()}')
    plt.show()

if __name__ == '__main__':
    # hmf2s, nmf2s, times = run1()
    # np.save('data/hmf2s_2022_v2.npy', np.array(hmf2s, dtype=object))
    # np.save('data/nmf2s_2022_v2.npy', np.array(nmf2s, dtype=object))
    # np.save('data/times_2022_v2.npy', np.array(times, dtype=object))
    # plot1(hmf2s, nmf2s, times, '2022')

    # hmf2s0 = np.load('data/hmf2s_v5.npy', allow_pickle=True)
    # nmf2s0 = np.load('data/nmf2s_v5.npy', allow_pickle=True)
    # times0 = np.load('data/times_v5.npy', allow_pickle=True)
    # hmf2s1 = np.load('data/hmf2s_2021.npy', allow_pickle=True)
    # nmf2s1 = np.load('data/nmf2s_2021.npy', allow_pickle=True)
    # times1 = np.load('data/times_2021.npy', allow_pickle=True)
    # hmf2s2 = np.load('data/hmf2s_2022.npy', allow_pickle=True)
    # nmf2s2 = np.load('data/nmf2s_2022.npy', allow_pickle=True)
    # times2 = np.load('data/times_2022.npy', allow_pickle=True)
    hmf2s = np.load('data/hmf2s_2022_v2.npy', allow_pickle=True)
    nmf2s = np.load('data/nmf2s_2022_v2.npy', allow_pickle=True)
    times = np.load('data/times_2022_v2.npy', allow_pickle=True)
    plot1(hmf2s, nmf2s, times, '2022')
