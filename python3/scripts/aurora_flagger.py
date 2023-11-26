import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import netCDF4, datetime
from dateutil import parser
from copy import deepcopy
from mpl_toolkits.basemap import Basemap
from skimage.draw import line
from iconfuv.plotting import loncorrect, connected_regions
from iconfuv.misc import lastfile
import glob, os
from tqdm import tqdm
path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

def flagger(file_l1=None, file_l2=None, stripe=2, png_stub=None, plot=False):
    nighttime_counter = 0
    flag_counter = 0
    qual1_counter = 0
    qual1flag_counter = 0
    peakbr_l = []
    hmf2s_all = []
    hmf2s_flagged = []

    l1 = netCDF4.Dataset(file_l1, mode='r')
    l2 = netCDF4.Dataset(file_l2, mode='r')
    mirror_dir = ['M9','M6','M3','P0','P3','P6']


    mode = l1.variables['ICON_L1_FUV_Mode'][:]

    # Get variables from netCDF file
    dn = [] # UTC
    for d in l2.variables['ICON_L25_UTC_Time']:
        dn.append(parser.parse(d))
    dn = np.array(dn)

    dn2 = l2.variables['ICON_L25_Local_Solar_Time'][:,stripe] # local time
    dn2_hour = dn2.astype(int)
    dn2_min = ((dn2-dn2_hour)*60).astype(int)

    orbits = l2.variables['ICON_L25_Orbit_Number'][:]
    quals = l2.variables['ICON_L25_Quality'][:]

    orbit_list = orbits

    fig, ax = plt.subplots()

    try:
        for orbit in np.unique(orbit_list):
            try:
                orbit_ind = np.squeeze(np.where(orbits == orbit))

                if orbit_ind.size < 2:
                    continue

                file_png = '_'.join(png_stub.split('_')[:-2]) + '-o%05d_' % orbit + '_'.join(png_stub.split('_')[-2:])
                ds = np.array([i.total_seconds() for i in dn-dn[orbit_ind][0]])
                orbit_ind = np.squeeze(np.where(abs(ds) < 2000.))

                target_tanalt = 300 #km
                tanaltinds = np.argmin(
                    abs(l2.variables['ICON_L25_O_Plus_Profile_Altitude'][orbit_ind,
                         :, stripe
                    ].squeeze() - target_tanalt), axis=1
                )
                tlats0 = l2.variables['ICON_L25_O_Plus_Profile_Latitude'][:,:,stripe].squeeze()[orbit_ind, tanaltinds]
                tlatmask = tlats0.mask==False
                tlons = loncorrect(
                    l2.variables['ICON_L25_O_Plus_Profile_Longitude'][:,:,stripe]
                .squeeze()[orbit_ind, tanaltinds])
                satlats0 = l2.variables['ICON_L25_Observatory_Position_Latitude'][orbit_ind].squeeze()
                satlats = l2.variables['ICON_L25_Observatory_Position_Latitude'][orbit_ind][tlatmask].squeeze()
                tlons = tlons[tlatmask].squeeze()
                tlats = tlats0[tlatmask].squeeze()
                hmf2s = l2.variables['ICON_L25_HMF2'][orbit_ind, stripe]

                idx = np.where(mode==2)[0][orbit_ind]

                X = np.transpose([dn,]*l2.dimensions['Altitude'].size)[orbit_ind]
                Y = l2.variables['ICON_L25_O_Plus_Profile_Altitude'][orbit_ind,:,stripe]
                Y = np.ma.filled(Y, fill_value = np.max(Y))
                T = l2.variables['ICON_L25_O_Plus_Profile_Latitude'][orbit_ind,:,stripe]
                T = np.ma.filled(T, fill_value = np.max(T))
                Z = l2.variables['ICON_L25_O_Plus_Density'][orbit_ind,:,stripe]
                brc = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[stripe]][idx,:]
                out = np.diff(X,axis=0)
                mask = np.vstack([out > datetime.timedelta(seconds=24),np.ones([1,np.size(out,1)],dtype=bool)])
                Zm = np.ma.MaskedArray(Z,mask)
                Bmc = deepcopy(Zm) # brightness counterpart
                maskcounts = np.sum(Zm.mask==True, axis=1)
                for i in range(Bmc.shape[0]):
                    Bmc[i,maskcounts[i]:] = brc[i][::-1][:Bmc.shape[1]-maskcounts[i]]

                min_alt = Y.min()
                max_alt = Y.max()

                # Peak brightness
                brmax = Bmc.max(axis=1)
                brmax_alt = Y[range(len(brmax)), Bmc.argmax(axis=1)]
                brmax_tlat = T[range(len(brmax)), Bmc.argmax(axis=1)]
                # aurora_flag = (satlats0>15) & (brmax>30) & (brmax_alt<250)
                aurora_flag = (brmax_tlat>30) & (brmax_alt<220)
                aurora_flag = aurora_flag.data

                min_dn = dn[orbit_ind[0]]
                max_dn = dn[orbit_ind[-1]]

                # Get the orbit(s) in this plot
                orbit_str = 'err'
                if len(np.unique(orbits[orbit_ind])) == 1:
                    orbit_str = '%d' % np.unique(orbits[orbit_ind])
                elif len(np.unique(orbits[orbit_ind])) == 2:
                    orbit_str = '%d-%d' % (np.unique(orbits[orbit_ind])[0],np.unique(orbits[orbit_ind])[1])

                nighttime_counter += orbit_ind.size
                qual = quals[orbit_ind, stripe]
                peakbr = brmax[(satlats0>15)&(brmax_alt<250)]
                peakbr_l.extend(peakbr)

                if orbit_ind.size < 2:
                    continue

                flag_counter += sum(aurora_flag)
                qual1_counter += sum(qual==1)
                flagqual1_locations = aurora_flag & (qual==1)
                qual1flag_counter += sum(flagqual1_locations)

                s0a,s1a = connected_regions(flagqual1_locations.astype(int))

                hmf2s_all.extend(hmf2s[qual==1])
                hmf2s_flagged.extend(hmf2s[flagqual1_locations])

                if plot==True and sum(flagqual1_locations)>0:
                    fig, axes = plt.subplots(nrows=3, figsize=(9,9))
                    fig.subplots_adjust(hspace=0.5)

                    im2 = axes[0].pcolormesh(X,Y,Bmc,vmin=None,vmax=None, cmap='jet')
                    axes[0].set_title('Brightness Profiles; Stripe #%d \n %s (Orbits: %s)' % (stripe,dn[-1].strftime('%Y-%m-%d'), orbit_str))

                    axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
                    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    axes[0].set_ylim([min_alt,max_alt])
                    # axes[1].set_xlim([min_dn,max_dn])
                    axes[0].set_ylabel('Tangent Altitude [km]')

                    # The electron density estimates
                    im3 = axes[1].pcolormesh(X,Y,Zm,vmin=None,vmax=None)
                    axes[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
                    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    axes[1].set_title('Retrieved O+ Profiles')
                    axes[1].set_ylim([min_alt,max_alt])
                    axes[1].set_xlim([min_dn,max_dn])
                    axes[1].set_ylabel('Altitude [km]')

                    axes[0].set_xlabel('   SLT')
                    axes[0].xaxis.set_label_coords(-0.07, -0.060)

                    axes[1].set_xlabel('    UTC')
                    axes[1].xaxis.set_label_coords(-0.07, -0.06)

                    axes[2].plot(brmax_tlat, '-o', color='blue', label='Peak Br Tlat')
                    axes[2].plot(satlats0, '-o', color='red', label='Sat Lats')
                    axes[2].set_ylim((0,50))
                    ax2 = axes[2].twinx()
                    ax2.plot(brmax_alt, '-o', color='purple', label='Peak Br Alt')
                    ax2.set_ylabel('Altitude')
                    ax2.set_ylim((150,350))
                    axes[2].set_title('Peak Br Tlat / Peak Br Alt / Satlat vs Epoch')
                    axes[2].set_xlabel('Epoch')
                    axes[2].set_ylabel('Peak Br & Satlat')
                    axes[2].grid(which='both', axis='both')
                    if len(s0a)>0:
                        for i,j in zip(s0a,s1a):
                            axes[0].axvspan(X[i,0],X[j,0],color='purple', fill=False, alpha=0.5, hatch='xx')
                            axes[1].axvspan(X[i,0],X[j,0],color='purple', fill=False, alpha=0.5, hatch='xx')
                            axes[2].axvspan(i,j,color='red', alpha=0.5)

                    lines, labels = axes[2].get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2)

                    box = axes[0].get_position()
                    pad, width = 0.02, 0.02
                    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
                    fig.colorbar(im2, cax=cax, label='Br [R]', extend='max')

                    box = axes[1].get_position()
                    pad, width = 0.02, 0.02
                    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
                    fig.colorbar(im3, cax=cax,format='%.0e',label='Ne [cm^-3]',extend='max')

                    ######## Add Local Time to xticks #########
                    fig.canvas.draw()
                    labels_x = [item.get_text() for item in axes[1].get_xticklabels()]
                    minlist = np.array([j.hour*60+j.minute for j in dn[orbit_ind]])
                    labels_x2 = []
                    for lbl in labels_x:
                        hh,mm = [int(i) for i in lbl.split(':')]
                        tick_ind = np.argmin(abs(minlist - (hh*60+mm)))
                        if dn2_hour.mask.size==1:
                            if dn2_hour.mask==False:
                                locstr = u'{:02d}:{:02d}'.format(dn2_hour[orbit_ind[tick_ind]], dn2_min[orbit_ind[tick_ind]])
                                labels_x2.append('{}'.format(locstr))
                            else:
                                labels_x2.append('')
                        elif dn2_hour.mask[orbit_ind[tick_ind]]==False:
                            locstr = u'{:02d}:{:02d}'.format(dn2_hour[orbit_ind[tick_ind]], dn2_min[orbit_ind[tick_ind]])
                            labels_x2.append('{}'.format(locstr))
                        else:
                            labels_x2.append('')

                    axes[0].set_xticklabels(labels_x2)


                    fig.savefig(file_png, bbox_inches='tight') #, dpi=400)
                    plt.close(fig)

            except:
                raise
                # pass
    except:
        l2.close()
        raise

    l2.close()
    plt.close('all')

    return  nighttime_counter, flag_counter, qual1_counter, qual1flag_counter, peakbr_l, hmf2s_all, hmf2s_flagged


if __name__=='__main__':
    # date = '2022-10-08'
    # file_l1 = lastfile(path_dir+'l1/ICON_L1_FUV_SWP_{}_v05r*'.format(date))
    # file_l2 = lastfile(path_dir+'l2/ICON_L2-5_FUV_Night_{}_v05r*'.format(date))
    # stripe=2
    # path_fig = path_dir + 'figures/aurora_flagging/aurora_flag_br30_satlat15_alt250'

    # (nighttime_counter, 
    # flag_counter, 
    # qual1_counter,
    # qual1flag_counter 
    # ) = flagger(file_l1=file_l1, file_l2=file_l2, stripe=stripe,
    #     png_stub=path_fig+'/'+file_l2[-41:-3]+'_str%d'%stripe+'.png')

    # print('Nighttime Counter: {}'.format(nighttime_counter))
    # print('Flag Counter: {} ({:.1f}%)'.format(flag_counter, 100*flag_counter/nighttime_counter))
    # print('Qual=1 Counter: {}'.format(qual1_counter))
    # print('Flag (Qual=1) Counter: {} ({:.1f}%)'.format(qual1flag_counter, 100*qual1flag_counter/qual1_counter))

    # # plt.hist(metric)
    # # plt.title('Normalized Metric Histogram')
    # # plt.show()

    # %% cell0
    files_l1 = glob.glob(path_dir + 'l1/*SWP_2022*')
    files_l2 = glob.glob(path_dir + 'l2/2022/*')
    files_l1.sort()
    files_l2.sort()
    stripe=2
    path_fig = path_dir + 'figures/aurora_flagging/aurora_flag_tlat30N_alt220_qual1'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)

    nighttime_counter = 0
    flag_counter = 0
    qual1_counter = 0
    qual1flag_counter = 0
    peakbrs = []
    hmf2s_all = []
    hmf2s_flagged = []

    for file_l1,file_l2 in tqdm(zip(files_l1[:], files_l2[:])):
        (nighttime_ctr, 
        flag_ctr, 
        qual1_ctr,
        qual1flag_ctr,
        peakbr,
        hmf2_all,
        hmf2_flagged
        ) = flagger(file_l1=file_l1, file_l2=file_l2, stripe=stripe, plot=True,
            png_stub=path_fig+'/'+file_l2[-41:-3]+'_str%d'%stripe+'.png')

        nighttime_counter += nighttime_ctr
        flag_counter += flag_ctr
        qual1_counter += qual1_ctr
        qual1flag_counter += qual1flag_ctr
        peakbrs.extend(peakbr)
        hmf2s_all.extend(hmf2_all)
        hmf2s_flagged.extend(hmf2_flagged)

    peakbrs = np.array(peakbrs)

    print('Nighttime Counter: {}'.format(nighttime_counter))
    print('Flag Counter: {} ({:.1f}%)'.format(flag_counter, 100*flag_counter/nighttime_counter))
    print('Qual=1 Counter: {}'.format(qual1_counter))
    print('Flag (Qual=1) Counter: {} ({:.1f}%)'.format(qual1flag_counter, 100*qual1flag_counter/qual1_counter))

    plt.figure()
    plt.hist(peakbrs)
    plt.show()

    plt.figure()
    (cts1, bins, bar) = plt.hist(hmf2s_all, 40, edgecolor='black', label='All')
    (cts2, bins) = np.histogram(hmf2s_flagged, bins=bins)
    (cts3, bins, bar) = plt.hist(bins[:-1], bins[:], weights=cts1-cts2, edgecolor='black', label='Flagged')
    plt.title('Histogram of All vs Aurora-Flagged HmF2s of 2022')
    plt.xlabel('L2.5 HmF2 (km)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig(path_fig+'/hmf2_hist_all_vs_aurora_flagged.png', dpi=300)

# # %% plot
# if 'metric' not in locals():
#     metric = np.load('metric.npy')
# if 'latlons' not in locals():
#     latlons = np.load('latlons.npy')
# # fig, ax = plt.subplots()
# # m = Basemap(
# #     llcrnrlon=-180.,llcrnrlat=-87.5,urcrnrlon=180.,urcrnrlat=87.5,
# #     resolution='l', projection='cyl', ax=ax
# # )
# # m.drawcoastlines(linewidth=0.5, color='gray')
# # m.drawparallels(np.arange(-80,81,20),labels=[1,0,0,1])
# # m.drawmeridians(np.arange(-180,180,30),labels=[1,0,0,1])
# # x, y = m(np.array(latlons[1]),np.array(latlons[0]))
# # m.scatter(x,y,0.2,marker='o',color='r')
# # # im=m.hexbin(x,y, gridsize=80, mincnt=1, cmap='jet')
# # # fig.colorbar(im, label='Frequency')
# # plt.title('300km Tanalt Locations of the Raised Asymmetry Flags in the 2022 Data')
# # plt.tight_layout()
# # plt.show()

# plt.figure(figsize=[7.8,3.6])
# plt.hist(metric, bins=200)
# plt.title('Histogram of the Asymmetry Metric for the 707156 Nighttime Indices')
# plt.axvline(0.5, color='red')
# plt.xlabel('Metric Value')
# plt.ylabel('Number of Occurences')
# plt.tight_layout()
# plt.show()