import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import netCDF4, datetime
from dateutil import parser
from copy import deepcopy
from mpl_toolkits.basemap import Basemap
# from airglow.FUV_L2 import l1_correction_orbit

def loncorrect(lon):
    lon = np.array(lon)
    if lon.size==1:
        if lon > 180:
            lon -= 360
    else:
        lon[lon>180] -= 360
    return lon

def tohban(l2=None, l1=None, anc=None, epoch=None, stripe=None):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']

    # Get variables from netCDF file
    dn = [] # UTC
    for d in l2.variables['ICON_L25_UTC_Time']:
        dn.append(parser.parse(d))
    dn = np.array(dn)

    dn2 = l2.variables['ICON_L25_Local_Solar_Time'][:,stripe] # local time
    dn2_hour = dn2.astype(np.int)
    dn2_min = ((dn2-dn2_hour)*60).astype(np.int)

    mode = l1.variables['ICON_L1_FUV_Mode'][:]

    orbits = l2.variables['ICON_L25_Orbit_Number'][:]
    orbit = orbits[epoch]
    Op_lat = l2.variables['ICON_L25_Latitude'][:, stripe] # NmF2 latitudes
    Op_lon = l2.variables['ICON_L25_Longitude'][:, stripe] # NmF2 longitudes

    orbit_ind = np.squeeze(np.where(orbits == orbit))
    print('orbit indices:[{},{}]'.format(orbit_ind[0], orbit_ind[-1]))

    ds = np.array([i.total_seconds() for i in dn-dn[orbit_ind][0]])
    orbit_ind = np.squeeze(np.where(abs(ds) < 2000.))
    print('orbit indices:[{},{}]'.format(orbit_ind[0], orbit_ind[-1]))

    if epoch > orbit_ind[-1]:
        orbit += 1
        orbit_ind = np.squeeze(np.where(orbits == orbit))

        ds = np.array([i.total_seconds() for i in dn-dn[orbit_ind][0]])
        orbit_ind = np.squeeze(np.where(abs(ds) < 2000.))
        print('new orbit indices:[{},{}]'.format(orbit_ind[0], orbit_ind[-1]))

    idx = np.where(mode==2)[0][orbit_ind]

    X = np.transpose([dn,]*l2.dimensions['Altitude'].size)[orbit_ind]
    Y = l2.variables['ICON_L25_O_Plus_Profile_Altitude'][orbit_ind,:,stripe]
    Y = np.ma.filled(Y, fill_value = np.max(Y))
    Z = l2.variables['ICON_L25_O_Plus_Density'][orbit_ind,:,stripe]
    Ze = l2.variables['ICON_L25_O_Plus_Density_Error'][orbit_ind,:,stripe]

    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][idx,:]
    br_er = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idx,:]

    out = np.diff(X,axis=0)
    mask = np.vstack([out > datetime.timedelta(seconds=24),np.ones([1,np.size(out,1)],dtype=bool)])
    Zm = np.ma.MaskedArray(Z,mask)
    Zem = np.ma.MaskedArray(Ze,mask)

    Bm = deepcopy(Zm) # brightness counterpart
    Bem = deepcopy(Zem) # brightness counterpart
    maskcounts = np.sum(Zm.mask==True, axis=1)
    for i in range(Bm.shape[0]):
        Bm[i,maskcounts[i]:] = br[i][::-1][:Bm.shape[1]-maskcounts[i]]
        Bem[i,maskcounts[i]:] = br_er[i][::-1][:Bm.shape[1]-maskcounts[i]]

    min_alt = Y.min()
    max_alt = Y.max()

    min_dn = dn[orbit_ind[0]]
    max_dn = dn[orbit_ind[-1]]

    # Get the orbit(s) in this plot
    orbit_str = 'err'
    if len(np.unique(orbits[orbit_ind])) == 1:
        orbit_str = '%d' % np.unique(orbits[orbit_ind])
    elif len(np.unique(orbits[orbit_ind])) == 2:
        orbit_str = '%d-%d' % (np.unique(orbits[orbit_ind])[0],np.unique(orbits[orbit_ind])[1])

    fig, axes = plt.subplots(nrows=4, figsize=(12,9))
    fig.subplots_adjust(hspace=0.5)

    # Brightness Plot
    im1 = axes[0].pcolormesh(X,Y,Bm,vmin=None,vmax=None)
    axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0].set_ylim([min_alt,max_alt])
    axes[0].set_xlim([min_dn,max_dn])
    axes[0].set_title('Brightness Profile; Stripe #%d \n %s (Orbits: %s)' % (stripe,dn[0].strftime('%Y-%m-%d'), orbit_str))
    axes[0].set_ylabel('Tangent Altitude [km]')

    im2 = axes[1].pcolormesh(X,Y,Bem,vmin=None,vmax=None)
    axes[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1].set_ylim([min_alt,max_alt])
    axes[1].set_title('Brightness Error Profile')
    axes[1].set_ylabel('Tangent Altitude [km]')

    # The electron density estimates
    im3 = axes[2].pcolormesh(X,Y,Zm,vmin=None,vmax=None)
    axes[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[2].set_title('Estimated Ne')
    axes[2].set_ylim([min_alt,max_alt])
    axes[2].set_ylabel('Altitude [km]')

    im4 = axes[3].pcolormesh(X,Y,np.log10(Zem),vmin=None,vmax=None)
    axes[3].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[3].set_ylim([min_alt,max_alt])
    axes[3].set_title('Estimated Ne Error')
    axes[3].set_ylabel('Altitude [km]')

    for i in range(4):
        axes[i].axvline(dn[epoch], color='red')
        # axes[i].axvline(dn[993], color='red')

    box = axes[0].get_position()
    pad, width = 0.02, 0.02
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    fig.colorbar(im1, cax=cax, label='Br [R]', extend='max')

    box = axes[1].get_position()
    pad, width = 0.02, 0.02
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    fig.colorbar(im2, cax=cax, label='Br error [R]', extend='max')

    box = axes[2].get_position()
    pad, width = 0.02, 0.02
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    fig.colorbar(im3, cax=cax,format='%.0e',label='Ne [cm^-3]',extend='max')

    box = axes[3].get_position()
    pad, width = 0.02, 0.02
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    fig.colorbar(im4, cax=cax,format='%.0e',label='log10(Ne error) [cm^-3]',extend='max')
    plt.show()

    ######## Add Local Time to xticks #########
    fig.canvas.draw()
    labels_x = [item.get_text() for item in axes[3].get_xticklabels()]
    minlist = np.array([j.hour*60+j.minute for j in dn[orbit_ind]])
    labels_x2 = []
    for lbl in labels_x:
        hh,mm = [np.int(i) for i in lbl.split(':')]
        tick_ind = np.argmin(abs(minlist - (hh*60+mm)))
        locstr = u'{:02d}:{:02d}'.format(dn2_hour[orbit_ind[tick_ind]], dn2_min[orbit_ind[tick_ind]])
        labels_x2.append('{}'.format(locstr))

    axes[3].set_xticklabels(labels_x2)
    # axes[2].set_xticklabels(labels_x2)
    # axes[1].set_xticklabels(labels_x2)
    # axes[0].set_xticklabels(labels_x2)

def tohban2(file_l2=None, png_stub=None, file_l1=None, stripe=None, max_ne=None, max_br=None):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    l1 = netCDF4.Dataset(file_l1, mode='r')
    l2 = netCDF4.Dataset(file_l2, mode='r')

    # Get variables from netCDF file
    dn = [] # UTC
    for d in l2.variables['ICON_L25_UTC_Time']:
        dn.append(parser.parse(d))
    dn = np.array(dn)

    dn2 = l2.variables['ICON_L25_Local_Solar_Time'][:,stripe] # local time
    dn2_hour = dn2.astype(np.int)
    dn2_min = ((dn2-dn2_hour)*60).astype(np.int)

    mode = l1.variables['ICON_L1_FUV_Mode'][:]

    orbits = l2.variables['ICON_L25_Orbit_Number'][:]
    Op_lat = l2.variables['ICON_L25_Latitude'][:, stripe] # NmF2 latitudes
    Op_lon = l2.variables['ICON_L25_Longitude'][:, stripe] # NmF2 longitudes

    for orbit in np.unique(orbits):
        try:
            file_png = '_'.join(png_stub.split('_')[:-2]) + '-o%05d_' % orbit + '_'.join(png_stub.split('_')[-2:])
            orbit_ind = np.squeeze(np.where(orbits == orbit))
            ds = np.array([i.total_seconds() for i in dn-dn[orbit_ind][0]])
            orbit_ind = np.squeeze(np.where(abs(ds) < 2000.))
            tlons = loncorrect(
                l2.variables['ICON_L25_O_Plus_Profile_Longitude'][orbit_ind, -1, stripe]
            )
            tlats = l2.variables['ICON_L25_O_Plus_Profile_Latitude'][orbit_ind, -1, stripe]
            satlons = loncorrect(
                l2.variables['ICON_L25_Observatory_Position_Longitude'][orbit_ind]
            )
            satlats = l2.variables['ICON_L25_Observatory_Position_Latitude'][orbit_ind]

            if orbit_ind.size > 1:
                t0 = parser.parse(l2.variables['ICON_L25_UTC_Time'][orbit_ind[0]])
                t1 = parser.parse(l2.variables['ICON_L25_UTC_Time'][orbit_ind[-1]])
            else:
                t0 = parser.parse(l2.variables['ICON_L25_UTC_Time'][orbit_ind])
                t1 = t0

            idx = np.where(mode==2)[0][orbit_ind]

            X = np.transpose([dn,]*l2.dimensions['Altitude'].size)[orbit_ind]
            Y = l2.variables['ICON_L25_O_Plus_Profile_Altitude'][orbit_ind,:,stripe]
            Y = np.ma.filled(Y, fill_value = np.max(Y))
            Z = l2.variables['ICON_L25_O_Plus_Density'][orbit_ind,:,stripe]
            Ze = l2.variables['ICON_L25_O_Plus_Density_Error'][orbit_ind,:,stripe]

            mirror_dir = ['M9','M6','M3','P0','P3','P6']
            br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[stripe]][idx,:]
            br_er = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idx,:]

            out = np.diff(X,axis=0)
            mask = np.vstack([out > datetime.timedelta(seconds=24),np.ones([1,np.size(out,1)],dtype=bool)])
            Zm = np.ma.MaskedArray(Z,mask)
            Zem = np.ma.MaskedArray(Ze,mask)

            Bm = deepcopy(Zm) # brightness counterpart
            Bem = deepcopy(Zem) # brightness counterpart
            maskcounts = np.sum(Zm.mask==True, axis=1)
            for i in range(Bm.shape[0]):
                Bm[i,maskcounts[i]:] = br[i][::-1][:Bm.shape[1]-maskcounts[i]]
                Bem[i,maskcounts[i]:] = br_er[i][::-1][:Bm.shape[1]-maskcounts[i]]

            min_alt = Y.min()
            max_alt = Y.max()

            min_dn = dn[orbit_ind[0]]
            max_dn = dn[orbit_ind[-1]]

            # Get the orbit(s) in this plot
            orbit_str = 'err'
            if len(np.unique(orbits[orbit_ind])) == 1:
                orbit_str = '%d' % np.unique(orbits[orbit_ind])
            elif len(np.unique(orbits[orbit_ind])) == 2:
                orbit_str = '%d-%d' % (np.unique(orbits[orbit_ind])[0],np.unique(orbits[orbit_ind])[1])

            fig, axes = plt.subplots(nrows=5, figsize=(6,13))
            fig.subplots_adjust(hspace=0.5)

            # Brightness Plot
            # im1 = axes[0].pcolormesh(X,Y,Bm,vmin=None,vmax=None)
            im1 = axes[0].pcolormesh(X,Y,Bm,vmin=None,vmax=max_br)
            axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[0].set_ylim([min_alt,max_alt])
            axes[0].set_title('Brightness Profile; Stripe #%d \n %s (Orbits: %s)' % (stripe,dn[-1].strftime('%Y-%m-%d'), orbit_str))
            axes[0].set_ylabel('Tangent Altitude [km]')
            # axes[0].set_xlim([min_dn,max_dn])

            # im2 = axes[1].pcolormesh(X,Y,Bem,vmin=None,vmax=None)
            im2 = axes[1].pcolormesh(X,Y,Bem,vmin=None,vmax=None)
            axes[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[1].set_ylim([min_alt,max_alt])
            axes[1].set_title('Brightness Error Profile')
            # axes[1].set_xlim([min_dn,max_dn])
            axes[1].set_ylabel('Tangent Altitude [km]')

            # The electron density estimates
            im3 = axes[2].pcolormesh(X,Y,Zm,vmin=None,vmax=max_ne)
            axes[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[2].set_title('Estimated Ne')
            axes[2].set_ylim([min_alt,max_alt])
            axes[2].set_xlim([min_dn,max_dn])
            axes[2].set_ylabel('Altitude [km]')

            im4 = axes[3].pcolormesh(X,Y,np.log10(Zem),vmin=None,vmax=None)
            axes[3].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[3].set_ylim([min_alt,max_alt])
            axes[3].set_title('Estimated Ne Error')
            axes[3].set_ylabel('Altitude [km]')

            m = Basemap(
                llcrnrlon=-180.,llcrnrlat=-50.,urcrnrlon=180.,urcrnrlat=50.,
                resolution='l', projection='merc', ax=axes[4]
            )
            m.drawcoastlines()
            m.fillcontinents(zorder=0)
            m.drawparallels(np.arange(-30,31,20),labels=[1,1,0,1])
            m.drawmeridians(np.arange(-180,180,60),labels=[1,1,0,1])
            m.nightshade(t0, alpha=0.3)
            m.nightshade(t1, alpha=0.3)
            x, y = m(satlons,satlats)
            m.scatter(x,y,15,marker='o',color='r')
            x, y = m(tlons,tlats)
            m.scatter(x,y,15,marker='o',color='b')


            box = axes[0].get_position()
            pad, width = 0.02, 0.02
            cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
            # fig.colorbar(im1, cax=cax, label='Br [R]', extend='max')
            fig.colorbar(im1, cax=cax, label='Br [R]', extend='max')

            box = axes[1].get_position()
            pad, width = 0.02, 0.02
            cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
            # fig.colorbar(im2, cax=cax, label='Br error [R]', extend='max')
            fig.colorbar(im2, cax=cax, label='Br error [R]', extend='max')

            box = axes[2].get_position()
            pad, width = 0.02, 0.02
            cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
            fig.colorbar(im3, cax=cax,format='%.0e',label='Ne [cm^-3]',extend='max')

            box = axes[3].get_position()
            pad, width = 0.02, 0.02
            cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
            fig.colorbar(im4, cax=cax,format='%.0e',label='log10(Ne error) [cm^-3]',extend='max')

            ######## Add Local Time to xticks #########
            fig.canvas.draw()
            labels_x = [item.get_text() for item in axes[3].get_xticklabels()]
            minlist = np.array([j.hour*60+j.minute for j in dn[orbit_ind]])
            labels_x2 = []
            for lbl in labels_x:
                hh,mm = [np.int(i) for i in lbl.split(':')]
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
            axes[1].set_xticklabels(labels_x2)
            axes[2].set_xticklabels(labels_x2)
            # axes[3].set_xticklabels(labels_x2)

            fig.savefig(file_png, bbox_inches='tight')
            plt.close(fig)

        except:
            pass

    l1.close()
    l2.close()

def tohban_l1(file_l1=None, png_dir=None, stripes=None, both=True):
    if stripes is None:
        stripes = [0,1,2,3,4,5]
    l1 = netCDF4.Dataset(file_l1, mode='r')
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    dn = parser.parse(l1.variables['ICON_L1_FUVA_SWP_Center_Times'][10])
    mode_night = (mode == 2).astype(np.int)
    nights = np.diff(mode_night, prepend=0)
    nights[nights==-1] = 0
    idxs = np.where(mode==2)[0][:]
    nights = np.cumsum(nights)[idxs]
    alt_size = 150
    for night in np.unique(nights):
        night_ind = np.where(nights==night)[0]
        br = np.zeros((6, len(night_ind), alt_size))
        br_err = np.zeros((6, len(night_ind), alt_size))
        for i in range(6):
            br[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]][idxs[night_ind],-alt_size:].filled(fill_value=0)
            br_err[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]][idxs[night_ind],-alt_size:].filled(fill_value=0)
        if both is True:
            br_corrected = np.zeros((6, len(night_ind), alt_size))
            br_err_modified = np.zeros((6, len(night_ind), alt_size))
            for i in range(6):
                br_corrected[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_CLEAN' % mirror_dir[i]][idxs[night_ind],-alt_size:].filled(fill_value=0)
                br_err_modified[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]][idxs[night_ind],-alt_size:].filled(fill_value=0)
        for stripe in stripes:
            file_png = png_dir + 'stripe_{}_orbit_{}.png'.format(stripe, night)
            if both is True:
                fig, ax = plt.subplots(nrows=4, figsize=(6.4,8.4))
            else:
                fig, ax = plt.subplots(nrows=2, figsize=(6.4,8.4))
            im0=ax[0].imshow(np.flipud(br[stripe].swapaxes(0,1)), aspect='auto')
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, cax=cax, orientation='vertical')
            ax[0].set_title('{} - Stripe: {} - Orbit: {} \n Brightness'.format(dn.strftime('%Y-%m-%d'), stripe, night))
            im1=ax[1].imshow(np.flipud(br_err[stripe].swapaxes(0,1)), aspect='auto')
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')
            ax[1].set_title('Uncertainty')
            if both is True:
                im2=ax[2].imshow(np.flipud(br_corrected[stripe].swapaxes(0,1)), aspect='auto', vmin=0)
                divider = make_axes_locatable(ax[2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im2, cax=cax, orientation='vertical')
                ax[2].set_title('Brightness - Artifact Removed')
                im3=ax[3].imshow(np.flipud(br_err_modified[stripe].swapaxes(0,1)), aspect='auto')
                divider = make_axes_locatable(ax[3])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im3, cax=cax, orientation='vertical')
                ax[3].set_title('Uncertainty - Artifact Removed')
            plt.tight_layout()
            fig.savefig(file_png, bbox_inches='tight')
            plt.close(fig)
    l1.close()

def l1_plotter(date='2019-12-24', num=10):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']

    file_l1='nc_files/ICON_L1_FUV_SWP_{}_v02r000.NC'.format(date),
    file_anc='nc_files/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date),
    anc = netCDF4.Dataset(file_anc, mode='r')
    l1 = netCDF4.Dataset(file_l1, mode='r')

    # Get variables from netCDF file
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    total_num = sum(mode==2)
    epoch = np.random.randint(total_num, size=num)
    stripe = np.random.randint(6, size=num)
    idx = np.where(mode==2)[0][epoch]

    br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][idx,:]
    br_er = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idx,:]
