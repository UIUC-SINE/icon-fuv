from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil import parser
import netCDF4, os
from iconfuv.plotting import tohban, tohban2
from iconfuv.misc import loncorrect, lastfile
import matplotlib

matplotlib.use('Agg')
savedir = '/home/kamo/resources/icon-fuv/pres/pres52_2023_05_18_Bottomside_Anomaly_Cases_by_Hysell/analyzer_figs/'
path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

def analyser(date, time, stripe, figdir):
    if date.count('-')==1:
        date = (datetime(int(date.split('-')[0]),1,1)+
        timedelta(days=int(date.split('-')[1])-1)).strftime('%Y-%m-%d')
    file_anc = lastfile(path_dir + 'l0/ICON_L0P_FUV_Ancillary_{}_v*'.format(date))
    file_l1 = lastfile(path_dir + 'l1/ICON_L1_FUV_SWP_{}_v*'.format(date))
    file_l2 = lastfile(path_dir + 'l2/ICON_L2-5_FUV_Night_{}_v*'.format(date))
    # file_l2 = '/home/kamo/resources/icon-fuv/ncfiles/l2/ICON_L2-5_FUV_Night_orb_2022-03-31_v05r000_reg_100000.NC'
    # file_l1 = path_dir + 'ICON_L1_FUV_SWP_{}_v77r000.NC'.format(date)
    # file_l2 = path_dir + 'ICON_L2-5_FUV_Night_{}_v01r000.NC'.format(date)
    # file_l1='/home/kamo/icon/nc_files/ICON_L1_FUV_SWP_{}_v99r000.NC'.format(date)
    # file_l2='/home/kamo/icon/nc_files/ICON_L2_FUV_Oxygen-Profile-Night_{}_v99r000.NC'.format(date)

    anc = netCDF4.Dataset(file_anc, mode='r')
    l1 = netCDF4.Dataset(file_l1, mode='r')
    l2 = netCDF4.Dataset(file_l2, mode='r')

    mirror_dir = ['M9','M6','M3','P0','P3','P6']

    # %% params ---------------------------------
    utctime = time
    utc = '{} {}'.format(date, utctime)
    dn = []
    for d in l2.variables['ICON_L25_UTC_Time']:
        dn.append(parser.parse(d))
    dn = np.array(dn)
    epoch = np.argmin(abs(parser.parse(utc)-dn))

    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    idxs = np.where(mode==2)[0][:]
    epoch_l1 = idxs[epoch]

    # print('Epoch L1: {}'.format(epoch_l1))
    print('Epoch L2: {}'.format(epoch))
    # print('Turret: {}'.format(l1.variables['ICON_L1_FUV_Turret'][epoch_l1]))

    # take the `epoch`^th night index of the `stripe`^th brightness profiles
    br = []
    for st in range(6):
        br.append(l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[st]][epoch_l1,:])
    br = np.array(br)
    oplus = l2.variables['ICON_L25_O_Plus_Density'][epoch, :, :]
    alt_l2 = l2.variables['ICON_L25_O_Plus_Profile_Altitude'][epoch, :, :]
    lon_l2 = l2.variables['ICON_L25_O_Plus_Profile_Longitude'][epoch, :, :]
    lat_l2 = l2.variables['ICON_L25_O_Plus_Profile_Latitude'][epoch, :, :]
    alt_l1 = anc.variables['ICON_ANCILLARY_FUVA_TANGENTPOINTS_LATLONALT'][epoch_l1, :, :, 2]
    err = l2.variables['ICON_L25_Quality_Flags'][epoch, stripe]
    lt = l2.variables['ICON_L25_Local_Solar_Time'][epoch, stripe]
    satlon = l2.variables['ICON_L25_Observatory_Position_Longitude'][epoch]
    satlat = l2.variables['ICON_L25_Observatory_Position_Latitude'][epoch]
    t = parser.parse(l2.variables['ICON_L25_UTC_Time'][epoch])

    ap = sum(alt_l2[:, stripe].mask==False) #num of active pixels (>150)
    ap=256

    # plt.figure(1); plt.plot(alt_l1[-ap:]); plt.title('Tangent Altitudes for 6 Stripes')

    fig, ax = plt.subplots(2,1, figsize=(5,8))
    for st in range(6):
        if st==stripe:
            ax[0].plot(br[st, -ap:], alt_l1[-ap:, st], label='Stripe %d'%st, linewidth=3, color='k')
        else:
            ax[0].plot(br[st, -ap:], alt_l1[-ap:, st], label='Stripe %d'%st)
    ax[0].set_title('Brightness Profiles')
    ax[0].set_xlabel('Brightness [R]')
    ax[0].set_ylabel('Tang. Altitudes [km]')
    ax[0].ticklabel_format(scilimits=(0,3))
    ax[0].grid(which='both', axis='both')
    ax[0].legend()
    for st in range(6):
        if st==stripe:
            ax[1].plot(oplus[:,st], alt_l2[:, st], label='Stripe %d'%st, linewidth=3, color='k')
        else:
            ax[1].plot(oplus[:,st], alt_l2[:, st], label='Stripe %d'%st)
    ax[1].set_title('$O^+$ Profiles')
    ax[1].set_xlabel('$O^+$ Density [$cm^{-3}$]')
    ax[1].set_ylabel('Tang. Altitudes [km]')
    ax[1].ticklabel_format(scilimits=(0,3))
    ax[1].grid(which='both', axis='both')
    ax[1].legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(figdir+'brightness_and_oplus_{}.png'.format(figdir.split('/')[-2]))

    # fig=plt.figure()
    # ax=fig.add_axes([0.1,0.1,0.8,0.8])
    # # setup mercator map projection.
    # m = Basemap(
    #     llcrnrlon=-180.,llcrnrlat=-50.,urcrnrlon=180.,urcrnrlat=50.,
    #     resolution='l', projection='merc'
    # )
    # m.drawcoastlines()
    # m.fillcontinents(zorder=0)
    # m.drawparallels(np.arange(-30,31,20),labels=[1,1,0,1])
    # m.drawmeridians(np.arange(-180,180,60),labels=[1,1,0,1])
    # m.nightshade(t, alpha=0.3)
    # x, y = m(loncorrect(satlon),satlat)
    # m.scatter(x,y,15,marker='o',color='r')
    # x, y = m(loncorrect(lon_l2[:,0].compressed()[0]), lat_l2[:,0].compressed()[0])
    # m.scatter(x,y,15,marker='o',color='b')
    # x, y = m(loncorrect(lon_l2[:,-1].compressed()[0]), lat_l2[:,-1].compressed()[0])
    # m.scatter(x,y,15,marker='o',color='b')
    # x, y = m(loncorrect(lon_l2[:,0].compressed()[-1]), lat_l2[:,0].compressed()[-1])
    # m.scatter(x,y,15,marker='o',color='b')
    # x, y = m(loncorrect(lon_l2[:,-1].compressed()[-1]), lat_l2[:,-1].compressed()[-1])
    # m.scatter(x,y,15,marker='o',color='b')
    # ax.set_title('SLT:{}'.format(str(timedelta(seconds=lt*3600))[:-7]))
    # plt.show()

    # tohban(l2=l2, l1=l1, epoch=epoch, stripe=stripe)
    tohban2(file_l2=file_l2, file_l1=file_l1, epoch=epoch, save=False, stripe=stripe, both_br=True)
    plt.savefig(figdir+'toh_{}.png'.format(figdir.split('/')[-2]))

    # import ipdb; ipdb.set_trace()

    # %% close ---------------------------------
    plt.close('all')

    anc.close()
    l1.close()
    l2.close()

if __name__ == '__main__':
    input_list = [
        ['2022-086', '17:42:15Z', 4],
        ['2022-086', '12:50:33Z', 5],
        # ['2022-276', '02:42:00Z', 0],
        # ['2022-277', '01:11:24Z', 0],
        # ['2022-277', '04:24:00Z', 0],
        # ['2022-278', '02:53:24Z', 0],
        # ['2022-278', '04:30:00Z', 0],
        # ['2022-278', '06:06:00Z', 0],
        # ['2022-279', '04:36:00Z', 0],
        # ['2022-279', '06:12:00Z', 0],
        # ['2022-279', '07:48:36Z', 0],
        # ['2022-279', '12:37:48Z', 0],
        # ['2022-280', '07:54:00Z', 0],
        # ['2022-280', '11:07:12Z', 0],
        # ['2022-280', '17:32:24Z', 0],
        # ['2022-281', '08:00:00Z', 0],
        # ['2022-281', '09:36:36Z', 0]
    ]

    for inn in input_list:
        print(inn)
        figdir = savedir + inn[0] + '_' + inn[1] + '_st_' + str(inn[2]) + '/'
        if not os.path.exists(figdir):
            os.mkdir(figdir)
        analyser(inn[0], inn[1], inn[2], figdir)
