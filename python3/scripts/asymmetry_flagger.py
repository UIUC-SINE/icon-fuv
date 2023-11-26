import matplotlib.pyplot as plt
import numpy as np
import netCDF4
from dateutil import parser
from copy import deepcopy
from mpl_toolkits.basemap import Basemap
from skimage.draw import line
from iconfuv.ionex import get_tecmap
from iconfuv.plotting import loncorrect, xy_to_rc
from iconfuv.misc import lastfile
import glob
from tqdm import tqdm
path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

def flagger(file_l2=None, stripe=2):
    nighttime_counter = 0
    flag_counter = 0
    qual1_counter = 0
    qual1flag_counter = 0
    metric = []
    latlist = []
    lonlist = []

    l2 = netCDF4.Dataset(file_l2, mode='r')

    # Get variables from netCDF file
    dn = [] # UTC
    for d in l2.variables['ICON_L25_UTC_Time']:
        dn.append(parser.parse(d))
    dn = np.array(dn)

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

                ds = np.array([i.total_seconds() for i in dn-dn[orbit_ind][0]])
                orbit_ind = np.squeeze(np.where(abs(ds) < 2000.))

                target_tanalt = 300 #km
                tanaltinds = np.argmin(
                    abs(l2.variables['ICON_L25_O_Plus_Profile_Altitude'][orbit_ind,
                         :, stripe
                    ].squeeze() - target_tanalt), axis=1
                )
                tlats = l2.variables['ICON_L25_O_Plus_Profile_Latitude'][:,:,stripe].squeeze()[orbit_ind, tanaltinds]
                tlons = loncorrect(
                    l2.variables['ICON_L25_O_Plus_Profile_Longitude'][:,:,stripe]
                .squeeze()[orbit_ind, tanaltinds])
                satlons = loncorrect(
                    l2.variables['ICON_L25_Observatory_Position_Longitude'][orbit_ind]
                )[tlats.mask==False].squeeze()
                satlats = l2.variables['ICON_L25_Observatory_Position_Latitude'][orbit_ind][tlats.mask==False].squeeze()
                tlons = tlons[tlats.mask==False].squeeze()
                orbit_ind = orbit_ind[tlats.mask==False].squeeze()
                tlats = tlats[tlats.mask==False].squeeze()

                nighttime_counter += orbit_ind.size
                qual = quals[orbit_ind, stripe]

                if orbit_ind.size < 2:
                    continue

                min_dn = dn[orbit_ind[0]]

                tecmap, tecutc = get_tecmap(utc=min_dn)

                m = Basemap(
                    llcrnrlon=-180.,llcrnrlat=-87.5,urcrnrlon=180.,urcrnrlat=87.5,
                    resolution='l', projection='cyl', ax=ax
                )
                # m.drawcoastlines(linewidth=0.5, color='gray')
                # m.drawparallels(np.arange(-80,81,20),labels=[1,0,0,1])
                # m.drawmeridians(np.arange(-180,180,30),labels=[1,0,0,1])
                # m.nightshade(min_dn, alpha=0.2)
                x1, y1 = m(satlons,satlats)
                m.scatter(x1,y1,15,marker='o',color='r')
                x2, y2 = m(tlons,tlats)
                m.scatter(x2,y2,15,marker='o',color='b')

                thrs = 0.5
                x3 = (2*x2 - x1)%360
                x3 = x3 - 360*(x3//180)
                y3 = (2*y2 - y1)%180
                y3 = y3 - 180*(y3//90)
                m.scatter(x3,y3,15,marker='o',color='yellow')
                r1,c1 = xy_to_rc(x1,y1)
                r3,c3 = xy_to_rc(x3,y3)
                rows = [] ; cols = []
                stds = []
                means = []
                tcs = tecmap.shape[1]
                for i in range(len(x1)):
                    rr,cc = line(r1[i], c1[i], r3[i], c3[i])
                    if len(np.unique(cc))>tcs/2:
                        if c1[i] > c3[i]:
                            c1h = c1[i]
                            c3h = c3[i] + tcs
                        else:
                            c1h = c1[i] + tcs
                            c3h = c3[i]
                        rr,cc = line(r1[i], c1h, r3[i], c3h)
                    rows.append(rr)
                    cols.append(cc%tcs)
                    stds.append(np.std(tecmap[rr,cc%tcs]))
                    means.append(np.mean(tecmap[rr,cc%tcs]))
                std_norm = np.divide(stds,means) 

                flag_locations = std_norm > thrs
                flag_counter += sum(flag_locations)
                qual1_counter += sum(qual==1)
                flagqual1_locations = flag_locations & (qual==1)
                qual1flag_counter += sum(flagqual1_locations)
                metric.extend(std_norm)

                assert std_norm.shape == tlats.shape, "Shape don't match"

                latlist.extend(tlats[flagqual1_locations])
                lonlist.extend(tlons[flagqual1_locations])

            except:
                raise
                # pass
    except:
        l2.close()
        raise

    l2.close()
    plt.close('all')

    return  nighttime_counter, flag_counter, qual1_counter,qual1flag_counter, np.array(metric), [latlist,lonlist]


if __name__=='__main__':
    date = '2022-10-08'
    file_l2 = lastfile(path_dir+'l2/ICON_L2-5_FUV_Night_{}_v05r*'.format(date))
    (nighttime_counter, 
    flag_counter, 
    qual1_counter,
    qual1flag_counter, 
    metric,
    latlon) = flagger(file_l2=file_l2)

    print('Nighttime Counter: {}'.format(nighttime_counter))
    print('Flag Counter: {} ({:.1f}%)'.format(flag_counter, 100*flag_counter/nighttime_counter))
    print('Qual=1 Counter: {}'.format(qual1_counter))
    print('Flag (Qual=1) Counter: {} ({:.1f}%)'.format(qual1flag_counter, 100*qual1flag_counter/qual1_counter))

    # plt.hist(metric)
    # plt.title('Normalized Metric Histogram')
    # plt.show()

# %% cell0
# files = glob.glob(path_dir + 'l2/2022/*')
# files.sort()

# nighttime_counter = 0
# flag_counter = 0
# qual1_counter = 0
# qual1flag_counter = 0
# metric = []
# latlons = [[],[]]

# for file in tqdm(files[:]):
#     (nighttime_ctr, 
#     flag_ctr, 
#     qual1_ctr,
#     qual1flag_ctr, 
#     metric_,
#     latlon_) = flagger(file_l2=file)

#     nighttime_counter += nighttime_ctr
#     flag_counter += flag_ctr
#     qual1_counter += qual1_ctr
#     qual1flag_counter += qual1flag_ctr
#     metric.extend(metric_[:])
#     latlons[0].extend(latlon_[0][:])
#     latlons[1].extend(latlon_[1][:])


# print('Nighttime Counter: {}'.format(nighttime_counter))
# print('Flag Counter: {} ({:.1f}%)'.format(flag_counter, 100*flag_counter/nighttime_counter))
# print('Qual=1 Counter: {}'.format(qual1_counter))
# print('Flag (Qual=1) Counter: {} ({:.1f}%)'.format(qual1flag_counter, 100*qual1flag_counter/qual1_counter))
# np.save('metric.npy', np.array(metric))
# np.save('latlons.npy', np.array(latlons))


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
