import numpy as np
import matplotlib.pyplot as plt
import re, datetime
import wget, glob, netCDF4
from dateutil import parser
from skimage.draw import line
import os, subprocess

path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

def parse_map(tecmap, exponent = -1):
    tecmap = re.split('.*END OF TEC MAP', tecmap)[0]
    return np.stack([np.fromstring(l, sep=' ') for l in re.split('.*LAT/LON1/LON2/DLON/H\\n',tecmap)[1:]])*10**exponent
    
def get_tecmaps(filename):
    with open(filename) as f:
        ionex = f.read()
        return [parse_map(t) for t in ionex.split('START OF TEC MAP')[1:]]

def get_tecmap(utc=None, year=None, day=None, utc_hours=None,
               output_dir='/home/kamo/resources/icon-fuv/python3/asymmetry_flag/data'):
    if utc is not None:
        year = utc.year
        day = utc.timetuple().tm_yday
        utc_hours = utc.hour + utc.minute/float(60)
    idx = int(round(utc_hours/2))
    download_ionex(year, day, output_dir=output_dir)
    tecmaps = get_tecmaps(ionex_local_path(year, day, directory=output_dir, zipped=False))
    tecmap = tecmaps[idx]
    return tecmap, idx*2

def ionex_filename(year, day, zipped=True):
    return 'igsg{:03d}0.{:02d}i{}'.format(day, year % 100, '.Z' if zipped else '')

def ionex_https_path(year, day):
    # return 'ftp://cddis.gsfc.nasa.gov/gnss/products/ionex/{:04d}/{:03d}/{}'.format(year, day, ionex_filename(year, day, centre))
    return 'https://cddis.nasa.gov/archive/gnss/products/ionex/{:04d}/{:03d}/{}'.format(year, day, ionex_filename(year, day))

def ionex_local_path(year, day, directory = './data', zipped=True):
    return directory + '/' + ionex_filename(year, day, zipped)
    
def download_ionex(year, day, output_dir = './data'):
    file = ionex_https_path(year, day)
    name = file.split('/')[-1][:-2]
    if os.path.exists(output_dir+'/'+name):
        # print('File already exists!')
        return 0
    # wget.download(file, output_dir)
    subprocess.call(['wget', '--auth-no-challenge', file, '-P', output_dir])
    subprocess.call(['gzip', '-d', ionex_local_path(year, day, output_dir)])

def xy_to_rc(x,y):
    col = np.round((x+180)/5).astype(int)
    row = 70-np.round((y+87.5)/2.5).astype(int)
    return row, col

def loncorrect(lon):
    lon = np.array(lon)
    if lon.size==1:
        if lon > 180:
            lon -= 360
    else:
        lon[lon>180] -= 360
    return lon

def lastfile(x):
    """
    Sort all the files complying with `x` alphabetically and return the last.
    """
    # y = glob.glob(x, recursive=True)
    y = glob.glob(x)
    y.sort()
    assert len(y) > 0, 'No file found with the given name'
    return y[-1]

def flagger2(file_l2=None, stripe=2):
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

                x1, y1 = (satlons,satlats)
                x2, y2 = (tlons,tlats)

                thrs = 0.5
                x3 = (2*x2 - x1)%360
                x3 = x3 - 360*(x3//180)
                y3 = (2*y2 - y1)%180
                y3 = y3 - 180*(y3//90)
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


if __name__ == "__main__":
    date = '2022-10-08'
    file_l2 = lastfile(path_dir+'l2/ICON_L2-5_FUV_Night_{}_v05r*'.format(date))
    (nighttime_counter, 
    flag_counter, 
    qual1_counter,
    qual1flag_counter, 
    metric,
    latlon) = flagger2(file_l2=file_l2)

    print('Nighttime Counter: {}'.format(nighttime_counter))
    print('Flag Counter: {} ({:.1f}%)'.format(flag_counter, 100*flag_counter/nighttime_counter))
    print('Qual=1 Counter: {}'.format(qual1_counter))
    print('Flag (Qual=1) Counter: {} ({:.1f}%)'.format(qual1flag_counter, 100*qual1flag_counter/qual1_counter))

    # plt.hist(metric)
    # plt.title('Normalized Metric Histogram')
    # plt.show()