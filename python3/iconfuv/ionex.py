import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re, datetime
import cartopy.crs as ccrs
from tqdm import tqdm
import os, subprocess

# Larger figure size
fig_size = [7.5, 3.6]
plt.rcParams['figure.figsize'] = fig_size

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
        utc_hours = utc.hour + utc.minute/60
    idx = round(utc_hours/2)
    download_ionex(year, day, output_dir=output_dir)
    tecmaps = get_tecmaps(ionex_local_path(year, day, directory=output_dir, zipped=False))
    tecmap = tecmaps[idx]
    return tecmap, idx*2

def get_tec(tecmap, lat, lon):
    i = round((87.5 - lat)*(tecmap.shape[0]-1)/(2*87.5))
    j = round((180 + lon)*(tecmap.shape[1]-1)/360)
    return tecmap[i,j]

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
    
def plot_tec_map(tecmap, datestr=''):
    proj = ccrs.PlateCarree()
    f, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj))
    ax.coastlines()
    h = plt.imshow(tecmap, cmap='inferno', extent = (-180, 180, -87.5, 87.5), transform=proj)
    plt.title('VTEC map - {}'.format(datestr))
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    f.add_axes(ax_cb)
    cb = plt.colorbar(h, cax=ax_cb)
    # plt.rc('text', usetex=True)
    cb.set_label('TECU ($10^{16} \\mathrm{l}/\\mathrm{m}^2$)')
    plt.tight_layout()

def tecmap_plotter(year, day, time_utc):
    datestr = (datetime.datetime(year,1,1)+datetime.timedelta(days=day, hours=time_utc)).strftime("%Y-%m-%d__%H:%M:%S")
    idx = round(time_utc/2)
    download_ionex(year, day)
    tecmaps = get_tecmaps(ionex_local_path(year, day, zipped=False))
    tecmap = tecmaps[idx]
    plot_tec_map(tecmap, datestr=datestr)
    plt.title('{:02d}:00 UTC'.format(idx*2))
    plt.show()
    return tecmap

if __name__ == "__main__":
    # year = 2022
    # day = 182
    # time_utc = 0.9
    # idx = round(time_utc/2)
    # download_ionex(year, day)
    # tecmaps = get_tecmaps(ionex_local_path(year, day, zipped=False))
    # tecmap = tecmaps[idx]
    # plot_tec_map(tecmap)
    # plt.title('{:02d}:00 UTC'.format(idx*2))
    # plt.show()
    # for i, tecmap in enumerate(get_tecmaps(ionex_local_path(year, day, zipped=False))):
    #     plot_tec_map(tecmap)
    #     plt.title(i)
    #     plt.show()

    dd = {}

    for year in [2019,2020,2021,2022]:
        dmax=366 if year==2020 else 365
        dmax=329 if year==2022 else dmax
        dmin=320 if year==2019 else 1
        for day in tqdm(np.arange(dmin,dmax+1)):
            tm = get_tecmaps('../asymmetry_flag/data/igsg{:03d}0.{:02d}i'.format(day, year % 100))
            dd['{},{}'.format(year,day)] = tm

