import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re, datetime
import cartopy.crs as ccrs
import wget
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

def get_tecmap(year, day, utc_hours):
    idx = round(utc_hours/2)
    download_ionex(year, day)
    tecmaps = get_tecmaps(ionex_local_path(year, day, zipped=False))
    tecmap = tecmaps[idx]
    return tecmap

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
        print('File already exists!')
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
    year = 2022
    day = 182
    time_utc = 0.9
    idx = round(time_utc/2)
    download_ionex(year, day)
    tecmaps = get_tecmaps(ionex_local_path(year, day, zipped=False))
    tecmap = tecmaps[idx]
    plot_tec_map(tecmap)
    plt.title('{:02d}:00 UTC'.format(idx*2))
    plt.show()
    # for i, tecmap in enumerate(get_tecmaps(ionex_local_path(year, day, zipped=False))):
    #     plot_tec_map(tecmap)
    #     plt.title(i)
    #     plt.show()
