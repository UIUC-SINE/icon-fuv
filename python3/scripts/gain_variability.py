import numpy as np
import matplotlib.pyplot as plt
import netCDF4, glob, datetime

path_dir = '/home/kamo/resources/icon-fuv/ncfiles/l1/2020/'
files = glob.glob(path_dir+'*')
files.sort()
gdays = []
gnights = []

for file in files:
    print('{}'.format(file))
    l1 = netCDF4.Dataset(file, 'r')
    gdays.append(l1.variables['ICON_L1_FUVA_SWP_GAIN_DAY'][:])
    gnights.append(l1.variables['ICON_L1_FUVA_SWP_GAIN_NIGHT'][:])
    l1.close()

gdays = np.array(gdays)
gnights = np.array(gnights)
