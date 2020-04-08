import numpy as np
import matplotlib.pyplot as plt
import netCDF4, sys

path_dir = '/home/kamo/resources/iconfuv/nc_files/'

def descrep(date):
    file_l1 = path_dir + 'ICON_L1_FUV_SWP_{}_v03r000.NC'.format(date)
    file_anc = path_dir + 'ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)

    l1 = netCDF4.Dataset(file_l1, mode='r')
    anc = netCDF4.Dataset(file_anc, mode='r')

    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    ancac = anc.variables['ICON_ANCILLARY_FUV_STATUS'][:]
    # ancac = anc.variables['ICON_ANCILLARY_FUV_ACTIVITY'][:]

    ancind_all = (ancac==258) | (ancac==2)
    l1ind = mode==2

    plt.figure()
    plt.plot(ancind_all.astype(int), label='Science Mode: ON')
    plt.plot(ancac==2, 'r', label='Science Mode: OFF')
    plt.xlabel('Epoch')
    plt.ylabel('Earth Night View (Binary)')
    plt.title('{} - ANCILLARY_FUV_STATUS - Night View: ON'.format(date))
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(l1ind.astype(int), label='L1 Nightside Science')
    plt.plot(ancind_all.astype(int) - l1ind.astype(int), 'r', label='Extra Indices in Ancillary')
    plt.xlabel('Epoch')
    plt.ylabel('Nightside Science Mode (Binarized)')
    plt.title('{} - L1_FUV_Mode - Nightside science: ON'.format(date))
    plt.legend()
    plt.show()

    l1.close()
    anc.close()

if __name__== "__main__":
    descrep(str(sys.argv[1]))
