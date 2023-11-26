## Ulas Kamaci - 2022-11-12

import numpy as np
import matplotlib.pyplot as plt
import netCDF4, glob, datetime
import pandas as pd
from dateutil import parser
from iconfuv.misc import profiler
from scipy.ndimage import convolve1d

def run1(
    file_dir_l2='/home/kamo/resources/icon-fuv/ncfiles/l2/2021/',
    save=False,
    savedir=None,
    save_suffix=None
):

    files_l2 = glob.glob(file_dir_l2+'*')
    files_l2.sort()

    # dim0:[1:n, 0:s]
    times = []
    quals = []
    slts = []

    for file in files_l2:
        print(file)
        l2 = netCDF4.Dataset(file, 'r')
        qual = l2.variables['ICON_L25_Quality'][:]
        time = l2.variables['ICON_L25_UTC_Time'][:]
        time = np.array([parser.parse(i) for i in time])
        slt = l2.variables['ICON_L25_Local_Solar_Time'][:]
        l2.close()

        times.extend(time)
        quals.extend(qual)
        slts.extend(slt)

    if save:
        np.save(savedir+f'times_{save_suffix}', np.array(times))
        np.save(savedir+f'quals_{save_suffix}', np.array(quals))
        np.save(savedir+f'slts_{save_suffix}', np.array(slts))

    return times, quals, slts

def slt_comparison_plotter(times1, quals1, slts1, times2, quals2, slts2):
    qual1_1 = []
    qual5_1 = []
    qual1p_1 = []
    qual5p_1 = []
    qual1_2 = []
    qual5_2 = []
    qual1p_2 = []
    qual5p_2 = []
    slt_bins = [18,19,20,21,22,23,0,1,2,3,4,5]
    for i,slt in enumerate(slt_bins):
        print(slt)
        tind1 = (slts1[:,2]>=slt) & (slts1[:,2]<slt+1)
        tind2 = (slts2[:,2]>=slt) & (slts2[:,2]<slt+1)
        qual1_1.append(np.sum(quals1[tind1]==1))
        qual1_2.append(np.sum(quals2[tind2]==1))
        qual5_1.append(np.sum(quals1[tind1]>=0.5))
        qual5_2.append(np.sum(quals2[tind2]>=0.5))
        if sum(tind1)>0:
            qual1p_1.append(qual1_1[i] / sum(tind1) / 6)
            qual5p_1.append(qual5_1[i] / sum(tind1) / 6)
        else:
            qual1p_1.append(0)
            qual5p_1.append(0)
        if sum(tind2)>0:
            qual1p_2.append(qual1_2[i] / sum(tind2) / 6)
            qual5p_2.append(qual5_2[i] / sum(tind2) / 6)
        else:
            qual1p_2.append(0)
            qual5p_2.append(0)

    slt_bins = [str(i) for i in slt_bins]
    plt.figure()
    plt.plot(slt_bins, qual1_1, '-o', label='2021')
    plt.plot(slt_bins, qual1_2, '-o', label='2022')
    plt.xlabel('SLTs')
    plt.ylabel('# of Quality 1 Retrievals')
    plt.title('# of Quality 1 Retrievals per SLT 2021 vs 2022')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(slt_bins, qual1p_1, '-o', label='2021')
    plt.plot(slt_bins, qual1p_2, '-o', label='2022')
    plt.xlabel('SLTs')
    plt.ylabel('% of Quality 1 Retrievals')
    plt.title('% of Quality 1 Retrievals per SLT 2021 vs 2022')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(slt_bins, qual5_1, '-o', label='2021')
    plt.plot(slt_bins, qual5_2, '-o', label='2022')
    plt.xlabel('SLTs')
    plt.ylabel('# of Quality >= 0.5 Retrievals')
    plt.title('# of Quality >= 0.5 Retrievals per SLT 2021 vs 2022')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(slt_bins, qual5p_1, '-o', label='2021')
    plt.plot(slt_bins, qual5p_2, '-o', label='2022')
    plt.xlabel('SLTs')
    plt.ylabel('% of Quality >= 0.5 Retrievals')
    plt.title('% of Quality >= 0.5 Retrievals per SLT 2021 vs 2022')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()
    

def monthly_comparison_plotter(times1, quals1, times2, quals2):
    qual1_1 = []
    qual5_1 = []
    qual1p_1 = []
    qual5p_1 = []
    qual1_2 = []
    qual5_2 = []
    qual1p_2 = []
    qual5p_2 = []
    for month in np.arange(12):
        tind1 = [True if times1[i].month==month+1 else False for i in range(len(times1))]
        tind2 = [True if times2[i].month==month+1 else False for i in range(len(times2))]
        if sum(tind1)>0:
            qual1_1.append(np.sum(quals1[tind1]==1))
            qual5_1.append(np.sum(quals1[tind1]>=0.5))
            qual1p_1.append(qual1_1[month] / sum(tind1) / 6)
            qual5p_1.append(qual5_1[month] / sum(tind1) / 6)
        else:
            qual1_1.append(np.nan)
            qual5_1.append(np.nan)
            qual1p_1.append(np.nan)
            qual5p_1.append(np.nan)
        if sum(tind2)>0:
            qual1_2.append(np.sum(quals2[tind2]==1))
            qual5_2.append(np.sum(quals2[tind2]>=0.5))
            qual1p_2.append(qual1_2[month] / sum(tind2) / 6)
            qual5p_2.append(qual5_2[month] / sum(tind2) / 6)
        else:
            qual1_2.append(np.nan)
            qual5_2.append(np.nan)
            qual1p_2.append(np.nan)
            qual5p_2.append(np.nan)
    
    month_bins = [str(i) for i in np.arange(12)+1]
    plt.figure()
    plt.plot(month_bins, qual1_1, '-o', label='2021')
    plt.plot(month_bins, qual1_2, '-o', label='2022')
    plt.xlabel('Months')
    plt.ylabel('# of Quality 1 Retrievals')
    plt.title('# of Quality 1 Retrievals per Month 2021 vs 2022')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(month_bins, qual1p_1, '-o', label='2021')
    plt.plot(month_bins, qual1p_2, '-o', label='2022')
    plt.xlabel('Months')
    plt.ylabel('% of Quality 1 Retrievals')
    plt.title('% of Quality 1 Retrievals per Month 2021 vs 2022')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(month_bins, qual5_1, '-o', label='2021')
    plt.plot(month_bins, qual5_2, '-o', label='2022')
    plt.xlabel('Months')
    plt.ylabel('# of Quality >= 0.5 Retrievals')
    plt.title('# of Quality >= 0.5 Retrievals per Month 2021 vs 2022')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(month_bins, qual5p_1, '-o', label='2021')
    plt.plot(month_bins, qual5p_2, '-o', label='2022')
    plt.xlabel('Months')
    plt.ylabel('% of Quality >= 0.5 Retrievals')
    plt.title('% of Quality >= 0.5 Retrievals per Month 2021 vs 2022')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dirr = '/home/kamo/resources/icon-fuv/ncfiles/l2/quality_trend_data/'
    times1 = np.load(dirr+'times_2021.npy', allow_pickle=True)
    quals1 = np.load(dirr+'quals_2021.npy')
    slts1 = np.load(dirr+'slts_2021.npy')
    times2 = np.load(dirr+'times_2022.npy', allow_pickle=True)
    quals2 = np.load(dirr+'quals_2022.npy')
    slts2 = np.load(dirr+'slts_2022.npy')
    monthly_comparison_plotter(times1, quals1, times2, quals2)
    # slt_comparison_plotter(times1, quals1, slts1, times2, quals2, slts2)
    # times, quals, slts = run1(file_dir_l2='/home/kamo/resources/icon-fuv/ncfiles/l2/2022/')
