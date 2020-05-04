from l1_destarring import destarrer
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import datetime, netCDF4, glob
import numpy as np
path_dir = '/home/kamo/resources/iconfuv/nc_files/'

# %% runner
def peakmin(date, day=False):
    file_input = path_dir + 'l1/ICON_L1_FUV_SWP_{}_v77r000.NC'.format(date)
    # file_input = path_dir + 'l1/ICON_L1_FUV_SWP_{}_v03r*'.format(date)
    # file_input = glob.glob(file_input)
    # file_input.sort()
    # file_input = file_input[-1]
    l1 = netCDF4.Dataset(file_input, mode='r')
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    qual = l1.variables['ICON_L1_FUVA_SWP_Quality_Flag'][:]
    idx = ((mode==2) & ((qual==0) | (qual==1)))
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    brmax = []
    brmin = []
    if day is True:
        for ind, d in enumerate(mirror_dir):
            brmax.append(np.max(l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % d][idx]))
            brmin.append(np.min(l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % d][idx]))
        l1.close()
        return np.max(brmax), np.min(brmin)
    for ind, d in enumerate(mirror_dir):
        brmax.append(np.max(l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % d][idx], axis=1))
        brmin.append(np.min(l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % d][idx], axis=1))
    l1.close()
    return np.concatenate(brmax), np.concatenate(brmin)

def dailystat(date):
    file_input = path_dir + 'l1/ICON_L1_FUV_SWP_{}_v77r000.NC'.format(date)
    # file_input = path_dir + 'l1/ICON_L1_FUV_SWP_{}_v03r*'.format(date)
    # file_input = glob.glob(file_input)
    # file_input.sort()
    # file_input = file_input[-1]
    l1 = netCDF4.Dataset(file_input, mode='r')
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    qual = l1.variables['ICON_L1_FUVA_SWP_Quality_Flag'][:]
    idx = ((mode==2) & ((qual==0) | (qual==1)))
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    brmean = []
    brmed = []
    brmax = []
    for ind, d in enumerate(mirror_dir):
        br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % d][idx]
        brm = np.max(br, axis=1)
        brmean.append(np.mean(brm))
        brmed.append(np.median(brm))
        brmax.append(np.median(np.sort(brm)[int(0.95*len(brm)):]))
    l1.close()
    return np.mean(brmean), np.median(brmed), np.median(brmax)

brmax = []
brmin = []
brmean = []
brmed = []
day = True
for i in np.arange(1,79):
    try:
        date=datetime.datetime.strptime('2020 {}'.format(i), '%Y %j').strftime('%Y-%m-%d')
        # bmax, bmin = peakmin(date, day=day)
        # if day is True:
        #     brmax.append((i, bmax))
        #     brmin.append((i, bmin))
        # else:
        #     brmax.append(bmax)
        #     brmin.append(bmin)
        bmean, bmed, bmax = dailystat(date)
        brmean.append((i, bmean))
        brmed.append((i, bmed))
        brmax.append((i, bmax))
        print('{} completed'.format(i))
    except:
        pass

brmax = np.array(brmax)
brmean = np.array(brmean)
brmed = np.array(brmed)
np.save(path_dir + 'l1/brmax_01_to_78_day.npy', brmax)
np.save(path_dir + 'l1/brmean_01_to_78_day.npy', brmean)
np.save(path_dir + 'l1/brmed_01_to_78_day.npy', brmed)
# if day is True:
#     brmax = np.array(brmax)
#     np.save(path_dir + 'l1/brmax_01_to_78_day.npy', brmax.data)
# else:
#     brmax = np.concatenate(brmax)
#     brmin = np.concatenate(brmin)
#     np.save(path_dir + 'l1/brmax_01_to_78.npy', brmax.data)
#     np.save(path_dir + 'l1/brmin_01_to_78.npy', brmin.data)

# %% plotting
# brmax = np.load(path_dir + 'l1/brmax_01_to_78_raw.npy')
# brmin = np.load(path_dir + 'l1/brmin_01_to_78_raw.npy')
# brmax = np.load(path_dir + 'l1/brmax_01_to_78.npy')
# brmin = np.load(path_dir + 'l1/brmin_01_to_78.npy')
#
# fig, ax = plt.subplots()
# ax.hist(brmax, bins=np.arange(0,301,25), density=True, edgecolor='black')
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=0.04))
# ax.grid(which='both', axis='both')
# ax.set_title('Peak 1365 Distribution for Nighttime Profiles, Day 1 to 78 - 2020')
# ax.set_xlabel('Peak Brightness [R]')
#
# fig, ax = plt.subplots()
# ax.hist(brmin, bins=np.arange(0,11,1), density=True, edgecolor='black')
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
# ax.grid(which='both', axis='both')
# ax.set_title('Minimum 1365 Distribution for Nighttime Profiles, Day 1 to 78 - 2020')
# ax.set_xlabel('Minimum Brightness [R]')

# brmax = np.load(path_dir + 'l1/brmax_01_to_78_day.npy')
brmean = np.load(path_dir + 'l1/brmean_01_to_78_day.npy')
brmed = np.load(path_dir + 'l1/brmed_01_to_78_day.npy')
#
plt.figure()
# plt.plot(brmax[:,0], brmax[:,1], '-o', label='Max')
plt.plot(brmean[:,0], brmean[:,1], '-o', label='Mean')
plt.plot(brmed[:,0], brmed[:,1], '-o', label='Median')
plt.title('Day to Day Peak Brightness Variablility')
plt.xlabel('Day of Year 2020')
plt.ylabel('Peak Brightness [R]')
plt.grid(which='both', axis='both')
plt.legend()
