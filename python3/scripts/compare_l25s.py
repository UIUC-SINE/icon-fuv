# Ulas Kamaci - 2022-03-04
# Compare the retrieved nmf2 and hmf2s of two versions of L2.5 files from the
# same day.
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from dateutil import parser
import datetime
from hmf2_variability_plotter import clusterer
import matplotlib.dates as mdates

def index_finder(ref_times, ref_inds, dst_times):
    dst_inds = np.zeros((len(dst_times),6)).astype(bool)
    ref_inds0 = np.where(np.sum(ref_inds, axis=1) > 0)[0]
    ref_times0 = ref_times[ref_inds0]
    diff = np.abs(dst_times[np.newaxis,:] - ref_times0[:,np.newaxis])
    ref_inds_match, dst_inds_match = np.where(diff<datetime.timedelta(seconds=3))
    for i in range(len(ref_inds_match)):
        dst_inds[dst_inds_match[i],ref_inds[ref_inds0[ref_inds_match[i]]]] = True
    return dst_inds

path_dir = '/home/kamo/resources/icon-fuv/ncfiles/l2/'

file_new = path_dir + 'ICON_L2-5_FUV_Night_2020-06-06_v04r000.NC'
file_old = path_dir + '2020/ICON_L2-5_FUV_Night_2020-06-06_v04r000.NC'
# file_new = path_dir + 'ICON_L2-5_FUV_Night_2020-11-29_v04r000.NC'
# file_old = path_dir + '2020/ICON_L2-5_FUV_Night_2020-11-29_v04r000.NC'
date = file_new[-21:-11]

l2_new = netCDF4.Dataset(file_new, 'r')
l2_old = netCDF4.Dataset(file_old, 'r')

qual_threshold = 1
match_indices = True #if true, use the profiles only where file_new satisfies
# the quality threshold

try:
    qual_new = l2_new.variables['ICON_L25_Quality'][:]
    time_new = l2_new.variables['ICON_L25_UTC_Time'][:]
    time_new = np.array([parser.parse(i) for i in time_new])
    ind_new = qual_new >= qual_threshold

    qual_old = l2_old.variables['ICON_L25_Quality'][:]
    time_old = l2_old.variables['ICON_L25_UTC_Time'][:]
    time_old = np.array([parser.parse(i) for i in time_old])
    if match_indices is True:
        ind_old = index_finder(ref_times=time_new, ref_inds=ind_new, dst_times=time_old)
    else:
        ind_old = qual_old >= qual_threshold

    hmf2_new = l2_new.variables['ICON_L25_HMF2'][:][ind_new]
    nmf2_new = l2_new.variables['ICON_L25_NMF2'][:][ind_new]
    time_new = np.repeat(time_new[:,np.newaxis],6,axis=1)[ind_new]

    hmf2_old = l2_old.variables['ICON_L25_HMF2'][:][ind_old]
    nmf2_old = l2_old.variables['ICON_L25_NMF2'][:][ind_old]
    time_old = np.repeat(time_old[:,np.newaxis],6,axis=1)[ind_old]

    l2_new.close()
    l2_old.close()

except:
    l2_new.close()
    l2_old.close()

#%% plotting
gap = datetime.timedelta(minutes=45)
meds_new, tims_new = clusterer(nmf2_new, time_new, gap)
meds_old, tims_old = clusterer(nmf2_old, time_old, gap)
diff = (meds_old - meds_new)/meds_old*100
print('Median old: {:.0f} , Median new: {:.0f}'.format(np.median(meds_old),np.median(meds_new)))
print('Decrease in NmF2 w.r.t. old: {:.02f}%'.format(np.median(diff)))

plt.figure()
plt.title('NmF2 Comparison on {}'.format(date))
plt.xlabel('UTC Hour')
plt.ylabel('NmF2 [cm^-3]')
plt.scatter(time_new, nmf2_new, label='new', s=1, c='b')
plt.scatter(time_old, nmf2_old, label='old', s=1, c='r')
plt.scatter(tims_new, meds_new, label='new_medians', s=40, c='g')
plt.scatter(tims_old, meds_old, label='old_medians', s=40, c='m')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
plt.tight_layout()
# plt.savefig('nmf2_comparison_{}.png'.format(date), dpi=300)
plt.show()

# plt.figure()
# plt.title('hmF2 Comparison')
# plt.scatter(time_new, hmf2_new, label='new', s=1)
# plt.scatter(time_old, hmf2_old, label='old', s=1)
# plt.legend()
# plt.show()
