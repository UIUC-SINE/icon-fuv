import numpy as np
import matplotlib.pyplot as plt
from iconfuv.misc import profiler, lastfile
import netCDF4

l1dir = '/home/kamo/resources/icon-fuv/ncfiles/l1/'
daynight = 2
channel = 1

for i in np.arange(1,8):
    file_s0 = lastfile(l1dir + '*SWP_2022-01-0{}_v05*'.format(i))
    file_s9 = lastfile(l1dir + '*SWP_2022-01-0{}_v99*'.format(i))
    file_l0 = lastfile(l1dir + '*LWP_2022-01-0{}_v05*'.format(i))
    file_l9 = lastfile(l1dir + '*LWP_2022-01-0{}_v99*'.format(i))
    if channel == 1:
        l1_0 = netCDF4.Dataset(file_s0, 'r')
        l1_9 = netCDF4.Dataset(file_s9, 'r')
        print('Gain Day = ', l1_0.variables['ICON_L1_FUVA_SWP_GAIN_DAY'][:])
        print('Gain Night = ', l1_0.variables['ICON_L1_FUVA_SWP_GAIN_NIGHT'][:])
    elif channel == 2:
        l1_0 = netCDF4.Dataset(file_l0, 'r')
        l1_9 = netCDF4.Dataset(file_l9, 'r')
        print('Gain Day = ', l1_0.variables['ICON_L1_FUVB_LWP_GAIN_DAY'][:])

    mode = l1_0.variables['ICON_L1_FUV_Mode'][:]
    br = profiler(l1_0)[:,:,mode==daynight]
    brc = profiler(l1_0, clean=True)[:,:,mode==daynight]
    brc2 = profiler(l1_9, clean=True)[:,:,mode==daynight]
    fig, ax = plt.subplots(1,3, figsize=(13.9, 4.8))
    maxx = max(brc[i%6:,:,:800].max(), brc2[i%6:,:,:800].max())
    minn = min(brc[i%6:,:,:800].min(), brc2[i%6:,:,:800].min())
    i1=ax[0].imshow(br[i%6,:,:800], aspect='auto', origin='lower', vmax=maxx, vmin=minn)
    i2=ax[1].imshow(brc[i%6,:,:800], aspect='auto', origin='lower', vmax=maxx, vmin=minn)
    i3=ax[2].imshow(brc2[i%6,:,:800], aspect='auto', origin='lower', vmax=maxx, vmin=minn)
    fig.colorbar(i1, ax=ax[0])
    fig.colorbar(i2, ax=ax[1])
    fig.colorbar(i3, ax=ax[2])
    plt.suptitle('SW Day Day: {}, Str: {}'.format(i,i%6))
    plt.show()

    l1_0.close()
    l1_9.close()
