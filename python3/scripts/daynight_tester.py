# Ulas Kamaci - 2022/04/15
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from iconfuv.misc import lastfile, get_br_nights
from scipy.ndimage import convolve1d

def daynight_index(br, threshold=30):
    index = br.shape[1]
    y = np.mean(br, axis=(0,2))
    y = convolve1d(y, 0.05*np.ones(20), mode='reflect')
    y1 = np.diff(y)
    ind = np.sort(np.where(y>threshold)[0])
    for i in ind:
        if i >= len(y1):
            break
        k = min(len(y1)-i, 30) # take the mean of last 30 epochs
        if (y1[i] > 5) & (y[i] < y[-k:].mean()) & ((y1[i:].max()>10) | (i>0.75*len(y1))):
        # if (y1[i] > 5):
            index = i
            break
    return index, y, y1

def ploter(file):
    date = file[-21:-11]
    l1 = netCDF4.Dataset(file, 'r')
    brs,_,_,_,_,_ = get_br_nights(l1)

    inds = [];  br1d = [];  br1dd = []
    rows = 5 if len(brs) < 16 else 6
    figi, axi = plt.subplots(rows, 3, figsize=(11.8,10.2))
    figl, axl = plt.subplots(rows, 3, figsize=(11.8,10.2))
    figd, axd = plt.subplots(rows, 3, figsize=(11.8,10.2))
    figi.suptitle('Br - {}'.format(date))
    figl.suptitle('Mean 1d - {}'.format(date))
    figd.suptitle('Diff 1d - {}'.format(date))
    for i,br in enumerate(brs):
        o1,o2,o3 = daynight_index(br)
        inds.append(o1)
        br1d.append(o2)
        br1dd.append(o3)
        im=axi[i%rows,i//rows].imshow(br.mean(axis=0).T, aspect='auto', origin='lower')
        figi.colorbar(im, ax=axi[i%rows,i//rows])
        axl[i%rows,i//rows].plot(br1d[i])
        axd[i%rows,i//rows].plot(br1dd[i])
        col = 'r' if inds[i] < len(br1d[i]) else 'b'
        axi[i%rows,i//rows].axvline(inds[i], color=col)
        axl[i%rows,i//rows].axvline(inds[i], color=col)
        axd[i%rows,i//rows].axvline(inds[i], color=col)
        axl[i%rows,i//rows].grid(which='major', axis='both')
        axd[i%rows,i//rows].grid(which='major', axis='both')
    plt.close('all')
    figi.savefig('figs/{}_figi.png'.format(date))
    figl.savefig('figs/{}_figl.png'.format(date))
    figd.savefig('figs/{}_figd.png'.format(date))
    l1.close()

filedir = '/home/kamo/resources/icon-fuv/ncfiles/l1/'

datetest = ['2022-03-27']

fileref = lastfile(filedir + '2020/*SWP*2020-07-11*')
filetest = lastfile(filedir + '*SWP*2022-03-26*v0*')

for i in ['27','28']:
    # ploter(lastfile(filedir + '2020/*SWP*2020-07-{}*'.format(i)))
    ploter(lastfile(filedir + '*SWP*2022-03-{}*v0*'.format(i)))
