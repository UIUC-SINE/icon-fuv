import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import glob, os
from iconfuv.misc import get_br_nights
path_dir = '/home/kamo/resources/icon-fuv/ncfiles/l1/'

def profile_extractor(files_path):
    files = glob.glob(path_dir + files_path + '/*')
    files.sort()
    mirror_dir = ['M9','M6','M3','P0','P3','P6']

    for file in files:
        print(file[-21:-11])
        l1 = netCDF4.Dataset(file, 'r')
        date = file[-21:-11]
        brs = np.zeros((l1.dimensions['Epoch'].size,256,6))
        ind = np.random.permutation(brs.shape[0])[:500]
        # ind = np.arange(560)
        for i,d in enumerate(mirror_dir):
            brs[:,:,i] = l1.variables['ICON_L1_FUVB_LWP_PROF_{}'.format(d)][:]
        fig, ax = plt.subplots(2,1,figsize=[6.4,8.4])
        plt.tight_layout()
        for i in ind:
            ax[0].imshow(brs[i], aspect='auto', origin='lower', cmap='jet')
            ax[1].plot(brs[i], np.arange(256))
            ax[1].legend(['0','1','2','3','4','5'])
            fig.savefig(path_dir+'dset2/figs/{}_Ep_{}'.format(date,i))
            ax[0].cla()
            ax[1].cla()

def orbit_extractor(files_path, out_path, target_mode='day', stripe=3):
    files = glob.glob(files_path + '/IC*')
    files.sort()
    mirror_dir = ['M9','M6','M3','P0','P3','P6']

    for file in files:
        print(file[-21:-11])
        l1 = netCDF4.Dataset(file, 'r')
        date = file[-21:-11]
        brs, brsc,_,_,_,_=get_br_nights(l1, target_mode=target_mode)
        inds = np.linspace(0,len(brs[0][0]),5).astype(int)
        for i in range(len(inds)-1):
            ind = np.arange(inds[i], inds[i+1])
            plt.clf()
            plt.imshow(np.log10(brs[0][stripe, ind].T+1e2),
                aspect='auto', origin='lower', cmap='jet')
            plt.title(f'{date} , Orbs: {i+1}')
            plt.colorbar()
            plt.savefig(out_path+'/{}_orbs_{}.png'.format(date,i+1), dpi=150)

if __name__ == '__main__':
    files_path = path_dir + 'dset2'
    target_mode = 'day'
    stripe = 3
    out_path = os.path.join(files_path , 'figs_orb')
    orbit_extractor(
        files_path=files_path,
        out_path=out_path,
        target_mode=target_mode,
        stripe=stripe
    )
