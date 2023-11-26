import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch, glob, netCDF4
from keras.models import load_model
from iconfuv.misc import get_br_nights, profiler
from iconfuv.artifact_removal2 import hot_pixel_correction, medfilt3d, artifact_removal
from star_removal.unet import UNet

def predictor(br, model, platform):
    if platform=='torch':
        with torch.no_grad():
            return model(torch.from_numpy(br[:,None,:,:]).float().to(device))[:,0].cpu().numpy()
    elif platform=='keras':
        br2 = np.where(br>-50, br, -50)
        return model.predict(br2[:,:,:,None])[:,:,:,0]

if __name__ == '__main__':
    channel = 'SW'
    channelnum = 1 if channel=='SW' else 2
    target_mode = 'night'
    target_mode_num = 1 if target_mode=='day' else 2
    nc_prefix = 'ICON_L1_FUV_SWP' if channel=='SW' else 'ICON_L1_FUV_LWP'
    orbit = 0
    stripe = 3
    # matplotlib.rcParams['lines.linewidth'] = 0.3

    model_keras = '/home/kamo/resources/icon-fuv/python3/scripts/cnn_model_SW_old_v5.4'
    model_torch = '/home/kamo/resources/icon-fuv/python3/star_removal/saved/2021_12_23__14_47_24_NF_32_LR_0.0005_EP_30_L1_LOSS/best_model.pth'
    path_save = '/home/kamo/resources/icon-fuv/python3/scripts/artifact_removal_results/{}_{}/'.format(channel,target_mode)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(in_channels=1,
                 start_filters=32,
                 bilinear=True,
                 residual=True).to(device)

    net.load_state_dict(torch.load(model_torch, map_location=device))
    net.eval()

    model = load_model(model_keras)

    if target_mode=='day':
        model2 = { # displays on bottom right
            'model'     : net,
            'platform'  : 'torch',
            'title'     : 'Unet'
        }
    elif target_mode=='night':
        model2 = { # displays on bottom right
            'model'     : model,
            'platform'  : 'keras',
            'title'     : 'Keras'
        }

    dates_day = [
    # '2020-06-18',
    # '2020-07-28',
    '2020-08-07',
    # '2020-08-27',
    # '2020-10-06',
    # '2020-11-15',
    # '2021-05-20'
    ]
    dates_night = [
    '2020-01-01',
    # '2020-06-18',
    # '2020-10-17',
    # '2021-03-16',
    # '2021-03-17',
    # '2021-03-18',
    # '2021-04-25'
    ]
    dates = dates_day if target_mode=='day' else dates_night

    for date in dates:
        path_dir = '/home/kamo/resources/icon-fuv/ncfiles/l1/'
        files = (glob.glob(path_dir+'{}_{}_v0*'.format(nc_prefix,date)) +
            glob.glob(path_dir+'*/{}_{}_v0*'.format(nc_prefix,date))) # for dset1 & dset2
        files.sort()
        file = files[-1]
        l1 = netCDF4.Dataset(file, 'r')

        profiles = profiler(l1)
        mode = l1.variables['ICON_L1_FUV_Mode'][:]
        profiles_clean = artifact_removal(profiles, channelnum, mode,
            '/home/kamo/resources/icon-fuv/python3/iconfuv/neural_networks')

        break

        brs, brsc,_,_,_,_ = get_br_nights(l1, target_mode=target_mode)
        l1.close()

        brs = np.concatenate(brs, axis=1)
        brsc = np.concatenate(brsc, axis=1)
        inds = np.linspace(0,len(brs[0]),5).astype(int)
        ind = np.arange(inds[orbit], inds[orbit+1])
        br0 = brs[:, ind]
        brc = brsc[:, ind].transpose(1,2,0)
        if target_mode=='night':
            _, temp = medfilt3d(br0, threshold=50, mode=2)
            diff, _ = hot_pixel_correction(temp)
            br = br0 - diff[:,np.newaxis,:]
        else:
            br = br0
        br0 = br0.transpose(1,2,0)
        br = br.transpose(1,2,0)

        cutter = lambda x,y: np.where(x>y,x,y)

        br1 = profiles_clean[:,:,(mode==target_mode_num)][:,:,ind].transpose(2,1,0)
        br00 = profiles[:,:,(mode==target_mode_num)][:,:,ind].transpose(2,1,0)
        br2 = predictor(br, model2['model'], model2['platform'])

        for stripe in [0,3,5]:
            print('Date:{} - Stripe:{}'.format(date,stripe))

            if target_mode=='night':
                # vmin=-5
                vmin=np.nanmin(br1[:,:,stripe])
                vmax=np.nanmax(br1[:,:,stripe])
            else:
                vmax = np.nanmax(brc[:,:,stripe])
                vmin = min(
                    np.nanmin(br[:,:,stripe]),
                    np.nanmin(brc[:,:,stripe]),
                    np.nanmin(br1[:,:,stripe]),
                    np.nanmin(br2[:,:,stripe]),
                )

            # f, ax =plt.subplots(2,2, figsize=(12.7,7.3))
            # plt.suptitle(f'{date} , Channel: {channel} , Stripe: {stripe}')
            # i1=ax[0,0].imshow(br00[:,:,stripe].T,
            #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            # ax[0,0].set_title('Starry')
            # i2=ax[0,1].imshow(br1[:,:,stripe].T,
            #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            # ax[0,1].set_title('Artifact Removal')
            # i3=ax[1,0].imshow(brc[:,:,stripe].T,
            #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            # ax[1,0].set_title('Median Filter')
            # i4=ax[1,1].imshow(br2[:,:,stripe].T,
            #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            # ax[1,1].set_title(model2['title'])
            # plt.colorbar(i1, ax=ax[0,0])
            # plt.colorbar(i2, ax=ax[0,1])
            # plt.colorbar(i3, ax=ax[1,0])
            # plt.colorbar(i4, ax=ax[1,1])
            # # f.colorbar(im, ax=ax.ravel().tolist())
            # plt.tight_layout()
            # plt.savefig(path_save+f'{date}_ch_{channel}_st_{stripe}.png', dpi=250)
            # plt.close(f)
            #
            # f, ax =plt.subplots(2,2, figsize=(12.7,7.3))
            # plt.suptitle(f'{date} , Channel: {channel} , Stripe: {stripe}')
            # i1=ax[0,0].imshow(br00[:,:,stripe].T,
            #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            # vmin=-20; vmax=10
            # ax[0,0].set_title('Starry')
            # i2=ax[0,1].imshow(-(br00-br1)[:,:,stripe].T,
            #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            # ax[0,1].set_title('{} - Starry'.format('Artifact Removal'))
            # i3=ax[1,0].imshow(-(br00-brc)[:,:,stripe].T,
            #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            # ax[1,0].set_title('Median Filter - Starry')
            # i4=ax[1,1].imshow(-(br00-br2)[:,:,stripe].T,
            #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            # ax[1,1].set_title('{} - Starry'.format(model2['title']))
            # plt.colorbar(i1, ax=ax[0,0])
            # plt.colorbar(i2, ax=ax[0,1])
            # plt.colorbar(i3, ax=ax[1,0])
            # plt.colorbar(i4, ax=ax[1,1])
            # # f.colorbar(im, ax=ax.ravel().tolist())
            # plt.tight_layout()
            # plt.savefig(path_save+f'{date}_ch_{channel}_st_{stripe}_diff_br.png', dpi=250)
            # plt.close(f)
            continue

# %% plot
# stripe=3
# epoch=450
# f, ax =plt.subplots(2,2, figsize=(12.7,7.3))
# plt.suptitle('{} , Channel: {} , Stripe: {}'.format(date, channel, stripe))
# i1=ax[0,0].imshow(br00[:,:,stripe].T,
#     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
# ax[0,0].set_title('Starry')
# i2=ax[0,1].imshow(br1[:,:,stripe].T,
#     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
# ax[0,1].set_title('Artifact Removed')
# ax[0,1].axvline(epoch, color='m')
# i3=ax[1,0].imshow(-(br00-br1)[:,:,stripe].T,
#     aspect='auto', origin='lower', cmap='jet', vmax=10, vmin=-20)
# ax[1,0].set_title('Difference: Artifact Removed - Starry')
# ax[1,1].plot(br1[epoch,:,stripe],np.arange(256), label='Artifact Removed')
# ax[1,1].plot(br00[epoch,:,stripe],np.arange(256), label='Starry (Raw)')
# ax[1,1].set_title('1D Profiles for Daytime Epoch: {}'.format(epoch))
# ax[1,1].legend()
# ax[1,1].grid(axis='both', which='both')
# plt.colorbar(i1, ax=ax[0,0])
# plt.colorbar(i2, ax=ax[0,1])
# plt.colorbar(i3, ax=ax[1,0])
# # f.colorbar(im, ax=ax.ravel().tolist())
# plt.tight_layout()
# plt.show()
# plt.savefig('/home/kamo/amo.png', dpi=250)
# # plt.savefig(path_save+f'{date}_ch_{channel}_st_{stripe}.png', dpi=250)
# # plt.close(f)
