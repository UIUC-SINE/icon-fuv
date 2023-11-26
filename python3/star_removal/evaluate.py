import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch, glob, netCDF4, os
from keras.models import load_model
from keras import backend
from iconfuv.misc import get_br_nights
from iconfuv.artifact_removal2 import hot_pixel_correction, medfilt3d
from star_removal.unet import UNet

def plot_profiles(net, valloader, i, shift_stripes=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = next(iter(valloader))
    x = x.to(device=device, dtype=torch.float)
    y = y.to(device=device, dtype=torch.float)
    with torch.no_grad():
        out = net(x)
    if shift_stripes:
        for i in range(6):
            x[:,:,:,i] += 100*i
            y[:,:,:,i] += 100*i
            out[:,:,:,i] += 100*i

    fig, ax = plt.subplots(2,4, sharey=True)
    ax[0,0].imshow(x[i,0].cpu(), aspect='auto', origin='lower')
    ax[0,0].set_title('Starry')
    ax[0,1].imshow(y[i,0].cpu(), aspect='auto', origin='lower')
    ax[0,1].set_title('Gold Standard')
    ax[0,2].imshow(out[i,0].cpu(), aspect='auto', origin='lower')
    ax[0,2].set_title('Predicted')
    diff = out[i,0] - y[i,0]
    ax[0,3].imshow(diff.cpu(), aspect='auto', origin='lower')
    ax[0,3].set_title('Diff')
    ax[1,0].plot(x[i,0].cpu(), np.arange(256))
    ax[1,0].set_title('Starry')
    ax[1,1].plot(y[i,0].cpu(), np.arange(256))
    ax[1,1].set_title('Gold Standard')
    ax[1,2].plot(out[i,0].cpu(), np.arange(256))
    ax[1,2].set_title('Predicted')
    ax[1,3].plot(diff.cpu(), np.arange(256))
    ax[1,3].set_title('Diff')
    plt.show()

def plot_dataset_diff(x,y,th1=10,bs=512):
    if type(x)!=np.ndarray:
        x=x.cpu()
        y=y.cpu()
    f=plt.gcf()
    plt.close(f)
    fig, ax = plt.subplots(1,3, figsize=(14,5))
    i = np.random.randint(bs)
    i1 = ax[0].imshow((x[i,0]-y[i,0]), aspect='auto', origin='lower', cmap='jet', vmax=th1)
    i2 = ax[1].imshow(np.log10((x[i,0]-y[i,0])+10), aspect='auto', origin='lower', cmap='jet')
    ax[2].plot((x[i,0]-y[i,0]), np.arange(256))
    ax[2].set_xlim([-2,22])
    plt.colorbar(i1, ax=ax[0])
    plt.colorbar(i2, ax=ax[1])
    plt.show()


def calculate_valloss(net, valloader, criterion):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.eval()
        running_valloss = 0.0
        for i, data in enumerate(valloader):
            # get the inputs
            inputs = data[0].to(device=device, dtype=torch.float)
            true_outputs = data[1].to(device=device, dtype=torch.float)

            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(true_outputs, outputs)

                running_valloss += loss.item()

        running_valloss/=len(valloader)*valloader.batch_size
        print('Validation loss: %.3f' % (running_valloss))
        return running_valloss

def predictor(br, model, platform):
    if platform=='torch':
        with torch.no_grad():
            return model(torch.from_numpy(br[:,None,:,:]).float().to(device))[:,0].cpu().numpy()
    elif platform=='keras':
        br2 = np.where(br>-50, br, -50)
        return model.predict(br2[:,:,:,None])[:,:,:,0]

if __name__ == '__main__':
    channel = 'SW'
    target_mode = 'day'
    nc_prefix = 'ICON_L1_FUV_SWP' if channel=='SW' else 'ICON_L1_FUV_LWP'
    orbit = 2
    stripe = 3
    # matplotlib.rcParams['lines.linewidth'] = 0.3

    model_keras_new = '/home/kamo/resources/icon-fuv/python3/scripts/cnn_model\
_SW_new_v5.4'
    model_keras_old = '/home/kamo/resources/icon-fuv/python3/scripts/cnn_model\
_SW_old_v5.4'
    model_torch_new = '/home/kamo/resources/icon-fuv/python3/star_removal/saved\
/2021_12_23__14_47_24_NF_32_LR_0.0005_EP_30_L1_LOSS/best_model.pth'
    model_torch_base = '/home/kamo/resources/icon-fuv/python3/star_removal/saved\
/2021_12_22__15_43_43_NF_32_LR_0.0005_EP_30_L1_LOSS/best_model.pth'
    path_save = f'/home/kamo/resources/icon-fuv/python3/star_removal/saved\
/2021_12_23__14_47_24_NF_32_LR_0.0005_EP_30_L1_LOSS/results/{channel}_{target_mode}/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_base = UNet(in_channels=1,
                 start_filters=32,
                 bilinear=True,
                 residual=True).to(device)

    net_new = UNet(in_channels=1,
                 start_filters=32,
                 bilinear=True,
                 residual=True).to(device)

    net_base.load_state_dict(torch.load(model_torch_base))
    net_new.load_state_dict(torch.load(model_torch_new))
    net_base.eval()
    net_new.eval()

    model_old = load_model(model_keras_old)
    model_new = load_model(model_keras_new)

    model1 = { # displays on top right
        'model'     : net_new,
        'platform'  : 'torch',
        'title'     : 'Unet New'
    }

    model2 = { # displays on bottom right
        'model'     : net_base,
        'platform'  : 'torch',
        'title'     : 'Unet Base'
    }

    dates_day = [
    '2020-06-18',
    '2020-07-28',
    '2020-08-07',
    '2020-08-27',
    '2020-10-06',
    '2020-11-15',
    '2021-05-20'
    ]
    dates_night = [
    '2020-01-01',
    '2020-06-18',
    '2020-10-17',
    '2021-03-16',
    '2021-03-17',
    '2021-03-18',
    '2021-04-25'
    ]
    dates = dates_day if target_mode=='day' else dates_night

    for date in dates:
        # for stripe in [0]:
        for stripe in [0,3,5]:
            path_dir = '/home/kamo/resources/icon-fuv/ncfiles/l1/'
            files = (glob.glob(path_dir+f'{nc_prefix}_{date}_v0*') +
                glob.glob(path_dir+f'*/{nc_prefix}_{date}_v0*')) # for dset1 & dset2
            files.sort()
            file = files[-1]
            l1 = netCDF4.Dataset(file, 'r')

            brs, brsc,_,_,_,_ = get_br_nights(l1, target_mode=target_mode)
            l1.close()

            brs = np.concatenate(brs, axis=1)
            brsc = np.concatenate(brsc, axis=1)
            inds = np.linspace(0,len(brs[0]),5).astype(int)
            ind = np.arange(inds[orbit], inds[orbit+1])
            br = brs[:, ind]
            brc = brsc[:, ind].transpose(1,2,0)
            if target_mode=='night':
                _, temp = medfilt3d(br, threshold=50, mode=2)
                diff, _ = hot_pixel_correction(temp)
                br -= diff[:,np.newaxis,:]
            br = br.transpose(1,2,0)

            cutter = lambda x,y: np.where(x>y,x,y)

            br1 = predictor(br, model1['model'], model1['platform'])
            br2 = predictor(br, model2['model'], model2['platform'])

            if target_mode=='night':
                vmin=-5
                vmax=np.max(br1[:,:,stripe])
            else:
                vmax = np.max(brc[:,:,stripe])
                vmin = min(
                    np.min(br[:,:,stripe]),
                    np.min(brc[:,:,stripe]),
                    np.min(br1[:,:,stripe]),
                    np.min(br2[:,:,stripe]),
                )

            f, ax =plt.subplots(2,2, figsize=(12.7,7.3))
            plt.suptitle(f'{date} , Channel: {channel} , Stripe: {stripe}')
            i1=ax[0,0].imshow(br[:,:,stripe].T,
                aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            ax[0,0].set_title('Starry')
            i2=ax[0,1].imshow(br1[:,:,stripe].T,
                aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            ax[0,1].set_title(model1['title'])
            i3=ax[1,0].imshow(brc[:,:,stripe].T,
                aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            ax[1,0].set_title('Median Filter')
            i4=ax[1,1].imshow(br2[:,:,stripe].T,
                aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            ax[1,1].set_title(model2['title'])
            plt.colorbar(i1, ax=ax[0,0])
            plt.colorbar(i2, ax=ax[0,1])
            plt.colorbar(i3, ax=ax[1,0])
            plt.colorbar(i4, ax=ax[1,1])
            # f.colorbar(im, ax=ax.ravel().tolist())
            plt.tight_layout()
            plt.savefig(path_save+f'{date}_ch_{channel}_st_{stripe}.png', dpi=250)
            plt.close(f)

            f, ax =plt.subplots(2,2, figsize=(12.7,7.3))
            plt.suptitle(f'{date} , Channel: {channel} , Stripe: {stripe}')
            i1=ax[0,0].imshow(br[:,:,stripe].T,
                aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            vmin=-20; vmax=10
            ax[0,0].set_title('Starry')
            i2=ax[0,1].imshow(-(br-br1)[:,:,stripe].T,
                aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            ax[0,1].set_title('{} - Starry'.format(model1['title']))
            i3=ax[1,0].imshow(-(br-brc)[:,:,stripe].T,
                aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            ax[1,0].set_title('Median Filter - Starry')
            i4=ax[1,1].imshow(-(br-br2)[:,:,stripe].T,
                aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
            ax[1,1].set_title('{} - Starry'.format(model2['title']))
            plt.colorbar(i1, ax=ax[0,0])
            plt.colorbar(i2, ax=ax[0,1])
            plt.colorbar(i3, ax=ax[1,0])
            plt.colorbar(i4, ax=ax[1,1])
            # f.colorbar(im, ax=ax.ravel().tolist())
            plt.tight_layout()
            plt.savefig(path_save+f'{date}_ch_{channel}_st_{stripe}_diff_br.png', dpi=250)
            plt.close(f)

    # fig, ax = plt.subplots(1,2, sharey=True)
    # i0=ax[0].imshow(np.log10(cutter(br[:,:,stripe].T,10)),
    #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
    # ax[0].axvline(date[1][0], color='m')
    # ax[1].plot(br_torch[date[1][0],:,stripe],np.arange(256), label='unet')
    # ax[1].plot(br[date[1][0],:,stripe],np.arange(256), label='raw')
    # ax[1].legend()
    # ax[1].grid(axis='both', which='both')
    # fig.colorbar(i0, ax=ax[0])
    # plt.savefig(path_save+'unet_1d.png', dpi=250)
    # # plt.show()
    #
    # fig, ax = plt.subplots(1,2, sharey=True)
    # i0=ax[0].imshow(np.log10(cutter(br[:,:,stripe].T,10)),
    #     aspect='auto', origin='lower', cmap='jet', vmax=vmax, vmin=vmin)
    # ax[0].axvline(date[1][0], color='m')
    # ax[1].plot((br-br_torch)[date[1][0],:,stripe],np.arange(256), label='diff')
    # ax[1].grid(axis='both', which='both')
    # ax[1].legend()
    # fig.colorbar(i0, ax=ax[0])
    # plt.savefig(path_save+'unet_1d_diff.png', dpi=250)

    # plt.savefig(out_path+'/{}_orbs_{}.png'.format(date,i+1), dpi=150)
