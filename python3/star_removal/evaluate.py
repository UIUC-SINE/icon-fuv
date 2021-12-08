import numpy as np
import matplotlib.pyplot as plt
import torch, glob, netCDF4
from keras.models import load_model
from keras import backend
from iconfuv.misc import get_br_nights
from unet import UNet

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

if __name__ == '__main__':
    channel = 'SW'
    target_mode = 'day'
    nc_prefix = 'ICON_L1_FUV_SWP' if channel=='SW' else 'ICON_L1_FUV_LWP'
    orbit = 2
    stripe = 3

    model_keras = '/home/kamo/resources/icon-fuv/python3/scripts/cnn_model\
_SW_new_v5.4'
#     model_keras = '/home/kamo/resources/icon-fuv/python3/scripts/cnn_model\
# _SW_old_v5.4'
    model_torch = '/home/kamo/resources/icon-fuv/python3/star_removal/saved\
/2021_12_06__23_02_02_NF_32_LR_0.0005_EP_3/best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(in_channels=1,
                 out_channels=1,
                 start_filters=32,
                 bilinear=True,
                 residual=True).to(device)


    net.load_state_dict(torch.load(model_torch))
    net.eval()

    model = load_model(model_keras)

    date = [
    '2020-06-18',
    # '2020-07-28',
    # '2020-08-07',
    # '2020-08-27',
    # '2020-10-06',
    # '2020-11-15',
    # '2021-05-20'
    ]

    path_dir = '/home/kamo/resources/icon-fuv/ncfiles/l1/'
    files = (glob.glob(path_dir+f'{nc_prefix}_{date[0]}*') +
        glob.glob(path_dir+f'*/{nc_prefix}_{date[0]}*')) # for dset1 & dset2
    files.sort()
    file = files[-1]
    l1 = netCDF4.Dataset(file, 'r')

    brs, brsc,_,_,_,_ = get_br_nights(l1, target_mode=target_mode)
    l1.close()

    brs = np.concatenate(brs, axis=1)
    brsc = np.concatenate(brsc, axis=1)
    inds = np.linspace(0,len(brs[0]),5).astype(int)
    ind = np.arange(inds[orbit], inds[orbit+1])
    br = brs[:, ind].transpose(1,2,0)
    brc = brsc[:, ind].transpose(1,2,0)

    br_keras = np.where(br>-50, br, -50)
    br_keras = model.predict(br_keras[:,:,:,None])[:,:,:,0]
    with torch.no_grad():
        br_torch = net(torch.from_numpy(br[:,None,:,:]).float().to(device))[:,0].cpu()

    plt.figure()
    plt.imshow(np.log10(br[:,:,stripe].T+1e2),
        aspect='auto', origin='lower', cmap='jet')
    plt.title(f'Starry - {date[0]} , Orbs: {orbit}')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(np.log10(br_torch[:,:,stripe].T+1e2),
        aspect='auto', origin='lower', cmap='jet')
    plt.title(f'Torch - {date[0]} , Orbs: {orbit}')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(np.log10(br_keras[:,:,stripe].T+1e2),
        aspect='auto', origin='lower', cmap='jet')
    plt.title(f'Keras - {date[0]} , Orbs: {orbit}')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(np.log10(brc[:,:,stripe].T+1e2),
        aspect='auto', origin='lower', cmap='jet')
    plt.title(f'Medfilt - {date[0]} , Orbs: {orbit}')
    plt.colorbar()
    plt.show()


    # plt.savefig(out_path+'/{}_orbs_{}.png'.format(date,i+1), dpi=150)
