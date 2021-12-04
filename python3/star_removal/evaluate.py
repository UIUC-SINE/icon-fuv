import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_profiles(net, valloader, i):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = next(iter(valloader))
    x = x.to(device=device, dtype=torch.float) / 10.
    y = y.to(device=device, dtype=torch.float) / 10.
    with torch.no_grad():
        out = net(x)

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
