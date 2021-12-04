import torch
import sys
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import BasicDataset
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt

def train_net(net,
              device,
              trainloader,
              valloader,
              epochs,
              optimizer,
              criterion
):

    train_loss_over_epochs = []
    val_loss_over_epochs = []

    for epoch in range(epochs):  # loop over the dataset multiple times

        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs = data[0].to(device=device, dtype=torch.float)
            true_outputs = data[1].to(device=device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(true_outputs, outputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # Normalizing the loss by the total number of train batches
        running_loss/=len(trainloader)*trainloader.batch_size
        print('[%d] Train loss: %.3f' %
              (epoch + 1, running_loss))

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
        print('Validation loss: %.3f' %
              (running_valloss))

        train_loss_over_epochs.append(running_loss)
        val_loss_over_epochs.append(running_valloss)

    return train_loss_over_epochs, val_loss_over_epochs

# def main():
if __name__ == '__main__':
    # ---------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = BasicDataset(data_dir = './data/', fold='train')
    valset = BasicDataset(data_dir = './data/', fold='val')
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)

    NUM_FILT = 8
    LR = 1e-3
    EPOCHS = 50
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(in_channels=1,
                 out_channels=1,
                 start_filters=NUM_FILT,
                 bilinear=True).to(device)

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    try:
        trainloss, valloss = train_net(net=net,
                  device=device,
                  trainloader=trainloader,
                  valloader=valloader,
                  epochs=EPOCHS,
                  optimizer=optimizer,
                  criterion=criterion
                  )

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        sys.exit(0)

    torch.save(net.state_dict(), f'nf_{NUM_FILT}_LR_{LR}_EP_{EPOCHS}.pth')

    # Show results
    x, y = next(iter(valloader))
    x = x.to(device=device, dtype=torch.float)
    y = y.to(device=device, dtype=torch.float)
    with torch.no_grad():
        out = net(x)

    i = 1
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(x[i,0].cpu(), aspect='auto', origin='lower')
    ax[0].set_title('Starry')
    ax[1].imshow(y[i,0].cpu(), aspect='auto', origin='lower')
    ax[1].set_title('Gold Standard')
    ax[2].imshow(out[i,0].cpu(), aspect='auto', origin='lower')
    ax[2].set_title('Predicted')

# if __name__ == '__main__':
#     main()
