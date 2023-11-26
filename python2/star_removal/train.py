import torch
import sys, os, logging, time
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import datetime

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
              criterion,
              path
):

    train_loss_over_epochs = []
    val_loss_over_epochs = []

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

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
        logging.info('[%d] Train loss: %.3f' % (epoch + 1, running_loss))

        net.eval()
        running_valloss = 0.0
        best_valloss = 1e6
        for i, data in enumerate(valloader):
            # get the inputs
            inputs = data[0].to(device=device, dtype=torch.float)
            true_outputs = data[1].to(device=device, dtype=torch.float)

            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(true_outputs, outputs)

                running_valloss += loss.item()

        running_valloss/=len(valloader)*valloader.batch_size
        logging.info('Validation loss: %.3f' % (running_valloss))

        if running_valloss < best_valloss:
            best_valloss = running_valloss
            torch.save(net.state_dict(), 'saved/{}/best_model.pth'.format(name))

        train_loss_over_epochs.append(running_loss)
        val_loss_over_epochs.append(running_valloss)

    return train_loss_over_epochs, val_loss_over_epochs

# def main():
if __name__ == '__main__':
    # ---------
    NUM_FILT = 32
    LR = 5e-4
    EPOCHS = 30
    RESIDUAL = True
    BATCH_SIZE = 64
    BILINEAR = True
    OPTIMIZER = 'ADAM'
    LOSS = 'L1'
    LOAD = False
    loaded_model_path = 'saved/nf_32_LR_0.001_EP_15.pth'

    now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    name = '{}_NF_{}_LR_{}_EP_{}_{}_LOSS'.format(now,NUM_FILT,LR,EPOCHS,LOSS)
    os.mkdir('saved/'+name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [] # clean up the previous handlers to avoid problems
    fh = logging.FileHandler('saved/{}/output.log'.format(name))
    formatter = logging.Formatter('%(asctime)s; %(message)s','%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    trainset = BasicDataset(data_dir = './data/', fold='train')
    valset = BasicDataset(data_dir = './data/', fold='val')
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=True)

    net = UNet(in_channels=1,
                 start_filters=NUM_FILT,
                 bilinear=BILINEAR,
                 residual=RESIDUAL).to(device)

    if LOAD:
        net.load_state_dict(torch.load(loaded_model_path))

    if OPTIMIZER=='ADAM':
        optimizer = optim.Adam(net.parameters(), lr=LR)
    elif OPTIMIZER=='SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if LOSS=='MSE':
        criterion = nn.MSELoss()
    elif LOSS=='L1':
        criterion = nn.L1Loss()

    training_summary = [
    '############## Network Parameters ############## \n',
    'Number of starting filters = {} \n'.format(NUM_FILT),
    'Residual Layer (subtract input from the last layer) = {} \n'.format(RESIDUAL),
    'Bilinear Interpolation for Upsampling (if False, use transposed ',
    'convolution) = {} \n'.format(BILINEAR),
    '\n############## Optimization Parameters ############## \n',
    'Optimizer = {} \n'.format(OPTIMIZER),
    'Loss = {} \n'.format(LOSS),
    'Learning Rate = {} \n'.format(LR),
    'Num Epochs = {} \n'.format(EPOCHS),
    'Training Batch Size = {} \n'.format(BATCH_SIZE),
    '\n############## Data Parameters ############## \n',
    'Num of Tranining Images = {} \n'.format(len(trainset)),
    'Num of Validation Images = {} \n'.format(len(valset)),
    ]

    with open('saved/{}/summary.txt'.format(name), 'w') as file:
        for line in training_summary:
            file.write(line)

    try:
        t0 = time.time()
        trainloss, valloss = train_net(net=net,
                  device=device,
                  trainloader=trainloader,
                  valloader=valloader,
                  epochs=EPOCHS,
                  optimizer=optimizer,
                  criterion=criterion,
                  path=name
                  )
        train_time = datetime.timedelta(seconds=int(time.time() - t0))

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'saved/{}/INTERRUPTED.pth'.format(name))
        sys.exit(0)

    torch.save(net.state_dict(), 'saved/{}/nf_{}_LR_{}_EP_{}.pth'.format(name,NUM_FILT,LR,EPOCHS))

    training_summary = [
    '\n############## Results ############## \n',
    'Final Validation Loss: {:.3f} \n'.format(valloss[-1]),
    'Minimum Validation Loss: {:.3f} \n'.format(np.min(valloss)),
    'Final Training Loss: {:.3f} \n'.format(trainloss[-1]),
    'Minimum Training Loss: {:.3f} \n'.format(np.min(trainloss)),
    'Training Time: {} \n'.format(str(train_time))
    # '\n############## Notes ############## \n',
    ]

    with open('saved/{}/summary.txt'.format(name), 'w') as file:
        for line in training_summary:
            file.write(line)


    plt.figure()
    plt.semilogy(trainloss, label='Training Loss')
    plt.semilogy(valloss, label='Validation Loss')
    plt.title('Convergence Plot')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.savefig('saved/{}/convergence_plot.png'.format(name))
    plt.close()

# if __name__ == '__main__':
#     main()