import time
from data_loader import BasicDataset
from torch.utils.data import DataLoader

t0 = time.time()
trainset = BasicDataset(data_dir = './data/', fold='train')
print('TrainSet Time: {}'.format(time.time()-t0))
t0 = time.time()
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
print('TrainLoader Time: {}'.format(time.time()-t0))

t0 = time.time()
for i,data in enumerate(trainloader):
    x,y = data
print('Loop Time: {}'.format(time.time()-t0))
