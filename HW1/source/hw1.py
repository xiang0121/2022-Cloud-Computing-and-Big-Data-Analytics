import os
import time
from random import shuffle
from unittest.mock import NonCallableMagicMock
import math
from xmlrpc.client import boolean
import pandas as pd
import numpy as np


import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
import random



best_acc = 0
config = { 
    'EPOCHES' : 100,
    'SEED' : 8787,
    'SPLIT_RATIO': 0.8,
    'TRAIN_DIR' : '/home/david0121/CCBDA/data/train',
    'TEST_DIR' : '/home/david0121/CCBDA/data/test',
    'BATCH_SIZE' : 12,
    'LEARNING_RATE' : 0.0001
}


myseed = config['SEED']  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


train_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])


test_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class VideoDataset(Dataset):
    
    def __init__(self, tag:bool ,path, tfm, files=None):
        super(VideoDataset).__init__()
        self.path = path
        if tag is True:
            self.files = sorted([os.path.join(path, y, x) for y in os.listdir(path) for x in os.listdir(os.path.join(path, y)) if x.endswith('.mp4')])
        else:
            self.files = sorted([os.path.join(path, y) for y in os.listdir(path) if y.endswith('.mp4')])
        # print(self.files)

        if files != None:
            self.files = files 
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        frame, _, _ = read_video(self.files[idx], pts_unit='sec', output_format='TCHW') # [T, C, H, W]
        frame = frame[np.linspace(frame.size()[0]*(1/3), frame.size()[0]*(2/3), 20 ,dtype=np.int16)]
        frame = transforms.Resize([128, 128])(frame)
        # frame = self.transform(frame)
        
        try:
            label = int(self.files[idx].split(os.path.sep)[-2])
        except:
            label = -1  # test set has not lable
        return frame.float()/255, label



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),      # [64, 32, 32]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),      # [128, 8, 8]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),      # [256, 2, 2]

            nn.Dropout2d()
        )

        self.rnn = nn.LSTM(1024, 1024, batch_first=True) # 256 * 2 * 2

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 39)
        )

    def forward(self, x):
        B, N = x.size()[0], x.size()[1]
        x = x.reshape(-1, x.size()[2], x.size()[3], x.size()[4]) # [B*N, C, H, W]
        x = self.cnn(x)
        x = x.reshape(B, N, -1) # [B, N, C*H*W]
        x = self.rnn(x)[0]
        x = x[:,-1,:].squeeze(dim=1)  # [B, C*H*W]
        out = self.fc(x)
        
        return out

class Trainer(object):
    def __init__(self):

        self.checkpoint = torch.load('../weights/checkpoint.pth') if os.path.isfile('../weights/checkpoint.pth') else None    # load checkpoint file

        use_cuda =  torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(config['SEED'])
        else:
            torch.manual_seed(config['SEED'])  

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.train_set = VideoDataset(True, config['TRAIN_DIR'], tfm=train_tfm)
        self.train_set_size = int(len(self.train_set) * config['SPLIT_RATIO'])
        self.val_set_size = len(self.train_set) - self.train_set_size
        self.train_set, self.val_set = random_split(self.train_set, [self.train_set_size, self.val_set_size])

        self.train_dataloader = DataLoader(self.train_set, batch_size=config['BATCH_SIZE'], num_workers=8, shuffle=True)
        self.val_dataloader = DataLoader(self.val_set, batch_size=config['BATCH_SIZE'], num_workers=8, shuffle=True)

        self.model = Classifier().to(self.device) 

        if use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            epoch = self.checkpoint['epoch']
            print(f'Restart epoch = {self.checkpoint["epoch"] + 1}')


        for epoch in range(1 if self.checkpoint is None else self.checkpoint['epoch'] + 1, config['EPOCHES'] + 1):
            self.train(epoch)
            if epoch % 1 == 0:
                self.valid(epoch)

        torch.cuda.empty_cache()
        print("finish model training")

    def train(self, epoch):
        epoches = config['EPOCHES']
        self.model.train()
        average_loss = []
        pbar = tqdm(self.train_dataloader,
                    desc=f'Train Epoch{epoch}/{epoches}')

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()  
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            average_loss.append(loss.item())
            self.optimizer.step()
            pbar.set_description(
                f'Train Epoch:{epoch}/{epoches} train_loss:{round(np.mean(average_loss), 4)}')
        self.scheduler.step()

    def valid(self, epoch):

        global best_acc
        epoches = config['EPOCHES']
        self.model.eval()
        val_loss = 0
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()

        average_loss = []

        pbar = tqdm(self.val_dataloader,
                    desc=f'Valid Epoch{epoch}/{epoches}',
                    mininterval=0.3)
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                output = self.model(data)
            average_loss.append(self.criterion(output, target).item())
            val_loss += self.criterion(output, target).item()  # sum up batch loss
            pred = torch.argmax(output, 1)

            correct += (pred == target).sum().float()
            total += len(target)
            predict_acc = correct / total
            pbar.set_description(
                f'Val Epoch:{epoch}/{epoches} acc:{predict_acc}')

        if  predict_acc > best_acc:
            best_acc = predict_acc
            print(f"Best model found at epoch: {epoch}, acc = {predict_acc} ,saving model")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': round(np.mean(average_loss), 2)
            },
                '../weights/checkpoint.pth')

if __name__ == "__main__":
    train = Trainer()

    print('========END========')