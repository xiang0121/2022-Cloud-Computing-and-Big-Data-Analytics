import os
import argparse
import math
import pandas as pd
import numpy as np
import random
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from IPython.display import display
from PIL import Image
from pytorch_gan_metrics import get_fid
from abc import abstractmethod
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.io import read_image

config = {
    'EPOCHES': 50,
    'TRAIN_DIR': './mnist',
    'WEIGHT_DIR': './checkpoint.pt', 
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'TIMESTEPS' : 1000,
    'SAVE_PATH' : '../images',
}



class TrainDataset(Dataset):
    def __init__(self, train=True,root=config['TRAIN_DIR']):
       self.train = train
       if self.train:
            self.paths = sorted(glob.glob(os.path.join(root, "*.png"), recursive=True))
            self.train_transforms = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])

    def __len__(self):
        if self.train:
            return len(self.paths)
        else:
            return 10000

    def __getitem__(self, idx):
        if self.train:
            img = Image.open(self.paths[idx])  
            img = self.train_transforms(img)
            img = img * 2 - 1
            
            return img
        else:
            return torch.randn(3, 32, 32)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.down_1_1 = ConvBlock(3, 64) # [64, 32, 32]
        self.down_1_2 = ConvBlock(64, 64)
        self.pool_1 = nn.MaxPool2d(2)
        
        self.down_2_1 = ConvBlock(64, 128) 
        self.down_2_2 = ConvBlock(128, 128)
        self.pool_2 = nn.MaxPool2d(2)

        self.down_3_1 = ConvBlock(128, 256) 
        self.down_3_2 = ConvBlock(256, 256)
        self.pool_3 = nn.MaxPool2d(2)

        self.down_4_1 = ConvBlock(256, 512)
        self.down_4_2 = ConvBlock(512, 512)
        self.pool_4 = nn.MaxPool2d(2)
        
        self.down_5_1 = ConvBlock(512, 1024)
        self.down_5_2 = ConvBlock(1024, 1024)
        
        self.up_6_1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        
        self.up_6_2 = ConvBlock(1024, 512)
        self.up_6_3 = ConvBlock(512, 512)

        self.up_7_1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_7_2 = ConvBlock (512, 256)
        self.up_7_3 = ConvBlock (256, 256)

        self.up_8_1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_8_2 = ConvBlock(256, 128)
        self.up_8_3 = ConvBlock(128, 128)

        self.up_9_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_9_2 = ConvBlock(128, 64)
        self.up_9_3 = ConvBlock(64, 64)

        self.up_10 = nn.Conv2d(64, 3, 1) #[3, 32, 32]

    def TimeEmbedding(self, t, dim):
        emb = (1 / 10000 ** (2 * torch.arange(dim / 2, device = t.device) / dim))
        emb = t[:, None] / emb[None, :]
        timeembed = torch.zeros(len(t), dim, device = t.device)
        timeembed[:, 0::2] = torch.sin(emb)
        timeembed[:, 1::2] = torch.cos(emb)
        return timeembed[:, :, None, None]

    def forward(self, x, t):

        x = self.down_1_1(x)
        x = x + self.TimeEmbedding(t, x.size()[1]) 
        c1 = self.down_1_2(x)
        p1 = self.pool_1(c1)
        
        x = self.down_2_1(p1)
        x = x + self.TimeEmbedding(t, x.size()[1]) 
        c2 = self.down_2_2(x)
        p2 = self.pool_2(c2)
        
        x = self.down_3_1(p2)
        x = x + self.TimeEmbedding(t, x.size()[1]) 
        c3 = self.down_3_2(x)
        p3 = self.pool_3(c3)
        
        x = self.down_4_1(p3)
        x = x + self.TimeEmbedding(t, x.size()[1]) 
        c4 = self.down_4_2(x)
        p4 = self.pool_4(c4)

        x = self.down_5_1(p4)
        x = x + self.TimeEmbedding(t, x.size()[1]) 
        c5 = self.down_5_2(x)
        
        up6 = self.up_6_1(c5) 
        merge6 = torch.cat([up6, c4], dim=1)
        x = self.up_6_2(merge6)
        x = x + self.TimeEmbedding(t, x.size()[1])
        c6 = self.up_6_3(x)

        up7 = self.up_7_1(c6)
        merge7 = torch.cat([up7, c3], dim=1)
        x = self.up_7_2(merge7)
        x = x + self.TimeEmbedding(t, x.size()[1])
        c7 = self.up_7_3(x)

        up8 = self.up_8_1(c7)
        merge8 = torch.cat([up8, c2], dim=1)
        x = self.up_8_2(merge8)
        x = x + self.TimeEmbedding(t, x.size()[1])
        c8 = self.up_8_3(x)

        up9 = self.up_9_1(c8)
        merge9 = torch.cat([up9, c1], dim=1)
        x = self.up_9_2(merge9)
        x = x + self.TimeEmbedding(t, x.size()[1])
        c9 = self.up_9_3(x)

        c10 = self.up_10(c9)

        return c10

def train(model, dataloader, alpha_bar):

    checkpoint = torch.load('./checkpoint.pt') if os.path.isfile('./checkpoint.pt') else None
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Restart epoch = {checkpoint["epoch"] + 1}')
        
    best_loss = 9999
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    for epoch in range(0 if checkpoint is None else checkpoint['epoch'] + 1, config['EPOCHES']):
        model.train()
        train_loss = []
        pbar = tqdm(dataloader)
        # Training
        for x in pbar:

            x = x.to(config['device'])
            t = torch.randint(config['TIMESTEPS'], [len(x)], device=config['device'])        
            epsilon = torch.randn_like(x)
            eps_coef = alpha_bar[t] ** 0.5
            eps_theta = (1 - alpha_bar[t]) ** 0.5
            X = eps_coef * x + eps_theta * epsilon
            p = model(X, t)
            loss = criterion(p ,epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f'< Train | {epoch}/{config["EPOCHES"]} >')
            pbar.set_postfix({'loss': loss.item()})
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        print(f'< Train | {epoch}/{config["EPOCHES"]} >: loss = {train_loss:.4f}')

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'best_model.ckpt')
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'checkpoint.pt')
            print('Save best model, loss : ', best_loss)
      
    print("finish model training")

@torch.no_grad()
def generate_image(model, dataloader, beta, alpha, alpha_bar):
    cnt=1
    model.eval()
    pbar = tqdm(dataloader)
    for x in pbar:
        x = x.to(config['device'])
        # q_sample
        for t_step in range(config['TIMESTEPS'] - 1, -1, -1):
            t = torch.full([len(x)], t_step, device = config['device'])
            z = torch.randn_like(x) if t_step > 0 else torch.zeros_like(x)

            sqrt_recip_alpha = 1 / alpha[t] ** 0.5
            sqrt_1m_alpha_bar = (1 - alpha_bar[t]) ** 0.5
            epsilon = model(x, t)
            variance =  (1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * beta[t]
            x = sqrt_recip_alpha * (x - (1 - alpha[t]) / sqrt_1m_alpha_bar * epsilon) + variance ** 0.5 * z

            pbar.set_postfix({'t_step':t_step})
        
        x = torch.clamp(x, min=-1, max=1)
        x = (x + 1) / 2
        transform = transforms.Resize(28)
        x = transform(x)
        
        for img in x:
            save_image(img, f'{config["SAVE_PATH"]}/{cnt:05d}.png')
            cnt += 1

@torch.no_grad()
def generate_grid(model, beta, alpha, alpha_bar):
    x = torch.zeros(8, 3, 32, 32, device = config['device'])
    images = [x]
    model.eval()
    # Generate images

    for step in range(config['TIMESTEPS'] - 1, -1, -1):
        t = torch.full([len(x)], step, device = config['device'])
        z = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
        
        
        sqrt_recip_alpha = 1 / alpha[t] ** 0.5
        sqrt_1m_alpha_bar = (1 - alpha_bar[t]) ** 0.5
        epsilon = model(x, t)
        variance =  (1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * beta[t]
        x = sqrt_recip_alpha * (x - (1 - alpha[t]) / sqrt_1m_alpha_bar * epsilon) + variance ** 0.5 * z

        if step in np.linspace(config['TIMESTEPS'], 0, num = 8, dtype = np.int64):
            images.append(x)

    images = torch.cat(images)
    images = torch.clamp(images, min = -1, max = 1)
    images = (images + 1) / 2
    transform = transforms.Resize(28)
    images = transform(images)
    save_image(images, '311511056.png')

def FID():
    images = []
    for i in range(10000):
        path = os.path.join(f'../images/{(i+1):05d}.png')
        image = read_image(path) / 255.
        images.append(image)
    images = torch.stack(images, dim=0)
    FID = get_fid(images, '../mnist.npz')
    print(f'{FID:.5f}')

def main():
    os.makedirs(config['SAVE_PATH'], exist_ok=True)
    beta = torch.linspace(0.0001, 0.02, config['TIMESTEPS'], device = config['device'])[:, None, None, None]
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, 0)

    # Load data
    train_set = TrainDataset(True)
    train_dataloader = DataLoader(train_set, batch_size=config['BATCH_SIZE'], num_workers=16, shuffle=True, pin_memory=True)

    test_set = TrainDataset(False)
    test_dataloader = DataLoader(test_set, batch_size=config['BATCH_SIZE'], num_workers=16, shuffle=False, pin_memory=True)
     
    model = Unet().to(config['device'])
    train(model, train_dataloader, alpha_bar)
    model = Unet().to(config['device'])
    model.load_state_dict(torch.load('best_model.ckpt'))
    generate_image(model, test_dataloader, beta, alpha, alpha_bar)

    model.load_state_dict(torch.load('best_model.ckpt'))
    generate_grid(model, beta, alpha, alpha_bar)
    FID()

if __name__ == '__main__':
    main()
