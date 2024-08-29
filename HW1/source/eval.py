from cgi import test
from importlib.metadata import files
import hw1
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
from torchvision import transforms

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

test_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

config = { 
    'EPOCHES' : 100,
    'SEED' : 8787,
    'SPLIT_RATIO': 0.8,
    'TRAIN_DIR' : '/home/david0121/CCBDA/data/train',
    'TEST_DIR' : '/home/david0121/CCBDA/data/test',
    'BATCH_SIZE' : 8,
    'LEARNING_RATE' : 0.0001
}

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



checkpoint = torch.load('../weights/checkpoint.pth') if os.path.isfile('../weights/checkpoint.pth') else None 

test_set = hw1.VideoDataset(False, config['TEST_DIR'], tfm=test_tfm)
test_loader = hw1.DataLoader(test_set, batch_size = config['BATCH_SIZE'], num_workers=8, shuffle=False, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


name = checkpoint['model_state_dict']
keys = [i for i in name.keys()]
for key in keys:
    name[key[7:]] = name.pop(key)

model_best = Classifier().to(device)
model_best.load_state_dict(name)
model_best.eval()
predict = []

with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        predict += test_label.squeeze().tolist()

df = pd.DataFrame()
print(test_set.files)
df['name'] = [os.path.split(i)[-1] for i in test_set.files]
df['label'] = predict
df.to_csv("submission.csv", index=False)


