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
from tqdm.auto import tqdm
from IPython.display import display
from PIL import Image

config = {
    'EPOCHES': 100,
    'SEED': 8787,
    'TRAIN_DIR': '../data/unlabeled',
    'TEST_DIR': '../data/test',
    'WEIGHT_DIR': './checkpoint.pth', 
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 0.00015
}


random.seed(config['SEED'])
np.random.seed(config['SEED'])
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed_all(config['SEED'])


class TrainDataset(Dataset):
    def __init__(self, root=config['TRAIN_DIR']):
        self.paths = sorted(
            glob.glob(os.path.join(root, "*.jpg"), recursive=True))

        self.train_transforms = transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(96),
            transforms.RandomRotation(degrees=(0,10)),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            # transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        random.seed(config['SEED'])
        img = Image.open(self.paths[idx])  # 96 * 96
        img1 = self.train_transforms(img)
        img2 = self.train_transforms(img)
        return img1, img2


class TestDataset(Dataset):
    def __init__(self, root=config['TEST_DIR']):
        self.paths = sorted(
            glob.glob(os.path.join(root, "**/*.jpg"), recursive=True))
        self.labels = [
            int(os.path.basename(os.path.dirname(path)))
            for path in self.paths
        ]
        self.num_classes = len(set(self.labels))
        self.test_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.paths[idx])
        img = self.test_transforms(img)
        return img


class ConvModel(nn.Module):

    def __init__(self):
        super(ConvModel, self).__init__()

        self.encoder = nn.Sequential(# [3, 96, 96]
            #conv1
            nn.Conv2d(3, 64, 3, padding=1),  # [64, 96, 96]
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(4, 4, 0),  # [64, 24, 24]
           
            #conv2
            nn.Conv2d(64, 128, 3, padding=1),  # [128, 24, 24]
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),  # [128, 6, 6]


        )

        self.projection = nn.Sequential(
            nn.Linear(128*6*6, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection(x)
        return x


def xt_xent(
    u: torch.Tensor,                               # [N, C]
    v: torch.Tensor,                               # [N, C]
    temperature: float = 0.5,
):
    """
    N: batch size
    C: feature dimension
    """
    N, C = u.shape

    z = torch.cat([u, v], dim=0)                   # [2N, C]
    z = F.normalize(z, p=2, dim=1)                 # [2N, C]
    s = torch.matmul(z, z.t()) / temperature       # [2N, 2N] similarity matrix
    mask = torch.eye(2 * N).bool().to(z.device)    # [2N, 2N] identity matrix
    # fill the diagonal with negative infinity
    s = torch.masked_fill(s, mask, -float('inf'))
    label = torch.cat([                            # [2N]
        torch.arange(N, 2 * N),                    # {N, ..., 2N - 1}
        torch.arange(N),                           # {0, ..., N - 1}
    ]).to(z.device)

    loss = F.cross_entropy(s, label)               # NT-Xent loss
    return loss


def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(
            batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]
    return max(accs)


def train():
    # Load data
    checkpoint = torch.load(config['WEIGHT_DIR']) if os.path.isfile(
        config['WEIGHT_DIR']) else None    # load checkpoint file

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(config['SEED'])
    else:
        torch.manual_seed(config['SEED'])

    device = torch.device("cuda" if use_cuda else "cpu")
    model = ConvModel().to(device)
    train_set = TrainDataset()
    train_dataloader = DataLoader(
        train_set, batch_size=config['BATCH_SIZE'], num_workers=12, shuffle=True)

    test_set = TestDataset()
    test_dataloader = DataLoader(
        test_set, batch_size=config['BATCH_SIZE'], num_workers=12, shuffle=False)

    paths = sorted(
        glob.glob(os.path.join(config['TEST_DIR'], "**/*.jpg"), recursive=True))
    test_label = torch.tensor([
        int(os.path.basename(os.path.dirname(path)))
        for path in paths
    ])


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5, eta_min=1e-5)

    if checkpoint is not None:
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Restart epoch = {checkpoint["epoch"] + 1}')
    else:
        best_acc = 0

    for epoch in range(1 if checkpoint is None else checkpoint['epoch'] + 1, config['EPOCHES'] + 1):
        # Training
        epoches = config['EPOCHES']
        average_loss = []
        embedding = []
        train_pbar = tqdm(train_dataloader,
                          desc=f'Train Epoch{epoch}/{epoches}')
        for x1, x2 in train_pbar:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            y1, y2 = model(x1), model(x2)
            loss = xt_xent(y1, y2)
            loss.backward()
            average_loss.append(loss.item())
            optimizer.step()
            train_pbar.set_description(
                f'Train Epoch:{epoch}/{epoches} train_loss:{round(np.mean(average_loss), 4)}')

        scheduler.step()

        # Testing
        test_pbar = tqdm(test_dataloader, desc=f'Test Epoch:{epoch}/{epoches}')

        for data in test_pbar:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            test_pbar.set_description(f'Test Epoch:{epoch}/{epoches}')
            embedding.append(output)
        embedding = torch.cat(embedding)

        test_acc = KNN(embedding, test_label, batch_size=64)

        if test_acc > best_acc:
            best_acc = test_acc
            print(
                f"Best model found at epoch: {epoch}, acc = {test_acc} ,saving model")
            torch.save({
                'best_acc' : best_acc,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': round(np.mean(average_loss), 2)
            },
                config['WEIGHT_DIR'])

    torch.cuda.empty_cache()
    print("finish model training")


def main():
    train()


if __name__ == '__main__':
    main()
