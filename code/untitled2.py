# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:51:05 2019

@author: vaish
"""
import torch
import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

class Net(nn.Module):
      def __init__(self):
         super(Net, self).__init__()
         self.conv1 = nn.Conv2d(1, 32, 3, 1)
         self.conv2 = nn.Conv2d(32, 64, 3, 1)
         self.dropout1 = nn.Dropout2d(0.25)
         self.dropout2 = nn.Dropout2d(0.5)
         self.fc1 = nn.Linear(9216, 128)
         self.fc2 = nn.Linear(128, 10)
 
      def forward(self, x):
         x = self.conv1(x)
         x = F.relu(x)
         x = self.conv2(x)
         x = F.max_pool2d(x, 2)
         x = self.dropout1(x)
         x = torch.flatten(x, 1)
         x = self.fc1(x)
         x = F.relu(x)
         x = self.dropout2(x)
         x = self.fc2(x)
         output = F.log_softmax(x, dim=1)
         return output

class getData(Dataset):
    def __init__(self, data, transform=None):

        self.data = data[0]
        self.target = data[1]
        self.transform = transform
        print(self.data.shape)
        print(self.target.shape)
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.data[:,:,idx]
        target = self.target[idx]
        #print(data.shape)
        #print(target)
        sample = {'data': data, 'target': target}

        if self.transform:
            sample['data'] = self.transform(sample['data'])
        return sample


def CNN_1(data,trainSet):
    transformed_dataset = getData(data=(data['x'][:,:,data['set']==trainSet], data['y'][data['set']==trainSet]),
                                           transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))
    
    dataloader = DataLoader(transformed_dataset, batch_size=64,
                        shuffle=True)
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['data'].size(),
          sample_batched['target'].size())
        model = Net()