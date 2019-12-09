import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncDec(nn.Module):

    def __init__(self):
        super(EncDec, self).__init__()
        #raise NotImplementedError()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(1)
        
        self.fc1 = nn.Linear(6 * 6 * 32, 120, bias=True)
        self.fc2 = nn.Linear(120, 10, bias=True)

    def forward(self, x):
        #raise NotImplementedError()
        x = self.conv1(x)
        #print(x.shape)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #print(x.shape)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        #print(x.shape)
        m = nn.Upsample(scale_factor=8, mode='bilinear')
        x = m(x)
        #print(x.shape)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        #print(x.shape)
        
        return x
