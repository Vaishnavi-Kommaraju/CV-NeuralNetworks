# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dip import EncDec
from utils import imread

# Load clean and noisy image
#im = imread('../data/denoising/saturn.png')
#noise1 = imread('../data/denoising/saturn-noise1g.png')
im = imread('../data/denoising/lena.png')
noise1 = imread('../data/denoising/lena-noisy.png')

error1 = ((im - noise1)**2).sum()

print('Noisy image SE: {:.2f}'.format(error1))

plt.figure(1)

plt.subplot(121)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(122)
plt.imshow(noise1, cmap='gray')
plt.title('Noisy image SE {:.2f}'.format(error1))

plt.show(block=False)


################################################################################
# Denoising algorithm (Deep Image Prior)
################################################################################
lr = 0.01
numIter = 500
#Create network
net = EncDec()
optimizer = optim.Adam(net.parameters(), lr=lr)

# Loads noisy image and sets it to the appropriate shape
noisy_img = torch.FloatTensor(noise1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
clean_img = torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).transpose(2,3)
# Creates \eta (noisy input)
eta = torch.randn(*noisy_img.size())
eta.detach()


###
# Your training code goes here.

#loss_fn = F.mse_loss()
loss_fn = torch.nn.MSELoss(reduction='sum')
train_err, test_err = [], []
for itr in range(numIter):
    optimizer.zero_grad()
    out = net(eta)
    loss = loss_fn(out, noisy_img)   
    loss.backward()
    optimizer.step()
    out_img = out[0, 0, :, :].transpose(0,1).detach().numpy()
    train_err.append(((out_img - noise1)**2).sum())
    test_err.append(((out_img - im)**2).sum())
    print('Iterations: {} Loss: {:.6f}'.format(itr, loss.item())) 
###

# Shows final result
#print(eta.shape)
#out = net(eta)


out_img = out[0, 0, :, :].transpose(0,1).detach().numpy()

error1 = ((im - noise1)**2).sum()
error2 = ((im - out_img)**2).sum()
print(error1, error2)

plt.figure(3)
plt.axis('off')

plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1, cmap='gray')
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(out_img, cmap='gray')
plt.title('SE {:.2f}'.format(error2))

plt.savefig('../Figures/output_eta.jpg')
plt.show()

plt.plot(np.arange(numIter), train_err, 'r-', label = 'train error')
plt.plot(np.arange(numIter), test_err, 'b-', label = 'test error')
plt.xlabel('No of iterations')
plt.ylabel('MSE')
plt.legend()
plt.title('MSE vs training iterations')
plt.savefig('../Figures/Error.jpg')
