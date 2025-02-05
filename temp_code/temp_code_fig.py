import torch
import os
import open_clip

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from datetime import datetime
from datasets.flickr import Flickr30kDatasetRet, Flickr30kDatasetRetFix
from configs.data import Flickr30kCfg
from utils.model_util import AutoEncoderChannel, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader, random_split
from utils.utils import write_log, get_checkpoint_openclip, random_seed
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pylab import yticks, xticks

###################################################################################### plot learning rates and acc interpolation

ae_channel = AutoEncoderChannel(input_embed_dim=512, out_feature_dims=[[16], [16]],
                                        channel_snr=[24, 24], dtype=torch.float32, device='cuda', seeds=1234)
ae_channel.to('cuda')
lr=[]
lr_cont=[]
for i in range(50):
    if i==4:
        optimizer = optim.AdamW(ae_channel.parameters(), lr=2e-4 / 4, betas=(0.9, 0.99), eps=1e-6)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=5e-7)
    if i == 15:
        optimizer = optim.AdamW(ae_channel.parameters(), lr=1e-7, betas=(0.9, 0.99), eps=1e-6)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=5e-9)
    if i < 4:
        lr.append(2e-4)
    else:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr.append(current_lr)
for i in range(500):
    if i==40:
        optimizer = optim.AdamW(ae_channel.parameters(), lr=2e-4 / 4, betas=(0.9, 0.99), eps=1e-6)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=5e-7)
    if i == 150:
        optimizer = optim.AdamW(ae_channel.parameters(), lr=1e-7, betas=(0.9, 0.99), eps=1e-6)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=5e-9)
    if i<40:
        lr_cont.append(2e-4)
    else:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_cont.append(current_lr)

lr_array = np.array(lr)
lr_array_cont=np.array(lr_cont)
epochs=np.arange(50)
epochs_cont=np.arange(500)/10.0


lr_array=lr_array_cont[::10]
fig=plt.figure()
fig.set_size_inches(16, 9)
ax1=fig.add_subplot()
ax1.semilogy(epochs, lr_array, linestyle='solid',  marker='*', markersize=15, color='r', linewidth=0, label='learning rate for each epoch')
ax1.semilogy(epochs_cont, lr_array_cont, linestyle='solid', color='k', linewidth=4.0, label='learning rate schedule')
ax1.set_xlabel('epochs', fontsize=20)
ax1.set_ylabel('learning rate', fontsize=20)
yticks_labels=[f'lr_{i}' for i in range(5)]
yticks(np.array([2e-4, 2e-4/4, 5e-7, 1e-7, 5e-9]), yticks_labels, fontsize=15)
ax1.grid()
ax1.legend(loc='upper right', fontsize='xx-large')
plt.xticks(fontsize=15)
plt.show()

#######################################################################################
x=[34, 28, 49, 56]
x1=[34]
for i in range(4):
    x1.append(34+(28-34)/5.0*(i+1))
x1.append(28)
for i in range(4):
    x1.append(28+(49-28)/5.0*(i+1))
x1.append(49)
for i in range(4):
    x1.append(49+(56-49)/5.0*(i+1))
x1.append(56)
for i in range(4):
    x1.append(56)
x1=np.array(x1)
x=np.array(x)
n=np.arange(1, 21, 5)
n1=np.arange(1, 21, 1)

fig=plt.figure()
fig.set_size_inches(16, 9)
ax1=fig.add_subplot()
ax1.plot(n, x, linestyle='solid',  marker='*', markersize=25, color='b', linewidth=0, label='original data')
ax1.plot(n1, x1, linestyle='solid',  marker='o', markersize=10, color='b', linewidth=0, label='interpolated data')
ax1.plot(n1, x1, linestyle='solid', color='k', linewidth=4.0)
ax1.set_xlabel('$x\in \{d^{img}_i, \gamma^{img}_i, d^{cap}_i, \gamma^{cap}_i\}$', fontsize=20)
ax1.set_ylabel('data value', fontsize=20)
ax1.set_xticks(n1)
ax1.set_xticklabels(['$x_0$', '$x_{0,1}$', '$x_{0,2}$', '$x_{0,3}$', '$x_{0,4}$',
                     '$x_1$', '$x_{1,1}$', '$x_{1,2}$', '$x_{1,3}$', '$x_{1,4}$',
                     '$x_2$', '$x_{2,1}$', '$x_{2,2}$', '$x_{2,3}$', '$x_{2,4}$',
                     '$x_3$', '$x_{3,1}$', '$x_{3,2}$', '$x_{3,3}$', '$x_{3,4}$'], fontsize=18)
ax1.grid()
plt.yticks(fontsize=15)
ax1.legend(loc='upper left', fontsize='xx-large')
plt.show()