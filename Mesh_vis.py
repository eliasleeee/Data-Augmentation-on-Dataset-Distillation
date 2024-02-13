import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator

from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

#load the data


data_image_100  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_100it.pt'))
data_image_200  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_200it.pt'))
data_image_300  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_300it.pt'))
data_image_400  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_400it.pt'))
data_image_500  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_500it.pt'))
data_image_600  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_600it.pt'))
data_image_700  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_700it.pt'))
data_image_800  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_800it.pt'))
data_image_900  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_900it.pt'))
data_image_1000  =  torch.load(os.path.join('result/CIFAR10', 'differ_none_CIFAR10_ConvNet_10ipc_1000it.pt'))


data_image_000_100 = data_image_100['data']
data_image_100_200 = data_image_200['data'] - data_image_100['data']
data_image_200_300 = data_image_300['data'] - data_image_200['data']
data_image_300_400 = data_image_400['data'] - data_image_300['data']
data_image_400_500 = data_image_500['data'] - data_image_400['data']
data_image_500_600 = data_image_600['data'] - data_image_500['data']
data_image_600_700 = data_image_700['data'] - data_image_600['data']
data_image_700_800 = data_image_800['data'] - data_image_700['data']
data_image_800_900 = data_image_900['data'] - data_image_800['data']
data_image_900_1000 = data_image_1000['data'] - data_image_900['data']


image_100  =  data_image_000_100 
image_200  =  data_image_100_200 
image_300  =  data_image_200_300 
image_400  =  data_image_300_400 
image_500  =  data_image_400_500 
image_600  =  data_image_500_600 
image_700  =  data_image_600_700 
image_800  =  data_image_700_800 
image_900  =  data_image_800_900
image_1000  =  data_image_900_1000




differ_vis_unbias_100 = torch.sum(image_100, dim = 0, keepdim = False)
differ_vis_unbias_200 = torch.sum(image_200, dim = 0, keepdim = False)
differ_vis_unbias_300 = torch.sum(image_300, dim = 0, keepdim = False)
differ_vis_unbias_400 = torch.sum(image_400, dim = 0, keepdim = False)
differ_vis_unbias_500 = torch.sum(image_500, dim = 0, keepdim = False)
differ_vis_unbias_600 = torch.sum(image_600, dim = 0, keepdim = False)
differ_vis_unbias_700 = torch.sum(image_700, dim = 0, keepdim = False)
differ_vis_unbias_800 = torch.sum(image_800, dim = 0, keepdim = False)
differ_vis_unbias_900 = torch.sum(image_900, dim = 0, keepdim = False)
differ_vis_unbias_1000 = torch.sum(image_1000, dim = 0, keepdim = False)



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(0, 32)
Y = np.arange(0, 32)
X, Y = np.meshgrid(X, Y)


differ_vis_normalized_100 = differ_vis_unbias_100/torch.max(differ_vis_unbias_100)
differ_vis_normalized_200 = differ_vis_unbias_200/torch.max(differ_vis_unbias_100)
differ_vis_normalized_300 = differ_vis_unbias_300/torch.max(differ_vis_unbias_100)
differ_vis_normalized_400 = differ_vis_unbias_400/torch.max(differ_vis_unbias_100)
differ_vis_normalized_500 = differ_vis_unbias_500/torch.max(differ_vis_unbias_100)
differ_vis_normalized_600 = differ_vis_unbias_600/torch.max(differ_vis_unbias_100)
differ_vis_normalized_700 = differ_vis_unbias_700/torch.max(differ_vis_unbias_100)
differ_vis_normalized_800 = differ_vis_unbias_800/torch.max(differ_vis_unbias_100)
differ_vis_normalized_900 = differ_vis_unbias_900/torch.max(differ_vis_unbias_100)
differ_vis_normalized_1000 = differ_vis_unbias_1000/torch.max(differ_vis_unbias_100)

# Plot the surface.


z_max = torch.max(differ_vis_normalized_100)
z_min = 0
# Customize the z axis.
ax.set_zlim(z_min, z_max)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')


# Add a color bar which maps values to colors.
surf_100 = ax.plot_surface(X, Y, differ_vis_normalized_100[X,Y], cmap=cm.coolwarm,vmin = z_min ,vmax = z_max ,linewidth=0, antialiased=False)
fig.colorbar(surf_100, shrink=0.5, aspect=5) 
plt.savefig("Heatmap_000_100.png")

# Add a color bar which maps values to colors.
surf_300 = ax.plot_surface(X, Y, differ_vis_normalized_300[X,Y], cmap=cm.coolwarm,vmin = z_min ,vmax = z_max ,linewidth=0, antialiased=False)
fig.colorbar(surf_300, shrink=0.5, aspect=5) 
plt.savefig("Heatmap_200_300.png")

# Add a color bar which maps values to colors.
surf_500 = ax.plot_surface(X, Y, differ_vis_normalized_500[X,Y], cmap=cm.coolwarm,vmin = z_min ,vmax = z_max ,linewidth=0, antialiased=False)
fig.colorbar(surf_500, shrink=0.5, aspect=5) 
plt.savefig("Heatmap_400_500.png")

# Add a color bar which maps values to colors.
surf_700 = ax.plot_surface(X, Y, differ_vis_normalized_700[X,Y], cmap=cm.coolwarm,vmin = z_min ,vmax = z_max ,linewidth=0, antialiased=False)
fig.colorbar(surf_700, shrink=0.5, aspect=5) 
plt.savefig("Heatmap_600_700.png")

# Add a color bar which maps values to colors.
surf_900 = ax.plot_surface(X, Y, differ_vis_normalized_900[X,Y], cmap=cm.coolwarm,vmin = z_min ,vmax = z_max ,linewidth=0, antialiased=False)
fig.colorbar(surf_900, shrink=0.5, aspect=5) 
plt.savefig("Heatmap_800_900.png")



