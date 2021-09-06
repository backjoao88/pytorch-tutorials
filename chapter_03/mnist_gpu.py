#!/usr/bin/env python3

'''
Copyright (C) 2021 João Paulo Back
 
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
 
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
 
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

'''

This file contains a simple tutorial on how to use PyTorch on Python.

'''

import torch
from torch.utils.data.dataset import random_split
import torchvision
import matplotlib.pyplot as plt

# Download dataset 
dataset = torchvision.datasets.MNIST(root='data/', download=False, transform=torchvision.transforms.ToTensor())
image, lbl = dataset[0]

# Premute image to be like 28x28x1 (plt requires)
p_image = image.permute(1, 2, 0)

# Show img
# plt.imshow(p_image, cmap='gray')
# plt.show()
# print(lbl)

# Serate train and val images
train_size = 50000
val_size   = 10000

train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Create data loaders

batch_size = 128

train_loader = torch.utils.data.dataloader.DataLoader(train_ds, batch_size, shuffle=True)
val_loader = torch.utils.data.dataloader.DataLoader(val_size, batch_size)

# Show images as grids
for images, _ in train_loader:
    # print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    #plt.imshow(torchvision.utils.make_grid(images, nrow=16).permute((1, 2, 0)))
    # plt.show()
    break

# Defining the model
for images, label in train_loader:
    inputs = images.reshape(-1, 784)
    break

input_size = inputs.shape[-1]
hidden_size = 32
output_size = 10

layer1 = torch.nn.Linear(input_size, hidden_size)
layer1_output = layer1(inputs)
relu_outputs = torch.nn.functional.relu(layer1_output)
layer2 = torch.nn.Linear(hidden_size, output_size)
layer2_outputs = layer2(relu_outputs)
# print(layer2_outputs.shape)

# Expanded version of layer2(F.relu(layer1(inputs)))
outputs = (torch.nn.functional.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias
torch.allclose(outputs, layer2_outputs, 1e-3)

