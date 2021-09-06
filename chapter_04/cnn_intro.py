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

# To understand how a CNN works, lets first define what is the Conv2D layer.

'''
The 2D convolution is a fairly simple operation at heart: you start with a kernel, 
which is simply a small matrix of weights. This kernel “slides” over the 2D input data, 
performing an elementwise multiplication with the part of the input it is currently on, 
and then summing up the results into a single output pixel.
'''

def apply_kernel(image, kernel):
    ri, ci = image.shape       # image dimensions
    rk, ck = kernel.shape      # kernel dimensions
    ro, co = ri-rk+1, ci-ck+1  # output dimensions
    output = torch.zeros([ro, co])
    for i in range(ro): 
        for j in range(co):
            print(image[i:i+rk,j:j+ck] * kernel)
            output[i,j] = torch.sum(image[i:i+rk,j:j+ck] @ kernel)
    return output

sample_image = torch.tensor([
    [3, 3, 2, 1, 0], 
    [0, 0, 1, 3, 1], 
    [3, 1, 2, 2, 3], 
    [2, 0, 0, 2, 2], 
    [2, 0, 0, 0, 1]
], dtype=torch.float32)

sample_kernel = torch.tensor([
    [0, 1, 2], 
    [2, 2, 0], 
    [0, 1, 2]
], dtype=torch.float32)

output = apply_kernel(sample_image, sample_kernel)
print(output)