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

# How to define a Tensor - A tensor is a number, vector or a matrix.
tensor1 = torch.tensor(4.) # Shorthand of 4.0
print(tensor1)
print(tensor1.dtype)

tensor2 = torch.tensor([1., 2., 3., 4.])
print(tensor2)
print(tensor2.dtype)

tensor3 = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
print(tensor3)
print(tensor3.dtype)

tensor4 = torch.tensor([[1.2, 3.4], [1.3, 4.4], [4.5, 6.0], [5.0, 7.7]])
print(tensor4)
print(tensor4.dtype)

# Verify the dimension of a tensor. Tensors can have any number of dimensions.
print(tensor1.shape)
print(tensor2.shape)
print(tensor3.shape)
print(tensor4.shape)









