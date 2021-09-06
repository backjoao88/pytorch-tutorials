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

# Create a tensor with the x-dimension and with y-number
t1 = torch.full((3,2), 15)
print(t1)
t2 = torch.full((3,2), 10)
print(t2)

# Concatenate two tensors
t3 = torch.cat((t1, t2))
print(t3)

# Reshape tensor
t4 = t2.reshape(3, 2)
print(t4)