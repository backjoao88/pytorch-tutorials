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

# Operations between tensors
x = torch.tensor(3., requires_grad=True)
w = torch.tensor(7., requires_grad=True)
b = torch.tensor(10., requires_grad=True)

y = w * x + b
print(y)

# Compute the derivatives
y.backward()

print('dy/dx = ', x.grad)
print('dy/dw = ', w.grad)
print('dy/db = ', b.grad)







