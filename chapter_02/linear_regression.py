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
import numpy

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# Defining the Linear Regression Model
def model(x):
    return x @ w.t() + b

# Define the loss function. (Difference^2) / total of elements (MSE loss function)
def compute_loss(tensor1, tensor2):
    difference = tensor1 - tensor2
    return torch.sum(difference * difference) / difference.numel()

inputs = numpy.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
outputs = numpy.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')
inputs = torch.from_numpy(inputs)
outputs = torch.from_numpy(outputs)

# Generate predictions
preds = model(inputs)

# Compute loss
loss = compute_loss(preds, outputs)

# Compute the derivatives of loss function (gradients)
loss.backward()

# no_grad function tells to PyTorch to not modify the grad while modifying the weights or bias.
# the .grad is set to zero because the pytorch backward function is accumulative.
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()


# Lets train for 100 epochs.
lr = 1e-5
epochs = 100
for i in range(epochs):
    preds = model(inputs)
    loss = compute_loss(preds, outputs)
    print('epoch: {0} - mse: {1}'.format(i, loss))
    loss.backward()
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr
        w.grad.zero_()
        b.grad.zero_()

print('testing model...')
preds = model(inputs)
loss = compute_loss(preds, outputs)
print('result mse: {0}'.format(loss))