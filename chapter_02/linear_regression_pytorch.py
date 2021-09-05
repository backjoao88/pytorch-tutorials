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

inputs = numpy.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
outputs = numpy.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
outputs = torch.from_numpy(outputs)

# TrainDataset allow us to access data as tuples.
train_dataset = torch.utils.data.TensorDataset(inputs, outputs)

# Show first three inputs and first three outputs.
# print(train_dataset[0:3])

# DataLoader split the data into batches of a predefined size while training.
batch_size = 5
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

# Show the data using the data loader. It will show based on the batch size.
# for xb, yb in train_dataloader:
#     print(xb)
#     print(yb)

# Define the model
model = torch.nn.Linear(3, 2)
#print(list(model.parameters()))

# Generate outputs
preds = model(inputs)

# Define the loss function
compute_loss = torch.nn.functional.mse_loss
loss = compute_loss(preds, outputs)

# Instead of manually manipulating the model's weights & biases using gradients, 
# we can use the optimizer optim.SGD. SGD is short for "stochastic gradient descent". 
# The term stochastic indicates that samples are selected in random batches instead of as a single group.

opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Train the model

# We are now ready to train the model. We'll follow the same process to implement gradient descent:

# Generate predictions
# Calculate the loss
# Compute gradients w.r.t the weights and biases
# Adjust the weights by subtracting a small quantity proportional to the gradient
# Reset the gradients to zero

epochs = 100
lr = 1e-5

for i in range(epochs):
    for x_batch, y_batch in train_dataloader:
        # Generate de output based on the batch size
        pred = model(x_batch)
        # Calculate de Loss (MSE)
        loss = compute_loss(pred, y_batch)
        # Update the Gradient of the weights (updating .grad values)
        loss.backward()
        # Update the weights using the gradients and the learning rate
        opt.step()
        # Reset the accumulative gradients
        opt.zero_grad()
        
    # Print the progress
    print('epoch [{}/{}], MSE: {:.4f}'.format(i+1, 100, loss.item()))