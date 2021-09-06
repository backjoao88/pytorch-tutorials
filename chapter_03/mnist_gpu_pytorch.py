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
import torchvision
import matplotlib.pyplot as plt

# Let's define a helper function to ensure that our code uses the GPU if available and defaults to using the CPU if it isn't.
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Data Loader

dataset = torchvision.datasets.MNIST(root='data/', download=False, transform=torchvision.transforms.ToTensor())

# Serate train and val images
train_size = 50000
val_size   = 10000

train_ds, val_ds = torch.utils.data.dataset.random_split(dataset, [train_size, val_size])

# Create data loaders

batch_size = 128

train_loader = torch.utils.data.dataloader.DataLoader(train_ds, batch_size, shuffle=True)
val_loader = torch.utils.data.dataloader.DataLoader(val_ds, 2*batch_size)

# Defining a Accuracy function

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Defining a Pytorch Custom Neural Network model.

class MnistNeuralNetwork(torch.nn.Module):
    
    '''
    torch.nn.Module - Base class for all neural network modules.
    '''
    
    """Feedfoward neural network with 1 hidden layer"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(input_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x_batch):
        x_batch = x_batch.view(x_batch.size(0), -1)
        hidden_layer_output = self.hidden_layer(x_batch)
        hidden_layer_output = torch.nn.functional.relu(hidden_layer_output)
        output_layer_output = self.output_layer(hidden_layer_output)
        return output_layer_output

    def training_step(self, x_batch):
        images, labels = x_batch
        out = self(images)
        loss = torch.nn.functional.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                                      # Generate predictions
        loss = torch.nn.functional.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)                             # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
            
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

input_size = 784
hidden_size = 32 
num_classes = 10

# Lets see the initial Weights and Bias of the model

model = MnistNeuralNetwork(input_size, hidden_size=32, output_size=num_classes)
# for t in model.parameters():
#     print(t.shape)
    

# Defining device data loader

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Trying to generate some batches to our model

train_loader = DeviceDataLoader(train_loader, device)
val_loader   = DeviceDataLoader(val_loader, device)


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# Model (on GPU)
model = MnistNeuralNetwork(input_size, hidden_size=hidden_size, output_size=num_classes)
to_device(model, device)
history = [evaluate(model, val_loader)]
history += fit(2, 0.5, model, train_loader, val_loader)
print(history)

losses = [x['val_loss'] for x in history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs');

test_dataset = torchvision.datasets.MNIST(root='data/', 
                     train=False,
                     transform=torchvision.transforms.ToTensor())

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))