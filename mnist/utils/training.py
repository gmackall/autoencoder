import torch.optim as optim
import numpy as np

from data_loader import load_data

def train(network, epochs, learning_rate, momentum, batch_size, data_root):
    train_loader = load_data(np.prod(network.image_dims), batch_size, data_root)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    
    for i in range(1, epochs + 1):
        train_epoch(network, epoch_index, data, target, train_loader, optimizer)

def train_epoch(network, epoch_index, data, target, loader, optimizer):
    netork.train()
    
