import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch

from data_loader import load_data

#TODO: -replace hardcoded loss w/ model defined loss
#       currently just hardcoded to get off the ground
#       with MLP, will need MSE(out, base) for autoenc

def train(network, epochs, learning_rate, momentum, batch_size, data_root):
    train_loader = load_data(np.prod(network.image_dims), batch_size, data_root)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    self_eval(network, train_loader)
    for i in range(1, epochs + 1):
        train_epoch(network, i, train_loader, optimizer)
        self_eval(network, train_loader)

def train_epoch(network, epoch_index, loader, optimizer):
    network.train()
    print_interval = 10
    
    for batch_num, (base, target) in enumerate(loader):
        optimizer.zero_grad()
        out = network(base)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        if batch_num % print_interval == 0:
            if network.verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_index, batch_num * len(base), len(loader.dataset),
                    100. * batch_num / len(loader), loss.item()))

def self_eval(network, loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            out = network(data)
            loss_fn = nn.CrossEntropyLoss()
            test_loss += loss_fn(out, target)
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(loader.dataset),
                100. * correct / len(loader.dataset)))
