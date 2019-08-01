import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch

from .data_loader import load_data

#TODO: -(mostly completed) replace hardcoded loss w/ model defined loss
#       currently just hardcoded to get off the ground
#       with MLP, will need MSE(out, base) for autoenc

#Train on GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(network, epochs, learning_rate, momentum, batch_size, data_root):
    network.to(device)
    train_loader = load_data(batch_size, data_root)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    self_eval(network, train_loader)
    for i in range(1, epochs + 1):
        train_epoch(network, i, train_loader, optimizer)
        self_eval(network, train_loader)

def train_epoch(network, epoch_index, loader, optimizer):
    network.train()
    print_interval = 10
    
    for batch_num, (base, target) in enumerate(loader):
        base = base.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        out = network(base)
        if network.arch == "autoencoder":
            ins = base.view(-1, np.prod(network.image_dims)).to(device)
            loss = network.loss_fn(ins, out, network.hidden_size)
        else:
            loss = network.loss_fn(out, target)
        loss.backward()
        optimizer.step()

        if batch_num % print_interval == 0:
            if network.verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_index, batch_num * len(base), len(loader.dataset),
                    100. * batch_num / len(loader), loss))#loss.item()

def self_eval(network, loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for base, target in loader:
            base = base.to(device)
            target = target.to(device)
            out = network(base)
            if network.arch == "autoencoder":
                ins = base.view(-1, np.prod(network.image_dims)).to(device)
                test_loss += network.loss_fn(ins, out, network.hidden_size)
            else:
                out = out.to(device)
                test_loss += network.loss_fn(out, target)
                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(loader.dataset)
        if network.arch == "autoencoder":
            print('\nTest set: Avg. loss: {:.4f}\n'.format(
                    test_loss))
        else:
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(loader.dataset),
                    100. * correct / len(loader.dataset)))
