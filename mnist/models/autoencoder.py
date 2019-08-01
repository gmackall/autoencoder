import numpy as np
import math
import torch.nn as nn
import torch

from .alternate_losses import init_loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self, image_dims=(1,28,28), bottleneck_factor=2, loss="softmax_kl", verbosity=True, sparsity=False, load_point=None):
        super(AutoEncoder, self).__init__()

        self.image_dims = image_dims
        self.input_size = np.prod(self.image_dims)
        self.hidden_size = int(self.input_size/bottleneck_factor)
        self.verbose = verbosity
        self.sparse = sparsity
        self.load_point = load_point
        self.loss_fn = init_loss(loss, sparsity)
        self.arch = "autoencoder"

        self.fc_down = nn.Linear(self.input_size, self.hidden_size)
        self.fc_up = nn.Linear(self.hidden_size, self.input_size)
        self.activation = nn.ReLU()

        self.model = nn.Sequential(self.fc_down,
                                   self.activation,
                                   self.fc_up,
                                   self.activation)

        if load_point is not None:
            self.load_state_dict(torch.load(load_point))
            model.eval()

    def full_pass(self, x):
        x = x.to(device)
        x = x.view(-1, np.prod(self.image_dims))
        out = self.model(x)
        return out

    def encode(self, x):
        x = x.to(device)
        x = x.view(-1, np.prod(self.image_dims))
        out = self.activation(self.fc_down(x))
        return out.to(device)

    def decode(self, x):
        return self.activation(self.fc_up(x)).to(device)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
