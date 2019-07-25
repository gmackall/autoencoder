import numpy as np
import math

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.image_dims = (1,28,28)
        self.input_size = np.prod(self.image_dims)
        self.hidden_size = 500
        self.verbose = True

        self.fc_down = nn.Linear(self.input_size, self.hidden_size)
        self.fc_up = nn.Linear(self.hidden_size, self.input_size)
        self.activation = nn.ReLU()

        self.model = nn.Sequential(self.fc_down,
                                   self.activation,
                                   self.fc_up,
                                   self.activation)

    def forward(self, x):
        out = self.model(x)
        return out

        
