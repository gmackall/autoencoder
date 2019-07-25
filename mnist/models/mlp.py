import torch.nn as nn
import torch.optim as optim
import numpy as np

#TODO: -add argparse to control verbosity, image size, etc

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.image_dims = (1,28,28)
        self.input_size = np.prod(self.image_dims)
        self.hidden_size = int(self.input_size / 2)
        self.classes = 10
        self.verbose = True

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.classes)
        self.activation = nn.ReLU()
        self.final = nn.Softmax()

        self.model = nn.Sequential(self.fc1,
                                   self.activation,
                                   self.fc2,
                                   self.final)

    def forward(self, x):
        x = x.view(-1, np.prod(self.image_dims))
        out = self.model(x)
        return out

