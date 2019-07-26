import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

#TODO: -add argparse to control verbosity, image size, etc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.image_dims = (1,28,28)
        self.input_size = np.prod(self.image_dims)
        self.hidden_size = int(self.input_size * 2)
        self.classes = 10
        self.verbose = True
        self.arch = "classifier"
        self.loss_fn = nn.CrossEntropyLoss()

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_final = nn.Linear(self.hidden_size, self.classes)

        self.activation = nn.ReLU()
        self.final = nn.Softmax(dim=0)

        self.model = nn.Sequential(self.fc1,
                                   self.activation,
                                   self.fc2,
                                   self.activation,
                                   self.fc_final,
                                   self.final)

    def forward(self, x):
        x = x.to(device)
        x = x.view(-1, np.prod(self.image_dims))
        out = self.model(x.to(device)).to(device)
        return out.to(device)

