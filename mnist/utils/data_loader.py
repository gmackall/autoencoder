import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

def load_data(batch_size, data_root):
    transform = transforms.Compose(
        [transforms.ToTensor()]#, transforms.Normalize([0.5], [0.5])]
        )
    trainset = datasets.MNIST(data_root, transform=transform)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
    return dataloader

#def load_feature_extracted_data(batch_size, data_root, trained_encoder):
    
