import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(batch_size=256, num_workers=4):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,  download=False, transform=transform)
    test_data  = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_data,  batch_size=4, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

