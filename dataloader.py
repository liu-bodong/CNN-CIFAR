import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(download=False, batch_size=256, num_workers=4):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,  download=download, transform=transform)
    test_data  = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_data,  batch_size=4, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def download_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

if __name__ == "__main__":
    print("Download data from external source")
    download_data()