import torch
import torch.nn as nn
import torch.nn.functional as F 

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(36 * 6 * 6, 128)
        self.fc2   = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(-1, 36 * 6 * 6)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
if __name__ == "__main__":
    model = CNN()
    print(model)
    x = torch.randn(4, 3, 32, 32)  # Batch size of 4, 3 channels, 32x32 images
    output = model(x)
    print(output.shape)  # Should be [4, 10] for CIFAR-10 classes
    