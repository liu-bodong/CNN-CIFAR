import torch
import torch.nn as nn
import torch.nn.functional as F 

class Residual(nn.Module):
    def __init__(self, in_channels, num_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5, 
                                 nn.AdaptiveAvgPool2d((1,1)),
                                 nn.Flatten(), nn.Linear(512, 10))
    
    def resnet_block(self, in_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, num_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        return self.net(x)


        
        

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

    
class CNN_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.aap   = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc3   = nn.Linear(36, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
                
        x = self.aap(x)
                
        x = torch.squeeze(x)
                
        x = self.fc3(x)
        
        return x
    
if __name__ == "__main__":
    model = CNN()
    print(model)
    x = torch.randn(4, 3, 32, 32)  # Batch size of 4, 3 channels, 32x32 images
    output = model(x)
    print(output.shape)  # Should be [4, 10] for CIFAR-10 classes
    