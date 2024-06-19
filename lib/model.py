import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*32, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*32)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

vgg11_CIFAR10 = torchvision.models.vgg11(weights=None)
vgg11_CIFAR10.classifier.add_module("7", nn.ReLU(inplace=True))
vgg11_CIFAR10.classifier.add_module("8", nn.Dropout(p=0.5,inplace=False))
vgg11_CIFAR10.classifier.add_module("9", nn.Linear(1000, 10))
vgg11_CIFAR100 = torchvision.models.vgg11(weights=None)
vgg11_CIFAR100.classifier.add_module("7", nn.ReLU(inplace=True))
vgg11_CIFAR100.classifier.add_module("8", nn.Dropout(p=0.5,inplace=False))
vgg11_CIFAR100.classifier.add_module("9", nn.Linear(1000, 100))