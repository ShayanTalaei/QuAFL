import torch
import torchvision
from torchvision.models import resnet18
from torch import nn
import torch.nn.functional as F
from resnet_cifar10 import *


CRITERIONS  = {'mnist': F.cross_entropy,
               'fashion mnist': nn.CrossEntropyLoss(),
               'cifar 10': nn.CrossEntropyLoss(),
               'celeba': nn.CrossEntropyLoss()}

OPTIMIZERS = {'mnist': torch.optim.SGD,
              'fashion mnist': torch.optim.Adam,
              'cifar 10': torch.optim.SGD,
              'celeba': torch.optim.SGD}

def get_criterion(dataset_name):
    return CRITERIONS[dataset_name]

def get_optimizer(dataset_name): 
    return OPTIMIZERS[dataset_name]

def get_model(dataset_name):
    if dataset_name == "mnist":
        return MnistNet()
    elif dataset_name == "fashion mnist":
        return FashionMnistNet()
    elif dataset_name == "cifar 10":
        return resnet20_cifar()
    elif dataset_name == 'celeba':
        model = resnet18()
        model.load_state_dict(torch.load('resnet18-5c106cde.pth'))
        new_fc = nn.Linear(512, 2, True)
        model.fc = new_fc
        return  model

class MnistNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=32, num_classes=10):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class FashionMnistNet(nn.Module):
    
    def __init__(self):
        super(FashionMnistNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    
class CelebaNet(nn.Module):
    
        def __init__(self):
            super(CelebaNet, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.fc1 = nn.Linear(in_features=193600, out_features=10000)
            self.drop = nn.Dropout2d(0.25)
            self.fc2 = nn.Linear(in_features=10000, out_features=500)
            self.fc3 = nn.Linear(in_features=500, out_features=1)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            print(out.shape)
            out = self.fc1(out)
            out = self.drop(out)
            out = self.fc2(out)
            out = self.fc3(out)

            return out
#     def __init__(self):
#         super(CelebaNet, self).__init__()
        
#         self.layer_1 = self.make_block()
#         self.layer_2 = self.make_block()
#         self.layer_3 = self.make_block()
#         self.layer_4 = self.make_block()
#         self.fc = nn.Linear(in_features=1, out_features=2)
    
#     def make_block(self):
#         return nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0),
#                              nn.BatchNorm2d(32),
#                              nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#                              nn.ReLU())
    
#     def forward(self, x):

#         x = self.layer_1(x)
#         x = self.layer_2(x)
#         x = self.layer_3(x)
#         x = self.layer_4(x)
#         out = self.fc(x)
#         return out