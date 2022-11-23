from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CelebA
from torchvision.transforms import ToTensor, InterpolationMode, transforms
from torch.utils.data import random_split
from torch.utils import data
from model_manager import *

import random
from PIL import Image
import numpy as np
import os

ROOT = "../data/"
BATCH_SIZES = {'mnist': 128,
               'fashion mnist': 400, 
               'cifar 10': 128,
               'celeba': 128} 

def get_batch_size(dataset_name):
    return BATCH_SIZES[dataset_name]

def get_datasets(dataset_name, **kwargs):
    print(f"Loading {dataset_name} dataset.")
    global ROOT
    if dataset_name == "mnist":
        dataset = MNIST(root= ROOT, download=True, transform=ToTensor())
        val_size = 10000
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    elif dataset_name == "cifar 10":
        train_ds, val_ds = cifar10_get_datasets(ROOT)
    elif dataset_name == 'fashion mnist':
        train_ds = FashionMNIST(ROOT + "Fashion MNIST/train_data", download=True, train=True,  transform=ToTensor())
        val_ds   = FashionMNIST(ROOT + "Fashion MNIST/test_data" , download=True, train=False, transform=ToTensor()) 
    elif dataset_name == 'celeba':
        return load_celeba_for_population(**kwargs)
    return [train_ds,], val_ds

def cifar10_get_datasets(data_dir, use_data_aug=True):
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_dataset = CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)
    return train_dataset, test_dataset

def celeba_get_datasets(data_dir, use_data_aug=True, label_indices=[31], **kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    # If we are only using some of the labels, remove all the ones we don't need.
    target_transform=None
    if label_indices:
        target_transform = lambda x: int(x[label_indices])#.float()

    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    
    train_dataset = CelebA(root=data_dir, split='train',
            target_type='attr',transform=train_transform,target_transform =target_transform,download=False)

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = CelebA(root=data_dir, split='test',
            target_type='attr', transform=test_transform, target_transform=target_transform, download=False)

    return train_dataset, test_dataset

def evaluate_on_dataloader(model, dataset_name, dataloader):
    model_initial_training = model.training
    model.eval()
    
    device = next(model.parameters()).device
    total_loss = 0
    count = 0
    corrects = 0
    data_count = 0
    criterion = get_criterion(dataset_name)
    result = {}
    
    with torch.no_grad():
        for data in dataloader:
            Xb, yb = data
            Xb, yb = Xb.to(device), yb.to(device)
            outputs = model(Xb)
#             print(outputs.shape, yb.shape)
            loss = criterion(outputs, yb)
            preds = torch.argmax(outputs, dim=1)
            corrects += float(torch.sum((preds == yb).float()))
            data_count += float(Xb.shape[0])
            total_loss += float(loss)
            count += 1
            del Xb, yb
    result["Loss"] = (total_loss / count)
    result["Accuracy"] = (corrects / data_count)
    
    model.train(model_initial_training)
    return result   

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert len(X) == len(y)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CelebaDataset(Dataset):
    
    def __init__(self, img_id_list, attr_list, train):
        self.img_id_list = img_id_list
        self.attr_list = attr_list
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
        if train:
            self.transform = transforms.Compose([transforms.RandomResizedCrop(100),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize])
        else:
            self.transform = transforms.Compose([
                                transforms.Resize(100),
                                transforms.CenterCrop(100),
                                transforms.ToTensor(),
                                normalize])
        assert len(img_id_list) == len(attr_list)

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_name = os.path.join("data", "celeba", "img_align_celeba", self.img_id_list[idx])
        image = Image.open(img_name)
        x = self.transform(image)
        y = self.attr_list[idx]
        return x, torch.tensor(y, dtype=torch.int64)

def load_celeba_for_population(count, train_percent=0.8, **kwargs):
    ## This function loads celeba dataset into "count" groups of disjoint celebrities randomly for train_set of clients
    ## and a validation_set for validating server model
    
    data = torch.load("data/celeba/celeba_min_10.torch") ## This file containes a dictionary for those celebrities with more than 10 images 
    user_num_sample = data['user_num_sample']
    random.shuffle(user_num_sample)
    all_data = data['user_data']
    train_dictionaries = {}
    for i in range(count):
        train_dictionaries[i] = {'img_id_list':[],
                                 'attr_list':[]}
    test_img_id_list, test_attr_list = [], []
    group_id = 0
    for pair in user_num_sample:
        user, num_sample = pair
        train_count = int(train_percent*num_sample)
        user_data = all_data[user]
        x, y = user_data['x'], user_data['y']
        train_group = train_dictionaries[group_id]
        train_group['img_id_list'] += x[:train_count]
        train_group['attr_list']   += y[:train_count]
        test_img_id_list += x[train_count:]
        test_attr_list   += y[train_count:]
        group_id += 1
        if group_id == count:
            group_id = 0
    
    train_sets_list = []
    for group_id, train_data in train_dictionaries.items():
        train_sets_list.append(CelebaDataset(img_id_list=train_data['img_id_list'],
                                             attr_list  =train_data['attr_list'],
                                             train=True))
    validation_set = CelebaDataset(test_img_id_list, test_attr_list, False)
    return train_sets_list, validation_set
    