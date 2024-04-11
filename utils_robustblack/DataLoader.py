import torch
import torch.nn as nn
import torchvision.datasets as dset
import torch.utils.data
import torchvision.transforms as transforms
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ImageFolderWithGPU(dset.ImageFolder):
    def __init__(self, gpu, root, transform=None, target_transform=None):
        super(ImageFolderWithGPU, self).__init__(root, transform=transform, target_transform=target_transform)
        self.gpu = gpu  # Add the state attribute

    def __getitem__(self, index):
        sample, target = super(ImageFolderWithGPU, self).__getitem__(index)

        device = torch.device(self.gpu)
        sample = sample.to(device)
        target = torch.tensor(target).to(device)

        return sample, target


def imagenet(state):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    test_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(state['data_path'], transform=transform),
        batch_size=state['batch_size'], shuffle=False, pin_memory=True)

    nlabels = 1000

    return test_loader, nlabels, mean, std


def imagenet_train_test(state):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(state['train_path'], transform=transform),
        batch_size=state['train_batch_size'], shuffle=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(state['data_path'], transform=transform),
        batch_size=state['test_batch_size'], shuffle=False, pin_memory=True)

    nlabels = 1000

    return train_loader, test_loader, nlabels, mean, std


def imagenet_gpu(state):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        ImageFolderWithGPU(state['gpu'], state['train_path'], transform=transform),
        batch_size=state['train_batch_size'], shuffle=True, pin_memory=False)

    nlabels = 1000

    return train_loader, nlabels, mean, std