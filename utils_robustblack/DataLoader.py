import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np

class ImageNetSubset(torch.utils.data.Dataset):
    def __init__(self, img_list_file, class_label_map_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_list = []
        with open(img_list_file, 'r') as f:
            for line in f:
                img_path = line.strip()  # Extract image path
                self.image_list.append(img_path)
        with open(class_label_map_file, 'r') as f:
            self.class_label_map = json.load(f)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        idx_image = self.image_list[idx]
        img_path = os.path.join(self.img_dir, idx_image)
        image = Image.open(img_path).convert('RGB')
        label = self.class_label_map.get(idx_image.split('/')[0])
        if self.transform:
            image = self.transform(image)
        return image, label


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


def imagenet_robustbench(state):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    dataset = ImageNetSubset(img_list_file=state['helpers_path']+'/imagenet_test_image_ids.txt',
                             class_label_map_file=state['helpers_path']+'/imagenet_class_to_id_map.json',
                             img_dir=state['data_path'],
                             transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=state['batch_size'], shuffle=False, pin_memory=True
    )

    nlabels = 1000

    return test_loader, nlabels, mean, std

def imagenet_robustbench_bases(state):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.PILToTensor() #transform PIL to tensor without scaling to 0-1
         ])

    dataset = ImageNetSubset(img_list_file=state['helpers_path']+'/imagenet_test_image_ids.txt',
                             class_label_map_file=state['helpers_path']+'/imagenet_class_to_id_map.json',
                             img_dir=state['data_path'],
                             transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=state['batch_size'], shuffle=False, pin_memory=True
    )

    nlabels = 1000

    return test_loader, nlabels, mean, std
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