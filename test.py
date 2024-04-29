import torch
import torch.nn as nn
from robustbench.utils import load_model, clean_accuracy

import argparse

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import torchvision.models as models
from utils_robustblack import set_random_seed
from utils_robustblack.Normalize import Normalize
from utils_robustblack import DataLoader


def load_model_torchvision(model_name, device, mean, std):
    model = getattr(models, model_name)(pretrained=True)
    model = nn.Sequential(
        Normalize(mean, std),
        model
    )
    model.to(device).eval()
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--target', type=str, default= 'Standard_R50')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--gpu", type=str, default='cuda:0', help="GPU ID: 0,1")
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    set_random_seed(args.seed)

    device = torch.device(args.gpu)

    target_model = load_model(args.target, dataset = 'imagenet', threat_model = 'Linf')
    target_model.to(device)

    loader, nlabels, mean, std = DataLoader.imagenet({'helpers_path': '',
                                                      'data_path': '~/Desktop/ImageNet_robustbench_VAL/MINE',
                                                      'batch_size': 1}
                                                     )


    print(len(loader))

