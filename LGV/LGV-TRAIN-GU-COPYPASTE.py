import torch
import torch.nn as nn
from robustbench.utils import load_model, clean_accuracy
import torchattacks
import argparse

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import torchvision.models as models
from utils_robustblack import DataLoader, set_random_seed
from utils_robustblack.Normalize import Normalize
from robustbench.utils import load_model
from torchvision import transforms
from torchvision import datasets
import numpy as np


import os
import glob
import random
import torch
import torchvision
import numpy as np
from tqdm.notebook import tqdm
from collections import OrderedDict
from torchvision.models import resnet50
from torchvision import transforms
from torchvision import datasets
from torchattacks import LGV, BIM, MIFGSM, DIFGSM, TIFGSM




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Liu2023Comprehensive_Swin-B')
    parser.add_argument('--eps', type = float, default=4/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)
    parser.add_argument('--lgv_epochs', type=int, default=5)
    parser.add_argument('--lgv_nb_models_epoch', type=int, default=2)
    parser.add_argument("--lgv_lr", type=float, default=0.05)
    parser.add_argument('--lgv_batch_size', type=int, default=256)
    parser.add_argument('--train_path', type=str, default= '/raid/data/mdjilani/dataset/Imagenet/Sample_49000')
    parser.add_argument('--save_models', type=str, default= '/raid/data/mdjilani/')
    parser.add_argument("--gpu", type=str, default='cuda:0', help="GPU ID: 0,1")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("-robust", action='store_true', help="use robust models")


    args = parser.parse_args()
    set_random_seed(args.seed)

    device = torch.device(args.gpu)

#********
    DATA_PATH = args.train_path
    BATCH_SIZE_TRAIN = args.lgv_batch_size  # changing batch-size to collect models might require you to tune the LGV learning rate hyperparameter
    BATCH_SIZE_TEST = 64
    N_WORKERS = 5


    def add_normalization_layer(model, mean, std):
        """
        Add a data normalization layer to a model
        """
        return torch.nn.Sequential(
            transforms.Normalize(mean=mean, std=std),
            model
        )

    if args.robust:
        base_model = load_model(args.model, dataset = 'imagenet', threat_model = 'Linf').to(device)

    else:
        base_model = resnet50(pretrained=True)
        base_model = add_normalization_layer(model=base_model,
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        base_model = base_model.eval().to(device)



    traindir = os.path.join(DATA_PATH,'')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    trainset = datasets.ImageFolder(traindir, transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE_TRAIN,
                                              shuffle=True, num_workers=N_WORKERS,
                                              pin_memory=True)


    atk = LGV(base_model, trainloader, lr=0.05, epochs=10, nb_models_epoch=4,
              wd=1e-4, attack_class=BIM, eps=4 / 255, alpha=4 / 255 / 10,
              steps=50, verbose=True)

    # uncomment the next 2 lines and comment the last one to collect models yourself (10 ImageNet epochs)
    path_save_models = args.save_models + "_FullTrain_"+ str(args.seed) + '_lgv_models_' + args.model + "_" + str(args.lgv_batch_size)

    atk.collect_models()
    atk.save_models(path_save_models)


#********

