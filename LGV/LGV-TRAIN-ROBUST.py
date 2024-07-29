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
from utils_robustblack.lgv_robust import LGV

from robustbench.utils import load_model

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

    parser.add_argument('--model', type=str, default='Liu2023Comprehensive_Swin-B')
    parser.add_argument('--eps', type = float, default=4/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)
    parser.add_argument('--lgv_epochs', type=int, default=5)
    parser.add_argument('--lgv_nb_models_epoch', type=int, default=2)
    parser.add_argument("--lgv_lr", type=float, default=0.05)
    parser.add_argument('--lgv_batch_size', type=int, default=64)
    parser.add_argument('--train_path', type=str, default= '/raid/data/mdjilani/dataset/Imagenet/Sample_49000')
    parser.add_argument('--save_models', type=str, default= '/raid/data/mdjilani/')
    parser.add_argument("--gpu", type=str, default='cuda:0', help="GPU ID: 0,1")
    parser.add_argument("--train_mode", type=str, default='know_dist', help="Custom lgv training mode on robust surrogate")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("-robust", action='store_true', help="use robust models")


    args = parser.parse_args()
    set_random_seed(args.seed)

    device = torch.device(args.gpu)

    train_loader, nlabels, mean, std = DataLoader.imagenet_gpu({ 'train_path': args.train_path,
                                                      'train_batch_size': args.lgv_batch_size,
                                                      'gpu': args.gpu,
                                                      })
    if args.robust:
        source_model = load_model(args.model, dataset = 'imagenet', threat_model = 'Linf').to(device)
        path_save_models = args.save_models + 'lgv_robust_models_robust' + args.model + "_" + args.train_mode
        
    else:
        source_model = load_model_torchvision(args.model, device, mean, std)
        path_save_models = args.save_models + 'lgv_robust_models' + args.model
        
    attack = LGV(source_model, train_loader, lr=args.lgv_lr, epochs=args.lgv_epochs,
                              nb_models_epoch=args.lgv_nb_models_epoch, wd=1e-4, n_grad=1,
                              train_mode=args.train_mode,
                              attack_class=torchattacks.attacks.mifgsm.MIFGSM, eps=args.eps, alpha=args.alpha,
                              steps=args.steps, decay=args.decay, verbose=True)
    attack.collect_models()
    attack.save_models(path_save_models)
