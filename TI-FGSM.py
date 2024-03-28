from comet_ml import Experiment
import torch
import torch.nn as nn
from robustbench.utils import load_model, clean_accuracy
import torchattacks
import json
import argparse

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT

import torchvision.models as models
from utils_robustblack import set_random_seed
from utils_robustblack import DataLoader



def load_model_torchvision(model_name, device):
    model = getattr(models, model_name)(pretrained=True).to(device).eval()
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--target', type=str, default= 'Standard_R50')
    parser.add_argument('--eps', type = float, default=8/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)
    parser.add_argument('--resize_rate', type=float, default=0.9)
    parser.add_argument('--diversity_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--kernel_name", type=str, default='gaussian', help='kernel name')
    parser.add_argument('--len_kernel', type=int, default=15, help='kernel length')
    parser.add_argument('--nsig', type=int, default=3, help='radius of gaussian kernel')
    parser.add_argument("--gpu", type=str, default='cuda:0', help="GPU ID: 0,1")
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    set_random_seed(args.seed)

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT,
        workspace=COMET_WORKSPACE,
    )

    parameters = {'attack': 'TI-FGSM', **vars(args)}
    experiment.log_parameters(parameters)
    experiment.set_name("TI-FGSM_"+args.model+"_"+args.target)

    device = torch.device(args.gpu)

    source_model = load_model_torchvision(args.model, device)
    target_model = load_model(args.target, dataset = 'imagenet', threat_model = 'Linf')
    target_model.to(device)

    loader, nlabels, mean, std = DataLoader.imagenet({'train_path': '', 'data_path':'dataset/Imagenet/Sample_1000', 'batch_size':args.batch_size})

    suc_rate_steps = 0
    images_steps = 0
    for batch_ndx, (x_test, y_test) in enumerate(loader):

        x_test, y_test = x_test.to(device), y_test.to(device)

        print('Running TI-FGSM attack on batch ', batch_ndx)
        attack = torchattacks.TIFGSM(source_model, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay,
                                     resize_rate=args.resize_rate, diversity_prob=args.diversity_prob,
                                     kernel_name=args.kernel_name, len_kernel=args.len_kernel, nsig=args.nsig)
        adv_images_TI = attack(x_test, y_test)


        acc = clean_accuracy(target_model, x_test, y_test)
        rob_acc = clean_accuracy(target_model, adv_images_TI, y_test)
        print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
        print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))

        with torch.no_grad():
            predictions = target_model(x_test)
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions = (predicted_classes == y_test).sum().item()
            correct_batch_indices = (predicted_classes == y_test).nonzero().squeeze(-1)
        
        suc_rate = 1 - clean_accuracy(target_model, adv_images_TI[correct_batch_indices,:,:,:], y_test[correct_batch_indices])
        print(args.target, 'Success Rate: %2.2f %%'%(suc_rate*100))
        if correct_batch_indices.size(0) != 0:
            suc_rate_steps = suc_rate_steps*images_steps + suc_rate*correct_batch_indices.size(0)
            images_steps += correct_batch_indices.size(0)
            suc_rate_steps = suc_rate_steps/images_steps
        metrics = {'suc_rate_steps':suc_rate_steps, 'clean_acc': acc, 'robust_acc': rob_acc, 'suc_rate': suc_rate, 'target_correct_pred': correct_predictions}
        experiment.log_metrics(metrics, step=batch_ndx+1)