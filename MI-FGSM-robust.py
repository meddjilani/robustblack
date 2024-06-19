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
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT_RQ3

import torchvision.models as models
from utils_robustblack import set_random_seed
from utils_robustblack import DataLoader
from utils_robustblack.Normalize import Normalize
import numpy as np
from PIL import Image





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Liu2023Comprehensive_Swin-B')
    parser.add_argument('--target', type=str, default= 'Liu2023Comprehensive_Swin-L')
    parser.add_argument('--eps', type = float, default=4/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--gpu", type=str, default='cuda', help="GPU ID: 0,1")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_path', type=str, default= '/raid/data/mdjilani/dataset/val')
    parser.add_argument('--helpers_path', type=str, default= '/home/mdjilani/robustblack/utils_robustblack')

    args = parser.parse_args()
    set_random_seed(args.seed)

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT_RQ3,
        workspace=COMET_WORKSPACE,
    )

    parameters = {'attack': 'MI-FGSM', **vars(args)}
    experiment.log_parameters(parameters)
    experiment.set_name("MI-FGSM_"+args.model+"_"+args.target)

    device = torch.device(args.gpu)

    loader, nlabels, mean, std = DataLoader.imagenet_robustbench({'helpers_path': args.helpers_path,
                                                      'data_path': args.data_path,
                                                      'batch_size': args.batch_size}
                                                     )

    source_model = load_model(args.model, dataset = 'imagenet', threat_model = 'Linf')
    source_model.to(device)
    target_model = load_model(args.target, dataset = 'imagenet', threat_model = 'Linf')
    target_model.to(device)

    suc_rate_steps = 0
    images_steps = 0
    for batch_ndx, (x_test, y_test) in enumerate(loader):

        x_test, y_test = x_test.to(device), y_test.to(device)

        print('Running MI-FGSM attack on batch ', batch_ndx)
        attack = torchattacks.MIFGSM(source_model, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay)
        adv_images_MI = attack(x_test, y_test)


        acc = clean_accuracy(target_model, x_test, y_test)
        print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))

        with torch.no_grad():
            predictions = target_model(x_test)
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions = (predicted_classes == y_test).sum().item()
            correct_batch_indices = (predicted_classes == y_test).nonzero().squeeze(-1)
        
        suc_rate = 1 - clean_accuracy(target_model, adv_images_MI[correct_batch_indices,:,:,:], y_test[correct_batch_indices])
        rob_acc = acc*(1-suc_rate)
        print(args.target, 'Robust Acc: %2.2f %%'%(acc*(1-suc_rate)*100))
        print(args.target, 'Success Rate: %2.2f %%'%(suc_rate*100))
        if correct_batch_indices.size(0) != 0:
            suc_rate_steps = suc_rate_steps*images_steps + suc_rate*correct_batch_indices.size(0)
            images_steps += correct_batch_indices.size(0)
            suc_rate_steps = suc_rate_steps/images_steps
        metrics = {'suc_rate_steps':suc_rate_steps, 'clean_acc': acc, 'robust_acc': rob_acc, 'suc_rate': suc_rate, 'target_correct_pred': correct_predictions}
        experiment.log_metrics(metrics, step=batch_ndx+1)

        adversarial_folder = "/raid/data/mdjilani/mi_remote"
        os.makedirs(adversarial_folder, exist_ok=True)
        for im_idx, image_tensor in enumerate(adv_images_MI[correct_batch_indices, :, :, :]):
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_np = image_np * 255
            image_np = image_np.astype(np.uint8)
            gt_label = y_test[correct_batch_indices][im_idx]

            adv_path = os.path.join(adversarial_folder, f"{batch_ndx}_{im_idx}_{gt_label}.png")

            adv_png = Image.fromarray(image_np)
            adv_png.save(adv_path)