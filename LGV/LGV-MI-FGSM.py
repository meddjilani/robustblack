from comet_ml import Experiment
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
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT_RQ1

import torchvision.models as models
from utils_robustblack import DataLoader, set_random_seed
from utils_robustblack.Normalize import Normalize


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
    parser.add_argument('--eps', type = float, default=8/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lgv_epochs', type=int, default=5)
    parser.add_argument('--lgv_nb_models_epoch', type=int, default=2)
    parser.add_argument("--lgv_lr", type=float, default=0.05)
    parser.add_argument('--lgv_batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default= '../dataset/Imagenet/Sample_1000')
    parser.add_argument('--train_path', type=str, default= '../dataset/Imagenet/Sample_49000')
    parser.add_argument('--lgv_models', type=str, default= './lgv_models')
    parser.add_argument("--gpu", type=str, default='cuda:0', help="GPU ID: 0,1")
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    set_random_seed(args.seed)

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT_RQ1,
        workspace=COMET_WORKSPACE,
    )

    parameters = {'attack': 'LGV', **vars(args)}
    experiment.log_parameters(parameters)
    experiment.set_name("LGV_"+args.model+"_"+args.target)

    device = torch.device(args.gpu)

    train_loader, loader, nlabels, mean, std = DataLoader.imagenet_train_test({'train_path': args.train_path,
                                                      'data_path': args.data_path,
                                                      'train_batch_size': args.lgv_batch_size,
                                                      'test_batch_size': args.batch_size,
                                                      'gpu': args.gpu,
                                                      })

    target_model = load_model(args.target, dataset = 'imagenet', threat_model = 'Linf')
    target_model.to(device)

    suc_rate_steps = 0
    images_steps = 0

    loaded_models = []
    source_model = load_model_torchvision(args.model, device, mean, std)
    attack = torchattacks.LGV(source_model, train_loader, lr=args.lgv_lr, epochs=args.lgv_epochs,
                              nb_models_epoch=args.lgv_nb_models_epoch, wd=1e-4, n_grad=1,
                              attack_class=torchattacks.attacks.mifgsm.MIFGSM, eps=args.eps, alpha=args.alpha,
                              steps=args.steps, decay=args.decay, verbose=True)
    for filename in os.listdir(args.lgv_models):
        source_model.load_state_dict(torch.load(os.path.join(args.lgv_models, filename))["state_dict"])
        source_model.eval()
        loaded_models.append(source_model)

    attack.load_models(loaded_models)

    for batch_ndx, (x_test, y_test) in enumerate(loader):

        x_test, y_test = x_test.to(device), y_test.to(device)

        print('Running LGV attack on batch ', batch_ndx)

        adv_images_LGV = attack(x_test, y_test)


        acc = clean_accuracy(target_model, x_test, y_test)
        rob_acc = clean_accuracy(target_model, adv_images_LGV, y_test)
        print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
        print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))

        with torch.no_grad():
            predictions = target_model(x_test)
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions = (predicted_classes == y_test).sum().item()
            correct_batch_indices = (predicted_classes == y_test).nonzero().squeeze(-1)
        
        suc_rate = 1 - clean_accuracy(target_model, adv_images_LGV[correct_batch_indices,:,:,:], y_test[correct_batch_indices])
        print(args.target, 'Success Rate: %2.2f %%'%(suc_rate*100))
        if correct_batch_indices.size(0) != 0:
            suc_rate_steps = suc_rate_steps*images_steps + suc_rate*correct_batch_indices.size(0)
            images_steps += correct_batch_indices.size(0)
            suc_rate_steps = suc_rate_steps/images_steps
        metrics = {'suc_rate_steps':suc_rate_steps, 'clean_acc': acc, 'robust_acc': rob_acc, 'suc_rate': suc_rate, 'target_correct_pred': correct_predictions}
        experiment.log_metrics(metrics, step=batch_ndx+1)
