from comet_ml import Experiment
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT_RQ2
import argparse
import torch
from gduap import gd_universal_adversarial_perturbation, get_fooling_rate, get_baseline_fooling_rate, load_model_torchvision
from utils_robustblack import set_random_seed
from utils_robustblack import DataLoader
from robustbench.utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50',
                        help='The network eg. vgg16')
    parser.add_argument('--target', type=str, default= 'Standard_R50',
                        help='target model')
    parser.add_argument('--eps', type=float, default=8/255,
                        help='maximum perturbation')
    parser.add_argument('--prior_type', default='no_data',
                        help='Which kind of prior to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use for training and testing')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--id',
                        help='An identification number (e.g. SLURM Job ID) that will prefix saved files')
    parser.add_argument('--baseline', action='store_true',
                        help='Obtain a fooling rate for a baseline random perturbation')
    parser.add_argument("--gpu", type=str, default='cuda:0',
                        help="GPU ID: 0,1")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_path', type=str, default= '../dataset/Imagenet/Sample_1000')
    parser.add_argument('--helpers_path', type=str, default= '/home/mdjilani/robustblack/utils_robustblack')

    args = parser.parse_args()
    set_random_seed(args.seed)

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT_RQ2,
        workspace=COMET_WORKSPACE,
    )

    parameters = {'attack': 'UAP', **vars(args)}
    experiment.log_parameters(parameters)
    experiment.set_name("UAP_"+args.model+"_"+args.target)

    device = torch.device(args.gpu)

    target_data_loader, _, mean, std = DataLoader.imagenet_robustbench({'helpers_path': args.helpers_path,
                                                      'data_path': args.data_path,
                                                      'batch_size': args.batch_size}
                                                     )

    model = load_model_torchvision(args.model, device, mean, std)

    if args.baseline:
        print("Obtaining baseline fooling rate...")
        baseline_fooling_rate = get_baseline_fooling_rate(model, target_data_loader, device, disable_tqdm=True)
        print(f"Baseline fooling rate for {args.model}: {baseline_fooling_rate}")
        return


    # create a universal adversarial perturbation
    uap = gd_universal_adversarial_perturbation(model, args.model, target_data_loader, args.prior_type, device, args.patience_interval, args.id, eps=args.eps,disable_tqdm=True)

    # perform a final evaluation
    target_model = load_model(args.target, dataset = 'imagenet', threat_model = 'Linf')
    target_model.to(device)
    target_fooling_rate,  target_suc_rate= get_fooling_rate(target_model, args.target, uap, target_data_loader, device, experiment, disable_tqdm=True)
    print(f"Target fooling rate on {args.target} : {target_fooling_rate}")
    print(f"Target success rate on {args.target} : {target_suc_rate}")


if __name__ == '__main__':
    main()
