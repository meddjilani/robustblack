from comet_ml import Experiment
import torch
from autoattack import AutoAttack
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT_LGVvsGHOST
from robustbench.utils import load_model, clean_accuracy
import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from utils_robustblack import DataLoader, set_random_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--attack', default='LGV', help='LGV or GHOST')
    parser.add_argument('--lgv_models', type=str, default= '/raid/data/mdjilani/')
    parser.add_argument("--gpu", type=str, default='mps', help="GPU ID: 0,1")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_path', type=str, default= '/raid/data/mdjilani/dataset/val')
    parser.add_argument('--helpers_path', type=str, default= '/home/mdjilani/robustblack/utils_robustblack')

    args = parser.parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.gpu)

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT_LGVvsGHOST,
        workspace=COMET_WORKSPACE,
    )

    parameters = {'attack': 'LGVvsGHOST', **vars(args)}
    experiment.log_parameters(parameters)
    experiment.set_name("LGVvsGHOST_"+'_'+'Peng2023Robust')

    loader, nlabels, mean, std = DataLoader.imagenet_robustbench({'helpers_path': args.helpers_path,
                                                                  'data_path': args.data_path,
                                                                  'batch_size': args.batch_size}
                                                                 )

    target_model = load_model('Peng2023Robust', dataset = 'imagenet', threat_model = 'Linf').to(device)

    list_targets = []
    if args.attack == 'LGV':
        for i, filename in enumerate(os.listdir(args.lgv_models + str(args.seed) + 'lgv_models_robustPeng2023Robust')):
            if filename[-3:] == ".pt":
                target_model.load_state_dict(torch.load(
                    os.path.join(args.lgv_models + str(args.seed) + 'lgv_models_robustPeng2023Robust', filename),
                    map_location=device)["state_dict"])
                target_model.eval()
                list_targets.append(target_model)
    # elif args.attack == 'GHOST':


    else:
        raise ValueError('Current comparison only supports LGV and GHOST.')


    for num_target in range(len(list_targets)):

        total_acc_rate = 0
        total_suc_rate = 0
        total_images = 0
        total_correct_predictions = 0
        for x_test,y_test in loader:

            x_test, y_test = x_test.to(device), y_test.to(device)

            target_model = list_targets[num_target]

            acc = clean_accuracy(target_model, x_test, y_test)
            print('=======',y_test.size(0))
            batch_images = y_test.size(0)
            total_acc = total_acc_rate * total_images + acc * batch_images
            total_images += batch_images
            total_acc_rate = total_acc / total_images

            adversary = AutoAttack(target_model, norm='Linf', eps=4 / 255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
            adversary.device = device
            adversary.apgd.device = device
            adversary.apgd.n_restarts = 1
            x_adv = adversary.run_standard_evaluation(x_test, y_test)

            with torch.no_grad():
                predictions = target_model(x_test)
                predicted_classes = torch.argmax(predictions, dim=1)
                correct_predictions = (predicted_classes == y_test).sum().item()
                correct_batch_indices = (predicted_classes == y_test).nonzero().squeeze(-1)

            if acc!=0:
                suc_rate = 1 - clean_accuracy(target_model, x_adv[correct_batch_indices,:,:,:], y_test[correct_batch_indices])
            else:
                suc_rate = 0

            #additinal check: whatever the value suc_rate takes when batch have no single correctly classified image, the total success rate should be correct
            if correct_predictions != 0:
                total_suc = total_suc_rate * total_correct_predictions + suc_rate * correct_predictions
                total_correct_predictions += correct_predictions
                total_suc_rate = total_suc / total_correct_predictions

        print('seed ',args.seed, ', target: ', num_target, " Clean accuracy: %2.2f %%" % (total_acc_rate * 100))
        print('seed ',args.seed, ', target: ', num_target, " Success rate : %2.2f %% \n" % (total_suc_rate * 100))
        metrics = {'suc_rate': total_suc_rate, 'clean_acc': total_acc_rate, 'target':target}
        experiment.log_metrics(metrics, step=num_target + 1)