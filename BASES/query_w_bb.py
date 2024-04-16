"""
This code is used for NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"

Attack blackbox victim model via querying weight space of ensemble models. 

Blackbox setting
"""

from comet_ml import Experiment
import argparse
from pathlib import Path

import torch
from robustbench.utils import load_model
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from class_names_imagenet import lab_dict as imagenet_names
from utils_bases import load_imagenet_1000, load_model_torchvision, get_adv_np, get_label_loss

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT_RQ1, COMET_PROJECT_RQ2, COMET_PROJECT_RQ3
from utils_robustblack import DataLoader, set_random_seed


def main():
    parser = argparse.ArgumentParser(description="BASES attack")
    parser.add_argument("--victim", nargs="?", default='Standard_R50', help="victim model")
    parser.add_argument("--n_wb", type=int, default=10, help="number of models in the ensemble: 4,10,20")
    parser.add_argument("--bound", default='linf', choices=['linf','l2'], help="bound in linf or l2 norm ball")
    parser.add_argument("--eps", type=int, default=8, help="perturbation bound: 10 for linf, 3128 for l2")
    parser.add_argument("--iters", type=int, default=10, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")

    parser.add_argument("--fuse", nargs="?", default='loss', help="the fuse method. loss or logit")
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate of w")
    parser.add_argument("--iterw", type=int, default=50, help="iterations of updating w")
    parser.add_argument("--n_im", type=int, default=1000, help="number of images")
    parser.add_argument("-untargeted", action='store_true', help="run untargeted attack")
    parser.add_argument('--data_path', type=str, default= '../dataset/Imagenet/Sample_1000')
    parser.add_argument('--exp_root', type=str, default= "bb_w_logs/")
    parser.add_argument('--adv_root', type=str, default= "bb_w_adv_images/")
    parser.add_argument("--gpu", type=str, default='cuda:0', help="GPU ID: 0,1")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--comet_proj', default='RQ1', type=str)

    args = parser.parse_args()
    set_random_seed(args.seed)

    comet_rq_proj = {'RQ1':COMET_PROJECT_RQ1, 'RQ2':COMET_PROJECT_RQ2,'RQ3':COMET_PROJECT_RQ3}
    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=comet_rq_proj[args.comet_proj],
        workspace=COMET_WORKSPACE,
    )

    n_wb = args.n_wb
    bound = args.bound
    eps = args.eps
    n_iters = args.iters
    alpha = args.x * eps / n_iters # step-size
    fuse = args.fuse
    loss_name = args.loss_name
    lr_w = float(args.lr)
    device = torch.device(args.gpu)

    # load surrogate models
    surrogate_names = ['vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
                'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']
    parameters = {'attack': 'BASES', **vars(args)}
    experiment.log_parameters(parameters)
    experiment.set_name("BASES_"+'_'.join(surrogate_names[:n_wb])+"_"+args.victim)

    wb = []
    for model_name in surrogate_names[:n_wb]:
        print(f"load: {model_name}")
        wb.append(load_model_torchvision(model_name, device))

    # load victim model
    victim_model = load_model(args.victim, dataset = 'imagenet', threat_model = 'Linf')
    victim_model.to(device)


    # create exp folders
    exp = f'victim_{args.victim}_{n_wb}wb_{bound}_{eps}_iters{n_iters}_x{args.x}_loss_{loss_name}_lr{lr_w}_iterw{args.iterw}_fuse_{fuse}_v2'
    if args.untargeted:
        exp = 'untargeted_' + exp

    exp_root = Path(args.exp_root) / exp
    exp_root.mkdir(parents=True, exist_ok=True)
    adv_root = Path(args.adv_root) / exp
    adv_root.mkdir(parents=True, exist_ok=True)

    loader, nlabels, mean, std = DataLoader.imagenet({'train_path': '', 'data_path':args.data_path, 'batch_size':1})

    success_idx_list = set()
    success_idx_list_pretend = set() # untargeted success
    query_list = []
    query_list_pretend = []
    queries = []
    count_correctly_classified_images = 0
    suc_rate_steps = 0
    for im_idx, (x_test, y_test) in enumerate(loader):
        x_test_sq_per = torch.squeeze(x_test, dim=0).permute(1,2,0)
        im_np = np.array(x_test_sq_per)
        lr_w = float(args.lr) # re-initialize
        gt_label = y_test.item()
        gt_label_name = imagenet_names[gt_label].split(',')[0]
        tgt_label = (y_test.item()+1)%1000
        exp_name = f"idx{im_idx}_f{gt_label}_t{tgt_label}"
        if args.untargeted:
            tgt_label = gt_label
            exp_name = f"idx{im_idx}_f{gt_label}_untargeted"

        pred_label = torch.argmax(victim_model(x_test.to(device)), dim=1).item()
        if args.untargeted and gt_label != pred_label:
            continue
            print('Image is already misclassified by victim model in UNTARGETED ATTACK')
        else:
            count_correctly_classified_images +=1

        # start from equal weights
        w_np = np.array([1 for _ in range(len(wb))]) / len(wb)
        adv_np, losses = get_adv_np(im_np, tgt_label, w_np, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=None)
        label_idx, loss, _ = get_label_loss(adv_np/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
        n_query = 1
        w_list = []
        loss_wb_list = losses   # loss of optimizing wb models
        loss_bb_list = []       # loss of victim model
        print(f"{label_idx, imagenet_names[label_idx]}, loss: {loss}")
        print(f"w: {w_np.tolist()}")
        print(label_idx != tgt_label, label_idx,tgt_label)

        if args.untargeted:
            if label_idx != tgt_label:
                suc_rate = 1
                suc_rate_steps += suc_rate
                queries.append(n_query)
            else:
                suc_rate = 0
            metrics = {'suc_rate_steps': suc_rate_steps/count_correctly_classified_images, 'suc_rate': suc_rate,
                   'queries_steps': np.mean(queries)}
        else:
            if label_idx == tgt_label:
                suc_rate = 1
                suc_rate_steps += suc_rate
                queries.append(n_query)
            else:
                suc_rate = 0
            metrics = {'suc_rate_steps': suc_rate_steps/im_idx+1, 'suc_rate': suc_rate,
                   'queries_steps': np.mean(queries)}

        # pretend
        if not args.untargeted and label_idx != gt_label:
            success_idx_list_pretend.add(im_idx)
            query_list_pretend.append(n_query)

        if (not args.untargeted and label_idx == tgt_label) or (args.untargeted and label_idx != tgt_label):
            # originally successful
            print('success')
            success_idx_list.add(im_idx)
            query_list.append(n_query)
            w_list.append(w_np.tolist())
            loss_bb_list.append(loss)
        else:
            idx_w = 0         # idx of wb in W
            last_idx = 0    # if no changes after one round, reduce the learning rate

            while n_query < args.iterw:
                w_np_temp_plus = w_np.copy()
                w_np_temp_plus[idx_w] += lr_w
                adv_np_plus, losses_plus = get_adv_np(im_np, tgt_label, w_np_temp_plus, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=adv_np)
                label_plus, loss_plus, _ = get_label_loss(adv_np_plus/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
                n_query += 1
                print(f"iter: {n_query}, {idx_w} +, {label_plus, imagenet_names[label_plus]}, loss: {loss_plus}")

                if args.untargeted:
                    if label_plus != tgt_label:
                        suc_rate = 1
                        suc_rate_steps += suc_rate
                        queries.append(n_query)
                    else:
                        suc_rate = 0
                    metrics = {'suc_rate_steps': suc_rate_steps / count_correctly_classified_images,
                               'suc_rate': suc_rate,
                               'queries_steps': np.mean(queries)}
                else:
                    if label_plus == tgt_label:
                        suc_rate = 1
                        suc_rate_steps += suc_rate
                        queries.append(n_query)
                    else:
                        suc_rate = 0
                    metrics = {'suc_rate_steps': suc_rate_steps / im_idx + 1, 'suc_rate': suc_rate,
                               'queries_steps': np.mean(queries)}

                # pretend
                if (not args.untargeted and label_plus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_plus) or args.untargeted*(tgt_label != label_plus):
                    print('success')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    loss_wb_list += losses_plus
                    break

                w_np_temp_minus = w_np.copy()
                w_np_temp_minus[idx_w] -= lr_w
                adv_np_minus, losses_minus = get_adv_np(im_np, tgt_label, w_np_temp_minus, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=adv_np)
                label_minus, loss_minus, _ = get_label_loss(adv_np_minus/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
                n_query += 1
                print(f"iter: {n_query}, {idx_w} -, {label_minus, imagenet_names[label_minus]}, loss: {loss_minus}")

                if args.untargeted:
                    if label_minus != tgt_label:
                        suc_rate = 1
                        suc_rate_steps += suc_rate
                        queries.append(n_query)
                    else:
                        suc_rate = 0
                    metrics = {'suc_rate_steps': suc_rate_steps / count_correctly_classified_images,
                               'suc_rate': suc_rate,
                               'queries_steps': np.mean(queries)}
                else:
                    if label_minus == tgt_label:
                        suc_rate = 1
                        suc_rate_steps += suc_rate
                        queries.append(n_query)
                    else:
                        suc_rate = 0
                    metrics = {'suc_rate_steps': suc_rate_steps / im_idx + 1, 'suc_rate': suc_rate,
                               'queries_steps': np.mean(queries)}

                # pretend
                if (not args.untargeted and label_minus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_minus) or args.untargeted*(tgt_label != label_minus):
                    print('success')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    loss_wb_list += losses_minus
                    break

                # update
                if loss_plus < loss_minus:
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    loss_wb_list += losses_plus
                    print(f"{idx_w} +")
                    last_idx = idx_w
                else:
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    loss_wb_list += losses_minus
                    print(f"{idx_w} -")
                    last_idx = idx_w


                idx_w = (idx_w+1)%n_wb
                if n_query > 5 and last_idx == idx_w:
                    lr_w /= 2 # decrease the lr
                    print(f"lr_w: {lr_w}")

                w_list.append(w_np.tolist())
                loss_bb_list.append(loss)


        print(f"im_idx: {im_idx}")
        if im_idx in success_idx_list:
            # save to txt
            info = f"im_idx: {im_idx}, iters: {query_list[-1]}, loss: {loss:.2f}, w: {w_np.squeeze().tolist()}\n"
            file = open(exp_root / f'{exp}.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"targeted. total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")

        if im_idx in success_idx_list_pretend:
            # save to txt
            info = f"im_idx: {im_idx}, iters: {query_list_pretend[-1]}, loss: {loss:.2f}, w: {w_np.squeeze().tolist()}\n"
            file = open(exp_root / f'{exp}_pretend.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"untargeted. total_success: {len(success_idx_list_pretend)}; success_rate: {len(success_idx_list_pretend)/(im_idx+1)}, avg queries: {np.mean(query_list_pretend)}")


        # save adv image
        adv_path = adv_root / f"{im_idx} {gt_label_name}.png"
        adv_png = Image.fromarray(adv_np.astype(np.uint8))
        adv_png.save(adv_path)

        # plot figs
        fig, ax = plt.subplots(1,5,figsize=(30,5))
        ax[0].plot(loss_wb_list)
        ax[0].set_xlabel('iters')
        ax[0].set_title('loss on surrogate ensemble')
        ax[1].imshow(im_np)
        ax[1].set_title(f"clean image:\n{gt_label_name}")
        victim_label_idx, loss, _ = get_label_loss(adv_np/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
        adv_label_name = imagenet_names[victim_label_idx].split(',')[0]
        ax[2].imshow(adv_np/255)
        ax[2].set_title(f"adv image:\n{adv_label_name}")
        ax[3].plot(loss_bb_list)
        ax[3].set_title('loss on victim model')
        ax[3].set_xlabel('iters')
        ax[4].plot(w_list)
        ax[4].legend(surrogate_names[:n_wb], shadow=True, bbox_to_anchor=(1, 1))
        ax[4].set_title('w of surrogate models')
        ax[4].set_xlabel('iters')
        plt.tight_layout()
        if im_idx in success_idx_list:
            plt.savefig(exp_root / f"{exp_name}_success.png")
        else:
            plt.savefig(exp_root / f"{exp_name}.png")
        plt.close()

        print(f"query_list: {query_list}")
        print(f"avg queries: {np.mean(query_list)}")

        if args.untargeted:
            experiment.log_metrics(metrics, step=count_correctly_classified_images)
        else:
            experiment.log_metrics(metrics, step=im_idx + 1)


if __name__ == '__main__':
    main()