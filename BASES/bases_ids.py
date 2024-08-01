from comet_ml import Experiment

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT_RQ1, COMET_PROJECT_RQ2, COMET_PROJECT_RQ3

from class_names_imagenet import lab_dict as imagenet_names

from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from robustbench.utils import load_model


def read_ids_from_file(filename):
    with open(filename, 'r') as file:
        # Read the entire file content
        content = file.read()

        # Split the content by commas and strip any whitespace or newline characters
        ids = [int(id.strip()) for id in content.split(',') if id!='']

    return ids


if __name__ == "__main__":
    device = torch.device('cuda')
    comet_rq_proj = {'RQ1':COMET_PROJECT_RQ1, 'RQ2':COMET_PROJECT_RQ2,'RQ3':COMET_PROJECT_RQ3}
    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=comet_rq_proj['RQ3'],
        workspace=COMET_WORKSPACE,
    )
    parameters = {'attack': 'bases_ids'}
    experiment.log_parameters(parameters)
    experiment.set_name("bases_ids")

    filename = '/raid/data/mdjilani/bases_ids.txt'  # replace with your filename

    # Read and transform the IDs
    ids_list = read_ids_from_file(filename)

    # Print the resulting list of IDs
    gt_names = []
    predicted_names = []
    base_path = '/raid/data/mdjilani/bases_adv_rob'
    folder_names = ['untargeted_victim_Bai2024MixedNUTS_2wb_linf_4_iters10_x3_loss_cw_lr0.005_iterw20_fuse_loss_v2', 'untargeted_victim_Liu2023Comprehensive_ConvNeXt-L_2wb_linf_4_iters10_x3_loss_cw_lr0.005_iterw20_fuse_loss_v2', 'untargeted_victim_Liu2023Comprehensive_Swin-L_2wb_linf_4_iters10_x3_loss_cw_lr0.005_iterw20_fuse_loss_v2']  # Replace with your actual folder names
    models = ['Bai2024MixedNUTS', 'Liu2023Comprehensive_ConvNeXt-L' , 'Liu2023Comprehensive_Swin-L']

    for j,folder_name in enumerate(folder_names):
        folder_path = os.path.join(base_path, folder_name)
        for image_filename in os.listdir(folder_path):
            image_path_adv = os.path.join(folder_path, image_filename)
            if image_filename[-4:] == '.png':
                sep = image_filename.split(' ')
                if int(sep[0]) in ids_list:
                    print(sep[1][:-4])
                    print(int(sep[0]))
                    gt_names.append(sep[1][:-4])

                    img_adv = Image.open(image_path_adv)

                    image_array_adv = np.array(img_adv)
                    image_tensor_adv = torch.from_numpy(image_array_adv).permute(2, 0, 1).unsqueeze(0).float().to(
                        device)
                    image_tensor_adv = image_tensor_adv / 255


                    victim_model = load_model(models[j], dataset='imagenet', threat_model='Linf')
                    victim_model.to(device)

                    pred = victim_model(image_tensor_adv).argmax(dim=1).item()

                    predicted_name = imagenet_names[int(pred)].split(',')[0]
                    predicted_names.append(predicted_name)



        cpt = 0
        len_list  = len(predicted_names)
        for i in range (len_list):
            if predicted_names[i] != gt_names[i]:
                cpt +=1

        suc_rate = cpt / len_list

        metrics = {'suc_rate':suc_rate, 'folder': folder_name}
        experiment.log_metrics(metrics, step=j)

