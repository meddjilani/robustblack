from comet_ml import Experiment
import argparse
import torchvision.models as models
import json
from utils import *
from FCN import *
from Normalize import Normalize, Permute
from imagenet_model.Resnet import resnet152_denoise, resnet101_denoise
from robustbench.utils import load_model

import os
from PIL import Image
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT_RQ1, COMET_PROJECT_RQ2, COMET_PROJECT_RQ3
from utils_robustblack import set_random_seed, DataLoader



def EmbedBA(function, encoder, decoder, image, label, config, latent=None):
    device = image.device

    if latent is None:
        latent = encoder(image.unsqueeze(0)).squeeze().view(-1)
    momentum = torch.zeros_like(latent)
    dimension = len(latent)
    noise = torch.empty((dimension, config['sample_size']), device=device)
    origin_image = image.clone()
    last_loss = []
    lr = config['lr']
    for iter in range(config['num_iters']+1):

        perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0)*config['epsilon'], -config['epsilon'], config['epsilon'])
        logit, loss = function(torch.clamp(image+perturbation, 0, 1), label)
        if config['target']:
            success = torch.argmax(logit, dim=1) == label
        else:
            success = torch.argmax(logit, dim=1) !=label
        last_loss.append(loss.item())

        if function.current_counts > 50000:
            break
        
        if bool(success.item()):
            return True, torch.clamp(image+perturbation, 0, 1)

        nn.init.normal_(noise)
        noise[:, config['sample_size']//2:] = -noise[:, :config['sample_size']//2]
        latents = latent.repeat(config['sample_size'], 1) + noise.transpose(0, 1)*config['sigma']
        perturbations = torch.clamp(decoder(latents)*config['epsilon'], -config['epsilon'], config['epsilon'])
        _, losses = function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

        grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

        if iter % config['log_interval'] == 0 and config['print_log']:
            print("iteration: {} loss: {}, l2_deviation {}".format(iter, float(loss.item()), float(torch.norm(perturbation))))

        momentum = config['momentum'] * momentum + (1-config['momentum'])*grad

        latent = latent - lr * momentum

        last_loss = last_loss[-config['plateau_length']:]
        if (last_loss[-1] > last_loss[0]+config['plateau_overhead'] or last_loss[-1] > last_loss[0] and last_loss[-1]<0.6) and len(last_loss) == config['plateau_length']:
            if lr > config['lr_min']:
                lr = max(lr / config['lr_decay'], config['lr_min'])
            last_loss = []

    return False, origin_image


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/attack_untarget.json', help='config file')
parser.add_argument('--device', type=str, default='cuda:0', help="GPU ID: 0,1")
parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
parser.add_argument('--model_name', default='Wong2020Fast')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--comet_proj', default='RQ3', type=str)
parser.add_argument('--data_path', default='/raid/data/mdjilani/dataset/Sample_1000')
parser.add_argument('--generator_name', default='Imagenet_Wong2020Fast_Engstrom2019Robustness_Debenedetti2022Light_XCiT-M12_untarget')
parser.add_argument('--save_path', default='/raid/data/mdjilani/tremba_save_path')
parser.add_argument('--helpers_path', type=str, default='/home/mdjilani/robustblack/utils_robustblack')
parser.add_argument('--adversarial_folder', default="/raid/data/mdjilani/adversarials_tremba")



args = parser.parse_args()
set_random_seed(args.seed)

comet_rq_proj = {'RQ1': COMET_PROJECT_RQ1, 'RQ2': COMET_PROJECT_RQ2, 'RQ3': COMET_PROJECT_RQ3}
experiment = Experiment(
    api_key=COMET_APIKEY,
    project_name=comet_rq_proj[args.comet_proj],
    workspace=COMET_WORKSPACE,
)

with open(args.config) as config_file:
    state = json.load(config_file)

if args.save_prefix is not None:
    state['save_prefix'] = args.save_prefix
if args.model_name is not None:
    state['model_name'] = args.model_name
if args.data_path is not None:
    state['data_path'] = args.data_path
if args.generator_name is not None:
    state['generator_name'] = args.generator_name
if args.save_path is not None:
    state['save_path'] = args.save_path
state['helpers_path'] = args.helpers_path

new_state = state.copy()
new_state['batch_size'] = 1
new_state['test_bs'] = 1

parameters = {'attack': 'TREMBA', **vars(args), **new_state}
experiment.log_parameters(parameters)
experiment.set_name("TREMBA_" + new_state['generator_name'] + "_" + args.model_name)

device = torch.device(args.device)
weight = torch.load(os.path.join("G_weight", state['generator_name']+".pytorch"), map_location=device)

encoder_weight = {}
decoder_weight = {}
if args.generator_name == 'Imagenet_VGG16_Resnet18_Squeezenet_Googlenet_untarget':
    for key, val in weight.items():
        if key.startswith('0.'):
            encoder_weight[key[2:]] = val
        elif key.startswith('1.'):
            decoder_weight[key[2:]] = val
else:
    for key, val in weight['model_state_dict'].items():
        if key.startswith('0.'):
            encoder_weight[key[2:]] = val
        elif key.startswith('1.'):
            decoder_weight[key[2:]] = val

dataloader, nlabels, mean, std = DataLoader.imagenet_robustbench(new_state)

if 'OSP' in state:
    if state['source_model_name'] == 'Adv_Denoise_Resnet152':
        s_model = resnet152_denoise()
        loaded_state_dict = torch.load(os.path.join('weight', state['source_model_name']+".pytorch"))
        s_model.load_state_dict(loaded_state_dict)
    if 'defense' in state and state['defense']:
        source_model = nn.Sequential(
            Normalize(mean, std),
            Permute([2,1,0]),
            s_model
        )
    else:
        source_model = nn.Sequential(
            Normalize(mean, std),
            s_model
        )

model = load_model(args.model_name, dataset='imagenet', threat_model='Linf')

encoder = Imagenet_Encoder()
decoder = Imagenet_Decoder()
encoder.load_state_dict(encoder_weight)
decoder.load_state_dict(decoder_weight)

model.to(device)
model.eval()
encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()

if 'OSP' in state:
    source_model.to(device)
    source_model.eval()

F = Function(model, state['batch_size'], state['margin'], nlabels, state['target'])

count_success = 0
count_total = 0
if not os.path.exists(state['save_path']):
    os.mkdir(state['save_path'])

adversarial_folder = args.adversarial_folder
os.makedirs(adversarial_folder, exist_ok=True)
for i, (images, labels) in enumerate(dataloader):
    images = images.to(device)
    labels = int(labels)
    logits = model(images)
    correct = torch.argmax(logits, dim=1) == labels
    if correct:
        torch.cuda.empty_cache()
        if state['target']:
            labels = state['target_class']

        if 'OSP' in state:
            hinge_loss = MarginLoss_Single(state['white_box_margin'], state['target'])
            images.requires_grad = True
            latents = encoder(images)
            for k in range(state['white_box_iters']):     
                perturbations = decoder(latents)*state['epsilon']
                logits = source_model(torch.clamp(images+perturbations, 0, 1))
                loss = hinge_loss(logits, labels)
                grad = torch.autograd.grad(loss, latents)[0]
                latents = latents - state['white_box_lr'] * grad

            with torch.no_grad():
                success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state, latents.view(-1))

        else:
            with torch.no_grad():
                success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state)

        count_success += int(success)
        count_total += int(correct)
        print("image: {} eval_count: {} success: {} average_count: {} success_rate: {}".format(i, F.current_counts, success, F.get_average(), float(count_success) / float(count_total)))

        if success:
            F.new_counter_successful()
        else:
            F.new_counter()

        metrics = {'suc_rate_steps': float(count_success) / float(count_total), 'suc_rate': int(success),
                   'queries_steps': F.get_average_successful(), 'queries_steps_all_images': F.get_average()}
        experiment.log_metrics(metrics, step=count_total)

        image_np = adv.squeeze().permute(1, 2, 0).cpu().numpy()
        image_np = image_np * 255
        image_np = image_np.astype(np.uint8)
        adv_path = os.path.join(adversarial_folder, f"{i}_{labels}_{correct}_{args.seed}_{args.generator_name[:15]}_{args.model_name}.png")
        adv_png = Image.fromarray(image_np)
        adv_png.save(adv_path)

success_rate = float(count_success) / float(count_total)
if state['target']:
    np.save(os.path.join(state['save_path'], '{}_class_{}.npy'.format(state['save_prefix'], state['target_class'])), np.array(F.counts))
else:
    np.save(os.path.join(state['save_path'], '{}.npy'.format(state['save_prefix'])), np.array(F.counts))
print("success rate {}".format(success_rate))
print("average eval count {}".format(F.get_average()))
