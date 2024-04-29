import argparse
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import requests
import timm
import torch
from torch import nn
import gdown
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from robustbench_ghost.model_zoo import model_dicts as all_models
from robustbench_ghost.model_zoo.enums import BenchmarkDataset, ThreatModel


DATASET_CLASSES = {
    BenchmarkDataset.imagenet: 1000,
}

CANNED_USER_AGENT="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"  # NOQA


def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()

    # Fix from https://github.com/wkentaro/gdown/pull/294.
    session.headers.update(
        {"User-Agent": CANNED_USER_AGENT}
    )

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))
    

def download_gdrive_new(gdrive_id, fname_save):
    """Download checkpoints with gdown, see https://github.com/wkentaro/gdown."""
    
    if isinstance(fname_save, Path):
        fname_save = str(fname_save)
    print(f'Downloading {fname_save} (gdrive_id={gdrive_id}).')
    gdown.download(id=gdrive_id, output=fname_save)


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def add_substr_to_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[substr + k] = v
    return new_state_dict


def load_model(model_name: str,
               model_dir: Union[str, Path] = './models',
               dataset: Union[str,
                              BenchmarkDataset] = BenchmarkDataset.imagenet,
               threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
               custom_checkpoint: str = "",
               norm: Optional[str] = None) -> nn.Module:
    """Loads a model from the model_zoo.

     The model is trained on the given ``dataset``, for the given ``threat_model``.

    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it
      overrides ``threat_model``

    :return: A ready-to-used trained model.
    """
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    if norm is None:
        # since there is only `corruptions` folder for models in the Model Zoo
        threat_model = ThreatModel(threat_model).value.replace('_3d', '')
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_ = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    lower_model_name = model_name.lower().replace('-', '_')
    timm_model_name = f"{lower_model_name}_{dataset_.value.lower()}_{threat_model_.value.lower()}"
    
    if timm.is_model(timm_model_name):
        return timm.create_model(timm_model_name,
                                 num_classes=DATASET_CLASSES[dataset_],
                                 pretrained=True,
                                 checkpoint_path=custom_checkpoint).eval()

    model_dir_ = Path(model_dir) / dataset_.value / threat_model_.value
    model_path = model_dir_ / f'{model_name}.pt'

    # models = all_models[dataset_][threat_model_]
    models = list(all_models[dataset_].items())[0][1]

    if models[model_name]['gdrive_id'] is None:
        raise ValueError(
            f"Model `{model_name}` nor {timm_model_name} aren't a timm model and has no `gdrive_id` specified."
        )

    if not isinstance(models[model_name]['gdrive_id'], list):
        model = models[model_name]['model']()
        if dataset_ == BenchmarkDataset.imagenet and 'Standard' in model_name:
            return model.eval()

        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        if not os.path.isfile(model_path):
            download_gdrive_new(models[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if 'Kireev2021Effectiveness' in model_name or model_name == 'Andriushchenko2020Understanding':
            checkpoint = checkpoint[
                'last']  # we take the last model (choices: 'last', 'best')
        try:
            # needed for the model of `Carmon2019Unlabeled`
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'],
                                                   'module.')
            # needed for the model of `Chen2020Efficient`
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')

        if dataset_ == BenchmarkDataset.imagenet:
            # Adapt checkpoint to the model defition in newer versions of timm.
            if model_name in [
                'Liu2023Comprehensive_Swin-B',
                'Liu2023Comprehensive_Swin-L',
                ]:
                try:
                    from timm.models.swin_transformer import checkpoint_filter_fn
                    state_dict = checkpoint_filter_fn(state_dict, model.model)
                except:
                    pass

            # Some models need input normalization, which is added as extra layer.
            if model_name not in [
                'Singh2023Revisiting_ConvNeXt-T-ConvStem',
                'Singh2023Revisiting_ViT-B-ConvStem',
                'Singh2023Revisiting_ConvNeXt-S-ConvStem',
                'Singh2023Revisiting_ConvNeXt-B-ConvStem',
                'Singh2023Revisiting_ConvNeXt-L-ConvStem',
                'Peng2023Robust',
                'Chen2024Data_WRN-50-2',
                ]:
                state_dict = add_substr_to_state_dict(state_dict, 'model.')

        model = _safe_load_state_dict(model, model_name, state_dict, dataset_)

        return model.eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial, Diffenderfer2021Winning_LRR_CARD_Deck)
    else:
        model = models[model_name]['model']()
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        for i, gid in enumerate(models[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive_new(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i),
                                    map_location=torch.device('cpu'))
            try:
                state_dict = rm_substr_from_state_dict(
                    checkpoint['state_dict'], 'module.')
            except KeyError:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')

            if model_name.startswith('Bai2023Improving'):
                # TODO: make it cleaner.
                if i < 2:
                    model.comp_model.models[i] = _safe_load_state_dict(
                        model.comp_model.models[i], model_name, state_dict, dataset_)
                    model.comp_model.models[i].eval()
                else:
                    model.comp_model.policy_net = _safe_load_state_dict(
                        model.comp_model.policy_net, model_name, state_dict['model'], dataset_)
                    model.comp_model.bn = _safe_load_state_dict(
                        model.comp_model.bn, model_name, state_dict['bn'], dataset_)
            elif model_name.startswith('Bai2024MixedNUTS'):
                if i == 0:
                    model.std_model = _safe_load_state_dict(
                        model.std_model, model_name, state_dict, dataset_)
                elif i == 1:
                    if dataset_ == BenchmarkDataset.imagenet:
                        from timm.models.swin_transformer import checkpoint_filter_fn
                        state_dict = checkpoint_filter_fn(state_dict, model.rob_model.model)
                        state_dict = add_substr_to_state_dict(state_dict, 'model.')
                    model.rob_model = _safe_load_state_dict(
                        model.rob_model, model_name, state_dict, dataset_)
                else:
                    raise ValueError('Unexpected checkpoint.')
            else:
                model.models[i] = _safe_load_state_dict(model.models[i],
                                                        model_name, state_dict,
                                                        dataset_)
                model.models[i].eval()

        return model.eval()


def _safe_load_state_dict(model: nn.Module, model_name: str,
                          state_dict: Dict[str, torch.Tensor],
                          dataset_: BenchmarkDataset) -> nn.Module:
    known_failing_models = {
        "Andriushchenko2020Understanding", "Augustin2020Adversarial",
        "Engstrom2019Robustness", "Pang2020Boosting", "Rice2020Overfitting",
        "Rony2019Decoupling", "Wong2020Fast", "Hendrycks2020AugMix_WRN",
        "Hendrycks2020AugMix_ResNeXt",
        "Kireev2021Effectiveness_Gauss50percent",
        "Kireev2021Effectiveness_AugMixNoJSD", "Kireev2021Effectiveness_RLAT",
        "Kireev2021Effectiveness_RLATAugMixNoJSD",
        "Kireev2021Effectiveness_RLATAugMixNoJSD",
        "Kireev2021Effectiveness_RLATAugMix", "Chen2020Efficient",
        "Wu2020Adversarial", "Augustin2020Adversarial_34_10",
        "Augustin2020Adversarial_34_10_extra", "Diffenderfer2021Winning_LRR",
        "Diffenderfer2021Winning_LRR_CARD_Deck",
        "Diffenderfer2021Winning_Binary",
        "Diffenderfer2021Winning_Binary_CARD_Deck",
        "Huang2022Revisiting_WRN-A4",
        "Bai2024MixedNUTS",
    }

    failure_messages = [
        'Missing key(s) in state_dict: "mu", "sigma".',
        'Unexpected key(s) in state_dict: "model_preact_hl1.1.weight"',
        'Missing key(s) in state_dict: "normalize.mean", "normalize.std"',
        'Unexpected key(s) in state_dict: "conv1.scores"',
        'Missing key(s) in state_dict: "mean", "std".',
    ]

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        #with open('./log_new_models.txt', 'a') as f:
        #    f.write(str(e))
        if (model_name in known_failing_models
                or dataset_ == BenchmarkDataset.imagenet) and any(
                    [msg in str(e) for msg in failure_messages]):
            model.load_state_dict(state_dict, strict=False)
        else:
            raise e

    return model



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='Carmon2019Unlabeled')
    parser.add_argument('--custom_checkpoint',
                        type=str,
                        default="",
                        help='Path to custom checkpoint')
    parser.add_argument('--threat_model',
                        type=str,
                        default='Linf',
                        choices=[x.value for x in ThreatModel])
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=[x.value for x in BenchmarkDataset])
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--n_ex',
                        type=int,
                        default=100,
                        help='number of examples to evaluate on')
    parser.add_argument('--batch_size',
                        type=int,
                        default=500,
                        help='batch size for evaluation')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data',
                        help='where to store downloaded datasets')
    parser.add_argument('--corruptions_data_dir',
                        type=str,
                        default='',
                        help='where the corrupted data are stored')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./models',
                        help='where to store downloaded models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to use for computations')
    parser.add_argument('--to_disk', type=bool, default=True)
    args = parser.parse_args()
    return args