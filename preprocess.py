from dmri_data.preprocessor import *
import utils
import argparse
from enum import Enum, auto
from monai.transforms import DivisiblePad
from foundation.running.utils.normalization import normalize_positive
from torchvision import transforms
from foundation.modules.models.latencizer import AutoencoderLatencizer
from utils import config_tool

def _get_model_from_config(config, model_config):
    specific_type_str = model_config.type
    exec(f'from {model_config.package} import {specific_type_str}')
    specific_parameters_dict = vars(model_config.parameters)
    for key, item in specific_parameters_dict.items():
        if isinstance(item, config_tool.script_func):
            specific_parameters_dict[key] = item.call(config)
    specific_model = eval(specific_type_str)(**specific_parameters_dict)
    return specific_model

class PreprocessType(Enum):
    dmri = auto()
    latence = auto()
    
    @staticmethod
    def arg_type(arg_str: str):
        return eval(f'PreprocessType.{arg_str}')

def parse_func_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('-t', '--type',
                        type=PreprocessType.arg_type, default=PreprocessType.dmri,  help='The task type of this preprocess.')
    args, _ = parser.parse_known_args()
    if args.type == PreprocessType.dmri:
        parser.add_argument('-d', '--data-dir', '--dir', type=str, required=True,  help='The directory where the data is located.')
        parser.add_argument('-s', '--saving-dir', '--save', type=str, default='./preprocessed', help='The directory where the preprocessed data is saved.')
        parser.add_argument('--save-prefix', type=str, default='', help='The prefix of the name of each of the saved file.')
        # parser.add_argument('--transform', type=Optional[Callable], default=None, help='The transform of each saved image.')
        parser.add_argument('-r', '--registration-ref-path', '--ref', type=str, default=None, help='The path of the required registration reference file.')
        parser.add_argument('-p', '--position-slice', type=utils.argparse_type, default=110, help='The path of the required registration reference file.')
        # parser.add_argument('-u', '--unique-b-values-save-path', type=str, default='./preprocessed/unique_b_values.npy', help='The save path of each unique b-value.')
        parser.add_argument('-b', '--b-value-mean-max-limit', '--b-max-num', type=str, default=8, help='The max limit number of each b-values.')
        func = preprocess_dmri_data
        args = parser.parse_args()
    else:
        parser.add_argument('-c', '--latencizer-config', '--config', type=str, default='default', help='The latencizer config.')
        parser.add_argument('-s', '--saving-dir', '--save', type=str, default='./latencized', help='The directory where the latencized data is saved.')
        parser.add_argument('-b', '--batch-size', '--batch', type=int, default=48, help='The batch size of the data loader.')
        parser.add_argument('-n', '--num-workers', type=int, default=8, help='The num workers of the data loader.')
        parser.add_argument('--device', type=str, default='cuda:0', help='The main device of processing.')
        parser.add_argument('--diffusion-dir', type=str, default='./preprocessed/train/diffusion_images', help='The directory where the preprocessed diffusion images is located.')
        parser.add_argument('--base-dir', type=str, default='./preprocessed/train/base_images', help='The directory where the preprocessed base images is located.')
        parser.add_argument('--adc-atlases-path', type=str, default='./preprocessed/train/dmri_adc_atlases/adc_atlases.npk', help='The path of the ADC atlases.')
        parser.add_argument('--divisible-pad-key', type=eval, default=[8, 4], help='The divisible pad used to transform.')
        parser.add_argument('--std', type=bool, default=True, help='The main device of processing.')
        func = pre_latencize
        args = parser.parse_args()
        transform = transforms.Compose([
            transforms.Lambda(normalize_positive),
            DivisiblePad(k=[8, 4])
        ])
        latence_config_path = args.latencizer_config
        device = args.device
        if not latence_config_path.endswith('.yaml'):
            latence_config_path = f'./configs/latence/{latence_config_path}.yaml'
        latence_config = config_tool.load(latence_config_path)
        latencizer = _get_model_from_config(latence_config, latence_config.model.latencizer).to(device)
        latencizer.load_state_dict(torch.load(latence_config.model.latencizer.pretrained, map_location=device))
        latencizer = AutoencoderLatencizer(latencizer)
        setattr(args, 'transform', transform)
        setattr(args, 'latencizer', latencizer)
        delattr(args, 'divisible_pad_key')
        delattr(args, 'latencizer_config')
    
    delattr(args, 'type')
    return args, func

if __name__ == '__main__':
    args, func = parse_func_args()
    func(**vars(args))

# def parse_func_args():
#     parser = argparse.ArgumentParser(description=globals()['__doc__'])
#     parser.add_argument('-t', '--type',
#                         type=PreprocessType.arg_type, required=True,  help='The task type of this preprocess.')
#     args, _ = parser.parse_known_args()
#     if args.type == PreprocessType.adc:
#         parser.add_argument('-d', '--diffusion-dir', '--dir', type=str, default='./preprocessed/dmri_images', help='The directory where the preprocessed diffusion data is located.')
#         parser.add_argument('-b', '--base-dir', type=str, default='./preprocessed/mri_images', help='The directory where the preprocessed base data is located.')
#         parser.add_argument('-u', '--unique-b-values-path', type=str, default='./preprocessed/unique_b_values.npy', help='The directory where the unique b-values are located.')
#         parser.add_argument('-p', '--position-count', type=int, default=110, help='The position count of each 3D image under each b-value.')
#         parser.add_argument('-i', '--image-size', type=util.argparse_type, default=(218, 182), help='The size of each 2D image under each b-value and position.')
#         parser.add_argument('-s', '--save-dir', '--save', type=str, default='./preprocessed/dmri_adc_atlases', help='The directory where the preprocessed ADC atlases data is saved.')
#         func = compute_dmri_adc_atlases
#     else:
#         parser.add_argument('-d', '--data-dir', '--dir', type=str, required=True,  help='The directory where the data is located.')
#         parser.add_argument('--save-prefix', type=str, default='', help='The prefix of the name of each of the saved file.')
#         parser.add_argument('--transform', type=Optional[Callable], default=None, help='The transform of each saved image.')
#         parser.add_argument('-r', '--registration-ref-path', '--ref', type=str, default=None, help='The path of the required registration reference file.')
#         parser.add_argument('-p', '--position-slice', type=util.argparse_type, default=110, help='The path of the required registration reference file.')
#         if args.type == PreprocessType.dmri:
#             parser.add_argument('-s', '--save-dir', '--save', type=str, default='./preprocessed/dmri_images', help='The directory where the preprocessed data is saved.')
#             parser.add_argument('-u', '--unique-b-values-save-path', type=str, default='./preprocessed/unique_b_values.npy', help='The save path of each unique b-value.')
#             parser.add_argument('-b', '--b-value-mean-max-limit', '--b-max-num', type=str, default=8, help='The max limit number of each b-values.')
#             func = preprocess_dmri_data
#         else:
#             parser.add_argument('-s', '--save-dir', '--save', type=str, default='./preprocessed/mri_images', help='The directory where the preprocessed data is saved.')
#             func = preprocess_mri_data
#     args = parser.parse_args()
#     delattr(args, 'type')
#     return func, args