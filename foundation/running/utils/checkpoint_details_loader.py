import numpy as np
import os
import torch
from utils.str_ext import extract_number

def _get_training_details_path(checkpoints_dir: str):
    return os.path.join(checkpoints_dir, 'training_details', 'latest_details.npz')

def _get_training_optim_details_path(checkpoints_dir: str):
    return os.path.join(checkpoints_dir, 'training_details', 'optim', 'latest_optim.pt')

def load_checkpoints_details(checkpoints_dir: str):
    details_path = _get_training_details_path(checkpoints_dir)
    if os.path.exists(details_path):
        details = np.load(details_path)
        optim_details_path = _get_training_optim_details_path(checkpoints_dir)
        if os.path.exists(optim_details_path):
            optim_details = torch.load(optim_details_path, map_location='cpu')
        else:
            optim_details = None
        return details['epoch'] + int(details['is_epoch_finished']), details['global_step'] + 1, details['learning_rate'], optim_details
    else:
        return extract_number(checkpoints_dir) + 1, None, None, None

def checkpoint_sorted_key(path: str):
    details_path = _get_training_details_path(path)
    if os.path.exists(details_path):
        details = np.load(details_path)
        return (details['epoch'], details['global_step'])
    else:
        return extract_number(path)
    