import torch

def normalize_across_origin(tensor):
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    return ((tensor - min_value) / (max_value - min_value)) * 2 - 1

def normalize_positive(tensor):
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    return ((tensor - min_value) / (max_value - min_value))