import random
import numpy as np
import torch

def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_ids_from_file(filename):
    with open(filename, 'r') as file:
        # Read the entire file content
        content = file.read()

        # Split the content by commas and strip any whitespace or newline characters
        ids = [int(id.strip()) for id in content.split(',') if id!='']

    return ids