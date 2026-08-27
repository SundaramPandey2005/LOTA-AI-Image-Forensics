import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Centrally configure all stochastic processes across Python, NumPy, and PyTorch.

    Args:
        seed (int): Integer random seed.
        deterministic (bool): If True, configures cuDNN for strict determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Standard benchmark mode for performance while preserving seeded model initialization
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
