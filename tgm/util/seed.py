import random

import numpy as np
import torch

from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


def seed_everything(seed: int) -> None:
    """Sets the seed for generating random number in Pytorch, numpy and Python.

    Args:
        seed (int): The desired seed.

    Notes:
        - You may also want to set `torch.backends.cudnn.deterministic = True`
          and `torch.backends.cudnn.benchmark = False` for full determinism on GPU.
    """
    logger.debug('Seeding RNG with %d', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
