import random

import numpy as np
import torch

from tgm.util.seed import seed_everything


def test_seed_everything():
    seed_everything(seed=1337)

    N = 100
    assert random.randint(0, N) == 79
    assert np.random.randint(0, N) == 23
    assert torch.randint(0, N, size=(1,)).item() == 15
