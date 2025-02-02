import random

import numpy as np
import torch

from opendg.seed import seed_everything


def test_seed_everything():
    seed_everything(1337)

    assert random.randint(0, 100) == 79
    assert random.randint(0, 100) == 68
    assert np.random.randint(0, 100) == 23
    assert np.random.randint(0, 100) == 61
    assert int(torch.randint(0, 100, (1,))) == 15
    assert int(torch.randint(0, 100, (1,))) == 57
