import time

import torch

print('Mock testing edgebank...')

# Fake alloc to ensure we have gpu
_ = torch.rand(1, device='cuda')
time.sleep(5)

print('Done')
