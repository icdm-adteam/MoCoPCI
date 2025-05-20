

import torch

eps = 1e-08
t = torch.rand(1) * (1000 - eps) + eps
z0 = torch.rand(1, 8192, 3)
batch = torch.rand(1, 8192, 3)

t_expand = t.view(-1, 1, 1, 1).repeat(1, 1, 8192, 3)
perturbed_data = t_expand * batch + (1. - t_expand) * z0
target = batch - z0

score = model_fn(perturbed_data, t * 999)  ### Copy from models/utils.py