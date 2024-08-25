import torch
import math
from einops import rearrange, repeat, einsum
import pdb


def tensors_sparseloss(Ms: torch.tensor):
    Ms = rearrange(Ms, "n0 ... -> n0 (...)")
    Ms_expandOne = Ms.unsqueeze(1)
    Ms_expandTwo = Ms.unsqueeze(0)

    Delta = torch.sum((Ms_expandOne - Ms_expandTwo) ** 2, dim=2)

    return Delta
