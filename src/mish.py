import torch
from torch.nn import Module
import torch.nn.functional as F

def mish(x):
  return x * torch.tanh(F.softplus(x))

class Mish(Module):
  def forward(self, x):
    return mish(x)
