# cs336_basics/model.py
import math
import torch
from torch import nn

class Linear(nn.Module):
  def __init__ (self, in_features: int, out_features: int):
    super().__init__()

    std = math.sqry(2 / (in_features + out_features))
    self.weight = nn.Parameter(
      torch.empty(out_features, in_features)
    )
    nn.init.trunc_normal_(
      self.weight,
      mean = 0.0,
      std = std,
      a=-3 * std,
      b=3 * std,
    )
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.weight.T
  
def linear(in_dim, out_dim, weights, in_features):
  layer = Linear(in_dim, out_dim)
  with torch.no_grad():
    layer.weight.copy_(weights)

  return layer(in_features)