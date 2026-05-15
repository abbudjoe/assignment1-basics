# cs336_basics/model.py
import math
import torch
from torch import nn

class Linear(nn.Module):
  def __init__ (self, d_in: int, d_out: int):
    super().__init__()
    self.weight = nn.Parameter(torch.empty(d_out, d_in))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.weight.T

class Embedding(nn.Module):
  def __init__(self, num_embeddings: int, embedding_dim: int):
    super().__init__()
    self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.weight[token_ids]

class SwiGLU(nn.Module):
  def __init__(self, d_model: int, d_ff: int):
    super().__init__()
    self.w1 = Linear(d_model, d_ff)
    self.w2 = Linear(d_ff, d_model)
    self.w3 = Linear(d_model, d_ff)
  
  def silu(self, x: torch.Tensor) -> torch.Tensor: 
    return x * torch.sigmoid(x)

  def forward(self, in_features: torch.Tensor):
    gate =  self.w1(in_features)
    up = self.w3(in_features)
    hidden = self.silu(gate) * up
    return self.w2(hidden)
    

