# cs336_basics/model.py
import math
import torch
from torch import nn, softmax

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
    
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads

    self.q_proj = Linear(d_model, d_model)
    self.k_proj = Linear(d_model, d_model)
    self.v_proj = Linear(d_model, d_model)
    self.o_proj = Linear(d_model, d_model)

  def forward(self, x):
    # name runtime input dimensions
    *leading_dims, seq, d_model = x.shape

    # create query/key/value vectors for each token
    Q = self.q_proj(x)
    K = self.k_proj(x)
    V = self.v_proj(x)

    # split the feature dimension into heads, and move heads before sequence
    Q = Q.view(*leading_dims, seq, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(*leading_dims, seq, self.num_heads, self.head_dim).transpose(1, 2)
    V = V.view(*leading_dims, seq, self.num_heads, self.head_dim).transpose(1, 2)

    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
    
    #run attantion independently per head
    attn = scaled_dot_product_attention(Q, K, V, mask)
    attn = attn.transpose(1, 2)
    
    #recombine all heads into one [batch, seq, d_model]
    attn = attn.reshape(*leading_dims, seq, d_model)  
    return self.o_proj(attn)

def scaled_dot_product_attention(
  Q: torch.Tensor, 
  K: torch.Tensor, 
  V: torch.Tensor, 
  mask=None,
) -> torch.Tensor:
  scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])

  if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))

  weights = torch.softmax(scores, dim=-1)
  return weights @ V

