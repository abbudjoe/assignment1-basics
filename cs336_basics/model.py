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
    self.output_proj = Linear(d_model, d_model)

  def forward(
    self, 
    x: torch.Tensor,
    token_positions: torch.Tensor | None = None, 
    theta: float | None = None, 
    max_seq_len: int | None = None,
  ):
    # name runtime input dimensions
    *leading_dims, seq, d_model = x.shape

    # create query/key/value vectors for each token
    Q = self.q_proj(x)
    K = self.k_proj(x)
    V = self.v_proj(x)

    # split the feature dimension into heads, and move heads before sequence
    Q = Q.view(*leading_dims, seq, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(*leading_dims, seq, self.num_heads, self.head_dim).transpose(1, 2)
   

    if token_positions is not None:
      Q = apply_rope(Q, theta, max_seq_len, token_positions)
      K = apply_rope(K, theta, max_seq_len, token_positions)

    V = V.view(*leading_dims, seq, self.num_heads, self.head_dim).transpose(1, 2)

    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
    
    #run attantion independently per head
    attn = scaled_dot_product_attention(Q, K, V, mask)
    attn = attn.transpose(1, 2)
    
    #recombine all heads into one [batch, seq, d_model]
    attn = attn.reshape(*leading_dims, seq, d_model)  
    return self.output_proj(attn)

class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.empty(d_model))
  
  def forward(self, x: torch.Tensor):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
    normalized = x / rms
    return normalized * self.weight

class TransformerBlock(nn.Module):
  def __init__(
    self, 
    d_model: int, 
    num_heads: int, 
    d_ff: int, 
    max_seq_len: int, 
    theta: int
  ):
    super().__init__()
    self.ln1 = RMSNorm(d_model, eps=1e-5)
    self.attn = MultiHeadAttention(d_model, num_heads)
    self.ln2 = RMSNorm(d_model, eps=1e-5)
    self.ffn = SwiGLU(d_model, d_ff)

    self.max_seq_len = max_seq_len
    self.theta = theta

  def forward(self, x: torch.Tensor):
    *leading_dims, seq, d_model = x.shape

    token_positions = torch.arange(seq, device=x.device)

    attn_input = self.ln1(x)
    attn_update = self.attn(
      attn_input, 
      token_positions=token_positions,
      theta=self.theta,
      max_seq_len=self.max_seq_len,
    )
    x = x + attn_update

    ffn_input = self.ln2(x)
    ffn_update = self.ffn(ffn_input)
    x = x + ffn_update
    return x

class TransformerLM(nn.Module):
  def __init__(
    self, 
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
  ) -> None:
    super().__init__()
    self.token_embeddings = Embedding(vocab_size, d_model)
    self.layers = nn.ModuleList([
      TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=context_length,
        theta=rope_theta,
      )
      for _ in range(num_layers)
    ])
    self.ln_final = RMSNorm(d_model, eps=1e-5)
    self.lm_head = Linear(d_model, vocab_size)

  def forward(self, token_ids):
    x = self.token_embeddings(token_ids)
    for layer in self.layers:
      x = layer(x)
    x = self.ln_final(x)
    logits = self.lm_head(x)
    return logits

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

def apply_rope(
  x: torch.Tensor,
  theta: float,
  max_seq_len: int, # max_seq_len is used by cached RoPE implementations; this simple version computes angles directly.
  token_positions: torch.Tensor,
) -> torch.Tensor:
  d_k = x.shape[-1]

  x_even = x[..., 0::2]
  x_odd = x[..., 1::2]

  inv_freq = 1.0 / (
    theta ** (torch.arange(0, d_k, 2, device=x.device, dtype=x.dtype) / d_k)
  )
  angles = token_positions[..., None].to(device=x.device, dtype=x.dtype) * inv_freq
  cos = torch.cos(angles)
  sin = torch.sin(angles)

  rotated_even = x_even * cos - x_odd * sin
  rotated_odd = x_even * sin + x_odd * cos

  output = torch.empty_like(x)
  output[..., 0::2] = rotated_even
  output[..., 1::2] = rotated_odd

  return output



