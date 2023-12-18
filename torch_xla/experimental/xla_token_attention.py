from typing import List

import torch
import torch_xla
from torch.library import Library, impl

xla_token_attention_lib = Library("xla_token_attention", "DEF")

xla_token_attention_lib.define(
    "token_attention(Tensor q, Tensor k, Tensor v, Tensor kv_read_indices, int[] seq_lens, float scale, bool prefill) -> Tensor"
)


@impl(xla_token_attention_lib, "token_attention", "XLA")
def token_attention_impl_xla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_read_indices: torch.Tensor,
    seq_lens: List[int],
    scale: float,
    prefill: bool):
  return torch_xla._XLAC._xla_token_attention(q, k, v, kv_read_indices, seq_lens, scale, prefill)

# Default dispatch, work for all backends.
@impl(xla_token_attention_lib, "token_attention", "CompositeExplicitAutograd")
def token_attention_impl_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_read_indices: torch.Tensor,
    seq_lens: List[int],
    scale: float,
    prefill: bool):
  # Do nothing for non-xla tensor.
  return q

@impl(xla_token_attention_lib, "token_attention", "Meta")
def token_attention_impl_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_read_indices: torch.Tensor,
    seq_lens: List[int],
    scale: float,
    prefill: bool):
  return torch.empty_like(q)
