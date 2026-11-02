"""Two-dimensional rotary embeddings."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

def broadcat(tensors, dim: int = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim: int,
        pt_seq_len: Union[int, Tuple[int, int]] = 16,
        ft_seq_len: Optional[Union[int, Tuple[int, int]]] = None,
        custom_freqs=None,
        freqs_for: str = "lang",
        theta: int = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
    ) -> None:
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if isinstance(pt_seq_len, int):
            pt_h = pt_w = pt_seq_len
        else:
            pt_h, pt_w = pt_seq_len

        if ft_seq_len is None:
            ft_h = pt_h
            ft_w = pt_w
        elif isinstance(ft_seq_len, int):
            ft_h = ft_w = ft_seq_len
        else:
            ft_h, ft_w = ft_seq_len

        t_h = torch.arange(ft_h) / ft_h * pt_h
        t_w = torch.arange(ft_w) / ft_w * pt_w

        freqs_h = torch.einsum("..., f -> ... f", t_h, freqs)
        freqs_w = torch.einsum("..., f -> ... f", t_w, freqs)

        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        _, _, Lt, _ = t.shape
        L, _ = self.freqs_cos.shape
        repeat_factor = Lt // L
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        if repeat_factor != 1:
            freqs_cos = freqs_cos.repeat_interleave(repeat_factor, dim=0)
            freqs_sin = freqs_sin.repeat_interleave(repeat_factor, dim=0)
        return t * freqs_cos + rotate_half(t) * freqs_sin


