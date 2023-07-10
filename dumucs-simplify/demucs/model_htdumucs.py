"""

"""
import os
import random
import typing as tp
import tempfile
import functools
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import math
from einops import rearrange


class Model:
    """
    代表一个模块
    """
    ...


# transformer

def create_sin_embedding(
        length: int, dim: int, shift: int = 0, device="cpu", max_period=10000
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    )


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model)
    )
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1:: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe[None, :].to(device)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0, channel_last=False):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


class MyGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: (B, T, C)
        if num_groups=1: Normalisation on all T and C together for each B
        """
        x = x.transpose(1, 2)
        return super().forward(x).transpose(1, 2)


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            group_norm=0,
            norm_first=False,
            norm_out=False,
            layer_norm_eps=1e-5,
            layer_scale=False,
            init_values=1e-4,
            device=None,
            dtype=None,
            sparse=False,
            mask_type="diag",
            mask_random_seed=42,
            sparse_attn_window=500,
            global_window=50,
            auto_sparsity=False,
            sparsity=0.95,
            batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)
        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        if sparse:
            self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                auto_sparsity=sparsity if auto_sparsity else 0,
            )
            self.__setattr__("src_mask", torch.zeros(1, 1))
            self.mask_random_seed = mask_random_seed

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        if batch_first = False, src shape is (T, B, C)
        the case where batch_first=True is not covered
        """
        device = src.device
        x = src
        T, B, C = x.shape
        if self.sparse and not self.auto_sparsity:
            assert src_mask is None
            src_mask = self.src_mask
            if src_mask.shape[-1] != T:
                src_mask = get_mask(
                    T,
                    T,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.__setattr__("src_mask", src_mask)

        if self.norm_first:
            x = x + self.gamma_1(
                self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            )
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))

            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(
                x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask))
            )
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation=F.relu,
            layer_norm_eps: float = 1e-5,
            layer_scale: bool = False,
            init_values: float = 1e-4,
            norm_first: bool = False,
            group_norm: bool = False,
            norm_out: bool = False,
            sparse=False,
            mask_type="diag",
            mask_random_seed=42,
            sparse_attn_window=500,
            global_window=50,
            sparsity=0.95,
            auto_sparsity=None,
            device=None,
            dtype=None,
            batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity

        self.cross_attn: nn.Module
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1: nn.Module
        self.norm2: nn.Module
        self.norm3: nn.Module
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

        if sparse:
            self.cross_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                auto_sparsity=sparsity if auto_sparsity else 0)
            if not auto_sparsity:
                self.__setattr__("mask", torch.zeros(1, 1))
                self.mask_random_seed = mask_random_seed

    def forward(self, q, k, mask=None):
        """
        Args:
            q: tensor of shape (T, B, C)
            k: tensor of shape (S, B, C)
            mask: tensor of shape (T, S)

        """
        device = q.device
        T, B, C = q.shape
        S, B, C = k.shape
        if self.sparse and not self.auto_sparsity:
            assert mask is None
            mask = self.mask
            if mask.shape[-1] != S or mask.shape[-2] != T:
                mask = get_mask(
                    S,
                    T,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.__setattr__("mask", mask)

        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

    # self-attention block
    def _ca_block(self, q, k, attn_mask=None):
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class CrossTransformerEncoder(nn.Module):
    def __init__(
            self,
            dim: int,
            emb: str = "sin",
            hidden_scale: float = 4.0,
            num_heads: int = 8,
            num_layers: int = 6,
            cross_first: bool = False,
            dropout: float = 0.0,
            max_positions: int = 1000,
            norm_in: bool = True,
            norm_in_group: bool = False,
            group_norm: int = False,
            norm_first: bool = False,
            norm_out: bool = False,
            max_period: float = 10000.0,
            weight_decay: float = 0.0,
            lr: tp.Optional[float] = None,
            layer_scale: bool = False,
            gelu: bool = True,
            sin_random_shift: int = 0,
            weight_pos_embed: float = 1.0,
            cape_mean_normalize: bool = True,
            cape_augment: bool = True,
            cape_glob_loc_scale: list = [5000.0, 1.0, 1.4],
            sparse_self_attn: bool = False,
            sparse_cross_attn: bool = False,
            mask_type: str = "diag",
            mask_random_seed: int = 42,
            sparse_attn_window: int = 500,
            global_window: int = 50,
            auto_sparsity: bool = False,
            sparsity: float = 0.95,
    ):
        super().__init__()
        """
        """
        assert dim % num_heads == 0

        hidden_dim = int(dim * hidden_scale)

        self.num_layers = num_layers
        # classic parity = 1 means that if idx%2 == 1 there is a
        # classical encoder else there is a cross encoder
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_decay = weight_decay
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift
        if emb == "cape":
            self.cape_mean_normalize = cape_mean_normalize
            self.cape_augment = cape_augment
            self.cape_glob_loc_scale = cape_glob_loc_scale
        if emb == "scaled":
            self.position_embeddings = ScaledEmbedding(max_positions, dim, scale=0.2)

        self.lr = lr

        activation: tp.Any = F.gelu if gelu else F.relu

        self.norm_in: nn.Module
        self.norm_in_t: nn.Module
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        # spectrogram layers
        self.layers = nn.ModuleList()
        # temporal layers
        self.layers_t = nn.ModuleList()

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
            "mask_type": mask_type,
            "mask_random_seed": mask_random_seed,
            "sparse_attn_window": sparse_attn_window,
            "global_window": global_window,
            "sparsity": sparsity,
            "auto_sparsity": auto_sparsity,
            "batch_first": True,
        }

        kwargs_classic_encoder = dict(kwargs_common)
        kwargs_classic_encoder.update({
            "sparse": sparse_self_attn,
        })
        kwargs_cross_encoder = dict(kwargs_common)
        kwargs_cross_encoder.update({
            "sparse": sparse_cross_attn,
        })

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:

                self.layers.append(MyTransformerEncoderLayer(**kwargs_classic_encoder))
                self.layers_t.append(
                    MyTransformerEncoderLayer(**kwargs_classic_encoder)
                )

            else:
                self.layers.append(CrossTransformerEncoderLayer(**kwargs_cross_encoder))

                self.layers_t.append(
                    CrossTransformerEncoderLayer(**kwargs_cross_encoder)
                )

    def forward(self, x, xt):
        B, C, Fr, T1 = x.shape
        pos_emb_2d = create_2d_sin_embedding(
            C, Fr, T1, x.device, self.max_period
        )  # (1, C, Fr, T1)
        pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")
        x = rearrange(x, "b c fr t1 -> b (t1 fr) c")
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B, C, T2 = xt.shape
        xt = rearrange(xt, "b c t2 -> b t2 c")  # now T2, B, C
        pos_emb = self._get_pos_embedding(T2, B, C, x.device)
        pos_emb = rearrange(pos_emb, "t2 b c -> b t2 c")
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
        xt = rearrange(xt, "b t2 c -> b c t2")
        return x, xt

    def _get_pos_embedding(self, T, B, C, device):
        if self.emb == "sin":
            shift = random.randrange(self.sin_random_shift + 1)
            pos_emb = create_sin_embedding(
                T, C, shift=shift, device=device, max_period=self.max_period
            )
        elif self.emb == "cape":
            if self.training:
                pos_emb = create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    device=device,
                    max_period=self.max_period,
                    mean_normalize=self.cape_mean_normalize,
                    augment=self.cape_augment,
                    max_global_shift=self.cape_glob_loc_scale[0],
                    max_local_shift=self.cape_glob_loc_scale[1],
                    max_scale=self.cape_glob_loc_scale[2],
                )
            else:
                pos_emb = create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    device=device,
                    max_period=self.max_period,
                    mean_normalize=self.cape_mean_normalize,
                    augment=False,
                )

        elif self.emb == "scaled":
            pos = torch.arange(T, device=device)
            pos_emb = self.position_embeddings(pos)[:, None]

        return pos_emb

    def make_optim_group(self):
        group = {"params": list(self.parameters()), "weight_decay": self.weight_decay}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


# Attention Modules


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            auto_sparsity=None,
    ):
        super().__init__()
        assert auto_sparsity is not None, "sanity check"
        self.num_heads = num_heads
        self.q = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = torch.nn.Dropout(dropout)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias)
        self.proj_drop = torch.nn.Dropout(dropout)
        self.batch_first = batch_first
        self.auto_sparsity = auto_sparsity

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
    ):

        if not self.batch_first:  # N, B, C
            query = query.permute(1, 0, 2)  # B, N_q, C
            key = key.permute(1, 0, 2)  # B, N_k, C
            value = value.permute(1, 0, 2)  # B, N_k, C
        B, N_q, C = query.shape
        B, N_k, C = key.shape

        q = (
            self.q(query)
            .reshape(B, N_q, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q = q.flatten(0, 1)
        k = (
            self.k(key)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = k.flatten(0, 1)
        v = (
            self.v(value)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = v.flatten(0, 1)

        if self.auto_sparsity:
            assert attn_mask is None
            x = dynamic_sparse_attention(q, k, v, sparsity=self.auto_sparsity)
        else:
            x = scaled_dot_product_attention(q, k, v, attn_mask, dropout=self.attn_drop)
        x = x.reshape(B, self.num_heads, N_q, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x, None


# demucs.py

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)


def rescale_conv(conv, reference):
    """Rescale initial weight scale. It is unclear why it helps but it certainly does.
    """
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)


class DConv(nn.Module):
    """
    New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.
    """

    def __init__(self, channels: int, compress: float = 4, depth: int = 2, init: float = 1e-4,
                 norm=True, attn=False, heads=4, ndecay=4, lstm=False, gelu=True,
                 kernel=3, dilate=True):
        """
        Args:
            channels: input/output channels for residual branch.
            compress: amount of channel compression inside the branch.
            depth: number of layers in the residual branch. Each layer has its own
                projection, and potentially LSTM and attention.
            init: initial scale for LayerNorm.
            norm: use GroupNorm.
            attn: use LocalAttention.
            heads: number of heads for the LocalAttention.
            ndecay: number of decay controls in the LocalAttention.
            lstm: use LSTM.
            gelu: Use GELU activation.
            kernel: kernel size for the (dilated) convolutions.
            dilate: if true, use dilation, increasing with the depth.
        """

        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa

        hidden = int(channels / compress)

        act: tp.Type[nn.Module]
        if gelu:
            act = nn.GELU
        else:
            act = nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)
            mods = [
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden), act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels), nn.GLU(1),
                LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


# hdemucs.py

def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen."""
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    assert (out[..., padding_left: padding_left + length] == x0).all()
    return out


class ScaledEmbedding(nn.Module):
    """
    Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 10., smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out


class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=0, dconv_kw={}, pad=True,
                 rewrite=True):
        """Encoder layer. This used both by the time and the frequency branch.

        Args:
            chin: number of input channels.
            chout: number of output channels.
            norm_groups: number of groups for group norm.
            empty: used to make a layer with just the first conv. this is used
                before merging the time and freq. branches.
            freq: this is acting on frequencies.
            dconv: insert DConv residual branches.
            norm: use GroupNorm.
            context: context size for the 1x1 conv.
            dconv_kw: list of kwargs for the DConv class.
            pad: pad the input. Padding is done so that the output size is
                always the input size / stride.
            rewrite: add 1x1 conv at the end of the layer.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        """
        `inject` is used to inject the result from the time branch into the frequency branch,
        when both have the same stride.
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y
        return z


class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=1, dconv_kw={}, pad=True,
                 context_freq=True, rewrite=True):
        """
        Same as HEncLayer but for decoder. See `HEncLayer` for documentation.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1,
                                     [0, context])
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = F.gelu(z)
        return z, y


# utils.py

class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return


@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


# apply.py

class BagOfModels(nn.Module):
    def __init__(self, models: tp.List[Model],
                 weights: tp.Optional[tp.List[tp.List[float]]] = None,
                 segment: tp.Optional[float] = None):
        """
        Represents a bag of models with specific weights.
        You should call `apply_model` rather than calling directly the forward here for
        optimal performance.

        Args:
            models (list[nn.Module]): list of Demucs/HDemucs models.
            weights (list[list[float]]): list of weights. If None, assumed to
                be all ones, otherwise it should be a list of N list (N number of models),
                each containing S floats (S number of sources).
            segment (None or float): overrides the `segment` attribute of each model
                (this is performed inplace, be careful is you reuse the models passed).
        """
        super().__init__()
        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if segment is not None:
                other.segment = segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [[1. for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)
            for weight in weights:
                assert len(weight) == len(first.sources)
        self.weights = weights

    @property
    def max_allowed_segment(self) -> float:
        max_allowed_segment = float('inf')
        for model in self.models:
            if isinstance(model, HTDemucs):
                max_allowed_segment = min(max_allowed_segment, float(model.segment))
        return max_allowed_segment

    def forward(self, x):
        raise NotImplementedError("Call `apply_model` on this.")


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)


def apply_model(model: tp.Union[BagOfModels, Model],
                mix: tp.Union[th.Tensor, TensorChunk],
                shifts: int = 1, split: bool = True,
                overlap: float = 0.25, transition_power: float = 1.,
                progress: bool = False, device=None,
                num_workers: int = 0, segment: tp.Optional[float] = None,
                pool=None) -> th.Tensor:
    """
    Apply model to a given mixture.

    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress (bool): if True, show a progress bar (requires split=True)
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
        num_workers (int): if non zero, device is 'cpu', how many threads to
            use in parallel.
        segment (float or None): override the model segment parameter.
    """
    if device is None:
        device = mix.device
    else:
        device = th.device(device)
    if pool is None:
        if num_workers > 0 and device.type == 'cpu':
            pool = ThreadPoolExecutor(num_workers)
        else:
            pool = DummyPoolExecutor()
    kwargs: tp.Dict[str, tp.Any] = {
        'shifts': shifts,
        'split': split,
        'overlap': overlap,
        'transition_power': transition_power,
        'progress': progress,
        'device': device,
        'pool': pool,
        'segment': segment,
    }
    out: tp.Union[float, th.Tensor]
    if isinstance(model, BagOfModels):
        # Special treatment for bag of model.
        # We explicitely apply multiple times `apply_model` so that the random shifts
        # are different for each model.
        estimates: tp.Union[float, th.Tensor] = 0.
        totals = [0.] * len(model.sources)
        for sub_model, model_weights in zip(model.models, model.weights):
            original_model_device = next(iter(sub_model.parameters())).device
            sub_model.to(device)

            out = apply_model(sub_model, mix, **kwargs)
            sub_model.to(original_model_device)
            for k, inst_weight in enumerate(model_weights):
                out[:, k, :, :] *= inst_weight
                totals[k] += inst_weight
            estimates += out
            del out

        assert isinstance(estimates, th.Tensor)
        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] /= totals[k]
        return estimates

    model.to(device)
    model.eval()
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape
    if shifts:
        kwargs['shifts'] = 0
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(model, shifted, **kwargs)
            out += shifted_out[..., max_shift - offset:]
        out /= shifts
        assert isinstance(out, th.Tensor)
        return out
    elif split:
        kwargs['split'] = False
        out = th.zeros(batch, len(model.sources), channels, length, device=mix.device)
        sum_weight = th.zeros(length, device=mix.device)
        if segment is None:
            segment = model.segment
        assert segment is not None and segment > 0.
        segment_length: int = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        scale = float(format(stride / model.samplerate, ".2f"))
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = th.cat([th.arange(1, segment_length // 2 + 1, device=device),
                         th.arange(segment_length - segment_length // 2, 0, -1, device=device)])
        assert len(weight) == segment_length
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max()) ** transition_power
        futures = []
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment_length)
            future = pool.submit(apply_model, model, chunk, **kwargs)
            futures.append((future, offset))
            offset += segment_length
        if progress:
            futures = tqdm.tqdm(futures, unit_scale=scale, ncols=120, unit='seconds')
        for future, offset in futures:
            chunk_out = future.result()
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment_length] += (
                    weight[:chunk_length] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment_length] += weight[:chunk_length].to(mix.device)
        assert sum_weight.min() > 0
        out /= sum_weight
        assert isinstance(out, th.Tensor)
        return out
    else:
        valid_length: int
        if isinstance(model, HTDemucs) and segment is not None:
            valid_length = int(segment * model.samplerate)
        elif hasattr(model, 'valid_length'):
            valid_length = model.valid_length(length)  # type: ignore
        else:
            valid_length = length
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)
        padded_mix = mix.padded(valid_length).to(device)
        with th.no_grad():
            out = model(padded_mix)
        assert isinstance(out, th.Tensor)
        return center_trim(out, length)


# states.py
def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


# spec.py
def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps = x.device.type == 'mps'
    if is_mps:
        x = x.cpu()
    z = th.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=th.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps = z.device.type == 'mps'
    if is_mps:
        z = z.cpu()
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)
