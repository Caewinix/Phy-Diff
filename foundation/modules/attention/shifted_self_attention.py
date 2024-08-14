from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F
from functools import lru_cache
from .. import flops
from .basic_core import scale_for_cosine_sim, checkpoint
from ..modules import AdaRMSNorm, Linear, AxialRoPE, FeedForwardBlock, apply_wd, zero_init, apply_rotary_emb_

def window(window_size, x):
    *b, h, w, c = x.shape
    x = torch.reshape(
        x,
        (*b, h // window_size, window_size, w // window_size, window_size, c),
    )
    x = torch.permute(
        x,
        (*range(len(b)), -5, -3, -4, -2, -1),
    )
    return x


def unwindow(x):
    *b, h, w, wh, ww, c = x.shape
    x = torch.permute(x, (*range(len(b)), -5, -3, -4, -2, -1))
    x = torch.reshape(x, (*b, h * wh, w * ww, c))
    return x


def shifted_window(window_size, window_shift, x):
    x = torch.roll(x, shifts=(window_shift, window_shift), dims=(-2, -3))
    windows = window(window_size, x)
    return windows


def shifted_unwindow(window_shift, x):
    x = unwindow(x)
    x = torch.roll(x, shifts=(-window_shift, -window_shift), dims=(-2, -3))
    return x


@lru_cache
def make_shifted_window_masks(n_h_w, n_w_w, w_h, w_w, shift, device=None):
    ph_coords = torch.arange(n_h_w, device=device)
    pw_coords = torch.arange(n_w_w, device=device)
    h_coords = torch.arange(w_h, device=device)
    w_coords = torch.arange(w_w, device=device)
    patch_h, patch_w, q_h, q_w, k_h, k_w = torch.meshgrid(
        ph_coords,
        pw_coords,
        h_coords,
        w_coords,
        h_coords,
        w_coords,
        indexing="ij",
    )
    is_top_patch = patch_h == 0
    is_left_patch = patch_w == 0
    q_above_shift = q_h < shift
    k_above_shift = k_h < shift
    q_left_of_shift = q_w < shift
    k_left_of_shift = k_w < shift
    m_corner = (
        is_left_patch
        & is_top_patch
        & (q_left_of_shift == k_left_of_shift)
        & (q_above_shift == k_above_shift)
    )
    m_left = is_left_patch & ~is_top_patch & (q_left_of_shift == k_left_of_shift)
    m_top = ~is_left_patch & is_top_patch & (q_above_shift == k_above_shift)
    m_rest = ~is_left_patch & ~is_top_patch
    m = m_corner | m_left | m_top | m_rest
    return m


def apply_window_attention(window_size, window_shift, q, k, v, scale=None):
    # prep windows and masks
    q_windows = shifted_window(window_size, window_shift, q)
    k_windows = shifted_window(window_size, window_shift, k)
    v_windows = shifted_window(window_size, window_shift, v)
    b, heads, h, w, wh, ww, d_head = q_windows.shape
    mask = make_shifted_window_masks(h, w, wh, ww, window_shift, device=q.device)
    q_seqs = torch.reshape(q_windows, (b, heads, h, w, wh * ww, d_head))
    k_seqs = torch.reshape(k_windows, (b, heads, h, w, wh * ww, d_head))
    v_seqs = torch.reshape(v_windows, (b, heads, h, w, wh * ww, d_head))
    mask = torch.reshape(mask, (h, w, wh * ww, wh * ww))

    # do the attention here
    flops.op(flops.op_attention, q_seqs.shape, k_seqs.shape, v_seqs.shape)
    qkv = F.scaled_dot_product_attention(q_seqs, k_seqs, v_seqs, mask, scale=scale)

    # unwindow
    qkv = torch.reshape(qkv, (b, heads, h, w, wh, ww, d_head))
    return shifted_unwindow(window_shift, qkv)


class ShiftedWindowSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, window_size, window_shift, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.window_size = window_size
        self.window_shift = window_shift
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, window_size={self.window_size}, window_shift={self.window_shift}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
        theta = self.pos_emb(pos).movedim(-2, -4)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        x = apply_window_attention(self.window_size, self.window_shift, q, k, v, scale=1.0)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class ShiftedWindowTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, window_size, index, dropout=0.0):
        super().__init__()
        window_shift = window_size // 2 if index % 2 == 1 else 0
        self.self_attn = ShiftedWindowSelfAttentionBlock(d_model, d_head, cond_features, window_size, window_shift, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x