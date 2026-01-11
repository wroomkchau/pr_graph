# -*- coding: utf-8 -*-
"""
CPU-only educational stub for torch_scatter, compatible with older PyG calls.

Implements:
  - scatter_add
  - scatter_mean
  - scatter_min
  - scatter_max
  - scatter_std

Important:
  - Very slow (Python loops).
  - Supports the 6-argument call pattern used by torch_geometric:
        scatter_add(src, index, dim, out, dim_size, fill_value)
"""

import torch


def _infer_dim_size(index, dim_size):
    if dim_size is not None:
        return int(dim_size)
    if index is None or index.numel() == 0:
        return 1
    return int(index.max()) + 1


def scatter_add(src, index, dim=0, out=None, dim_size=None, fill_value=0):
    """
    Supports both signatures:
      scatter_add(src, index, dim=0, out=None, dim_size=None)
      scatter_add(src, index, dim=0, out=None, dim_size=None, fill_value=0)

    NOTE: fill_value is accepted for compatibility but not used.
    """
    # Flatten to 1D for simplicity (most PyG usage for aggregation is 1D per feature column)
    # If src is 2D (N, F), we aggregate along dim=0 by index (N,) into (dim_size, F)
    if dim != 0:
        # Minimal support: only dim=0
        raise NotImplementedError("stub scatter_add supports dim=0 only")

    dim_size = _infer_dim_size(index, dim_size)

    if src.dim() == 1:
        src_flat = src
        idx_flat = index.view(-1)
        if out is None:
            out = torch.zeros((dim_size,), dtype=src.dtype)
        for i in range(idx_flat.numel()):
            j = int(idx_flat[i])
            out[j] += src_flat[i]
        return out

    elif src.dim() == 2:
        N, F = src.size(0), src.size(1)
        idx_flat = index.view(-1)
        if out is None:
            out = torch.zeros((dim_size, F), dtype=src.dtype)
        for i in range(N):
            j = int(idx_flat[i])
            out[j] += src[i]
        return out

    else:
        # Fallback: flatten everything
        src_flat = src.contiguous().view(-1)
        idx_flat = index.contiguous().view(-1)
        if out is None:
            out = torch.zeros((dim_size,), dtype=src.dtype)
        for i in range(idx_flat.numel()):
            j = int(idx_flat[i])
            out[j] += src_flat[i]
        return out


def scatter_mean(src, index, dim=0, out=None, dim_size=None, fill_value=0):
    if dim != 0:
        raise NotImplementedError("stub scatter_mean supports dim=0 only")

    dim_size = _infer_dim_size(index, dim_size)

    if src.dim() == 1:
        sum_out = scatter_add(src, index, dim=dim, out=None, dim_size=dim_size, fill_value=fill_value)
        cnt = torch.zeros((dim_size,), dtype=src.dtype)
        idx_flat = index.view(-1)
        for i in range(idx_flat.numel()):
            cnt[int(idx_flat[i])] += 1
        cnt = torch.clamp(cnt, min=1)
        return sum_out / cnt

    elif src.dim() == 2:
        N, F = src.size(0), src.size(1)
        sum_out = scatter_add(src, index, dim=dim, out=None, dim_size=dim_size, fill_value=fill_value)
        cnt = torch.zeros((dim_size, 1), dtype=src.dtype)
        idx_flat = index.view(-1)
        for i in range(N):
            cnt[int(idx_flat[i])] += 1
        cnt = torch.clamp(cnt, min=1)
        return sum_out / cnt

    else:
        # Very minimal
        sum_out = scatter_add(src, index, dim=dim, out=None, dim_size=dim_size, fill_value=fill_value)
        return sum_out


def scatter_min(src, index, dim=0, out=None, dim_size=None, fill_value=0):
    if dim != 0:
        raise NotImplementedError("stub scatter_min supports dim=0 only")

    dim_size = _infer_dim_size(index, dim_size)

    if src.dim() == 1:
        if out is None:
            out = torch.full((dim_size,), float('inf'), dtype=src.dtype)
        arg = torch.full((dim_size,), -1, dtype=torch.long)
        idx_flat = index.view(-1)
        for i in range(idx_flat.numel()):
            j = int(idx_flat[i])
            if src[i] < out[j]:
                out[j] = src[i]
                arg[j] = i
        return out, arg

    elif src.dim() == 2:
        N, F = src.size(0), src.size(1)
        if out is None:
            out = torch.full((dim_size, F), float('inf'), dtype=src.dtype)
        arg = torch.full((dim_size, F), -1, dtype=torch.long)
        idx_flat = index.view(-1)
        for i in range(N):
            j = int(idx_flat[i])
            for f in range(F):
                if src[i, f] < out[j, f]:
                    out[j, f] = src[i, f]
                    arg[j, f] = i
        return out, arg

    else:
        raise NotImplementedError("stub scatter_min supports 1D/2D src only")


def scatter_max(src, index, dim=0, out=None, dim_size=None, fill_value=0):
    if dim != 0:
        raise NotImplementedError("stub scatter_max supports dim=0 only")

    dim_size = _infer_dim_size(index, dim_size)

    if src.dim() == 1:
        if out is None:
            out = torch.full((dim_size,), -float('inf'), dtype=src.dtype)
        arg = torch.full((dim_size,), -1, dtype=torch.long)
        idx_flat = index.view(-1)
        for i in range(idx_flat.numel()):
            j = int(idx_flat[i])
            if src[i] > out[j]:
                out[j] = src[i]
                arg[j] = i
        return out, arg

    elif src.dim() == 2:
        N, F = src.size(0), src.size(1)
        if out is None:
            out = torch.full((dim_size, F), -float('inf'), dtype=src.dtype)
        arg = torch.full((dim_size, F), -1, dtype=torch.long)
        idx_flat = index.view(-1)
        for i in range(N):
            j = int(idx_flat[i])
            for f in range(F):
                if src[i, f] > out[j, f]:
                    out[j, f] = src[i, f]
                    arg[j, f] = i
        return out, arg

    else:
        raise NotImplementedError("stub scatter_max supports 1D/2D src only")


def scatter_std(src, index, dim=0, out=None, dim_size=None, unbiased=True, fill_value=0):
    if dim != 0:
        raise NotImplementedError("stub scatter_std supports dim=0 only")

    dim_size = _infer_dim_size(index, dim_size)

    mean = scatter_mean(src, index, dim=dim, out=None, dim_size=dim_size, fill_value=fill_value)

    if src.dim() == 1:
        var = torch.zeros((dim_size,), dtype=src.dtype)
        cnt = torch.zeros((dim_size,), dtype=src.dtype)
        idx_flat = index.view(-1)
        for i in range(idx_flat.numel()):
            j = int(idx_flat[i])
            diff = src[i] - mean[j]
            var[j] += diff * diff
            cnt[j] += 1
        if unbiased:
            cnt = torch.clamp(cnt - 1, min=1)
        else:
            cnt = torch.clamp(cnt, min=1)
        return torch.sqrt(var / cnt)

    elif src.dim() == 2:
        N, F = src.size(0), src.size(1)
        var = torch.zeros((dim_size, F), dtype=src.dtype)
        cnt = torch.zeros((dim_size, 1), dtype=src.dtype)
        idx_flat = index.view(-1)
        for i in range(N):
            j = int(idx_flat[i])
            diff = src[i] - mean[j]
            var[j] += diff * diff
            cnt[j] += 1
        if unbiased:
            cnt = torch.clamp(cnt - 1, min=1)
        else:
            cnt = torch.clamp(cnt, min=1)
        return torch.sqrt(var / cnt)

    else:
        raise NotImplementedError("stub scatter_std supports 1D/2D src only")
