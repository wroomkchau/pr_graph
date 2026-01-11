# -*- coding: utf-8 -*-
"""
Educational stub for torch_cluster.

Provides minimal implementations of knn_graph and radius_graph
for small graphs and CPU-only execution.
"""

import torch


def knn_graph(x, k, batch=None, loop=False):
    """
    Very simple CPU fallback for k-NN graph.
    x: Tensor [N, F]
    returns edge_index [2, E]
    """
    N = x.size(0)
    edge_src = []
    edge_dst = []

    # Compute pairwise distances (VERY slow, but OK for small N)
    for i in range(N):
        dists = []
        for j in range(N):
            if i == j and not loop:
                continue
            dist = torch.norm(x[i] - x[j])
            dists.append((dist, j))

        dists.sort(key=lambda t: t[0])
        for _, j in dists[:k]:
            edge_src.append(i)
            edge_dst.append(j)

    if len(edge_src) == 0:
        edge_src = [0]
        edge_dst = [0]

    return torch.tensor([edge_src, edge_dst], dtype=torch.long)


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32):
    """
    Very simple CPU fallback for radius graph.
    x: Tensor [N, F]
    returns edge_index [2, E]
    """
    N = x.size(0)
    edge_src = []
    edge_dst = []

    for i in range(N):
        neighbors = 0
        for j in range(N):
            if i == j and not loop:
                continue
            dist = torch.norm(x[i] - x[j])
            if dist <= r:
                edge_src.append(i)
                edge_dst.append(j)
                neighbors += 1
                if neighbors >= max_num_neighbors:
                    break

    if len(edge_src) == 0:
        edge_src = [0]
        edge_dst = [0]

    return torch.tensor([edge_src, edge_dst], dtype=torch.long)

def graclus_cluster(edge_index, weight=None, num_nodes=None):
    """
    Educational stub for graclus_cluster.

    Returns a cluster assignment vector of shape [num_nodes].
    Fallback: each node is its own cluster.
    """
    if num_nodes is None:
        if edge_index is None or edge_index.numel() == 0:
            num_nodes = 1
        else:
            num_nodes = int(edge_index.max()) + 1

    return torch.arange(num_nodes, dtype=torch.long)

def grid_cluster(pos, size, batch=None, start=None, end=None):
    """
    Educational CPU fallback for grid_cluster.

    pos: Tensor [N, F] (usually coordinates)
    size: float or Tensor [F] - grid cell size per dimension
    returns: LongTensor [N] cluster id per node
    """
    if pos is None or pos.numel() == 0:
        return torch.zeros((1,), dtype=torch.long)

    if not torch.is_tensor(size):
        size = torch.tensor([float(size)] * pos.size(1))

    size = size.to(pos.dtype)
    coords = torch.floor(pos / size).to(torch.long)

    # map multi-dim coords -> single id (simple hashing)
    # id = (((c0 * M) + c1) * M + c2) ...
    M = 1000003  # large prime-ish base
    ids = coords[:, 0].clone()
    for d in range(1, coords.size(1)):
        ids = ids * M + coords[:, d]

    # if batch is provided, make clusters unique per batch
    if batch is not None:
        ids = ids + batch.to(torch.long) * (M ** 2)

    # compress ids to 0..num_clusters-1
    unique, inv = torch.unique(ids, return_inverse=True)
    return inv

def fps(pos, batch=None, ratio=0.5, random_start=True):
    """
    Educational CPU fallback for farthest point sampling (FPS).

    pos: Tensor [N, F]
    ratio: fraction of points to sample (0..1]
    returns: LongTensor indices of sampled points
    """
    if pos is None or pos.numel() == 0:
        return torch.zeros((1,), dtype=torch.long)

    N = pos.size(0)
    M = int(max(1, round(float(ratio) * N)))
    M = min(M, N)

    # Start point
    if random_start:
        start = int(torch.randint(low=0, high=N, size=(1,)).item())
    else:
        start = 0

    selected = [start]

    # distances to the nearest selected point
    dist = torch.full((N,), 1e30, dtype=pos.dtype)

    for _ in range(1, M):
        last = selected[-1]
        d = torch.norm(pos - pos[last], dim=1)
        dist = torch.min(dist, d)

        # pick farthest
        nxt = int(torch.argmax(dist).item())
        selected.append(nxt)

    return torch.tensor(selected, dtype=torch.long)

def knn(x, y, k, batch_x=None, batch_y=None, cosine=False):
    """
    Educational CPU fallback for k-NN between two point sets.

    x: Tensor [N_x, F] (database)
    y: Tensor [N_y, F] (queries)
    returns: LongTensor [2, N_y * k] with (idx_x, idx_y)
    """
    Nx = x.size(0)
    Ny = y.size(0)

    edge_src = []
    edge_dst = []

    for i in range(Ny):
        dists = []
        for j in range(Nx):
            if cosine:
                # 1 - cosine similarity
                num = torch.dot(y[i], x[j])
                den = (torch.norm(y[i]) * torch.norm(x[j]) + 1e-12)
                dist = 1.0 - (num / den)
            else:
                dist = torch.norm(y[i] - x[j])
            dists.append((dist, j))
        dists.sort(key=lambda t: t[0])

        for _, j in dists[:k]:
            edge_src.append(j)  # from x
            edge_dst.append(i)  # to y

    if len(edge_src) == 0:
        edge_src = [0]
        edge_dst = [0]

    return torch.tensor([edge_src, edge_dst], dtype=torch.long)

def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    """
    Educational CPU fallback for radius search between two point sets.

    x: Tensor [N_x, F] (database)
    y: Tensor [N_y, F] (queries)
    r: float radius
    returns: LongTensor [2, E] with (idx_x, idx_y)
    """
    Nx = x.size(0)
    Ny = y.size(0)

    edge_src = []
    edge_dst = []

    r = float(r)

    for i in range(Ny):
        neighbors = 0
        for j in range(Nx):
            dist = torch.norm(y[i] - x[j])
            if float(dist) <= r:
                edge_src.append(j)  # from x
                edge_dst.append(i)  # to y
                neighbors += 1
                if neighbors >= int(max_num_neighbors):
                    break

    if len(edge_src) == 0:
        edge_src = [0]
        edge_dst = [0]

    return torch.tensor([edge_src, edge_dst], dtype=torch.long)

def nearest(x, y, batch_x=None, batch_y=None):
    """
    Educational CPU fallback for nearest neighbor lookup.

    x: Tensor [N_x, F]
    y: Tensor [N_y, F]
    returns: LongTensor [N_y] where out[i] = argmin_j ||y[i] - x[j]||
    """
    Nx = x.size(0)
    Ny = y.size(0)

    out = torch.zeros((Ny,), dtype=torch.long)

    for i in range(Ny):
        best_j = 0
        best_d = None
        for j in range(Nx):
            d = torch.norm(y[i] - x[j])
            d_val = float(d)
            if best_d is None or d_val < best_d:
                best_d = d_val
                best_j = j
        out[i] = best_j

    return out
