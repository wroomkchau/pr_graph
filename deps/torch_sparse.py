# -*- coding: utf-8 -*-
"""
Educational stub for torch_sparse.

This is NOT a real torch_sparse implementation.
It provides a minimal subset of API so that torch_geometric can import
in a constrained CPU-only environment.

Includes: SparseTensor (very minimal), matmul, sum, mul, coalesce.
"""

import torch


def coalesce(index, value=None, m=None, n=None, op='add'):
    """
    Coalesce (merge) duplicate entries in a COO-style sparse representation.

    Parameters
    ----------
    index: LongTensor shape [2, E]
    value: Tensor shape [E] or [E, *] or None
    m, n: optional sizes (ignored in this stub)
    op: 'add' or 'max' (only 'add' is implemented safely)

    Returns
    -------
    index_coalesced: LongTensor [2, E']
    value_coalesced: Tensor [E', ...] or None
    """
    if index is None:
        raise ValueError("index is required")
    if index.dim() != 2 or index.size(0) != 2:
        raise ValueError("index must have shape [2, E]")

    E = int(index.size(1))
    if E == 0:
        return index, value

    row = index[0].contiguous().view(-1)
    col = index[1].contiguous().view(-1)

    # Build keys to identify duplicates: key = row * big + col
    # big chosen as (max_col+1) to avoid collisions
    max_col = int(col.max()) if col.numel() > 0 else 0
    big = max_col + 1
    keys = row * big + col

    # Sort by keys
    sorted_keys, perm = torch.sort(keys)
    row_s = row[perm]
    col_s = col[perm]

    if value is not None:
        val_s = value[perm]
    else:
        val_s = None

    # Merge duplicates
    out_rows = []
    out_cols = []
    out_vals = [] if val_s is not None else None

    i = 0
    while i < E:
        r = int(row_s[i])
        c = int(col_s[i])

        if val_s is None:
            # keep one copy
            out_rows.append(r)
            out_cols.append(c)
            # skip duplicates
            j = i + 1
            while j < E and int(row_s[j]) == r and int(col_s[j]) == c:
                j += 1
            i = j
        else:
            # aggregate values for duplicates
            agg = val_s[i].clone()
            j = i + 1
            while j < E and int(row_s[j]) == r and int(col_s[j]) == c:
                if op == 'add' or op == 'sum':
                    agg = agg + val_s[j]
                elif op == 'max':
                    agg = torch.max(agg, val_s[j])
                else:
                    # fallback to add
                    agg = agg + val_s[j]
                j += 1
            out_rows.append(r)
            out_cols.append(c)
            out_vals.append(agg)
            i = j

    idx = torch.tensor([out_rows, out_cols], dtype=torch.long)
    if out_vals is None:
        return idx, None

    # Stack values back
    try:
        val_out = torch.stack(out_vals, dim=0)
    except Exception:
        # if scalars
        val_out = torch.tensor(out_vals)
    return idx, val_out


class SparseTensor(object):
    def __init__(self, row=None, col=None, value=None, sparse_sizes=None):
        self.row = row
        self.col = col
        self.value = value
        self.sparse_sizes = sparse_sizes

    @staticmethod
    def from_edge_index(edge_index, edge_attr=None, sparse_sizes=None):
        row = edge_index[0]
        col = edge_index[1]
        return SparseTensor(row=row, col=col, value=edge_attr, sparse_sizes=sparse_sizes)

    def to_dense(self):
        if self.sparse_sizes is None:
            raise RuntimeError("sparse_sizes required for to_dense")
        mat = torch.zeros(self.sparse_sizes)
        if self.value is None:
            for r, c in zip(self.row.tolist(), self.col.tolist()):
                mat[r, c] = 1
        else:
            # value might be vector; write first element if so
            for i, (r, c) in enumerate(zip(self.row.tolist(), self.col.tolist())):
                v = self.value[i]
                mat[r, c] = v if torch.is_tensor(v) and v.dim() == 0 else (v[0] if torch.is_tensor(v) else v)
        return mat

    def matmul(self, x):
        dense = self.to_dense()
        return torch.matmul(dense, x)

    def __repr__(self):
        return "FakeSparseTensor(size={})".format(self.sparse_sizes)


def matmul(sparse, dense):
    return sparse.matmul(dense)


def sum(sparse, dim=0):
    return sparse.to_dense().sum(dim)


def mul(sparse, other):
    return sparse.to_dense() * other

def spspmm(indexA, valueA, indexB, valueB, m, k, n):
    """
    Very slow CPU fallback for sparse-sparse matrix multiplication in COO format.

    A: (m x k) with (indexA, valueA)
    B: (k x n) with (indexB, valueB)
    Returns C = A @ B in COO: (indexC, valueC)

    Notes:
    - This is an educational stub, not optimized.
    - Assumes value tensors are 1D (E,) or None (treated as 1s).
    """
    if valueA is None:
        valueA = torch.ones(indexA.size(1), dtype=torch.float)
    if valueB is None:
        valueB = torch.ones(indexB.size(1), dtype=torch.float)

    # Build adjacency dict for A rows: A_row[i] = list of (col, val)
    A_row = {}
    for e in range(indexA.size(1)):
        r = int(indexA[0, e])
        c = int(indexA[1, e])
        v = valueA[e]
        A_row.setdefault(r, []).append((c, v))

    # Build adjacency dict for B rows (since multiplying A(r, t) * B(t, c)):
    B_row = {}
    for e in range(indexB.size(1)):
        r = int(indexB[0, e])  # r is "t"
        c = int(indexB[1, e])
        v = valueB[e]
        B_row.setdefault(r, []).append((c, v))

    # Multiply: for each (r -> t) in A and (t -> c) in B accumulate into (r, c)
    out = {}  # (r,c) -> value
    for r, a_list in A_row.items():
        for t, va in a_list:
            if t not in B_row:
                continue
            for c, vb in B_row[t]:
                key = (r, c)
                out[key] = out.get(key, 0.0) + float(va) * float(vb)

    if not out:
        # return an explicit zero entry to keep downstream code alive
        indexC = torch.tensor([[0], [0]], dtype=torch.long)
        valueC = torch.tensor([0.0], dtype=torch.float)
        return indexC, valueC

    rows = []
    cols = []
    vals = []
    for (r, c), v in out.items():
        rows.append(r)
        cols.append(c)
        vals.append(v)

    indexC = torch.tensor([rows, cols], dtype=torch.long)
    valueC = torch.tensor(vals, dtype=torch.float)

    # Coalesce duplicates / sort (safe)
    indexC, valueC = coalesce(indexC, valueC, m=m, n=n, op='add')
    return indexC, valueC

def spmm(index, value, m, n, matrix):
    """
    Very slow CPU fallback for sparse-dense matrix multiplication.

    index: LongTensor [2, E] (row, col)
    value: Tensor [E] or None (treated as 1s)
    m, n: sparse matrix shape (m x n)
    matrix: dense matrix shape (n x F)  (or vector (n,))

    returns: dense (m x F)
    """
    if value is None:
        value = torch.ones(index.size(1), dtype=matrix.dtype)

    # Ensure matrix is 2D
    if matrix.dim() == 1:
        matrix2 = matrix.view(-1, 1)
    else:
        matrix2 = matrix

    out = torch.zeros((m, matrix2.size(1)), dtype=matrix2.dtype)

    row = index[0].contiguous().view(-1)
    col = index[1].contiguous().view(-1)
    val = value.contiguous().view(-1)

    for e in range(row.numel()):
        r = int(row[e])
        c = int(col[e])
        out[r] += val[e] * matrix2[c]

    # If original was vector, return vector
    if matrix.dim() == 1:
        return out.view(-1)
    return out
