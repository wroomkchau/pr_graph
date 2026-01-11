# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import glob
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


# Dataset: reads dict .pt

class PTGraphDataset(object):
    def __init__(self, processed_dir, max_files=None):
        self.files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
        if max_files is not None and int(max_files) > 0:
            self.files = self.files[:int(max_files)]

    def __len__(self):
        return len(self.files)

    def load_raw(self, idx):
        return torch.load(self.files[idx], map_location="cpu")

    def __getitem__(self, idx):
        d = self.load_raw(idx)
        if ("x" not in d) or ("edge_index" not in d) or ("y" not in d):
            raise KeyError("Each .pt must contain keys: 'x', 'edge_index', 'y'")
        
        # Each sample is a graph saved as dict in .pt
        x = d["x"].float()
        edge_index = d["edge_index"].long()
        y = int(d["y"].long().view(-1)[0].item())

        # edge_index shape must be [2, E]. If saved as [E, 2], transpose it.
        if edge_index.dim() != 2:
            raise ValueError("edge_index must be 2D, got shape %s" % (tuple(edge_index.shape),))

        if edge_index.size(0) == 2:
            pass
        elif edge_index.size(1) == 2:
            edge_index = edge_index.t().contiguous()
        else:
            raise ValueError("edge_index must be [2,E] or [E,2], got %s" % (tuple(edge_index.shape),))

        if x.dim() != 2:
            raise ValueError("x must be [N,F], got %s" % (tuple(x.shape),))

        n = x.size(0)
        if edge_index.numel() > 0:
            edge_index = edge_index.clamp(min=0, max=max(n - 1, 0))

        return x, edge_index, y


class RemapDataset(object):
    def __init__(self, base_ds, y_remap):
        self.base = base_ds
        self.map_old_to_new = y_remap

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, edge_index, y_old = self.base[idx]
        y_old = int(y_old)
        if y_old not in self.map_old_to_new:
            return None
        y_new = int(self.map_old_to_new[y_old])
        return x.float(), edge_index.long(), y_new


def build_label_remap(train_base):
    labels = set()
    for i in range(len(train_base)):
        d = train_base.load_raw(i)
        y = int(d["y"].long().view(-1)[0].item())
        labels.add(y)
    labels = sorted(list(labels))
    return {old: new for new, old in enumerate(labels)}, labels


def compute_class_weights(ds, num_classes, limit=20000):
    counts = [0 for _ in range(num_classes)]
    n = min(len(ds), int(limit))
    for i in range(n):
        it = ds[i]
        if it is None:
            continue
        _, _, y = it
        y = int(y)
        if 0 <= y < num_classes:
            counts[y] += 1

    counts_t = torch.tensor(counts, dtype=torch.float32)
    w = 1.0 / torch.sqrt(counts_t + 1.0)
    w = w / (w.mean() + 1e-12)
    w = torch.clamp(w, 0.25, 4.0)
    return w, counts


def get_first_valid_item(ds):
    for i in range(len(ds)):
        it = ds[i]
        if it is not None:
            return it
    return None



# Batching graphs

def collate_graphs(items):
    xs = []
    eis = []
    batch = []
    ys = []
    node_offset = 0

    for bi, (x, ei, y) in enumerate(items):
        n = x.size(0)
        xs.append(x)

        if ei.numel() > 0:
            ei2 = ei.clone()
            ei2[0] += node_offset
            ei2[1] += node_offset
            eis.append(ei2)

        batch.append(torch.full((n,), bi, dtype=torch.long))
        ys.append(int(y))
        node_offset += n

    X = torch.cat(xs, dim=0)
    EI = torch.cat(eis, dim=1) if len(eis) else torch.zeros((2, 0), dtype=torch.long)
    batch = torch.cat(batch, dim=0)
    Y = torch.tensor(ys, dtype=torch.long)
    return X, EI, batch, Y



# GCN without torch_scatter

def gcn_norm(edge_index, num_nodes, device):
    row = edge_index[0]
    col = edge_index[1]

    self_loops = torch.arange(0, num_nodes, dtype=torch.long, device=device)
    row = torch.cat([row, self_loops], dim=0)
    col = torch.cat([col, self_loops], dim=0)

    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float32, device=device))
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)

    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    ei = torch.stack([row, col], dim=0)
    return ei, norm


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        n = x.size(0)
        ei, norm = gcn_norm(edge_index, n, x.device)
        row = ei[0]
        col = ei[1]

        h = self.lin(x)
        out = torch.zeros_like(h)
        msg = h[col] * norm.view(-1, 1)
        out.index_add_(0, row, msg)
        return out


def pool_mean_max(x, batch, num_graphs):
    out_sum = torch.zeros(num_graphs, x.size(1), device=x.device)
    out_sum.index_add_(0, batch, x)
    cnt = torch.zeros(num_graphs, device=x.device)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float32, device=x.device))
    mean = out_sum / (cnt.view(-1, 1) + 1e-12)

    mx = []
    for g in range(num_graphs):
        mask = (batch == g)
        mx.append(x[mask].max(dim=0)[0])
    mx = torch.stack(mx, dim=0)

    return torch.cat([mean, mx], dim=1)


class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.1):
        super(GraphClassifier, self).__init__()
        self.g1 = GCNLayer(in_dim, hidden_dim)
        self.g2 = GCNLayer(hidden_dim, hidden_dim)
        self.g3 = GCNLayer(hidden_dim, hidden_dim)
        self.bn1 = Identity()
        self.bn2 = Identity()
        self.bn3 = Identity()
        self.dropout = float(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, X, EI, batch, num_graphs):
        x = self.g1(X, EI)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.g2(x, EI)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.g3(x, EI)
        x = self.bn3(x)
        x = F.relu(x)

        g = pool_mean_max(x, batch, num_graphs)
        return self.mlp(g)


def topk_acc(logits, y, k):
    k = min(k, logits.size(1))
    _, idx = torch.topk(logits, k=k, dim=1)
    return float((idx == y.view(-1, 1)).any(dim=1).float().mean().item())


def run_epoch(model, ds, optimizer, device, train, class_w, bs, max_steps, log_every):
    if train:
        model.train()
    else:
        model.eval()

    idxs = list(range(len(ds)))
    if train:
        random.shuffle(idxs)

    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    seen = 0
    steps = 0

    for start in range(0, len(idxs), bs):
        if max_steps is not None and steps >= max_steps:
            break

        batch_ids = idxs[start:start + bs]
        items = [ds[i] for i in batch_ids]
        items = [it for it in items if it is not None]
        if len(items) == 0:
            continue

        X, EI, B, Y = collate_graphs(items)
        X, EI, B, Y = X.to(device), EI.to(device), B.to(device), Y.to(device)

        if train:
            logits = model(X, EI, B, num_graphs=Y.size(0))
            loss = F.cross_entropy(logits, Y, weight=class_w.to(device) if class_w is not None else None)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(X, EI, B, num_graphs=Y.size(0))
                loss = F.cross_entropy(logits, Y)

        acc1 = float((logits.argmax(dim=1) == Y).float().mean().item())
        acc5 = topk_acc(logits, Y, 5)

        total_loss += float(loss.item()) * Y.size(0)
        total_acc1 += acc1 * Y.size(0)
        total_acc5 += acc5 * Y.size(0)
        seen += Y.size(0)

        steps += 1
        if train and log_every is not None and log_every > 0 and (steps % log_every == 0):
            print("  step %d  loss=%.4f  acc=%.3f  top5=%.3f" % (steps, float(loss.item()), acc1, acc5))

    if seen == 0:
        return 0.0, 0.0, 0.0
    return total_loss / seen, total_acc1 / seen, total_acc5 / seen


def main():
    BS = int(os.environ.get("BS", "16"))
    MAX_STEPS = int(os.environ.get("MAX_STEPS", "1000"))
    TEST_STEPS = int(os.environ.get("TEST_STEPS", "200"))
    EPOCHS = int(os.environ.get("EPOCHS", "80"))
    LR = float(os.environ.get("LR", "0.001"))
    LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))

    USE_CLASS_WEIGHTS = int(os.environ.get("USE_CLASS_WEIGHTS", "0"))
    OVERFIT_ONE_BATCH = int(os.environ.get("OVERFIT_ONE_BATCH", "0"))
    PRINT_DATA_CHECK = int(os.environ.get("PRINT_DATA_CHECK", "1"))

    random.seed(1)
    torch.manual_seed(1)

    train_dir = os.path.join("..", "data", "Traingraph", "processed")
    test_dir = os.path.join("..", "data", "Testgraph", "processed")

    train_base = PTGraphDataset(train_dir)
    test_base = PTGraphDataset(test_dir)

    print("Train files:", len(train_base))
    print("Test  files:", len(test_base))

    y_remap, labels_sorted = build_label_remap(train_base)
    num_classes = len(labels_sorted)
    print("Detected num_classes:", num_classes)

    train_ds = RemapDataset(train_base, y_remap)
    test_ds = RemapDataset(test_base, y_remap)

    it0 = get_first_valid_item(train_ds)
    if it0 is None:
        raise RuntimeError("No valid training samples found (train_ds returned only None).")

    x0, ei0, y0 = it0
    in_dim = int(x0.size(1))
    print("Feature dim:", in_dim)

    if PRINT_DATA_CHECK:
        print("Sample check:")
        print("  x:", tuple(x0.shape), x0.dtype)
        print("  edge_index:", tuple(ei0.shape), ei0.dtype,
              "min/max:",
              (int(ei0.min().item()) if ei0.numel() else None,
               int(ei0.max().item()) if ei0.numel() else None))
        print("  y:", y0, "mapped classes 0..", num_classes - 1)

    class_w = None
    if USE_CLASS_WEIGHTS:
        class_w, counts = compute_class_weights(train_ds, num_classes, limit=20000)
        print("Train label counts (first 30):", counts[:30])
        print("Class weights (first 30):", class_w[:30].tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphClassifier(in_dim=in_dim, hidden_dim=64, num_classes=num_classes, dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0)


    print("Config: BS=%d MAX_STEPS=%d TEST_STEPS=%d EPOCHS=%d LR=%g USE_CLASS_WEIGHTS=%d OVERFIT_ONE_BATCH=%d" %
          (BS, MAX_STEPS, TEST_STEPS, EPOCHS, LR, USE_CLASS_WEIGHTS, OVERFIT_ONE_BATCH))
    
    metrics_path = "training_metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,train_acc,train_top5,test_loss,test_acc,test_top5\n")


    if OVERFIT_ONE_BATCH:
        print("\n[OVERFIT_ONE_BATCH] Training on ONE batch only.\n")
        items = []
        i = 0
        while len(items) < BS and i < len(train_ds):
            it = train_ds[i]
            if it is not None:
                items.append(it)
            i += 1
        if len(items) == 0:
            raise RuntimeError("No items for overfit batch.")

        X, EI, B, Y = collate_graphs(items)
        X, EI, B, Y = X.to(device), EI.to(device), B.to(device), Y.to(device)

        model.train()
        for step in range(1, 501):
            logits = model(X, EI, B, num_graphs=Y.size(0))
            loss = F.cross_entropy(logits, Y, weight=class_w.to(device) if class_w is not None else None)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 50 == 0:
                acc = float((logits.argmax(dim=1) == Y).float().mean().item())
                print("step %d | loss %.4f | acc %.3f" % (step, float(loss.item()), acc))
        return

    best = 0.0
    bad = 0
    patience = 10

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_top5 = run_epoch(model, train_ds, opt, device, True, class_w, BS, MAX_STEPS, LOG_EVERY)
        te_loss, te_acc, te_top5 = run_epoch(model, test_ds, opt, device, False, None, BS, TEST_STEPS, 0)

        if te_acc > best:
            best = te_acc
            bad = 0
            torch.save(model.state_dict(), "best_gnn.pt")
        else:
            bad += 1

        print("epoch %d | train loss %.4f acc %.3f top5 %.3f | test loss %.4f acc %.3f top5 %.3f | best %.3f" %
              (epoch, tr_loss, tr_acc, tr_top5, te_loss, te_acc, te_top5, best))
        
        with open(metrics_path, "a") as f:
            f.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" %
                (epoch, tr_loss, tr_acc, tr_top5, te_loss, te_acc, te_top5))


        if bad >= patience:
            print("Early stop. Best:", best)
            break

    print("DONE. Best test_acc:", best)
    print("Saved: best_gnn.pt")
    print("Time:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
