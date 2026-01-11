# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from train_gnn_nopg import (
    PTGraphDataset, RemapDataset, build_label_remap, get_first_valid_item,
    GraphClassifier, collate_graphs, topk_acc
)


def load_metrics_csv(path):
    rows = []
    with open(path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 7:
                continue
            rows.append([float(x) for x in parts])
    arr = np.array(rows, dtype=np.float32)  # columns: epoch, tr_loss, tr_acc, tr_top5, te_loss, te_acc, te_top5
    return arr


def plot_training_curves(csv_path, out_dir):
    arr = load_metrics_csv(csv_path)
    if arr.shape[0] == 0:
        raise RuntimeError("CSV пустой: %s" % csv_path)

    epoch = arr[:, 0]
    tr_loss, tr_acc, tr_top5 = arr[:, 1], arr[:, 2], arr[:, 3]
    te_loss, te_acc, te_top5 = arr[:, 4], arr[:, 5], arr[:, 6]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epoch, tr_loss, label="train_loss")
    plt.plot(epoch, te_loss, label="test_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_vs_epoch.png"), dpi=200)
    plt.close()

    # Acc + Top5
    plt.figure(figsize=(10, 5))
    plt.plot(epoch, tr_acc, label="train_acc")
    plt.plot(epoch, te_acc, label="test_acc")
    plt.plot(epoch, tr_top5, label="train_top5")
    plt.plot(epoch, te_top5, label="test_top5")
    plt.xlabel("epoch"); plt.ylabel("metric"); plt.title("Accuracy / Top-5 vs epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_top5_vs_epoch.png"), dpi=200)
    plt.close()


def evaluate_model(best_ckpt_path, device, bs=16, max_steps=100000):
    train_dir = os.path.join("..", "data", "Traingraph", "processed")
    test_dir  = os.path.join("..", "data", "Testgraph", "processed")

    train_base = PTGraphDataset(train_dir)
    test_base  = PTGraphDataset(test_dir)

    y_remap, labels_sorted = build_label_remap(train_base)
    num_classes = len(labels_sorted)

    train_ds = RemapDataset(train_base, y_remap)
    test_ds  = RemapDataset(test_base,  y_remap)

    it0 = get_first_valid_item(train_ds)
    x0, _, _ = it0
    in_dim = int(x0.size(1))

    model = GraphClassifier(in_dim=in_dim, hidden_dim=64, num_classes=num_classes, dropout=0.0).to(device)
    sd = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # accumulators
    total = 0
    correct1 = 0
    correct5 = 0
    loss_sum = 0.0

    # confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    idxs = list(range(len(test_ds)))
    steps = 0

    with torch.no_grad():
        for start in range(0, len(idxs), bs):
            if steps >= max_steps:
                break
            batch_ids = idxs[start:start + bs]
            items = [test_ds[i] for i in batch_ids]
            items = [it for it in items if it is not None]
            if len(items) == 0:
                continue

            X, EI, B, Y = collate_graphs(items)
            X, EI, B, Y = X.to(device), EI.to(device), B.to(device), Y.to(device)

            logits = model(X, EI, B, num_graphs=Y.size(0))
            loss = F.cross_entropy(logits, Y)

            pred = logits.argmax(dim=1)
            total += int(Y.size(0))
            correct1 += int((pred == Y).sum().item())
            correct5 += int(topk_acc(logits, Y, 5) * Y.size(0))
            loss_sum += float(loss.item()) * int(Y.size(0))

            for t, p in zip(Y.cpu().numpy().tolist(), pred.cpu().numpy().tolist()):
                cm[int(t), int(p)] += 1

            steps += 1

    acc1 = correct1 / float(total + 1e-12)
    acc5 = correct5 / float(total + 1e-12)
    avg_loss = loss_sum / float(total + 1e-12)

    return avg_loss, acc1, acc5, cm, num_classes


def macro_f1_from_cm(cm):
    # Macro F1 from confusion matrix
    num_classes = cm.shape[0]
    f1s = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def balanced_acc_from_cm(cm):
    # mean recall across classes
    recalls = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        rec = tp / (tp + fn + 1e-12)
        recalls.append(rec)
    return float(np.mean(recalls))


def plot_confusion_matrix(cm, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    out_dir = "report_figures"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1) curves from CSV
    if os.path.exists("training_metrics.csv"):
        plot_training_curves("training_metrics.csv", out_dir)
        print("Saved curves to:", out_dir)
    else:
        print("No training_metrics.csv found. Run training with CSV logging first.")

    # 2) metrics + confusion matrix from best checkpoint
    ckpt = "best_gnn.pt"
    if not os.path.exists(ckpt):
        print("No best_gnn.pt found. Train first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss, acc1, acc5, cm, C = evaluate_model(ckpt, device, bs=16)

    macro_f1 = macro_f1_from_cm(cm)
    bal_acc = balanced_acc_from_cm(cm)

    random_acc = 1.0 / float(C)
    improvement = acc1 - random_acc
    norm_acc = (acc1 - random_acc) / (1.0 - random_acc + 1e-12)

    plot_confusion_matrix(cm, os.path.join(out_dir, "confusion_matrix.png"))

    print("\n=== FINAL METRICS (TEST) ===")
    print("Classes (C):", C)
    print("CrossEntropy loss:", loss)
    print("Top-1 accuracy:", acc1)
    print("Top-5 accuracy:", acc5)
    print("Macro-F1:", macro_f1)
    print("Balanced accuracy:", bal_acc)
    print("Random baseline (1/C):", random_acc)
    print("Improvement over random:", improvement)
    print("Normalized accuracy:", norm_acc)
    print("Saved confusion matrix to:", os.path.join(out_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
