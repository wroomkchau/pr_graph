```markdown
# Reproducibility Study: Graph-Based Learning for Neuromorphic Vision (ASL-DVS)

This repository is a **reproducibility + engineering study** based on the ICCV 2019 paper:

> Yin Bi, Aaron Chadha, Alhabib Abbas, Eirina Bourtsoulatze, Yiannis Andreopoulos  
> **Graph-Based Object Classification for Neuromorphic Vision Sensing**, ICCV 2019.

The original work proposes a graph-based pipeline that converts neuromorphic events to graphs and applies graph convolutional learning for object classification.

In this project I focused on the **most fragile part of reproduction**:  
**robust AEDAT preprocessing → windowing → graph construction → training pipeline**  
under realistic constraints (CPU-only training, missing PyG CUDA dependencies).

---

## Key Contributions (What’s new in this repo)

### 1) Robust AEDAT preprocessing pipeline (`code/preprocess_window.py`)
AEDAT files are not fully standardized and may vary across sensors/dataset sources.  
To make preprocessing reproducible, this repo implements:

- **AEDAT header skipping**
- **Endian detection**: decode both BE/LE and select the one with highest timestamp monotonicity
- **Address decoder selection** using spatial heuristics (event-cloud sanity checks)
- **Degenerate stream rejection** (skip broken files / dead coordinate streams)
- **Fixed-event temporal windowing** for sample generation:
  - long streams → many overlapping windows  
  - each window → one `.pt` graph sample
- **Graph construction without `pos`** (compatible with minimal `.pt` format)

Output graph samples contain:
- `x`: node features `[N,4] = [x_norm, y_norm, t_norm, polarity]`
- `edge_index`: `[2,E]`, bidirectional temporal chain edges
- `y`: label

Preprocessing is **resume-safe**: if a `.pt` sample already exists, it will be skipped.

---

### 2) Training without PyTorch-Geometric CUDA dependencies
The original implementation relies on:
- `torch_geometric`
- `torch_scatter`
- `torch_sparse`
- `SplineConv` (and other CUDA-heavy operators)

In this project:
- training scripts are designed to run **without `torch_scatter` / `torch_sparse`**
- graph batching + message passing + pooling are implemented in **pure PyTorch**
  using `index_add_` and lightweight pooling

This enables **CPU-only execution** and makes the code easier to run on standard laptops.

---

### 3) Reproducibility-focused evaluation
Training is evaluated using:
- **Top-1 accuracy**
- **Top-5 accuracy**
- early stopping (optional)
- runtime control knobs (`MAX_STEPS`, `TEST_STEPS`) for stable training on CPU

---

## Dataset: ASL-DVS (unchanged)

I use the dataset referenced by the original paper:

- Dropbox: https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0  
- Google Drive: https://drive.google.com/drive/folders/1tK5OY3pkjppYwAnLF8bnxGdaFbYEA8iY?usp=sharing

ASL-DVS contains **24 classes** (A–Y excluding J), recorded using a **DAVIS240c** neuromorphic sensor.  
The dataset contains multiple subjects (Subject1–Subject5) with natural variance.

---

## Repository Structure

```

NVS2Graph/
├── code/
│   ├── preprocess_window.py      # AEDAT → windows → graph .pt
│   ├── train_baseline.py         # baseline classifier (fast sanity check)
│   ├── train_gnn_nopg.py         # pure PyTorch GNN (no PyG scatter deps)
│   ├── fake_deps/                # optional stubs (if needed)
│   └── ...
├── data/
│   ├── AEDAT/Subject*/           # raw .aedat files
│   ├── Traingraph/processed/     # generated TRAIN*.pt graphs
│   └── Testgraph/processed/      # generated TEST*.pt graphs
└── Results/

```

---

## Setup / Requirements

This project was tested in a lightweight environment:
- Python 2.7 (to match the original repo)
- PyTorch (CPU is enough)
- NumPy

> Note: You do NOT need to install `torch_scatter`, `torch_sparse`, or `SplineConv`
> to run the preprocessing and the simplified training scripts.

---

## How to Run

### Step 1 — Put the dataset in the expected folders
After downloading ASL-DVS, place AEDAT files like this:

```

../data/AEDAT/Subject1/*.aedat
../data/AEDAT/Subject2/*.aedat
...
../data/AEDAT/Subject5/*.aedat

````

Filename assumptions:
- usually `a.aedat`, `b.aedat`, ...
- duplicates may include digits: `g1.aedat`, `z12.aedat`, etc.
- label is inferred from the **first letter** of the filename

---

### Step 2 — Preprocess AEDAT into graph windows
From `code/`:

```bash
cd code
python preprocess_window.py
````

Outputs:

* `../data/Traingraph/processed/*.pt`
* `../data/Testgraph/processed/*.pt`

---

### Step 3 — Train a fast baseline (sanity check)

```bash
python train_baseline.py
```

---

### Step 4 — Train a pure PyTorch GNN (no PyG CUDA)

Example run:

```bash
BS=16 MAX_STEPS=800 TEST_STEPS=200 EPOCHS=60 python train_gnn_nopg.py
```

Main knobs:

* `BS`: batch size (#graphs per batch)
* `MAX_STEPS`: train batches per epoch (runtime control)
* `TEST_STEPS`: test batches per epoch
* `EPOCHS`: max epochs

Outputs:

* best checkpoint: `best_gnn.pt`

---

## Metrics

Top-1 Accuracy:

[
Accuracy = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\hat{y}_i = y_i)
]

Top-5 Accuracy:

[
Top5 = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(y_i \in Top5(\hat{y}_i))
]

Top-5 is informative in this setting because:

* the number of classes is large relative to sample count
* the dataset is challenging and highly variable

---

## Notes on Results

This repository focuses on **reproducibility under constraints**, not on matching GPU-optimized PyG performance.

Observed behavior:

* stable training convergence
* best top-1 accuracy typically around **36–38%**
* top-5 accuracy is higher (often substantially above chance)

Main limitations:

* simplified model architecture (no `SplineConv`, no radius graph)
* CPU-only training
* limited examples per class / strong intra-class variance
* sensitivity of performance to AEDAT decoding correctness

---

## Citation

If you use this dataset or refer to the original method, please cite:

**MLA**

> Bi, Yin, et al. “Graph-Based Object Classification for Neuromorphic Vision Sensing.” ICCV 2019.

**BibTeX**

```bibtex
@inproceedings{bi2019graph,
  title={Graph-based Object Classification for Neuromorphic Vision Sensing},
  author={Bi, Y and Chadha, A and Abbas, A and Bourtsoulatze, E and Andreopoulos, Y},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019},
  organization={IEEE},
  doi={10.1109/ICCV.2019.00058}
}
```

---

## Acknowledgements

This project is based on the ICCV 2019 work by Bi et al. and their released dataset.
This fork focuses on reproducibility and robust preprocessing to enable training even when the full PyTorch-Geometric stack is unavailable.

---

## Contact

For issues related to this reproduction study:

* open a GitHub issue in this repository

Original paper contact (dataset/authors):

* Yin Bi: [yin.bi.16@ucl.ac.uk](mailto:yin.bi.16@ucl.ac.uk)

```
::contentReference[oaicite:0]{index=0}
```
