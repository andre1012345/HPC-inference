# Parallel Ensemble Inference Engine
### Hybrid MPI + OpenMP · HPC Project · University of Messina

A two-level parallel inference engine for a soft-voting ensemble classifier
(K-Nearest Neighbours, Naive Bayes, Logistic Regression).
MPI distributes the test set across processes; OpenMP parallelises each local slice
across threads. Total concurrency is **p × t** workers.

According to **Flynn's taxonomy** this is a **MIMD** (Multiple Instruction, Multiple Data) architecture: every core executes its own instruction stream on its own data slice independently. The parallelisation strategy applies **hybrid decomposition** at two levels — MPI performs coarse-grained data decomposition across independent address spaces; OpenMP performs fine-grained data decomposition within each rank's shared-memory domain.

> **Full write-up:** [`report/hpc_inference_report.tex`](./report/hpc_inference_report.tex) (compile on Overleaf)

---

## Results at a glance

### Strong scaling — fixed problem size (20,000 test samples)

Baseline T₁ = 20.60 s (hybrid binary, p=1, t=1). Each row is the median of 3 cold-cache runs.

| Ranks (p) | Threads (t) | p · t | Time (s) | Speedup | Efficiency |
|:---------:|:-----------:|:-----:|:--------:|:-------:|:----------:|
| 1 | 1 | 1 | 20.60 | 1.00× | 1.00 |
| 1 | 2 | 2 | 11.27 | 1.83× | 0.91 |
| 2 | 1 | 2 | 11.10 | 1.86× | 0.93 |
| 2 | 2 | 4 | 7.93 | 2.60× | 0.65 |
| 3 | 1 | 3 | 8.28 | 2.49× | 0.83 |
| 3 | 2 | 6 | 7.05 | 2.92× | 0.49 |
| 4 | 1 | 4 | 7.21 | 2.86× | 0.71 |
| **4** | **2** | **8** | **5.40** | **3.81×** | **0.48** |

Efficiency decreases as workers increase, reflecting the growing share of serial work
(MPI coordination, rank-0 gather, accuracy reduction) relative to total compute time.
The KNN working set (50,000 × 20 × 8 bytes ≈ 8 MB) exceeds the 6 MB L3 cache,
so every rank streams training data from DRAM — memory bandwidth is the primary
bottleneck limiting further speedup.

**Scalability** (E(n₂)/E(n₁)) measures how well efficiency is retained as workers increase. For the t=1 MPI series:

| Transition | E before | E after | Retention |
|:----------:|:--------:|:-------:|:---------:|
| p=1 → p=2  |  1.000   |  0.928  |   92.8%   |
| p=2 → p=3  |  0.928   |  0.830  |   89.4%   |
| p=3 → p=4  |  0.830   |  0.714  |   86.0%   |

Each step retains roughly 86–93% of prior efficiency. The declining trend confirms progressive memory-bandwidth saturation as more ranks stream independent 8 MB training sets from the shared DRAM bus.

### Weak scaling — fixed work per rank (1,000 samples/rank, t=1)

Efficiency = T(p=1) / T(p). Speedup = p × efficiency.

| Ranks | Total samples | Min (s) | Time (s) | Max (s) | Efficiency | Speedup |
|:-----:|:------------:|:-------:|:--------:|:-------:|:----------:|:-------:|
| 1 | 1,000 | 0.98 | 1.03 | 1.09 | 1.000 | 1.00× |
| 2 | 2,000 | 1.39 | 1.40 | 1.56 | 0.731 | 1.46× |
| 3 | 3,000 | 1.07 | 1.39 | 1.60 | 0.735 | 2.21× |
| 4 | 4,000 | 1.20 | 1.21 | 1.24 | 0.846 | 3.38× |

The min/max columns show measurement variance across 3 runs. The p=3 run has a 50%
spread (1.07–1.60 s), reflecting OS scheduling noise on a single WSL2 machine where
MPI uses shared-memory IPC rather than a real network interconnect. The non-monotonic
efficiency at p=4 is a consequence of this variance and should not be interpreted as
a genuine performance gain.

All experiments run with **cold page cache** (`sync && echo 3 > /proc/sys/vm/drop_caches`).  
Machine: Intel Core i5-8350U (4 cores / 8 HT, 6 MB L3) · WSL2 / Ubuntu 22.04  
Ensemble accuracy: **88.44 %** — identical across all configurations.

**Limitations:** all MPI ranks run on the same machine, so inter-rank communication
goes through shared memory (near-zero cost). Results would differ on a real cluster
where network latency and bandwidth dominate at higher rank counts.

---

## How it works

Training is done **offline** in Python (`train.py`). The C++ engine only runs inference.

```
train.py                            inference_dist_knn.cpp
────────────────────────────        ──────────────────────────────────────────
scikit-learn trains KNN,     ──►    Rank 0 loads model_params/ and X_test
Naive Bayes, Logistic Reg.          ↓
on 70k synthetic samples            MPI_Scatter  →  each rank gets Ntest/p samples
                                    ↓
exports model_params/               #pragma omp parallel for  (schedule static)
  X_train.txt  (50k × 20)           └─ knn_predict_proba  (dominant cost)
  lr_weights.txt                    └─ nb_predict_proba
  nb_means.txt  …                   └─ lr_predict_proba
                                    ↓
                                    soft vote  →  local_preds[i]
                                    ↓
                                    MPI_Isend/Irecv + MPI_Waitall  →  rank 0
                                    ↓
                                    MPI_Reduce  →  global accuracy
```

**Note on profiling:** hotspot percentages (e.g. KNN dominates runtime) were measured
with a `-O0 -pg` build for gprof symbol visibility. Absolute times at `-O0` are not
meaningful; only the relative breakdown between functions is indicative.

### MPI patterns used

| Pattern | Role |
|---------|------|
| `MPI_Scatter` | distribute X\_test and y\_test from rank 0 to all ranks |
| `MPI_Isend` / `MPI_Irecv` + `MPI_Waitall` | non-blocking gather of predictions at rank 0 |
| `MPI_Reduce` | sum per-rank correct counts for a single global accuracy |

### Why not parallelise within a single sample?

KNN, NB, and LR are independent within each sample and could run concurrently.
In practice KNN dominates runtime (gprof, see `results/profile_report.txt`),
so intra-sample parallelism would save only a small fraction per sample while paying a
thread fork/join overhead on every one of the 20,000 samples per run.
Parallelism across samples is strictly more effective.

---

## Repository structure

```
hpc-inference/
├── src/
│   ├── inference_seq.cpp         # sequential baseline (p=1, t=1)
│   └── inference_dist_knn.cpp    # hybrid MPI+OpenMP inference engine
├── report/
│   └── hpc_inference_report.tex  # LaTeX source (compile on Overleaf)
├── results/
│   ├── speedup.csv               # strong scaling summary
│   ├── weak_scaling.csv          # weak scaling summary
│   ├── profile_report.txt        # gprof hotspot summary
│   └── raw/                      # full stdout for each configuration
├── model_params/                 # exported model files (large files gitignored)
│   ├── X_train.txt               #   gitignored — regenerate with train.py
│   ├── X_test.bin                #   gitignored — regenerate with train.py
│   ├── y_test.bin                #   gitignored — regenerate with train.py
│   ├── lr_weights.txt
│   ├── lr_bias.txt
│   ├── nb_means.txt
│   ├── nb_vars.txt
│   ├── nb_priors.txt
│   ├── scaler_mean.txt
│   ├── scaler_std.txt
│   └── config.txt
├── inference_seq                 # compiled sequential binary
├── inference_dist_knn            # compiled hybrid binary
├── train.py                      # offline training — generates model_params/
├── benchmark.sh                  # strong scaling matrix → results/speedup.csv
├── weak_scaling.sh               # weak scaling experiment → results/weak_scaling.csv
└── .gitignore
```

---

## Prerequisites

| Dependency | Version tested |
|------------|---------------|
| `mpic++` / OpenMPI | 4.x |
| OpenMP | included with GCC ≥ 9 |
| Python ≥ 3.8 | for `train.py` |
| numpy, scikit-learn | `pip install numpy scikit-learn` |

On Ubuntu / WSL2:

```bash
sudo apt install openmpi-bin libopenmpi-dev
pip install numpy scikit-learn
```

---

## Setup

The large dataset files (`X_train.txt`, `X_test.bin`, `y_test.bin`) are not tracked
in git. Generate them by running the training script once:

```bash
python train.py
```

This trains KNN, Naive Bayes, and Logistic Regression on 70,000 synthetic samples
(`random_state=42`) and writes all parameter files to `model_params/`.

---

## Build

```bash
# Sequential baseline
mpic++ -O2 -o inference_seq src/inference_seq.cpp

# Hybrid MPI+OpenMP engine
mpic++ -O2 -fopenmp -o inference_dist_knn src/inference_dist_knn.cpp
```

Profiling build (hotspot identification only — absolute times not meaningful at `-O0`):

```bash
mpic++ -O0 -pg -o inference_seq_prof src/inference_seq.cpp
mpirun -n 1 ./inference_seq_prof
gprof inference_seq_prof gmon.out > profile_report.txt
```

---

## Run

**Sequential baseline:**

```bash
OMP_NUM_THREADS=1 mpirun -n 1 ./inference_seq
```

**Hybrid engine (example: p=4 ranks, t=2 threads each):**

```bash
OMP_NUM_THREADS=2 mpirun -n 4 ./inference_dist_knn
```

**Strong scaling experiment** (reproduces `results/speedup.csv`):

```bash
bash benchmark.sh
```

**Weak scaling experiment** (reproduces `results/weak_scaling.csv`):

```bash
bash weak_scaling.sh
```

---

## Design notes

**Why model_params/ exists — the Python-C++ bridge.**  
Training happens in Python (scikit-learn); inference runs in C++. The files in
`model_params/` are the only bridge between the two. Each file has a specific role:

| File | Why it's needed |
|------|-----------------|
| `X_train.txt`, `y_train.txt` | KNN has no compact model — the training set **is** the model. C++ must search all 50,000 points at inference time. |
| `nb_means.txt`, `nb_vars.txt`, `nb_priors.txt` | Gaussian NB needs per-class feature statistics learned during training. |
| `lr_weights.txt`, `lr_bias.txt` | LR needs the weight vector and bias produced by gradient descent. |
| `scaler_mean.txt`, `scaler_std.txt` | C++ must apply the exact same z-score transform fitted in Python, or distances and dot products are wrong. |
| `config.txt` | Tells C++ the array dimensions and k at startup. |

The only large file is `X_train.txt` (~8 MB) — unavoidable for exact KNN. NB and LR
export less than 1 KB combined. Training is a one-time offline cost; what the engine
parallelises and measures is **inference only**.

**Why each rank loads X\_train independently.**  
Broadcasting X\_train from rank 0 would require a collective on the critical path
and would force all ranks to share the same memory region. Loading independently
lets each rank stream training data from its own DRAM region without contention,
and keeps inference exact — every rank searches the full 50,000-point training set.

**Why OpenMP is on the outer loop, not the inner KNN distance loop.**  
One `#pragma omp parallel for` on the outer test-sample loop gives a single
fork/join per run. Parallelising the inner distance loop instead would produce
one fork/join per sample — 20,000+ times more synchronisation overhead with no
change in total work.

**Memory bottleneck.**  
The KNN working set is 50,000 × 20 × 8 bytes = **8 MB**, which exceeds the
6 MB L3 cache. Every KNN call streams from DRAM. This is the primary reason
efficiency degrades as more ranks compete for the same memory bus.
