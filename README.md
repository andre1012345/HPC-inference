# Parallel Ensemble Inference Engine
### Hybrid MPI + OpenMP · HPC Project · University of Messina

A two-level parallel inference engine for a soft-voting ensemble classifier
(K-Nearest Neighbours, Naive Bayes, Logistic Regression).
MPI distributes the test set across processes; OpenMP parallelises each local slice
across threads. Total concurrency is **p × t** workers.

> **Full write-up:** [`hpc_inference_report.pdf`](./hpc_inference_report.pdf)

---

## Results at a glance

| Ranks (p) | Threads (t) | p · t | Time (s) | Speedup | Efficiency |
|:---------:|:-----------:|:-----:|:--------:|:-------:|:----------:|
| 1 | 1 | 1 | 28.32 | 1.00× | 1.00 |
| 1 | 2 | 2 | 15.07 | 1.88× | 1.88 |
| 2 | 1 | 2 | 14.91 | 1.90× | 0.95 |
| 2 | 2 | 4 | 9.57 | 2.96× | 1.48 |
| 3 | 1 | 3 | 12.66 | 2.24× | 0.75 |
| 3 | 2 | 6 | 8.99 | 3.15× | 1.05 |
| 4 | 1 | 4 | 10.30 | 2.75× | 0.69 |
| **4** | **2** | **8** | **8.65** | **3.27×** | **0.82** |

Baseline T₁ = 28.32 s · Machine: Intel Core i5-8350U (4 cores / 8 HT threads, 6 MB L3) · WSL2 / Ubuntu 22.04  
Ensemble accuracy: **88.44 %** — identical across all configurations.

---

## How it works

Training is done **offline** in Python. The C++ engine only runs inference.

```
parallel.ipynb                      inference_dist_knn.cpp
─────────────────                   ──────────────────────────────────────────
scikit-learn trains KNN,     ──►    Rank 0 loads model_params/ and X_test
Naive Bayes, Logistic Reg.          ↓
on 70k synthetic samples            MPI_Scatter  →  each rank gets Ntest/p samples
                                    ↓
exports model_params/               #pragma omp parallel for  (schedule static)
  X_train.txt  (50k × 20)           └─ knn_predict_proba  (87 % of compute)
  lr_weights.txt                    └─ nb_predict_proba   (< 0.02 %)
  nb_means.txt  …                   └─ lr_predict_proba   (< 0.01 %)
                                    ↓
                                    soft vote  →  local_preds[i]
                                    ↓
                                    MPI_Isend/Irecv + MPI_Waitall  →  rank 0
                                    ↓
                                    MPI_Reduce  →  global accuracy
```

### MPI patterns used

| Pattern | Role |
|---------|------|
| `MPI_Scatter` | distribute X\_test and y\_test from rank 0 to all ranks |
| `MPI_Isend` / `MPI_Irecv` + `MPI_Waitall` | non-blocking gather of predictions at rank 0 |
| `MPI_Reduce` | sum per-rank correct counts for a single global accuracy |

### Why not parallelise within a single sample?

KNN, NB, and LR are independent within each sample and could run concurrently.
In practice KNN alone accounts for **87.15 %** of runtime (gprof, Table 1 in the report),
so intra-sample parallelism would save at most 13 % per sample while paying a
thread fork/join overhead on every one of the 20,000 samples per run.
Parallelism across samples is strictly more effective.

---

## Repository structure

```
hpc_inference/
├── parallel.ipynb            # offline training + parameter export (run in Colab)
├── inference_seq.cpp         # sequential baseline (p=1, t=1)
├── inference_dist_knn.cpp    # hybrid MPI+OpenMP inference engine
├── run_experiments.sh        # automated experiment matrix → results/speedup.csv
├── hpc_inference_report.pdf  # full project report
├── model_params/             # exported model files — gitignored, see Setup
│   ├── X_train.txt
│   ├── X_test.bin
│   ├── y_test.bin
│   ├── lr_weights.txt
│   ├── lr_bias.txt
│   ├── nb_means.txt
│   ├── nb_vars.txt
│   ├── nb_priors.txt
│   ├── scaler_mean.txt
│   ├── scaler_std.txt
│   └── config.txt
└── results/
    ├── speedup.csv           # summary table
    ├── experiment.log        # timestamped run log
    └── raw/                  # full stdout for each configuration
```

---

## Prerequisites

| Dependency | Version tested |
|------------|---------------|
| `mpic++` / OpenMPI | 4.x |
| OpenMP | included with GCC ≥ 9 |
| Python + scikit-learn | for `parallel.ipynb` only |

On Ubuntu / WSL2:

```bash
sudo apt install openmpi-bin libopenmpi-dev
```

---

## Setup: offline training

The `model_params/` directory is **not** tracked in git (large flat files).
Generate it by running `parallel.ipynb` in Google Colab or locally, then download
the exported `model_params.zip` and unzip it next to the binaries:

```bash
unzip model_params.zip          # produces model_params/
```

The notebook trains on 70,000 synthetic samples (scikit-learn `make_classification`,
`random_state=42`) and exports all parameters as plain-text or binary files.

---

## Build

```bash
# Sequential baseline
mpic++ -O2 -o inference_seq inference_seq.cpp

# Hybrid MPI+OpenMP engine
mpic++ -O2 -fopenmp -o inference_dist_knn inference_dist_knn.cpp
```

Profiling build (Section 2.4 of the report — absolute times not meaningful, hotspot only):

```bash
mpic++ -O0 -pg -o inference_seq_prof inference_seq.cpp
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

**Full experiment matrix** (reproduces `results/speedup.csv`):

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

The script runs the sequential baseline, then all `(p, t)` combinations with
`p ∈ {1,2,3,4}` and `t ∈ {1,2}`, writes raw output to `results/raw/`, and
appends a summary row to `results/speedup.csv`.

---

## Design notes

**Why each rank loads X\_train independently.**  
Broadcasting X\_train from rank 0 would require exact KNN to search the same
full training set seen at training time. Loading independently keeps inference
exact, avoids an extra collective on the critical path, and lets all ranks
stream training data in parallel from their own memory region without contending
for the same cache lines.

**Why OpenMP is on the outer loop, not the inner KNN distance loop.**  
One `#pragma omp parallel for` on the outer test-sample loop gives a single
fork/join per run. Parallelising the inner distance loop instead would produce
one fork/join per sample per run — 20,000+ times more synchronisation overhead,
with no change in total work.

**Memory bottleneck.**  
The KNN working set is 50,000 × 20 × 8 bytes = **8 MB**, which exceeds the
6 MB L3 cache. Every KNN call streams from DRAM. Arithmetic intensity is
≈ 0.25 FLOP/byte; the DDR4 bus (~30 GB/s) limits peak throughput to ~7.5 GFLOP/s.
This is the primary reason speedup saturates beyond 4 physical cores.
See Section 7.3 of the report for the full Amdahl analysis (estimated serial
fraction f ≈ 0.15, ceiling ≈ 6.7× on infinite cores).

---
