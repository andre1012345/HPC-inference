[README (4).md](https://github.com/user-attachments/files/27513154/README.4.md)
# HPC Parallel Ensemble Inference Engine

**University of Messina · HPC Course Project · Andrea**

A hybrid MPI + OpenMP inference engine for a three-model soft-voting ensemble classifier (KNN, Naive Bayes, Logistic Regression), built to study parallelisation strategies on a four-core UMA node.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andre1012345/HPC-inference/blob/main/parallel.ipynb)

---

## Repository Layout

```
HPC-inference/
├── parallel.ipynb          # Colab notebook — generates model_params/ (run this first)
├── model_params/           # exported parameters + test set (output of the notebook)
├── inference_seq.cpp       # sequential baseline      (p=1, t=1)
├── inference_sections.cpp  # OpenMP task parallel     (omp sections)
├── inference_omp.cpp       # OpenMP data parallel     (omp parallel for)
├── inference_mpi.cpp       # MPI data parallel        (p ranks, t=1)
├── inference_nested.cpp    # nested / oversubscription experiment
├── inference_hybrid.cpp    # Hybrid MPI+OpenMP        (p ranks × t threads)
└── README.md
```

---

## Step 1 — Offline Training Phase (Google Colab)

> **The notebook must be run once before any C++ binary can be executed.**
> It generates the `model_params/` directory that the inference engine reads at startup.
> No training happens at C++ runtime.

Click the badge at the top of this page, or open the notebook directly:

```
https://colab.research.google.com/github/andre1012345/HPC-inference/blob/main/parallel.ipynb
```

Run all cells in order. When the last cell finishes it produces a `model_params.zip` in the
Colab file browser (left panel). Download it, unzip it, and place the `model_params/` folder
in the same directory as the compiled C++ binaries.

### What the notebook does

| Step | Code | Output |
|------|------|--------|
| Dataset generation | `make_classification(n_samples=70000, n_features=20, n_informative=10, random_state=42)` | 70k synthetic samples, 2 balanced classes |
| Train/test split | `train_test_split(test_size=20000, stratify=y)` | 50k train · 20k test |
| Feature scaling | `StandardScaler().fit(X_train)` → transform both splits | Zero mean, unit variance |
| Model training | `KNeighborsClassifier(k=5)`, `GaussianNB()`, `LogisticRegression()` | Three fitted models |
| Soft-vote validation | Average probabilities, threshold at 0.5 | Accuracy **0.8844** on test set |
| Export | `np.savetxt` / `struct.pack` | `model_params/` folder |

### Exported files

| File | Contents | Size |
|------|----------|------|
| `config.txt` | `n_features`, `n_train`, `knn_k`, `random_state` | < 1 KB |
| `scaler_mean.txt` / `scaler_std.txt` | 20 float64 values each | < 1 KB |
| `X_train.txt` | 50,000 × 20 float64 — the KNN "model" | **~8 MB** |
| `y_train.txt` | 50,000 integer labels | ~300 KB |
| `nb_means.txt` / `nb_vars.txt` | 2 × 20 float64 values each | < 1 KB |
| `nb_priors.txt` | 2 class prior probabilities | < 1 KB |
| `lr_weights.txt` / `lr_bias.txt` | 20 weights + 1 scalar bias | < 1 KB |
| `X_test.bin` | Binary: `[int32 rows][int32 cols][float64 × 20000 × 20]` | ~3.2 MB |
| `y_test.bin` | Binary: `[int32 n][int32 × 20000]` | ~80 KB |

> **Why ~8 MB?** KNN is a lazy learner — the entire training set *is* the model. At inference
> time the engine computes the Euclidean distance from every query point to all 50,000 training
> points. This array is the primary source of memory pressure in every experiment.

---

## Step 2 — Compilation

All variants compile with `mpicxx`, the MPI C++ wrapper around `g++`.
The `-fopenmp` flag is required only for variants that use OpenMP.

```bash
# Sequential baseline
mpicxx -O2 -std=c++17 -o inference_seq      inference_seq.cpp

# OpenMP variants
mpicxx -O2 -fopenmp -std=c++17 -o inference_sections  inference_sections.cpp
mpicxx -O2 -fopenmp -std=c++17 -o inference_omp       inference_omp.cpp

# MPI-only
mpicxx -O2 -std=c++17 -o inference_mpi      inference_mpi.cpp

# Nested / oversubscription
mpicxx -O2 -fopenmp -std=c++17 -o inference_nested    inference_nested.cpp

# Hybrid MPI+OpenMP (best configuration)
mpicxx -O2 -fopenmp -std=c++17 -o inference_hybrid    inference_hybrid.cpp
```

| Flag | Effect |
|------|--------|
| `-O2` | Standard optimisations (inlining, loop unrolling, constant folding) |
| `-fopenmp` | Enables OpenMP directives and links the runtime |
| `-std=c++17` | C++17 standard |

---

## Step 3 — Running Experiments

`P` = MPI ranks · `T` = OpenMP threads per rank · `P × T` = total logical execution units.

```bash
# Sequential baseline (must use -n 1)
mpirun -n 1 ./inference_seq

# OpenMP data-parallel — 4 threads
OMP_NUM_THREADS=4 mpirun -n 1 ./inference_omp

# OpenMP task-parallel — 3 sections (one per model)
OMP_NUM_THREADS=3 mpirun -n 1 ./inference_sections

# MPI only — 4 ranks
mpirun -n 4 ./inference_mpi

# Hybrid: 2 ranks × 2 threads = 4 logical units (no oversubscription)
OMP_NUM_THREADS=2 mpirun -n 2 ./inference_hybrid

# Hybrid: 4 ranks × 2 threads = 8 logical units (oversubscription on 4-core node)
OMP_NUM_THREADS=2 mpirun -n 4 ./inference_hybrid
```

> `P × T ≤ physical_cores` should be respected to avoid oversubscription.
> The last command is an explicit oversubscription experiment.

---

## How Each Variant Works

### `inference_seq.cpp` — Sequential baseline
Single loop over all 20,000 samples. Calls KNN → Naive Bayes → Logistic Regression in
sequence per sample, averages probabilities, writes the predicted class.
Reference time: **T₁ = 17.73 s**.

### `inference_sections.cpp` — OpenMP task parallelism
Three `#pragma omp parallel sections`, one model per section. One thread runs KNN on all
samples, a second runs Naive Bayes, a third runs Logistic Regression.
⚠️ KNN is ~3,000× slower than the other two — the other threads sit idle at the barrier.
Result: **0.70× speedup** (slower than sequential). Intentional experiment to demonstrate
Amdahl's Law under load imbalance.

### `inference_omp.cpp` — OpenMP data parallelism
`#pragma omp parallel for schedule(static)` distributes the 20,000 samples evenly across
threads. Eliminates load imbalance, but all threads share the same `X_train` in memory.
The UMA memory bus saturates between t=2 and t=4 → **scaling plateau at ~1.30×**.

### `inference_mpi.cpp` — MPI data parallelism
Each rank loads its own **private copy** of all parameters. `MPI_Scatter` sends N/p samples
per rank; each rank processes its chunk sequentially; `MPI_Gather` collects results.
Private address spaces eliminate cache-coherency bus traffic.
Result: **1.80× at p=4** — better than OpenMP's 1.30× at t=4 on the same hardware.

### `inference_hybrid.cpp` — Hybrid MPI+OpenMP *(best performing)*
MPI initialised with `MPI_THREAD_FUNNELED`. `MPI_Scatter` → `omp parallel for` per rank →
`MPI_Gather`. Per-thread KNN distance buffers pre-allocated to avoid heap contention.
Best result: **2.89× speedup** = 73% of the Amdahl theoretical maximum (3.95×).

### `inference_nested.cpp` — Oversubscription experiment
Creates more logical units than physical cores. Included to quantify the performance
penalty when P × T > physical_cores.

---

## Results Summary

| Configuration | P | T | P×T | Median (s) | Speedup |
|---|---|---|---|---|---|
| Sequential | 1 | 1 | 1 | 17.73 | 1.00× |
| OpenMP sections | 1 | 3 | 3 | 25.39 | 0.70× |
| OpenMP parallel-for (t=2) | 1 | 2 | 2 | 13.52 | 1.31× |
| OpenMP parallel-for (t=4) | 1 | 4 | 4 | 13.64 | 1.30× |
| MPI only (p=4) | 4 | 1 | 4 | 9.85 | 1.80× |
| Hybrid (p=2, t=2) | 2 | 2 | 4 | 7.42 | 2.39× |
| **Hybrid (p=4, t=2) \*** | **4** | **2** | **8** | **6.14** | **2.89×** |

\* Oversubscription experiment. Amdahl theoretical maximum on this hardware: **3.95×**.

---

## Requirements

- GCC 9+ with C++17 support
- OpenMPI or MPICH
- OpenMP (included with GCC via `-fopenmp`)
- Python 3.8+ with `scikit-learn` and `numpy` — only needed for the Colab notebook;
  no local Python install required if you use the pre-built `model_params/` in the repo
