[README (3).md](https://github.com/user-attachments/files/27512713/README.3.md)
# HPC Parallel Ensemble Inference Engine

**University of Messina · HPC Course Project · Andrea**

A hybrid MPI + OpenMP inference engine for a three-model soft-voting ensemble classifier (KNN, Naive Bayes, Logistic Regression), built to study parallelisation strategies on a four-core UMA node.

---

## Repository Layout

```
HPC-inference/
├── model_params/           # exported by Colab (parameters + test set)
├── inference_seq.cpp       # sequential baseline      (p=1, t=1)
├── inference_sections.cpp  # OpenMP task parallel     (omp sections)
├── inference_omp.cpp       # OpenMP data parallel     (omp parallel for)
├── inference_mpi.cpp       # MPI data parallel        (p ranks, t=1)
├── inference_nested.cpp    # nested / oversubscription experiment
└── inference_hybrid.cpp    # Hybrid MPI+OpenMP        (p ranks × t threads)
```

Each file is self-contained — no shared library between variants. Compile, run, and time each one independently.

---

## Step 1 — Offline Training Phase (Google Colab)

Run the provided Colab notebook **once** to generate all model parameters. No training happens at C++ runtime.

### What the notebook does

| Step | Code | Output |
|------|------|--------|
| Dataset generation | `make_classification(n_samples=70000, n_features=20, n_informative=10, random_state=42)` | 70k synthetic samples, 2 balanced classes |
| Train/test split | `train_test_split(test_size=20000, stratify=y)` | 50k train · 20k test |
| Feature scaling | `StandardScaler().fit(X_train)` → transform both splits | Zero mean, unit variance |
| Model training | `KNeighborsClassifier(k=5)`, `GaussianNB()`, `LogisticRegression()` | Three fitted models |
| Soft-vote validation | Average probabilities, threshold at 0.5 | Accuracy **0.8844** on test set |
| Export | `np.savetxt / struct.pack` | `model_params/` folder |

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

> **Why ~8 MB?** KNN is a lazy learner — the entire training set *is* the model. At inference time the engine computes the Euclidean distance from every query point to all 50,000 training points. This array is loaded once into a contiguous `std::vector<double>` and kept in memory for the whole batch, making it the primary source of memory pressure in every experiment.

After the notebook finishes, download `model_params.zip` from the Colab file browser and place the unzipped folder next to the compiled binaries.

---

## Step 2 — Compilation

All variants compile with `mpicxx`, the MPI C++ wrapper around `g++`. The `-fopenmp` flag is required only for variants that use OpenMP.

```bash
# Sequential baseline
mpicxx -O2 -std=c++17 -o inference_seq      inference_seq.cpp

# OpenMP variants (sections and parallel-for)
mpicxx -O2 -fopenmp -std=c++17 -o inference_sections  inference_sections.cpp
mpicxx -O2 -fopenmp -std=c++17 -o inference_omp       inference_omp.cpp

# MPI-only
mpicxx -O2 -std=c++17 -o inference_mpi      inference_mpi.cpp

# Nested / oversubscription
mpicxx -O2 -fopenmp -std=c++17 -o inference_nested    inference_nested.cpp

# Hybrid MPI+OpenMP (best configuration)
mpicxx -O2 -fopenmp -std=c++17 -o inference_hybrid    inference_hybrid.cpp
```

**Compiler flags explained:**

| Flag | Effect |
|------|--------|
| `-O2` | Standard optimisations (inlining, loop unrolling, constant folding) without aggressive floating-point reordering |
| `-fopenmp` | Enables OpenMP directives and links the OpenMP runtime |
| `-std=c++17` | Required for structured bindings and `<filesystem>` (if used) |

---

## Step 3 — Running Experiments

`P` = number of MPI ranks · `T` = number of OpenMP threads per rank · `P × T` = total logical execution units.

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

> **Resource constraint:** `P × T ≤ physical_cores` should be respected to avoid oversubscription. Configurations that violate this rule (like the last example above) are included as explicit oversubscription experiments and their results are interpreted accordingly.

---

## How Each Variant Works

### `inference_seq.cpp` — Sequential baseline
A single loop iterates over all 20,000 test samples. For each sample it calls KNN, Naive Bayes, and Logistic Regression in sequence, averages the three probabilities, and writes the predicted class to a result array. Provides the reference time **T₁**.

### `inference_sections.cpp` — OpenMP task parallelism
Three `#pragma omp parallel sections` assign one base model per section: one thread runs KNN on all samples, a second runs Naive Bayes, a third runs Logistic Regression. Results are combined in a sequential reduction step after the parallel region.  
⚠️ KNN dominates execution time by a factor of ~100×, so the two faster threads sit idle while KNN finishes. This causes severe **load imbalance** and produces a slowdown relative to sequential.

### `inference_omp.cpp` — OpenMP data parallelism
A single `#pragma omp parallel for schedule(static)` distributes the 20,000 test samples evenly across threads. Each thread evaluates all three models on its own share and writes predictions to disjoint indices — no synchronisation needed. Eliminates load imbalance, but all threads share the same copy of `X_train` in the same address space. On a UMA node with a shared memory bus, bus saturation limits speedup beyond 2 threads.

### `inference_mpi.cpp` — MPI data parallelism
Each rank loads its own **private copy** of all model parameters from disk. Rank 0 distributes the test set with `MPI_Scatter` (N/p rows per rank) and collects results with `MPI_Gather`. Because each rank is a separate OS process, the cache-coherence protocol does not see the model parameters as a shared resource — each rank accesses its own copy in its own physical address space, eliminating the bus contention of the OpenMP variant.

### `inference_hybrid.cpp` — Hybrid MPI+OpenMP *(best performing)*
Combines both levels of parallelism. MPI initialised with `MPI_THREAD_FUNNELED` (only the master thread makes MPI calls). `MPI_Scatter` distributes N/p samples to each rank; within each rank a `#pragma omp parallel for` further splits the local chunk across `t` threads. The implicit OpenMP barrier ensures all threads complete before the master thread calls `MPI_Gather`. Per-thread KNN distance buffers are pre-allocated before the timing loop to avoid dynamic allocation in the hot path.

### `inference_nested.cpp` — Nested / oversubscription
Deliberately creates more logical execution units than physical cores. Included to quantify the performance penalty of oversubscription (context-switch overhead, cache thrashing) and to validate the `P × T ≤ physical_cores` guideline experimentally.

---

## Common Infrastructure

All six files share the same three inference functions and the same I/O helpers; only `main()` differs.

**Inference functions (thread-safe — all state on the stack):**
- `knn_predict_proba` — brute-force Euclidean distance + `std::partial_sort`, returns class-1 vote fraction
- `nb_predict_proba` — Gaussian log-posterior with numerically stable softmax
- `lr_predict_proba` — sigmoid of dot product `w · x + b`

**Timing discipline:**  
`MPI_Wtime()` wraps only the inference phase (Scatter + parallel loop + Gather). File I/O and model loading are excluded. Each configuration runs 5 times; the **median** is reported to reduce system noise.

**Correctness invariant:**  
Every parallel configuration must reproduce the exact same 20,000 predictions as the sequential baseline. Accuracy must match **0.8844**. Any deviation indicates a race condition or a floating-point aggregation error.

---

## Results Summary

| Configuration | P | T | P×T | Median time (s) | Speedup |
|---|---|---|---|---|---|
| Sequential | 1 | 1 | 1 | 17.73 | 1.00× |
| OpenMP sections | 1 | 3 | 3 | 25.30 | 0.70× |
| OpenMP parallel-for | 1 | 4 | 4 | 13.52 | 1.31× |
| MPI only | 4 | 1 | 4 | 9.85 | 1.80× |
| **Hybrid MPI+OpenMP** | **4** | **2** | **8\*** | **6.13** | **2.89×** |

\* Oversubscription experiment (8 logical units on 4 physical cores).  
Amdahl's Law theoretical maximum on this hardware: **3.95×**.  
Best achieved: **2.89×** = 73% of theoretical maximum.

---

## Requirements

- C++17-compatible compiler (GCC 9+ recommended)
- MPI implementation (OpenMPI or MPICH)
- OpenMP (included with GCC via `-fopenmp`)
- Python 3.8+ with `scikit-learn`, `numpy` (for the Colab notebook only)
