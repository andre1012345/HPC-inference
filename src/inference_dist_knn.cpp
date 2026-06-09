// =============================================================================
// inference_dist_knn.cpp — Hybrid MPI + OpenMP soft-voting ensemble inference
// HPC Project · Andrea · University of Messina
//
// Two-level parallelism hierarchy (see report Section 1.4 and Figure 1):
//
//   MPI level (coarse-grained):
//     • Each MPI rank is an independent OS process with its own address space.
//     • Rank 0 distributes the test set via MPI_Scatter; each rank classifies
//       its local slice independently — no inter-rank communication during inference.
//     • Rank 0 collects predictions with non-blocking MPI_Isend/MPI_Irecv.
//     • MPI is the only mechanism that scales across multiple nodes in a cluster.
//
//   OpenMP level (fine-grained):
//     • Threads within a rank share one address space (and one X_train copy).
//     • A single #pragma omp parallel for subdivides the local test slice.
//     • One fork/join per timed run — not one per sample — keeps overhead < 0.1%.
//     • Each thread has its own dist/idx scratch buffers to avoid false sharing.
//
//   Total workers = p (MPI ranks) × t (OMP threads per rank).
//
// MPI communication patterns used (see report Section 4):
//   MPI_Scatter           — distribute test samples (before timing)
//   MPI_Isend/Irecv       — non-blocking gather of predictions (inside timing)
//   MPI_Reduce            — sum per-rank correct counts for global accuracy
//
// Compile:
//   mpic++ -O2 -fopenmp -o inference_dist_knn src/inference_dist_knn.cpp
//
// Run (e.g. 4 ranks, 2 threads each):
//   OMP_NUM_THREADS=2 mpirun -np 4 --bind-to none ./inference_dist_knn
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr int N_RUNS = 3;

// TILE_Q: queries processed per batch.  Each thread handles TILE_Q queries at
// once, loading each training-row stripe once and reusing it for all TILE_Q
// queries before eviction — cuts DRAM reads by ~TILE_Q× vs per-query scanning.
// TILE_T: training rows per stripe.  512 × 20 × 8 B = 80 KB fits in L2 (256 KB).
static constexpr int TILE_Q = 4;
static constexpr int TILE_T = 512;

// =============================================================================
// I/O helpers  (same logic as inference_seq; duplicated to keep each file self-contained)
// =============================================================================

static std::vector<double> load_text_matrix(const std::string& path,
                                             int rows, int cols) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    std::vector<double> m(rows * cols);
    for (auto& v : m) f >> v;
    return m;
}

static std::vector<int> load_text_intvec(const std::string& path, int n) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    std::vector<int> v(n);
    for (auto& x : v) f >> x;
    return v;
}

static int read_config(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    std::string k; int v;
    while (f >> k >> v) {
        if (k == key) return v;
    }
    std::cerr << "Key '" << key << "' not found in " << path << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return -1;
}

// Binary formats match train.py exactly.
static std::vector<double> load_binary_matrix(const std::string& path,
                                               int& rows, int& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    int32_t r, c;
    f.read(reinterpret_cast<char*>(&r), sizeof(r));
    f.read(reinterpret_cast<char*>(&c), sizeof(c));
    rows = r; cols = c;
    std::vector<double> data(r * c);
    f.read(reinterpret_cast<char*>(data.data()), r * c * sizeof(double));
    return data;
}

static std::vector<int> load_binary_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    int32_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<int32_t> tmp(n);
    f.read(reinterpret_cast<char*>(tmp.data()), n * sizeof(int32_t));
    return std::vector<int>(tmp.begin(), tmp.end());
}

// =============================================================================
// Per-thread KNN kernel (TILE_Q queries × TILE_T training rows, cache-tiled)
// =============================================================================
//
// Each OpenMP thread calls this with its own dist[] buffer (TILE_Q × N_TRAIN).
// The outer loop tiles over TILE_T training rows: each stripe is loaded once
// and reused for all n_q queries in the batch before being evicted from L2.
// Threads write to disjoint heap allocations — no false sharing.
//
// dists[q * n_train + t] = ||Xq[q] - Xtrain[t]||²
static void knn_distances_tiled(const double* Xq,       // [n_q × n_feat]
                                  const double* Xtrain,  // [n_train × n_feat]
                                  double*       dists,   // [n_q × n_train]
                                  int n_q, int n_train, int n_feat) {
    std::fill(dists, dists + n_q * n_train, 0.0);
    for (int t0 = 0; t0 < n_train; t0 += TILE_T) {
        int t_end = std::min(t0 + TILE_T, n_train);
        for (int t = t0; t < t_end; ++t) {
            const double* xt = Xtrain + t * n_feat; // loaded once, reused for all n_q queries
            for (int q = 0; q < n_q; ++q) {
                const double* xq = Xq + q * n_feat;
                double d = 0.0;
                for (int f = 0; f < n_feat; ++f) {
                    double diff = xq[f] - xt[f];
                    d += diff * diff;
                }
                dists[q * n_train + t] = d;
            }
        }
    }
}

// partial_sort extracts K nearest neighbours in O(n log k); only K elements need ordering.
static double knn_predict_proba(double* dist, int* idx,
                                  const int* y_train, int n_train, int K) {
    std::iota(idx, idx + n_train, 0);
    std::partial_sort(idx, idx + K, idx + n_train,
                      [&](int a, int b) { return dist[a] < dist[b]; });
    int c1 = 0;
    for (int j = 0; j < K; ++j) c1 += y_train[idx[j]];
    return static_cast<double>(c1) / K;
}

// =============================================================================
// Naive Bayes (Gaussian log-posterior with log-sum-exp normalisation)
// =============================================================================
static double nb_predict_proba(const double* x,
                                 const double* means, const double* vars,
                                 const double* priors, int n_feat) {
    double log_p[2];
    for (int c = 0; c < 2; ++c) {
        double lp = std::log(priors[c]);
        for (int f = 0; f < n_feat; ++f) {
            double mu  = means[c * n_feat + f];
            double var = vars [c * n_feat + f];
            double d   = x[f] - mu;
            lp += -0.5 * std::log(2.0 * M_PI * var) - (d * d) / (2.0 * var);
        }
        log_p[c] = lp;
    }
    double lmax  = std::max(log_p[0], log_p[1]);
    double denom = std::exp(log_p[0] - lmax) + std::exp(log_p[1] - lmax);
    return std::exp(log_p[1] - lmax) / denom;
}

// =============================================================================
// Logistic Regression
// =============================================================================
static double lr_predict_proba(const double* x, const double* w, double b,
                                 int n_feat) {
    double z = b;
    for (int f = 0; f < n_feat; ++f) z += w[f] * x[f];
    return 1.0 / (1.0 + std::exp(-z));
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    // MPI_THREAD_FUNNELED: only the master thread (rank's main thread) calls MPI.
    // OpenMP parallel regions run after MPI_Scatter and before MPI_Isend, so
    // MPI calls are never made from inside a parallel region.
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Optional --n_test N argument: limits the test set to N samples (weak scaling).
    // If omitted, the full test set is used.
    int n_test_limit = -1;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--n_test")
            n_test_limit = std::stoi(argv[i + 1]);
    }

    const std::string mp = "model_params/";

    // ── Configuration (every rank reads the same small file) ──────────────────
    const int N_TRAIN = read_config(mp + "config.txt", "n_train");
    const int N_FEAT  = read_config(mp + "config.txt", "n_features");
    const int K       = read_config(mp + "config.txt", "knn_k");

    // ── Load model parameters — every rank loads its own copy from disk ────────
    // Avoids broadcasting the 8 MB training matrix and keeps each rank's copy
    // in a separate memory region, so all p ranks stream from DRAM in parallel
    // rather than through one process's address space.
    auto X_train  = load_text_matrix (mp + "X_train.txt",   N_TRAIN, N_FEAT);
    auto y_train  = load_text_intvec (mp + "y_train.txt",   N_TRAIN);
    auto nb_means = load_text_matrix (mp + "nb_means.txt",  2, N_FEAT);
    auto nb_vars  = load_text_matrix (mp + "nb_vars.txt",   2, N_FEAT);
    auto nb_prior = load_text_matrix (mp + "nb_priors.txt", 2, 1);
    auto lr_w     = load_text_matrix (mp + "lr_weights.txt",1, N_FEAT);
    double lr_b   = load_text_matrix (mp + "lr_bias.txt",   1, 1)[0];

    // ── Rank 0 loads the full test set — single read, N_TEST_ORIG from header ──
    // We read here (not earlier) so X_train is already resident and the file
    // system cache can serve X_test.bin with minimal seek overhead.
    std::vector<double> X_test_all;
    std::vector<int>    y_test_all;
    int N_TEST_ORIG = 0;
    if (rank == 0) {
        int n_feat_chk;
        X_test_all = load_binary_matrix(mp + "X_test.bin", N_TEST_ORIG, n_feat_chk);
        y_test_all = load_binary_labels(mp + "y_test.bin");
    }

    // Broadcast N_TEST so all ranks can compute their local slice size.
    // Trim to a multiple of p so MPI_Scatter sends perfectly equal chunks.
    // At p=3 this drops at most 2 samples (0.01% of 20,000).
    MPI_Bcast(&N_TEST_ORIG, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // If --n_test was given, cap N_TEST_ORIG before trimming (weak scaling).
    if (n_test_limit > 0 && n_test_limit < N_TEST_ORIG)
        N_TEST_ORIG = n_test_limit;
    int N_TEST  = (N_TEST_ORIG / size) * size;
    int local_n = N_TEST / size;   // samples this rank will classify

    // ── Distribute test data — this is outside the timed region ───────────────
    // At p=4, each scatter message carries 5,000 × 20 × 8 = 800 KB.
    std::vector<double> X_test_local(local_n * N_FEAT);
    MPI_Scatter(X_test_all.data(),   local_n * N_FEAT, MPI_DOUBLE,
                X_test_local.data(), local_n * N_FEAT, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Labels are also scattered so each rank can count its own correct predictions.
    std::vector<int> y_test_local(local_n);
    MPI_Scatter(y_test_all.data(),   local_n, MPI_INT,
                y_test_local.data(), local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    // ── Pre-allocate per-thread scratch buffers — before the timed loop ────────
    // Thread i writes only to dist_bufs[i] and idx_bufs[i].
    // Because each buffer is an independent heap allocation, their first elements
    // land on different cache lines — no false sharing between threads.
    int n_threads = omp_get_max_threads();
    // dist_bufs: TILE_Q × N_TRAIN per thread — holds distances for a full batch.
    // idx_bufs:  N_TRAIN per thread — reset with iota before each partial_sort.
    std::vector<std::vector<double>> dist_bufs(n_threads,
                                                std::vector<double>(TILE_Q * N_TRAIN));
    std::vector<std::vector<int>>    idx_bufs (n_threads,
                                                std::vector<int>(N_TRAIN));
    std::vector<int> local_preds(local_n);

    // ── Benchmark loop ────────────────────────────────────────────────────────
    std::vector<double> run_times(N_RUNS);
    for (int run = 0; run < N_RUNS; ++run) {
        // Synchronise all ranks before starting the clock so that one slow
        // rank's load time does not inflate another rank's measured time.
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        // One parallel region per run (one fork/join, not one per batch).
        // The outer loop steps by TILE_Q so each iteration processes a batch of
        // up to TILE_Q queries with a single knn_distances_tiled call.
        // schedule(static) preserves contiguous access order within each thread's
        // slice of X_test_local, keeping hardware prefetchers effective.
#pragma omp parallel
        {
            int tid      = omp_get_thread_num();
            double* dist = dist_bufs[tid].data();  // TILE_Q × N_TRAIN, private
            int*    idx  = idx_bufs [tid].data();  // N_TRAIN, private

#pragma omp for schedule(static)
            for (int i = 0; i < local_n; i += TILE_Q) {
                int batch = std::min(TILE_Q, local_n - i);

                // Compute distances for all `batch` queries in one tiled pass.
                knn_distances_tiled(X_test_local.data() + i * N_FEAT,
                                    X_train.data(), dist, batch, N_TRAIN, N_FEAT);

                // Classify each query in the batch sequentially (NB and LR are free).
                for (int q = 0; q < batch; ++q) {
                    const double* xq = X_test_local.data() + (i + q) * N_FEAT;

                    // dist row q holds squared distances for query i+q.
                    double p_knn = knn_predict_proba(dist + q * N_TRAIN, idx,
                                                      y_train.data(), N_TRAIN, K);
                    double p_nb  = nb_predict_proba(xq, nb_means.data(), nb_vars.data(),
                                                     nb_prior.data(), N_FEAT);
                    double p_lr  = lr_predict_proba(xq, lr_w.data(), lr_b, N_FEAT);

                    local_preds[i + q] = ((p_knn + p_nb + p_lr) / 3.0 >= 0.5) ? 1 : 0;
                }
            }
        } // end parallel

        double t_end = MPI_Wtime();
        run_times[run] = t_end - t_start;

        if (rank == 0)
            std::cout << "  run " << run + 1 << ": " << run_times[run] << " s\n";
    }

    std::sort(run_times.begin(), run_times.end());
    double median_t = run_times[N_RUNS / 2];

    // ── Gather predictions at rank 0 (non-blocking) ───────────────────────────
    // Rank 0 posts all receives simultaneously so multiple transfers can proceed
    // concurrently (MPI_Gather would serialize them internally on some implementations).
    std::vector<int> global_preds;
    if (rank == 0) {
        global_preds.resize(N_TEST);
        // Copy rank 0's own slice — no MPI needed for self.
        std::copy(local_preds.begin(), local_preds.end(), global_preds.begin());

        std::vector<MPI_Request> reqs(size - 1);
        for (int r = 1; r < size; ++r)
            MPI_Irecv(global_preds.data() + r * local_n, local_n,
                      MPI_INT, r, 0, MPI_COMM_WORLD, &reqs[r - 1]);
        // Block until every non-blocking receive completes.
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    } else {
        // Non-blocking send lets the worker proceed immediately (it has nothing
        // else to do, so MPI_Wait follows right away, but the pattern is correct).
        MPI_Request req;
        MPI_Isend(local_preds.data(), local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    // ── Accuracy — per-rank counts reduced to rank 0 ──────────────────────────
    // Each rank counts its local correct predictions; MPI_Reduce sums them.
    // This avoids re-sending individual labels (which were already gathered above).
    long local_correct = 0;
    for (int i = 0; i < local_n; ++i)
        local_correct += (local_preds[i] == y_test_local[i]);

    long global_correct = 0;
    MPI_Reduce(&local_correct, &global_correct, 1,
               MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // ── Report (rank 0 only) ──────────────────────────────────────────────────
    if (rank == 0) {
        double acc = 100.0 * global_correct / N_TEST;
        std::cout << "\nRanks    : " << size
                  << "  Threads/rank : " << n_threads << "\n";
        std::cout << "Samples  : " << N_TEST
                  << " (original " << N_TEST_ORIG << ")\n";
        std::cout << "Accuracy : " << acc << " %\n";
        std::cout << "Median t : " << median_t << " s\n";
        std::cout << "Speedup  : T_1 / " << median_t
                  << "  (divide by sequential baseline)\n";
    }

    MPI_Finalize();
    return 0;
}
