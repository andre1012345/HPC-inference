// =============================================================================
// inference_seq.cpp — Sequential soft-voting ensemble inference
// HPC Project · Andrea · University of Messina
//
// Sequential baseline used for all speedup measurements:
//   speedup = T_seq / T_parallel
//
// Compile:
//   g++ -O2 -o inference_seq src/inference_seq.cpp
//
// Profile (finds the hotspot; times are meaningless at -O0):
//   g++ -O0 -pg -o inference_seq_prof src/inference_seq.cpp
//   ./inference_seq_prof && gprof inference_seq_prof gmon.out | head -40
//
// Run:
//   ./inference_seq
// =============================================================================

#include <algorithm>    // std::partial_sort, std::min, std::fill, std::sort
#include <chrono>
#include <cmath>        // std::exp, std::log, M_PI
#include <cstdint>      // int32_t
#include <fstream>
#include <iostream>
#include <numeric>      // std::iota
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Number of timed runs; the median is reported to reduce noise.
// Can be overridden at compile time with -DN_RUNS=<n> (e.g. 1 for profiling).
#ifndef N_RUNS
#define N_RUNS 3
#endif

// ── KNN tiling parameters ─────────────────────────────────────────────────────
// The training matrix (50 000 × 20 × 8 B = 8 MB) exceeds the 6 MB L3 cache.
// Without tiling, every query re-reads the full 8 MB from DRAM.
//
// With TILE_Q=4 and TILE_T=512:
//   • We process 4 queries at once (TILE_Q).
//   • We iterate over training rows in stripes of 512 (TILE_T).
//   • Each stripe occupies 512 × 20 × 8 = 80 KB, which fits in the 256 KB L2.
//   • The stripe is loaded once and reused for all 4 queries before eviction,
//     cutting effective DRAM reads by ~TILE_Q×.
static constexpr int TILE_Q = 4;   // queries processed per batch
static constexpr int TILE_T = 512; // training rows per L2 tile

// =============================================================================
// I/O helpers
// =============================================================================

// Read a space-separated text matrix (rows × cols doubles).
static std::vector<double> load_text_matrix(const std::string& path,
                                             int rows, int cols) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    std::vector<double> m(rows * cols);
    for (auto& v : m) f >> v;
    return m;
}

// Read a text file containing one integer per line.
static std::vector<int> load_text_intvec(const std::string& path, int n) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    std::vector<int> v(n);
    for (auto& x : v) f >> x;
    return v;
}

// Parse config.txt which has "key value\n" lines (written by train.py).
static int read_config(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    std::string k;
    int v;
    while (f >> k >> v) {
        if (k == key) return v;
    }
    std::cerr << "Key '" << key << "' not found in " << path << "\n";
    std::exit(1);
}

// Read X_test.bin: [int32 rows][int32 cols][float64 × rows × cols]
// This binary format is written by train.py to avoid loading an 8 MB text file.
static std::vector<double> load_binary_matrix(const std::string& path,
                                               int& rows, int& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    int32_t r, c;
    f.read(reinterpret_cast<char*>(&r), sizeof(r));
    f.read(reinterpret_cast<char*>(&c), sizeof(c));
    rows = r; cols = c;
    std::vector<double> data(r * c);
    f.read(reinterpret_cast<char*>(data.data()), r * c * sizeof(double));
    return data;
}

// Read y_test.bin: [int32 n][int32 × n]
static std::vector<int> load_binary_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    int32_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<int32_t> tmp(n);
    f.read(reinterpret_cast<char*>(tmp.data()), n * sizeof(int32_t));
    return std::vector<int>(tmp.begin(), tmp.end());
}

// =============================================================================
// KNN — cache-tiled distance kernel
// =============================================================================
//
// Computes squared Euclidean distances for n_q queries (≤ TILE_Q) against the
// full training set.
//
// Memory layout: dists[q * n_train + t] = ||Xq[q] - Xtrain[t]||²
//
// The outer loop tiles over TILE_T training rows so that each tile fits in L2
// and is reused for all n_q queries before being evicted.  A naïve per-query
// implementation would reload the full 8 MB from DRAM on every query.
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

// Extract the K nearest neighbours and return P(y=1|x) = (# class-1 neighbours) / K.
// partial_sort runs in O(n log k) rather than O(n log n) — only the K smallest
// distances need to be ordered, so the sort stops after placing K elements.
static double knn_predict_proba(const double* dists,  // [n_train] for this query
                                  int*         idx,    // scratch [n_train], reused
                                  const int*   y_train,
                                  int n_train, int K) {
    // Reset index array to [0, 1, …, n_train-1].
    std::iota(idx, idx + n_train, 0);
    std::partial_sort(idx, idx + K, idx + n_train,
                      [&](int a, int b) { return dists[a] < dists[b]; });
    int c1 = 0;
    for (int j = 0; j < K; ++j) c1 += y_train[idx[j]];
    return static_cast<double>(c1) / K;
}

// =============================================================================
// Naive Bayes — Gaussian log-posterior
// =============================================================================
//
// log P(c|x) = log P(c) + Σ_j [ -½ log(2π σ²_cj) - (xj - μ_cj)² / (2σ²_cj) ]
//
// Working in log-space avoids underflow when multiplying many small probabilities.
// Log-sum-exp normalisation converts the two raw log-posteriors to a valid
// probability that sums to 1 across classes.
static double nb_predict_proba(const double* x,
                                 const double* means,   // [2 × n_feat]
                                 const double* vars,    // [2 × n_feat]
                                 const double* priors,  // [2]
                                 int n_feat) {
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
    // Numerically stable: subtract the maximum before exponentiating.
    double lmax = std::max(log_p[0], log_p[1]);
    double denom = std::exp(log_p[0] - lmax) + std::exp(log_p[1] - lmax);
    return std::exp(log_p[1] - lmax) / denom;  // P(y=1|x)
}

// =============================================================================
// Logistic Regression — dot product + sigmoid
// =============================================================================
//
// P(y=1|x) = σ(w·x + b),   σ(z) = 1 / (1 + e^{-z})
static double lr_predict_proba(const double* x, const double* w, double b,
                                 int n_feat) {
    double z = b;
    for (int f = 0; f < n_feat; ++f) z += w[f] * x[f];
    return 1.0 / (1.0 + std::exp(-z));
}

// =============================================================================
// main
// =============================================================================
int main() {
    const std::string mp = "model_params/";

    // ── Load configuration ────────────────────────────────────────────────────
    const int N_TRAIN = read_config(mp + "config.txt", "n_train");
    const int N_FEAT  = read_config(mp + "config.txt", "n_features");
    const int K       = read_config(mp + "config.txt", "knn_k");

    // ── Load model parameters ─────────────────────────────────────────────────
    // X_train is the KNN "model" — brute-force search needs the full matrix.
    auto X_train  = load_text_matrix (mp + "X_train.txt",   N_TRAIN, N_FEAT);
    auto y_train  = load_text_intvec (mp + "y_train.txt",   N_TRAIN);
    auto nb_means = load_text_matrix (mp + "nb_means.txt",  2, N_FEAT);
    auto nb_vars  = load_text_matrix (mp + "nb_vars.txt",   2, N_FEAT);
    auto nb_prior = load_text_matrix (mp + "nb_priors.txt", 2, 1);
    auto lr_w     = load_text_matrix (mp + "lr_weights.txt",1, N_FEAT);
    double lr_b   = load_text_matrix (mp + "lr_bias.txt",   1, 1)[0];

    // ── Load test set (binary for fast I/O) ───────────────────────────────────
    int N_TEST, N_FEAT_CHK;
    auto X_test = load_binary_matrix(mp + "X_test.bin", N_TEST, N_FEAT_CHK);
    auto y_test = load_binary_labels(mp + "y_test.bin");

    // ── Pre-allocate scratch buffers (reused across every run and sample) ─────
    // dist_buf: TILE_Q rows of length N_TRAIN — holds distances for the current batch.
    // idx_buf:  length N_TRAIN — reset with iota before each partial_sort.
    // Allocating once avoids ~100,000 heap calls per run.
    std::vector<double> dist_buf(TILE_Q * N_TRAIN);
    std::vector<int>    idx_buf(N_TRAIN);
    std::vector<int>    preds(N_TEST);

    // ── Benchmark loop ────────────────────────────────────────────────────────
    std::vector<double> run_times(N_RUNS);
    for (int run = 0; run < N_RUNS; ++run) {
        auto t_start = std::chrono::steady_clock::now();

        // Process test samples in batches of TILE_Q.
        // knn_distances_tiled computes distances for the whole batch at once;
        // then we classify each sample individually (NB, LR are negligible).
        for (int i = 0; i < N_TEST; i += TILE_Q) {
            int batch = std::min(TILE_Q, N_TEST - i);

            knn_distances_tiled(X_test.data() + i * N_FEAT,
                                X_train.data(), dist_buf.data(),
                                batch, N_TRAIN, N_FEAT);

            for (int q = 0; q < batch; ++q) {
                const double* xq = X_test.data() + (i + q) * N_FEAT;

                // dist_buf row q holds distances for query i+q.
                double p_knn = knn_predict_proba(dist_buf.data() + q * N_TRAIN,
                                                  idx_buf.data(),
                                                  y_train.data(), N_TRAIN, K);
                double p_nb  = nb_predict_proba(xq, nb_means.data(),
                                                 nb_vars.data(),
                                                 nb_prior.data(), N_FEAT);
                double p_lr  = lr_predict_proba(xq, lr_w.data(), lr_b, N_FEAT);

                // Soft vote: average probabilities, threshold at 0.5.
                preds[i + q] = ((p_knn + p_nb + p_lr) / 3.0 >= 0.5) ? 1 : 0;
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        run_times[run] = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "  run " << run + 1 << ": " << run_times[run] << " s\n";
    }

    // Median over N_RUNS to reduce noise from OS scheduling jitter.
    std::sort(run_times.begin(), run_times.end());
    double median_t = run_times[N_RUNS / 2];

    // ── Accuracy ──────────────────────────────────────────────────────────────
    long correct = 0;
    for (int i = 0; i < N_TEST; ++i) correct += (preds[i] == y_test[i]);

    std::cout << "\nAccuracy : " << 100.0 * correct / N_TEST << " %\n";
    std::cout << "Median t : " << median_t << " s\n";
    return 0;
}
