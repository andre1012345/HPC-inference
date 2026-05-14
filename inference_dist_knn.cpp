// =============================================================================
// Parallel Ensemble Inference Engine — Hybrid MPI + OpenMP
// HPC Project · Andrea · University of Messina
//
// Every rank loads the full X_train from file independently (no MPI for training
// data). MPI_Scatter distributes only X_test and y_test across ranks.
// OpenMP parallelises the outer loop over test samples inside each rank.
// One fork/join per run — minimal overhead vs. fork/join per sample.
//
// MPI patterns used (all three required by Prof. Distefano):
//   MPI_Scatter           — distribute X_test, y_test across ranks
//   MPI_Isend / MPI_Irecv — non-blocking gather of local predictions
//   MPI_Waitall           — synchronise non-blocking operations
//   MPI_Reduce            — aggregate correct-prediction counts for accuracy
//
// KNN is exact: every rank searches all N_TRAIN training points per test sample,
//   matching sequential accuracy. LR and NB are also exact and communication-free.
//
// Timing: MPI_Wtime() wraps the inference + communication phase only.
//   File I/O is excluded. Median of N_RUNS is reported.
// =============================================================================

#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Configuration  (must match inference_seq.cpp)
// -----------------------------------------------------------------------------
static const int    N_FEATURES = 20;
static const int    N_TRAIN    = 50000;
static const int    KNN_K      = 5;
static const int    N_CLASSES  = 2;
static const int    N_RUNS     = 5;
static const std::string MODEL_DIR = "model_params";

// -----------------------------------------------------------------------------
// I/O helpers  (identical to inference_seq.cpp)
// -----------------------------------------------------------------------------
std::vector<double> load_vector(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<double> v;
    double x;
    while (f >> x) v.push_back(x);
    return v;
}

std::vector<double> load_matrix(const std::string& path, int rows, int cols) {
    std::vector<double> m = load_vector(path);
    if ((int)m.size() != rows * cols) {
        std::cerr << "ERROR: " << path << " expected " << rows * cols
                  << " values, got " << m.size() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return m;
}

// Binary format: [int32 n_samples][int32 n_features][float64 * n_samples * n_features]
std::vector<double> load_X_binary(const std::string& path,
                                   long& n_samples, int& n_features) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int32_t rows, cols;
    f.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    n_samples  = rows;
    n_features = cols;
    std::vector<double> X(rows * cols);
    f.read(reinterpret_cast<char*>(X.data()), rows * cols * sizeof(double));
    return X;
}

// Binary format: [int32 n_samples][int32 * n_samples]
std::vector<int> load_y_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int32_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    std::vector<int32_t> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), n * sizeof(int32_t));
    return std::vector<int>(buf.begin(), buf.end());
}

// -----------------------------------------------------------------------------
// KNN — fully serial, thread-safe via pre-allocated per-thread buffers.
// Called from inside the OpenMP parallel region on the outer test loop.
// Exact: searches the full N_TRAIN training points loaded independently per rank.
//
// Why serial here: OpenMP is on the outer loop (one thread per test sample).
// Parallelising the inner distance loop too would cause nested parallelism
// with thousands of fork/join overheads — much worse than one outer fork/join.
// -----------------------------------------------------------------------------
double knn_predict_proba(const double* x,
                          const double* X_train_local,
                          const int*    y_train_local,
                          int n_train, int n_features, int k,
                          double* dist_buf, int* idx_buf) {

    // Distance computation — serial, private buffers passed by caller
    for (int i = 0; i < n_train; ++i) {
        double d = 0.0;
        const double* xt = X_train_local + i * n_features;
        for (int j = 0; j < n_features; ++j) {
            double diff = x[j] - xt[j];
            d += diff * diff;
        }
        dist_buf[i] = d;
    }

    // Top-k selection on pre-allocated index buffer
    int actual_k = std::min(k, n_train);
    std::iota(idx_buf, idx_buf + n_train, 0);
    std::partial_sort(idx_buf, idx_buf + actual_k, idx_buf + n_train,
                      [&](int a, int b){ return dist_buf[a] < dist_buf[b]; });

    int votes = 0;
    for (int i = 0; i < actual_k; ++i)
        votes += y_train_local[idx_buf[i]];

    return static_cast<double>(votes) / actual_k;
}

// -----------------------------------------------------------------------------
// Gaussian Naive Bayes  (identical to inference_seq.cpp)
// -----------------------------------------------------------------------------
double nb_predict_proba(const double* x,
                         const std::vector<double>& means,
                         const std::vector<double>& vars,
                         const std::vector<double>& priors,
                         int n_features) {
    double log_post[2];
    for (int c = 0; c < 2; ++c) {
        double lp = std::log(priors[c]);
        for (int j = 0; j < n_features; ++j) {
            double mu   = means[c * n_features + j];
            double var  = vars [c * n_features + j];
            double diff = x[j] - mu;
            lp += -0.5 * std::log(2.0 * M_PI * var)
                  - (diff * diff) / (2.0 * var);
        }
        log_post[c] = lp;
    }
    double max_lp = std::max(log_post[0], log_post[1]);
    double p1 = std::exp(log_post[1] - max_lp);
    double p0 = std::exp(log_post[0] - max_lp);
    return p1 / (p0 + p1);
}

// -----------------------------------------------------------------------------
// Logistic Regression  (identical to inference_seq.cpp)
// -----------------------------------------------------------------------------
double lr_predict_proba(const double* x,
                         const std::vector<double>& weights,
                         double bias, int n_features) {
    double z = bias;
    for (int j = 0; j < n_features; ++j)
        z += weights[j] * x[j];
    return 1.0 / (1.0 + std::exp(-z));
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {

    // ── MPI init ──────────────────────────────────────────────────────────
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ── OpenMP thread count ───────────────────────────────────────────────
    // Set via OMP_NUM_THREADS before launch. Read at runtime.
    const int n_threads = omp_get_max_threads();

    // ── All ranks load fixed model parameters from file ───────────────────
    if (rank == 0) std::cout << "Loading model parameters...\n";

    auto nb_means   = load_matrix(MODEL_DIR + "/nb_means.txt",   N_CLASSES, N_FEATURES);
    auto nb_vars    = load_matrix(MODEL_DIR + "/nb_vars.txt",    N_CLASSES, N_FEATURES);
    auto nb_priors  = load_vector(MODEL_DIR + "/nb_priors.txt");
    auto lr_weights = load_vector(MODEL_DIR + "/lr_weights.txt");
    auto lr_bias_v  = load_vector(MODEL_DIR + "/lr_bias.txt");
    double lr_bias  = lr_bias_v[0];

    // ── All ranks load full X_train from file independently (no MPI) ──────
    // Each rank searches all N_TRAIN training points per test sample — exact,
    // same accuracy as sequential.
    if (rank == 0) std::cout << "Loading training and test sets...\n";

    auto X_train_local = load_matrix(MODEL_DIR + "/X_train.txt", N_TRAIN, N_FEATURES);
    auto y_train_d     = load_vector(MODEL_DIR + "/y_train.txt");
    std::vector<int> y_train_local(y_train_d.begin(), y_train_d.end());

    // ── Rank 0 loads X_test and y_test, then scatters ────────────────────
    std::vector<double> X_test_all;
    std::vector<int>    y_test_all;
    long n_test = 0; int n_feat = 0;

    if (rank == 0) {
        X_test_all = load_X_binary(MODEL_DIR + "/X_test.bin", n_test, n_feat);
        y_test_all = load_y_binary(MODEL_DIR + "/y_test.bin");

        // Trim n_test to a multiple of size for uniform scatter
        n_test = (n_test / size) * size;
        X_test_all.resize(n_test * N_FEATURES);
        y_test_all.resize(n_test);
    }

    // Broadcast n_test so every rank can size its receive buffer
    MPI_Bcast(&n_test, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    const int local_n_test = (int)(n_test / size);

    if (rank == 0) {
        std::cout << "  N_train : " << N_TRAIN << " (full, per rank)\n";
        std::cout << "  N_test  : " << n_test  << "  →  " << local_n_test
                  << " per rank\n\n";
    }

    // ── MPI_Scatter X_test and y_test ─────────────────────────────────────
    std::vector<double> X_test_local(local_n_test * N_FEATURES);
    std::vector<int>    y_test_local(local_n_test);

    MPI_Scatter(rank == 0 ? X_test_all.data() : nullptr,
                local_n_test * N_FEATURES, MPI_DOUBLE,
                X_test_local.data(),
                local_n_test * N_FEATURES, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Scatter(rank == 0 ? y_test_all.data() : nullptr,
                local_n_test, MPI_INT,
                y_test_local.data(),
                local_n_test, MPI_INT,
                0, MPI_COMM_WORLD);

    // Free rank-0 X_test buffers now that scatter is done
    if (rank == 0) {
        std::vector<double>().swap(X_test_all);
        std::vector<int>().swap(y_test_all);
    }

    if (rank == 0) {
        std::cout << "Ranks: " << size
                  << "  Threads/rank: " << n_threads
                  << "  Total logical cores: " << size * n_threads << "\n\n";
    }

    // ── Pre-allocate per-thread KNN buffers (outside timing loop) ────────
    // Each thread gets its own dist and idx buffer — no allocation inside
    // the parallel region, no false sharing, no malloc overhead per sample.
    std::vector<std::vector<double>> dist_bufs(n_threads,
                                               std::vector<double>(N_TRAIN));
    std::vector<std::vector<int>>    idx_bufs (n_threads,
                                               std::vector<int>   (N_TRAIN));

    // ── Timing loop ───────────────────────────────────────────────────────
    std::vector<double> run_times(N_RUNS);
    std::vector<int>    local_preds(local_n_test);
    std::vector<int>    global_preds(rank == 0 ? n_test : 0);

    for (int run = 0; run < N_RUNS; ++run) {

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();      // ── timing starts ──

        // ── Local inference: OpenMP on the outer test loop ────────────────
        // One fork/join for the entire local test slice — minimal overhead.
        // Each thread processes a contiguous chunk of test samples (static).
        // Thread-safe: reads shared params, writes to distinct local_preds[i].
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < local_n_test; ++i) {
            int tid = omp_get_thread_num();
            const double* x = X_test_local.data() + i * N_FEATURES;

            double p_lr  = lr_predict_proba(x, lr_weights, lr_bias,   N_FEATURES);
            double p_nb  = nb_predict_proba(x, nb_means, nb_vars,
                                            nb_priors,                 N_FEATURES);
            double p_knn = knn_predict_proba(x,
                                             X_train_local.data(),
                                             y_train_local.data(),
                                             N_TRAIN, N_FEATURES, KNN_K,
                                             dist_bufs[tid].data(),
                                             idx_bufs[tid].data());

            local_preds[i] = ((p_knn + p_nb + p_lr) / 3.0 >= 0.5) ? 1 : 0;
        }
        // Implicit OpenMP barrier — all threads done before MPI communication.

        // ── Non-blocking gather: predictions → rank 0 ─────────────────────
        // Rank 0 posts MPI_Irecv for every non-root rank.
        // Non-root ranks post MPI_Isend to rank 0.
        // MPI_Waitall ensures all transfers complete before timing stops.
        if (rank == 0) {
            // Rank 0 copies its own slice directly
            std::copy(local_preds.begin(), local_preds.end(),
                      global_preds.begin());

            // Post non-blocking receives from ranks 1 .. size-1
            std::vector<MPI_Request> reqs(size - 1);
            for (int r = 1; r < size; ++r) {
                MPI_Irecv(global_preds.data() + r * local_n_test,
                          local_n_test, MPI_INT,
                          r, 0, MPI_COMM_WORLD, &reqs[r - 1]);
            }
            MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        } else {
            // Non-root: non-blocking send to rank 0
            MPI_Request req;
            MPI_Isend(local_preds.data(), local_n_test, MPI_INT,
                      0, 0, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }

        double t_end = MPI_Wtime();        // ── timing ends ──
        run_times[run] = t_end - t_start;

        if (rank == 0)
            std::cout << "  Run " << run + 1 << "/" << N_RUNS
                      << "  time = " << run_times[run] << " s\n";
    }

    // ── Median time ───────────────────────────────────────────────────────
    std::vector<double> sorted_times = run_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median_time = sorted_times[N_RUNS / 2];

    // ── Accuracy via MPI_Reduce ───────────────────────────────────────────
    // Each rank counts its own correct predictions independently.
    // MPI_Reduce sums them at rank 0 — no need to send individual labels.
    long local_correct = 0;
    for (int i = 0; i < local_n_test; ++i)
        local_correct += (local_preds[i] == y_test_local[i]);

    long global_correct = 0;
    MPI_Reduce(&local_correct, &global_correct, 1, MPI_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    // ── Report (rank 0 only) ──────────────────────────────────────────────
    if (rank == 0) {
        double accuracy = static_cast<double>(global_correct) / n_test;

        std::cout << "\n========================================\n";
        std::cout << "  HYBRID MPI+OpenMP RESULTS\n";
        std::cout << "========================================\n";
        std::cout << "  Ranks       : " << size                       << "\n";
        std::cout << "  Threads/rank: " << n_threads                  << "\n";
        std::cout << "  N_test      : " << n_test                     << "\n";
        std::cout << "  N_train     : " << N_TRAIN                    << "\n";
        std::cout << "  KNN k       : " << KNN_K                      << "\n";
        std::cout << "  Accuracy    : " << accuracy                   << "\n";
        std::cout << "  Median time : " << median_time << " s\n";
        std::cout << "========================================\n";
    }

    MPI_Finalize();
    return 0;
}