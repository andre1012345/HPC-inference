// =============================================================================
// Parallel Ensemble Inference Engine — Hybrid MPI+OpenMP
// HPC Project · Andrea · University of Messina
//
// Combines MPI data parallelism (across ranks) with OpenMP data parallelism
// (across threads within each rank).
//
// Architecture:
//   MPI_Scatter  → Rank 0 distributes N/p samples to each rank
//   omp parallel for → each rank processes its N/p samples across t threads
//   MPI_Gather   → Rank 0 collects predictions from all ranks
//
// Thread safety: MPI_THREAD_FUNNELED — only the master thread makes MPI calls.
// OpenMP parallel region runs entirely between Scatter and Gather.
// No MPI calls inside the parallel region — safe with FUNNELED level.
//
// Compile: mpic++ -O2 -fopenmp -o inference_hybrid inference_hybrid.cpp
// Run:     OMP_NUM_THREADS=2 mpirun -n 2 ./inference_hybrid
// =============================================================================

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cassert>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
static const int    N_FEATURES  = 20;
static const int    N_TRAIN     = 50000;
static const int    KNN_K       = 5;
static const int    N_CLASSES   = 2;
static const int    N_RUNS      = 5;
static const double T1_BASELINE = 17.73;
static const std::string MODEL_DIR = "model_params";

// -----------------------------------------------------------------------------
// I/O helpers
// -----------------------------------------------------------------------------
std::vector<double> load_vector(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "ERROR: cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    std::vector<double> v; double x;
    while (f >> x) v.push_back(x);
    return v;
}

std::vector<double> load_matrix(const std::string& path, int rows, int cols) {
    std::vector<double> m = load_vector(path);
    if ((int)m.size() != rows*cols) { std::cerr << "ERROR: size mismatch " << path << "\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    return m;
}

std::vector<double> load_X_binary(const std::string& path, long& n_samples, int& n_features) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "ERROR: cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    int32_t rows, cols;
    f.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    n_samples = rows; n_features = cols;
    std::vector<double> X(rows*cols);
    f.read(reinterpret_cast<char*>(X.data()), rows*cols*sizeof(double));
    return X;
}

std::vector<int> load_y_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "ERROR: cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    int32_t n; f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    std::vector<int32_t> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), n*sizeof(int32_t));
    return std::vector<int>(buf.begin(), buf.end());
}

// -----------------------------------------------------------------------------
// Model inference functions — thread-safe (all locals on stack)
// -----------------------------------------------------------------------------
double knn_predict_proba(const double* x,
                         const double* X_train, const int* y_train,
                         int n_train, int n_features, int k,
                         std::pair<double,int>* dist_buf) {
    for (int i = 0; i < n_train; ++i) {
        double d = 0.0;
        const double* xt = X_train + i * n_features;
        for (int j = 0; j < n_features; ++j) {
            double diff = x[j] - xt[j]; d += diff * diff;
        }
        dist_buf[i] = {d, y_train[i]};
    }
    std::partial_sort(dist_buf, dist_buf + k, dist_buf + n_train);
    int votes = 0;
    for (int i = 0; i < k; ++i) votes += dist_buf[i].second;
    return static_cast<double>(votes) / k;
}

double nb_predict_proba(const double* x, const double* means,
                        const double* vars, const double* priors, int n_features) {
    double log_post[2];
    for (int c = 0; c < 2; ++c) {
        double lp = std::log(priors[c]);
        for (int j = 0; j < n_features; ++j) {
            double mu = means[c*n_features+j], var = vars[c*n_features+j];
            double diff = x[j] - mu;
            lp += -0.5*std::log(2.0*M_PI*var) - (diff*diff)/(2.0*var);
        }
        log_post[c] = lp;
    }
    double max_lp = std::max(log_post[0], log_post[1]);
    double p1 = std::exp(log_post[1]-max_lp), p0 = std::exp(log_post[0]-max_lp);
    return p1/(p0+p1);
}

double lr_predict_proba(const double* x, const double* weights,
                        double bias, int n_features) {
    double z = bias;
    for (int j = 0; j < n_features; ++j) z += weights[j]*x[j];
    return 1.0/(1.0+std::exp(-z));
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {

    // MPI_THREAD_FUNNELED: only the master thread will make MPI calls.
    // OpenMP threads run between Scatter and Gather — no MPI inside parallel region.
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "WARNING: MPI_THREAD_FUNNELED not supported, got level "
                  << provided << "\n";
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_threads = omp_get_max_threads();

    // ── Load model parameters on all ranks ───────────────────────────────────
    auto X_train_v    = load_matrix(MODEL_DIR+"/X_train.txt", N_TRAIN, N_FEATURES);
    auto y_train_d    = load_vector(MODEL_DIR+"/y_train.txt");
    auto nb_means_v   = load_matrix(MODEL_DIR+"/nb_means.txt",  N_CLASSES, N_FEATURES);
    auto nb_vars_v    = load_matrix(MODEL_DIR+"/nb_vars.txt",   N_CLASSES, N_FEATURES);
    auto nb_priors_v  = load_vector(MODEL_DIR+"/nb_priors.txt");
    auto lr_weights_v = load_vector(MODEL_DIR+"/lr_weights.txt");
    auto lr_bias_v    = load_vector(MODEL_DIR+"/lr_bias.txt");

    const double* X_train    = X_train_v.data();
    const double* nb_means   = nb_means_v.data();
    const double* nb_vars    = nb_vars_v.data();
    const double* nb_priors  = nb_priors_v.data();
    const double* lr_weights = lr_weights_v.data();
    const double  lr_bias    = lr_bias_v[0];
    std::vector<int> y_train(y_train_d.begin(), y_train_d.end());
    const int* y_train_ptr = y_train.data();

    // ── Load test set on Rank 0 ──────────────────────────────────────────────
    long n_test = 0;
    std::vector<double> X_test_global;
    std::vector<int>    y_test;

    if (rank == 0) {
        int n_feat;
        X_test_global = load_X_binary(MODEL_DIR+"/X_test.bin", n_test, n_feat);
        y_test        = load_y_binary(MODEL_DIR+"/y_test.bin");
        std::cout << "Ranks       : " << size      << "\n";
        std::cout << "Threads/rank: " << n_threads << "\n";
        std::cout << "Total cores : " << size * n_threads << "\n";
        std::cout << "N_test      : " << n_test    << "\n";
        std::cout << "N per rank  : " << n_test/size << "\n\n";
    }

    MPI_Bcast(&n_test, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    long local_n     = n_test / size;
    int  local_elems = local_n * N_FEATURES;

    std::vector<double> X_local(local_n * N_FEATURES);
    std::vector<int>    local_preds(local_n);
    std::vector<int>    global_preds(rank == 0 ? n_test : 0);

    // Pre-allocate per-thread KNN distance buffers
    std::vector<std::vector<std::pair<double,int>>>
        dist_bufs(n_threads, std::vector<std::pair<double,int>>(N_TRAIN));

    // ── Timed inference loop ─────────────────────────────────────────────────
    std::vector<double> run_times(N_RUNS);

    for (int run = 0; run < N_RUNS; ++run) {

        // ── MPI_Scatter: master thread distributes data ───────────────────────
        double t_start = MPI_Wtime();  // ── timing starts ──

        MPI_Scatter(
            X_test_global.data(), local_elems, MPI_DOUBLE,
            X_local.data(),       local_elems, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // ── OpenMP: each rank parallelises its local N/p samples ──────────────
        // Master thread does not make MPI calls here — safe with FUNNELED.
        // All variables declared inside the loop are private by default.
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < local_n; ++i) {
            int tid = omp_get_thread_num();
            const double* x = X_local.data() + i * N_FEATURES;

            double p_knn = knn_predict_proba(x, X_train, y_train_ptr,
                                             N_TRAIN, N_FEATURES, KNN_K,
                                             dist_bufs[tid].data());
            double p_nb  = nb_predict_proba(x, nb_means, nb_vars,
                                            nb_priors, N_FEATURES);
            double p_lr  = lr_predict_proba(x, lr_weights, lr_bias, N_FEATURES);

            local_preds[i] = ((p_knn + p_nb + p_lr) / 3.0 >= 0.5) ? 1 : 0;
        }
        // ── Implicit OpenMP barrier: all threads complete before Gather ───────

        // ── MPI_Gather: master thread collects results ────────────────────────
        MPI_Gather(
            local_preds.data(),  local_n, MPI_INT,
            global_preds.data(), local_n, MPI_INT,
            0, MPI_COMM_WORLD);

        double t_end = MPI_Wtime();    // ── timing ends ──

        run_times[run] = t_end - t_start;

        if (rank == 0)
            std::cout << "  Run " << run+1 << "/" << N_RUNS
                      << "  time = " << run_times[run] << " s\n";
    }

    // ── Results on Rank 0 ────────────────────────────────────────────────────
    if (rank == 0) {
        std::vector<double> sorted_times = run_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        double median_time = sorted_times[N_RUNS/2];

        long correct = 0;
        for (long i = 0; i < n_test; ++i)
            correct += (global_preds[i] == y_test[i]);
        double accuracy = static_cast<double>(correct) / n_test;
        double speedup  = T1_BASELINE / median_time;

        std::cout << "\n========================================\n";
        std::cout << "  HYBRID MPI+OpenMP RESULTS\n";
        std::cout << "========================================\n";
        std::cout << "  Ranks       : " << size         << "\n";
        std::cout << "  Threads/rank: " << n_threads    << "\n";
        std::cout << "  N_test      : " << n_test        << "\n";
        std::cout << "  Accuracy    : " << accuracy      << "\n";
        std::cout << "  Median time : " << median_time   << " s\n";
        std::cout << "  Speedup     : " << speedup       << "x  (vs T1=" << T1_BASELINE << "s)\n";
        std::cout << "========================================\n";
    }

    MPI_Finalize();
    return 0;
}
