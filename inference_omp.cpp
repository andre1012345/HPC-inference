// =============================================================================
// Parallel Ensemble Inference Engine — OpenMP Data Parallelism
// HPC Project · Andrea · University of Messina
//
// Key fix vs previous version: KNN distance buffer pre-allocated per thread.
// Allocating std::vector inside the parallel loop caused heap contention —
// the allocator serializes concurrent allocations, inflating the serial
// fraction and killing speedup (Amdahl's Law in action).
//
// Solution: allocate [max_threads][N_TRAIN] buffer before the parallel region.
// Each thread indexes its own row via omp_get_thread_num() — zero contention.
//
// Compile: mpic++ -O2 -fopenmp -o inference_omp inference_omp.cpp
// Run:     OMP_NUM_THREADS=4 mpirun -n 1 ./inference_omp
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
static const int    N_FEATURES = 20;
static const int    N_TRAIN    = 50000;
static const int    KNN_K      = 5;
static const int    N_CLASSES  = 2;
static const int    N_RUNS     = 5;
static const std::string MODEL_DIR = "model_params";

// T1 baseline measured from sequential run — used to compute speedup
static const double T1_BASELINE = 17.73;

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
    if ((int)m.size() != rows*cols) { std::cerr << "ERROR: size mismatch in " << path << "\n"; MPI_Abort(MPI_COMM_WORLD,1); }
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
// KNN inference — uses a pre-allocated per-thread distance buffer.
// The buffer pointer is passed in from the parallel region so no heap
// allocation happens inside this function.
// -----------------------------------------------------------------------------
double knn_predict_proba(const double* x,
                         const double* X_train, const int* y_train,
                         int n_train, int n_features, int k,
                         std::pair<double,int>* dist_buf) {  // pre-allocated buffer
    for (int i = 0; i < n_train; ++i) {
        double d = 0.0;
        const double* xt = X_train + i * n_features;
        for (int j = 0; j < n_features; ++j) {
            double diff = x[j] - xt[j];
            d += diff * diff;
        }
        dist_buf[i] = {d, y_train[i]};
    }
    std::partial_sort(dist_buf, dist_buf + k, dist_buf + n_train);
    int votes = 0;
    for (int i = 0; i < k; ++i) votes += dist_buf[i].second;
    return static_cast<double>(votes) / k;
}

// Naive Bayes: log-posterior + numerically stable softmax
double nb_predict_proba(const double* x,
                        const double* means, const double* vars,
                        const double* priors, int n_features) {
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

// Logistic Regression: sigmoid(w · x + b)
double lr_predict_proba(const double* x, const double* weights,
                        double bias, int n_features) {
    double z = bias;
    for (int j = 0; j < n_features; ++j) z += weights[j] * x[j];
    return 1.0 / (1.0 + std::exp(-z));
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 1) {
        if (rank == 0) std::cerr << "OpenMP-only version: run with mpirun -n 1\n";
        MPI_Finalize(); return 1;
    }

    // ── Load model parameters ────────────────────────────────────────────────
    std::cout << "Loading model parameters...\n";

    auto X_train_v    = load_matrix(MODEL_DIR + "/X_train.txt", N_TRAIN, N_FEATURES);
    auto y_train_d    = load_vector(MODEL_DIR + "/y_train.txt");
    auto nb_means_v   = load_matrix(MODEL_DIR + "/nb_means.txt",  N_CLASSES, N_FEATURES);
    auto nb_vars_v    = load_matrix(MODEL_DIR + "/nb_vars.txt",   N_CLASSES, N_FEATURES);
    auto nb_priors_v  = load_vector(MODEL_DIR + "/nb_priors.txt");
    auto lr_weights_v = load_vector(MODEL_DIR + "/lr_weights.txt");
    auto lr_bias_v    = load_vector(MODEL_DIR + "/lr_bias.txt");

    const double* X_train    = X_train_v.data();
    const double* nb_means   = nb_means_v.data();
    const double* nb_vars    = nb_vars_v.data();
    const double* nb_priors  = nb_priors_v.data();
    const double* lr_weights = lr_weights_v.data();
    const double  lr_bias    = lr_bias_v[0];

    std::vector<int> y_train(y_train_d.begin(), y_train_d.end());
    const int* y_train_ptr = y_train.data();

    // ── Load test set ────────────────────────────────────────────────────────
    std::cout << "Loading test set...\n";
    long n_test; int n_feat;
    auto X_test_v = load_X_binary(MODEL_DIR + "/X_test.bin", n_test, n_feat);
    auto y_test   = load_y_binary(MODEL_DIR + "/y_test.bin");
    const double* X_test = X_test_v.data();

    int n_threads = omp_get_max_threads();
    std::cout << "OpenMP threads : " << n_threads << "\n";
    std::cout << "Test samples   : " << n_test    << "\n\n";

    // ── Pre-allocate per-thread KNN distance buffers ─────────────────────────
    // Each thread gets its own row of size N_TRAIN — allocated once here,
    // reused across all samples and all runs. Zero heap activity in the loop.
    std::vector<std::vector<std::pair<double,int>>>
        dist_bufs(n_threads, std::vector<std::pair<double,int>>(N_TRAIN));

    // ── Timed inference loop ─────────────────────────────────────────────────
    std::vector<double> run_times(N_RUNS);
    std::vector<int>    predictions(n_test);

    for (int run = 0; run < N_RUNS; ++run) {

        double t_start = MPI_Wtime();  // ── timing starts ──

        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n_test; ++i) {
            int tid = omp_get_thread_num();          // unique per thread
            const double* x = X_test + i * N_FEATURES;

            double p_knn = knn_predict_proba(x, X_train, y_train_ptr,
                                             N_TRAIN, N_FEATURES, KNN_K,
                                             dist_bufs[tid].data());  // thread-local buffer
            double p_nb  = nb_predict_proba(x, nb_means, nb_vars, nb_priors,
                                            N_FEATURES);
            double p_lr  = lr_predict_proba(x, lr_weights, lr_bias, N_FEATURES);

            predictions[i] = ((p_knn + p_nb + p_lr) / 3.0 >= 0.5) ? 1 : 0;
        }

        double t_end = MPI_Wtime();    // ── timing ends ──

        run_times[run] = t_end - t_start;
        std::cout << "  Run " << run+1 << "/" << N_RUNS
                  << "  time = " << run_times[run] << " s\n";
    }

    // ── Median time ──────────────────────────────────────────────────────────
    std::vector<double> sorted_times = run_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median_time = sorted_times[N_RUNS / 2];

    // ── Accuracy ─────────────────────────────────────────────────────────────
    long correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (long i = 0; i < n_test; ++i)
        correct += (predictions[i] == y_test[i]);
    double accuracy = static_cast<double>(correct) / n_test;

    // ── Report ───────────────────────────────────────────────────────────────
    double speedup = T1_BASELINE / median_time;

    std::cout << "\n========================================\n";
    std::cout << "  OPENMP DATA PARALLELISM RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "  Threads     : " << n_threads   << "\n";
    std::cout << "  N_test      : " << n_test       << "\n";
    std::cout << "  Accuracy    : " << accuracy     << "\n";
    std::cout << "  Median time : " << median_time  << " s\n";
    std::cout << "  Speedup     : " << speedup      << "x  (vs T1=" << T1_BASELINE << "s)\n";
    std::cout << "========================================\n";

    MPI_Finalize();
    return 0;
}