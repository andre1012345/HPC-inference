// =============================================================================
// Parallel Ensemble Inference Engine — Sequential Baseline
// HPC Project · Andrea · University of Messina
//
// Sequential reference implementation: p=1, t=1.
// Test set loaded from binary file exported by Colab — exact same distribution
// the models were trained on, enabling meaningful accuracy validation.
//
// Timing: MPI_Wtime() wraps only the inference phase.
//         File I/O is excluded from timing.
//         Median of N_RUNS reported to reduce system noise.
//
// Fix (v2): the KNN distance buffer is now allocated once before the run loop
//           and reused across all N_test x N_RUNS calls. The original version
//           allocated a 50k-element heap vector on every knn_predict_proba
//           call (20k times per run), inflating the baseline with allocator
//           overhead that does not reflect actual compute work.
// =============================================================================

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
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

// -----------------------------------------------------------------------------
// I/O helpers
// -----------------------------------------------------------------------------

// Load a flat vector from a plain-text file (one value per line or space-separated).
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

// Load a 2-D matrix from plain text (rows x cols values, row-major).
std::vector<double> load_matrix(const std::string& path, int rows, int cols) {
    std::vector<double> m = load_vector(path);
    if ((int)m.size() != rows * cols) {
        std::cerr << "ERROR: " << path << " expected " << rows * cols
                  << " values, got " << m.size() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return m;
}

// Load X_test from binary file exported by Colab.
// Format: [int32 n_samples][int32 n_features][float64 x n_samples x n_features]
std::vector<double> load_X_binary(const std::string& path, long& n_samples, int& n_features) {
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

// Load y_test from binary file exported by Colab.
// Format: [int32 n_samples][int32 x n_samples]
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
// KNN inference — brute-force Euclidean distance, majority vote over k neighbours.
// Complexity: O(N_train * n_features) per query — intentionally the heaviest model.
//
// 'dist_buf' (doubles) and 'idx_buf' (ints) are pre-allocated scratch buffers
// of size N_train passed by the caller. Reusing them avoids 20k heap allocations
// per run (one per test sample).
// -----------------------------------------------------------------------------
double knn_predict_proba(const double* x,
                         const std::vector<double>& X_train,
                         const std::vector<int>&    y_train,
                         double* dist_buf, int* idx_buf,
                         int n_train, int n_features, int k) {
    for (int i = 0; i < n_train; ++i) {
        double d = 0.0;
        const double* xt = X_train.data() + i * n_features;
        for (int j = 0; j < n_features; ++j) {
            double diff = x[j] - xt[j];
            d += diff * diff;
        }
        dist_buf[i] = d;
    }
    std::iota(idx_buf, idx_buf + n_train, 0);
    std::partial_sort(idx_buf, idx_buf + k, idx_buf + n_train,
                      [&](int a, int b){ return dist_buf[a] < dist_buf[b]; });
    int votes = 0;
    for (int i = 0; i < k; ++i) votes += y_train[idx_buf[i]];
    return static_cast<double>(votes) / k;
}

// -----------------------------------------------------------------------------
// Gaussian Naive Bayes inference.
// P(class=1|x) via log-posterior, numerically stabilised with log-sum-exp trick.
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
// Logistic Regression inference.
// P(class=1|x) = sigmoid(w · x + b)
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 1) {
        if (rank == 0)
            std::cerr << "Sequential baseline must run with mpirun -n 1\n";
        MPI_Finalize();
        return 1;
    }

    // Load model parameters once at startup.
    // Note: X_train.txt and X_test.bin are pre-scaled by the Python pipeline.
    // Scaler parameters are stored in model_params/ for reference but are
    // not applied here (data is already normalised).
    std::cout << "Loading model parameters...\n";

    auto X_train    = load_matrix(MODEL_DIR + "/X_train.txt", N_TRAIN, N_FEATURES);
    auto y_train_d  = load_vector(MODEL_DIR + "/y_train.txt");
    auto nb_means   = load_matrix(MODEL_DIR + "/nb_means.txt",  N_CLASSES, N_FEATURES);
    auto nb_vars    = load_matrix(MODEL_DIR + "/nb_vars.txt",   N_CLASSES, N_FEATURES);
    auto nb_priors  = load_vector(MODEL_DIR + "/nb_priors.txt");
    auto lr_weights = load_vector(MODEL_DIR + "/lr_weights.txt");
    auto lr_bias_v  = load_vector(MODEL_DIR + "/lr_bias.txt");

    std::vector<int> y_train(y_train_d.begin(), y_train_d.end());
    double lr_bias = lr_bias_v[0];

    // Load test set from binary file.
    std::cout << "Loading test set from binary file...\n";

    long n_test; int n_feat;
    auto X_test = load_X_binary(MODEL_DIR + "/X_test.bin", n_test, n_feat);
    auto y_test = load_y_binary(MODEL_DIR + "/y_test.bin");

    std::cout << "Test set loaded: " << n_test << " samples x "
              << n_feat << " features\n\n";

    // Pre-allocate KNN scratch buffers once, reused across all test samples
    // and all runs to eliminate repeated heap allocation.
    std::vector<double> dist_buf(N_TRAIN);
    std::vector<int>    idx_buf(N_TRAIN);

    // Timed inference loop.
    std::vector<double> run_times(N_RUNS);
    std::vector<int>    predictions(n_test);

    for (int run = 0; run < N_RUNS; ++run) {

        double t_start = MPI_Wtime();  // timing starts — I/O excluded

        for (long i = 0; i < n_test; ++i) {
            const double* x = X_test.data() + i * N_FEATURES;

            double p_knn = knn_predict_proba(x, X_train, y_train,
                                             dist_buf.data(), idx_buf.data(),
                                             N_TRAIN, N_FEATURES, KNN_K);
            double p_nb  = nb_predict_proba (x, nb_means, nb_vars, nb_priors,
                                             N_FEATURES);
            double p_lr  = lr_predict_proba (x, lr_weights, lr_bias, N_FEATURES);

            // Soft-voting ensemble: average probability, threshold at 0.5.
            predictions[i] = ((p_knn + p_nb + p_lr) / 3.0 >= 0.5) ? 1 : 0;
        }

        double t_end = MPI_Wtime();    // timing ends

        run_times[run] = t_end - t_start;
        std::cout << "  Run " << run + 1 << "/" << N_RUNS
                  << "  time = " << run_times[run] << " s\n";
    }

    // Median time across runs.
    std::vector<double> sorted_times = run_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median_time = sorted_times[N_RUNS / 2];

    // Accuracy on last run's predictions.
    long correct = 0;
    for (long i = 0; i < n_test; ++i)
        correct += (predictions[i] == y_test[i]);
    double accuracy = static_cast<double>(correct) / n_test;

    // Results report.
    std::cout << "\n========================================\n";
    std::cout << "  SEQUENTIAL BASELINE RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "  N_test      : " << n_test       << "\n";
    std::cout << "  N_train     : " << N_TRAIN      << "\n";
    std::cout << "  KNN k       : " << KNN_K        << "\n";
    std::cout << "  Accuracy    : " << accuracy     << "\n";
    std::cout << "  Median time : " << median_time  << " s  (T1 baseline)\n";
    std::cout << "========================================\n";

    MPI_Finalize();
    return 0;
}