// =============================================================================
// Parallel Ensemble Inference Engine — OpenMP Task Parallelism (sections)
// HPC Project · Andrea · University of Messina
//
// This version demonstrates TASK PARALLELISM: the three classifiers (KNN, NB, LR)
// run concurrently in separate omp sections — one thread per model.
//
// Expected result: speedup ≈ 1.0x regardless of thread count.
// Reason: KNN costs O(N × N_train × d), NB and LR cost O(N × d).
// NB and LR finish in milliseconds; their threads sit idle waiting for KNN.
// The implicit barrier at end of sections forces all threads to synchronize.
// This is the load imbalance finding — documented, not a bug.
//
// Each section processes ALL N_test samples for its model, storing probabilities
// in a shared array. Soft voting combines the three arrays after the barrier.
//
// Compile: mpic++ -O2 -fopenmp -o inference_sections inference_sections.cpp
// Run:     OMP_NUM_THREADS=3 mpirun -n 1 ./inference_sections
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
// Model inference — per-sample functions, identical to previous versions
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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 1) {
        if (rank == 0) std::cerr << "Sections version: run with mpirun -n 1\n";
        MPI_Finalize(); return 1;
    }

    // ── Load model parameters ────────────────────────────────────────────────
    std::cout << "Loading model parameters...\n";
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

    // ── Load test set ────────────────────────────────────────────────────────
    std::cout << "Loading test set...\n";
    long n_test; int n_feat;
    auto X_test_v = load_X_binary(MODEL_DIR+"/X_test.bin", n_test, n_feat);
    auto y_test   = load_y_binary(MODEL_DIR+"/y_test.bin");
    const double* X_test = X_test_v.data();

    // Probability arrays: one per model, filled by each section independently
    std::vector<double> p_knn_arr(n_test), p_nb_arr(n_test), p_lr_arr(n_test);

    // KNN distance buffer — one per thread (3 sections = 3 threads max)
    int max_t = omp_get_max_threads();
    std::vector<std::vector<std::pair<double,int>>>
        dist_bufs(max_t, std::vector<std::pair<double,int>>(N_TRAIN));

    // Per-section timing: measure how long each model actually takes
    double t_knn = 0, t_nb = 0, t_lr = 0;

    std::cout << "OpenMP threads available: " << max_t << "\n";
    std::cout << "Test samples            : " << n_test << "\n\n";

    // ── Timed inference loop ─────────────────────────────────────────────────
    std::vector<double> run_times(N_RUNS);
    std::vector<int>    predictions(n_test);

    for (int run = 0; run < N_RUNS; ++run) {

        double t_start = MPI_Wtime();  // ── timing starts ──

        // Three sections run concurrently — one thread per section.
        // Each section processes ALL N_test samples for its model.
        // The implicit barrier at the end forces synchronization:
        // the rank cannot proceed until the slowest section (KNN) finishes.
        #pragma omp parallel sections
        {
            // ── Section 0: KNN ───────────────────────────────────────────────
            // Dominates total time: O(N × N_train × d)
            // Thread assigned here will be busy until all other threads are idle
            #pragma omp section
            {
                int tid = omp_get_thread_num();
                double ts = MPI_Wtime();
                for (long i = 0; i < n_test; ++i) {
                    p_knn_arr[i] = knn_predict_proba(
                        X_test + i*N_FEATURES, X_train, y_train_ptr,
                        N_TRAIN, N_FEATURES, KNN_K, dist_bufs[tid].data());
                }
                t_knn = MPI_Wtime() - ts;
            }

            // ── Section 1: Naive Bayes ───────────────────────────────────────
            // Finishes in milliseconds: O(N × d)
            // Thread idles at the barrier waiting for KNN
            #pragma omp section
            {
                double ts = MPI_Wtime();
                for (long i = 0; i < n_test; ++i) {
                    p_nb_arr[i] = nb_predict_proba(
                        X_test + i*N_FEATURES, nb_means, nb_vars,
                        nb_priors, N_FEATURES);
                }
                t_nb = MPI_Wtime() - ts;
            }

            // ── Section 2: Logistic Regression ──────────────────────────────
            // Finishes in milliseconds: O(N × d)
            // Thread idles at the barrier waiting for KNN
            #pragma omp section
            {
                double ts = MPI_Wtime();
                for (long i = 0; i < n_test; ++i) {
                    p_lr_arr[i] = lr_predict_proba(
                        X_test + i*N_FEATURES, lr_weights, lr_bias, N_FEATURES);
                }
                t_lr = MPI_Wtime() - ts;
            }
        }
        // ── Implicit barrier: all threads synchronize here ───────────────────
        // Total wall time = max(t_knn, t_nb, t_lr) ≈ t_knn

        // Soft voting: combine the three probability arrays
        for (long i = 0; i < n_test; ++i)
            predictions[i] = ((p_knn_arr[i]+p_nb_arr[i]+p_lr_arr[i])/3.0 >= 0.5) ? 1 : 0;

        double t_end = MPI_Wtime();    // ── timing ends ──

        run_times[run] = t_end - t_start;
        std::cout << "  Run " << run+1 << "/" << N_RUNS
                  << "  time = " << run_times[run] << " s"
                  << "  [KNN=" << t_knn << "s  NB=" << t_nb
                  << "s  LR=" << t_lr << "s]\n";
    }

    // ── Median time ──────────────────────────────────────────────────────────
    std::vector<double> sorted_times = run_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median_time = sorted_times[N_RUNS/2];

    // ── Accuracy ─────────────────────────────────────────────────────────────
    long correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (long i = 0; i < n_test; ++i)
        correct += (predictions[i] == y_test[i]);
    double accuracy = static_cast<double>(correct)/n_test;

    // ── Report ───────────────────────────────────────────────────────────────
    double speedup = T1_BASELINE / median_time;
    double idle_fraction = 1.0 - (t_knn / (t_knn + t_nb + t_lr));

    std::cout << "\n========================================\n";
    std::cout << "  OMP SECTIONS (TASK PARALLELISM)\n";
    std::cout << "========================================\n";
    std::cout << "  Threads     : " << max_t         << "\n";
    std::cout << "  N_test      : " << n_test         << "\n";
    std::cout << "  Accuracy    : " << accuracy       << "\n";
    std::cout << "  Median time : " << median_time    << " s\n";
    std::cout << "  Speedup     : " << speedup        << "x  (vs T1=" << T1_BASELINE << "s)\n";
    std::cout << "  -- Load imbalance breakdown (last run) --\n";
    std::cout << "  KNN time    : " << t_knn          << " s\n";
    std::cout << "  NB time     : " << t_nb           << " s\n";
    std::cout << "  LR time     : " << t_lr           << " s\n";
    std::cout << "  Idle frac.  : " << idle_fraction*100 << " %  (NB+LR threads waiting)\n";
    std::cout << "========================================\n";

    MPI_Finalize();
    return 0;
}
