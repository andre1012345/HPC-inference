# =============================================================================
# train.py — Offline Training Pipeline
# HPC Project · Andrea · University of Messina
#
# Trains three classifiers on a synthetic binary classification dataset and
# exports all model parameters to model_params/.
# The C++ inference engine loads these files at startup — no training happens
# at runtime.
#
# Usage:
#   pip install numpy scikit-learn
#   python train.py
#
# Output: model_params/ directory with all parameter files (see bottom of script).
# =============================================================================

import os
import struct
import numpy as np
from sklearn.datasets        import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, classification_report

# =============================================================================
# Configuration — change these to regenerate a different dataset
# =============================================================================
RANDOM_STATE  = 42       # fixed seed → fully reproducible across runs
N_SAMPLES     = 70_000   # total samples (50k train + 20k test)
N_FEATURES    = 20       # feature dimensionality
N_INFORMATIVE = 10       # truly informative features
N_REDUNDANT   = 2        # linear combinations of informative ones
CLASS_SEP     = 1.0      # separation between classes (1.0 = non-trivial but solvable)
TEST_SIZE     = 20_000   # held-out samples for accuracy validation
KNN_K         = 5        # number of neighbours — must match KNN_K in inference_dist_knn.cpp

OUT_DIR = "model_params"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. Dataset generation
# =============================================================================
print("Generating dataset...")
X, y = make_classification(
    n_samples     = N_SAMPLES,
    n_features    = N_FEATURES,
    n_informative = N_INFORMATIVE,
    n_redundant   = N_REDUNDANT,
    n_classes     = 2,
    class_sep     = CLASS_SEP,
    random_state  = RANDOM_STATE,
)
print(f"  Shape: {X.shape}  |  Class balance: {np.bincount(y)}")

# =============================================================================
# 2. Train / test split
# =============================================================================
# stratify=y preserves class balance in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y,
)
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

# =============================================================================
# 3. Feature scaling (StandardScaler)
# =============================================================================
# KNN needs scaling because it uses Euclidean distance — unscaled features
# with large ranges would dominate the distance. LR is gradient-sensitive.
# NB is scale-invariant but we apply the same transform for consistency.
#
# Rule: fit ONLY on training data, then apply to both splits.
# The scaler parameters are exported so C++ applies the same transform.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =============================================================================
# 4. Model training
# =============================================================================
print("\nTraining models...")

# KNN is a lazy learner — "training" just stores X_train
knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean", algorithm="brute")
knn.fit(X_train, y_train)

nb = GaussianNB()
nb.fit(X_train, y_train)

# liblinear solver is well-suited for binary classification on small-to-medium datasets
lr = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
lr.fit(X_train, y_train)

# =============================================================================
# 5. Accuracy validation
# =============================================================================
# Soft-voting ensemble: average the predicted probabilities of all three models,
# then threshold at 0.5 — same logic used by the C++ inference engine.
p_knn = knn.predict_proba(X_test)[:, 1]
p_nb  = nb.predict_proba(X_test)[:, 1]
p_lr  = lr.predict_proba(X_test)[:, 1]
y_ensemble = ((p_knn + p_nb + p_lr) / 3 >= 0.5).astype(int)

print(f"\n  KNN accuracy      : {accuracy_score(y_test, knn.predict(X_test)):.4f}")
print(f"  Naive Bayes acc.  : {accuracy_score(y_test, nb.predict(X_test)):.4f}")
print(f"  Logistic Reg. acc.: {accuracy_score(y_test, lr.predict(X_test)):.4f}")
print(f"  Ensemble accuracy : {accuracy_score(y_test, y_ensemble):.4f}")
print()
print(classification_report(y_test, y_ensemble))

# =============================================================================
# 6. Export model parameters
# =============================================================================
print(f"Exporting parameters to {OUT_DIR}/ ...")

# -- Dataset metadata (read by C++ to know array dimensions and k) ------------
with open(f"{OUT_DIR}/config.txt", "w") as f:
    f.write(f"n_features {N_FEATURES}\n")
    f.write(f"n_train {X_train.shape[0]}\n")
    f.write(f"knn_k {KNN_K}\n")
    f.write(f"random_state {RANDOM_STATE}\n")
    f.write(f"n_classes 2\n")

# -- StandardScaler: one value per line, matching feature order ---------------
np.savetxt(f"{OUT_DIR}/scaler_mean.txt", scaler.mean_)
np.savetxt(f"{OUT_DIR}/scaler_std.txt",  scaler.scale_)

# -- KNN: the full training set IS the model (lazy learner) -------------------
# X_train.txt is ~8 MB — the C++ engine loads it once into a contiguous array.
np.savetxt(f"{OUT_DIR}/X_train.txt", X_train, fmt="%.8f")
np.savetxt(f"{OUT_DIR}/y_train.txt", y_train, fmt="%d")

# -- Gaussian Naive Bayes: per-class feature means, variances, and priors -----
np.savetxt(f"{OUT_DIR}/nb_means.txt",  nb.theta_,       fmt="%.8f")
np.savetxt(f"{OUT_DIR}/nb_vars.txt",   nb.var_,         fmt="%.8f")
np.savetxt(f"{OUT_DIR}/nb_priors.txt", nb.class_prior_, fmt="%.8f")

# -- Logistic Regression: weight vector and bias scalar -----------------------
# C++ inference: P(y=1|x) = sigmoid(w · x + b),  sigmoid(z) = 1/(1+exp(-z))
np.savetxt(f"{OUT_DIR}/lr_weights.txt", lr.coef_,      fmt="%.8f")
np.savetxt(f"{OUT_DIR}/lr_bias.txt",    lr.intercept_, fmt="%.8f")

# -- Test set as a flat binary file (fast to load in C++) ---------------------
# Format: [int32 n_samples][int32 n_features][float64 × n_samples × n_features]
def _write_binary_matrix(path, arr):
    with open(path, "wb") as f:
        rows, cols = arr.shape
        f.write(struct.pack("ii", rows, cols))
        f.write(arr.astype(np.float64).tobytes())

def _write_binary_labels(path, arr):
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(arr)))
        f.write(arr.astype(np.int32).tobytes())

_write_binary_matrix(f"{OUT_DIR}/X_test.bin", X_test)
_write_binary_labels(f"{OUT_DIR}/y_test.bin", y_test)

# =============================================================================
# 7. Summary
# =============================================================================
print(f"\n{'File':<25} {'Size':>10}")
print("-" * 37)
for fname in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, fname))
    unit, val = ("MB", size / 1e6) if size >= 1_000_000 else ("KB", size / 1e3)
    print(f"{fname:<25} {val:>8.1f} {unit}")
print("\nDone. Run ./inference_dist_knn to start inference.")
