#!/bin/bash
# weak_scaling.sh — Weak scaling experiment for the HPC inference engine.
#
# Protocol (matches report Section 5.2):
#   Work per rank fixed at SAMPLES_PER_RANK test samples, t=1 thread/rank.
#   p = 1, 2, 3, 4.  Total samples = p × SAMPLES_PER_RANK.
#
# Output:
#   results/raw/weak_p{p}_t1.txt   — raw run output
#   results/weak_scaling.csv       — summary with time, efficiency, speedup
#
# Usage:
#   bash weak_scaling.sh

set -e
cd "$(dirname "$0")"
mkdir -p results/raw

SAMPLES_PER_RANK=1000

flush() {
    sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
}

get_median() {
    grep "Median t" "$1" | awk '{print $4}'
}

get_min() {
    grep "run [0-9]*:" "$1" | awk '{print $3}' | sort -n | head -1
}

get_max() {
    grep "run [0-9]*:" "$1" | awk '{print $3}' | sort -n | tail -1
}

T1=0   # set from the p=1 run

run_weak() {
    local p=$1
    local n_test=$((p * SAMPLES_PER_RANK))
    local out="results/raw/weak_p${p}_t1.txt"
    echo "=========================================="
    echo "  Weak scaling  p=$p  samples=$n_test  (${SAMPLES_PER_RANK}/rank)"
    echo "=========================================="
    flush
    OMP_NUM_THREADS=1 mpirun -np $p --bind-to none \
        ./inference_dist_knn --n_test $n_test | tee "$out"
    echo ""
}

generate_csv() {
    local csv="results/weak_scaling.csv"
    printf "p,samples_per_rank,total_samples,min_s,time_s,max_s,efficiency,speedup\n" > "$csv"

    T1=$(get_median "results/raw/weak_p1_t1.txt")

    for p in 1 2 3 4; do
        local raw="results/raw/weak_p${p}_t1.txt"
        local min_t tp max_t efficiency speedup
        min_t=$(get_min "$raw")
        tp=$(get_median "$raw")
        max_t=$(get_max "$raw")
        efficiency=$(awk "BEGIN{printf \"%.4f\", $T1 / $tp}")
        speedup=$(awk "BEGIN{printf \"%.4f\", $p * $efficiency}")
        printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$p" "$SAMPLES_PER_RANK" "$((p * SAMPLES_PER_RANK))" \
            "$min_t" "$tp" "$max_t" "$efficiency" "$speedup" >> "$csv"
    done

    echo "=========================================="
    echo "  results/weak_scaling.csv"
    echo "=========================================="
    echo "  T1 (p=1 baseline) = ${T1} s"
    echo ""
    column -t -s, "$csv"
    echo ""
}

run_weak 1
run_weak 2
run_weak 3
run_weak 4
generate_csv

echo "Weak scaling results saved to results/weak_scaling.csv"
