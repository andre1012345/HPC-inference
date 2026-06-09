#!/bin/bash
# benchmark.sh — Full benchmark pipeline for the HPC inference engine.
#
# Steps:
#   1. Sequential baseline   → results/raw/seq_p1_t1.txt
#   2. Hybrid configurations → results/raw/hybrid_p{p}_t{t}.txt  (8 configs)
#   3. gprof profiler run    → results/profile_report.txt
#   4. CSV summary           → results/speedup.csv
#
# Usage:
#   bash benchmark.sh
#
# All output is also echoed to stdout so you can tee to a log if needed:
#   bash benchmark.sh | tee results/run_$(date +%Y%m%d_%H%M%S).log

set -e
cd "$(dirname "$0")"
mkdir -p results/raw

T1=0   # set from the sequential run; used when building the CSV

# ── Helpers ───────────────────────────────────────────────────────────────────

flush() {
    sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
}

# Extract the median time (fourth field of the "Median t : X s" line).
get_median() {
    grep "Median t" "$1" | awk '{print $4}'
}

# ── Run functions ─────────────────────────────────────────────────────────────

run_seq() {
    local out="results/raw/seq_p1_t1.txt"
    echo "=========================================="
    echo "  Sequential baseline"
    echo "=========================================="
    flush
    ./inference_seq | tee "$out"
    echo ""
}

run_hybrid() {
    local p=$1 t=$2
    local out="results/raw/hybrid_p${p}_t${t}.txt"
    echo "=========================================="
    echo "  p=$p  t=$t  (total workers: $((p*t)))"
    echo "=========================================="
    flush
    OMP_NUM_THREADS=$t mpirun -np $p --bind-to none ./inference_dist_knn | tee "$out"
    echo ""
}

run_profile() {
    echo "=========================================="
    echo "  Profiler  (g++ -O0 -pg, N_RUNS=1)"
    echo "=========================================="
    # -O0 keeps every function as a distinct symbol visible to the sampler.
    # -pg instruments the binary for gprof call-graph collection.
    # N_RUNS=1 limits runtime (absolute times are meaningless at -O0 anyway —
    # we only care about the relative hotspot breakdown, not wall-clock numbers).
    g++ -O0 -pg -DN_RUNS=1 -o inference_seq_prof src/inference_seq.cpp
    echo "  Running profiler binary (slow at -O0, please wait)..."
    ./inference_seq_prof > /dev/null
    gprof inference_seq_prof gmon.out > results/profile_report.txt
    rm -f gmon.out inference_seq_prof   # clean up profiler artefacts
    echo "  Profile saved → results/profile_report.txt"
    echo ""
    echo "  Flat profile (top entries):"
    # Print the header + first data rows of the flat profile section.
    awk '/Flat profile/,/Call graph/' results/profile_report.txt \
        | head -30
    echo ""
}

get_min() {
    grep "run [0-9]*:" "$1" | awk '{print $3}' | sort -n | head -1
}

get_max() {
    grep "run [0-9]*:" "$1" | awk '{print $3}' | sort -n | tail -1
}

generate_csv() {
    local csv="results/speedup.csv"

    # T1 = hybrid p=1,t=1 — same binary as all other rows, no parallelism active.
    T1=$(get_median "results/raw/hybrid_p1_t1.txt")

    printf "p,t,workers,min_s,time_s,max_s,speedup,efficiency\n" > "$csv"

    local configs=(
        "1 1 hybrid_p1_t1"
        "1 2 hybrid_p1_t2"
        "2 1 hybrid_p2_t1"
        "2 2 hybrid_p2_t2"
        "3 1 hybrid_p3_t1"
        "3 2 hybrid_p3_t2"
        "4 1 hybrid_p4_t1"
        "4 2 hybrid_p4_t2"
    )

    for entry in "${configs[@]}"; do
        read -r p t fname <<< "$entry"
        local raw="results/raw/${fname}.txt"
        local min_t tp max_t workers speedup efficiency
        min_t=$(get_min "$raw")
        tp=$(get_median "$raw")
        max_t=$(get_max "$raw")
        workers=$((p * t))
        speedup=$(awk "BEGIN{printf \"%.4f\", $T1 / $tp}")
        efficiency=$(awk "BEGIN{printf \"%.4f\", $speedup / $workers}")
        printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$p" "$t" "$workers" "$min_t" "$tp" "$max_t" "$speedup" "$efficiency" >> "$csv"
    done

    echo "=========================================="
    echo "  results/speedup.csv"
    echo "=========================================="
    echo "  T1 (hybrid p=1,t=1) = ${T1} s"
    echo ""
    column -t -s, "$csv"
    echo ""
}

# ── Pipeline ──────────────────────────────────────────────────────────────────

run_seq
run_hybrid 1 1
run_hybrid 1 2
run_hybrid 2 1
run_hybrid 2 2
run_hybrid 3 1
run_hybrid 3 2
run_hybrid 4 1
run_hybrid 4 2
run_profile
generate_csv

echo "All results saved to results/"
