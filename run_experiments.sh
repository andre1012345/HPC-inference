#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# HPC Project — Andrea — University of Messina
#
# Runs the full experiment matrix:
#   Sequential baseline     : inference_seq  (p=1, t=1)
#   Hybrid MPI+OpenMP       : inference_dist_knn  (p={1,2,3,4}, t={1,2})
#
# Output layout:
#   results/
#     raw/            ← full stdout of every run, one file per config
#     speedup.csv     ← summary table: ranks, threads, median_time, speedup, efficiency
#     experiment.log  ← timestamped log of this script
#
# Usage (from inside hpc_inference/):
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
RAW_DIR="${RESULTS_DIR}/raw"
CSV="${RESULTS_DIR}/speedup.csv"
LOG="${RESULTS_DIR}/experiment.log"

mkdir -p "${RAW_DIR}"

# ── Logging helper ─────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

log "============================================================"
log "  HPC Experiment Run — $(date '+%Y-%m-%d %H:%M:%S')"
log "  Host : $(hostname)"
log "  CPU  : $(lscpu | grep 'Model name' | sed 's/Model name.*: *//')"
log "============================================================"

# ── Helper: extract median time from binary stdout ─────────────────────────────
# Both binaries print:  "  Median time : X.XXXXXX s"
extract_median() {
    local file="$1"
    grep -oP 'Median time\s*:\s*\K[0-9]+\.[0-9]+' "${file}" || echo "NA"
}

# ── Step 1: Sequential baseline ───────────────────────────────────────────────
log "--- Running sequential baseline (pure serial, no MPI/OpenMP) ---"

SEQ_OUT="${RAW_DIR}/seq_p1_t1.txt"
{
    echo "# Sequential baseline — pure serial inference_seq"
    echo "# Command: OMP_NUM_THREADS=1 ./inference_seq"
    echo "# Date: $(date)"
    echo ""
    OMP_NUM_THREADS=1 ./inference_seq
} 2>&1 | tee "${SEQ_OUT}"

T1=$(extract_median "${SEQ_OUT}")
log "  T1 baseline = ${T1} s"

if [[ "${T1}" == "NA" ]]; then
    log "ERROR: could not extract T1 from sequential run. Aborting."
    exit 1
fi

# ── Step 2: Initialise CSV ─────────────────────────────────────────────────────
{
    echo "# HPC Experiment Results — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# T1 (sequential baseline) = ${T1} s"
    echo "ranks,threads,total_workers,median_time_s,speedup,efficiency"
    echo "1,1,1,${T1},1.0000,1.0000"
} > "${CSV}"

log "CSV initialised at ${CSV}"

# ── Step 3: Hybrid experiment matrix ──────────────────────────────────────────
# p = MPI ranks, t = OMP threads per rank
RANKS=(1 2 3 4)
THREADS=(1 2)

for p in "${RANKS[@]}"; do
    for t in "${THREADS[@]}"; do

        # Skip (1,1) — T1 baseline already recorded above
        if [[ "${p}" -eq 1 && "${t}" -eq 1 ]]; then
            continue
        fi

        LABEL="hybrid_p${p}_t${t}"
        OUT="${RAW_DIR}/${LABEL}.txt"
        TOTAL=$((p * t))

        log "--- Running ${LABEL}  (ranks=${p}, threads/rank=${t}, total=${TOTAL}) ---"

        {
            echo "# Hybrid MPI+OpenMP — p=${p} t=${t}"
            echo "# Command: OMP_NUM_THREADS=${t} mpirun -n ${p} ./inference_dist_knn"
            echo "# Date: $(date)"
            echo ""
            OMP_NUM_THREADS=${t} mpirun -n ${p} ./inference_dist_knn
        } 2>&1 | tee "${OUT}"

        Tp=$(extract_median "${OUT}")
        log "  Median time = ${Tp} s"

        if [[ "${Tp}" == "NA" ]]; then
            log "  WARNING: could not parse median time for ${LABEL} — writing NA"
            echo "${p},${t},${TOTAL},NA,NA,NA" >> "${CSV}"
        else
            # speedup  = T1 / Tp
            # efficiency = speedup / total_workers
            SPEEDUP=$(awk "BEGIN { printf \"%.4f\", ${T1} / ${Tp} }")
            EFFICIENCY=$(awk "BEGIN { printf \"%.4f\", ${T1} / (${Tp} * ${TOTAL}) }")
            echo "${p},${t},${TOTAL},${Tp},${SPEEDUP},${EFFICIENCY}" >> "${CSV}"
            log "  Speedup = ${SPEEDUP}   Efficiency = ${EFFICIENCY}"
        fi

    done
done

# ── Step 4: Print summary ──────────────────────────────────────────────────────
log ""
log "============================================================"
log "  EXPERIMENT COMPLETE"
log "  Raw outputs  →  ${RAW_DIR}/"
log "  Summary CSV  →  ${CSV}"
log "  Full log     →  ${LOG}"
log "============================================================"
log ""
log "  CSV contents:"
cat "${CSV}" | tee -a "${LOG}"
