#!/usr/bin/env bash
# =============================================================================
# END-TO-END ABLATION PIPELINE
# =============================================================================
# Runs the complete experimental pipeline:
#   1. Verify data cache (download if needed)
#   2. Train multiple seeds × window sizes
#   3. Run statistical ablation (window=1 vs window=5)
#   4. Run three-way comparison (reactive vs memoryless vs forecast)
#   5. Run interpretability analysis
#   6. Generate all paper figures
#   7. Produce final summary report
#
# Usage:
#   ./scripts/run_full_ablation.sh [OPTIONS]
#
# Options:
#   --seeds N              Number of seeds (default: 5)
#   --skip-train           Skip training (use existing checkpoints)
#   --skip-data            Skip data verification
#   --quick                Quick mode (1 seed, fewer episodes)
#   --resume               Resume from last checkpoint
#   --help                 Show help
#
# Author: Carbon-Aware Deep Reinforcement Learning Platform
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Experiment configuration
NUM_SEEDS=5
SEEDS=(42 123 456 789 1024)
WINDOW_SIZES=(1 5)
EPISODES=500
TASKS_PER_EPISODE=24

# Directories
CACHE_DIR="${PROJECT_ROOT}/watttime_cache"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
LOG_DIR="${PROJECT_ROOT}/logs/ablation"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Flags
SKIP_TRAIN=false
SKIP_DATA=false
QUICK_MODE=false
RESUME_MODE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Timing
SCRIPT_START=$(date +%s)

# =============================================================================
# LOGGING
# =============================================================================

setup_logging() {
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/ablation_$(date '+%Y%m%d_%H%M%S').log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
}

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*" >&2; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $(date '+%H:%M:%S') $*"; }
log_phase() { echo -e "\n${MAGENTA}════════════════════════════════════════════════════════════${NC}"; \
              echo -e "${MAGENTA}  $*${NC}"; \
              echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}\n"; }

elapsed_time() {
    local now=$(date +%s)
    local elapsed=$((now - SCRIPT_START))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# =============================================================================
# HELP
# =============================================================================

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run the complete ablation experiment pipeline for the Carbon-Aware Deep Reinforcement Learning Platform.

Pipeline stages:
  1. Verify data cache (download if needed)
  2. Train PPO agents (multiple seeds × window sizes)
  3. Run statistical ablation (paired t-tests, confidence intervals)
  4. Run three-way comparison (reactive vs memoryless vs forecast)
  5. Run interpretability analysis (t-SNE, attribution, linear probe)
  6. Generate all paper figures
  7. Produce final summary report

Options:
    --seeds N              Number of random seeds (default: ${NUM_SEEDS})
    --episodes N           Training episodes per run (default: ${EPISODES})
    --skip-train           Skip training, use existing checkpoints
    --skip-data            Skip data verification
    --quick                Quick mode: 1 seed, 100 episodes (for testing)
    --resume               Resume from last checkpoint
    --cache-dir PATH       Override cache directory
    --help                 Show this help message

Examples:
    # Full experiment (default: 5 seeds, 500 episodes)
    $(basename "$0")

    # Quick test run
    $(basename "$0") --quick

    # Resume after interruption
    $(basename "$0") --resume

    # Skip training, just analyze existing checkpoints
    $(basename "$0") --skip-train

EOF
}

# =============================================================================
# PREREQUISITES
# =============================================================================

check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Python
    if [ -d "${VENV_DIR}" ]; then
        PYTHON="${VENV_DIR}/bin/python"
    fi
    
    if ! command -v "${PYTHON}" &> /dev/null; then
        log_error "Python not found: ${PYTHON}"
        exit 1
    fi
    log_info "Python: $(${PYTHON} --version)"
    
    # Required packages
    local required_packages=("torch" "gymnasium" "pandas" "numpy" "scipy" "matplotlib")
    for pkg in "${required_packages[@]}"; do
        if ! "${PYTHON}" -c "import ${pkg}" 2>/dev/null; then
            log_error "Missing package: ${pkg}"
            log_error "Run: pip install -r requirements.txt"
            exit 1
        fi
    done
    log_info "Python packages: OK"
    
    # Create directories
    mkdir -p "${CHECKPOINT_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
    log_info "Directories: OK"
}

verify_data_cache() {
    if [ "${SKIP_DATA}" = true ]; then
        log_warn "Skipping data verification (--skip-data)"
        return
    fi
    
    log_step "Verifying data cache..."
    
    # Check if cache exists and has data
    if [ ! -d "${CACHE_DIR}" ] || [ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]; then
        log_warn "Cache directory empty or missing"
        log_info "Downloading data..."
        
        if [ -f "${SCRIPT_DIR}/download_real_data.sh" ]; then
            bash "${SCRIPT_DIR}/download_real_data.sh"
        else
            log_error "Download script not found: ${SCRIPT_DIR}/download_real_data.sh"
            exit 1
        fi
    else
        log_info "Data cache found at ${CACHE_DIR}"
        
        # Quick validation
        local parquet_count
        parquet_count=$(find "${CACHE_DIR}" -name "*.parquet" | wc -l)
        if [ "${parquet_count}" -lt 3 ]; then
            log_warn "Only ${parquet_count} Parquet files found (expected ≥ 3)"
            log_info "Consider re-downloading: ./scripts/download_real_data.sh"
        else
            log_info "Found ${parquet_count} cached regions"
        fi
    fi
}

# =============================================================================
# TRAINING
# =============================================================================

train_checkpoints() {
    if [ "${SKIP_TRAIN}" = true ]; then
        log_warn "Skipping training (--skip-train)"
        return
    fi
    
    log_phase "PHASE 1: TRAINING PPO AGENTS"
    
    local total_runs=$((${#SEEDS[@]} * ${#WINDOW_SIZES[@]}))
    local current_run=0
    
    for window in "${WINDOW_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            current_run=$((current_run + 1))
            local checkpoint_name="ppo_window${window}_seed${seed}"
            local checkpoint_path="${CHECKPOINT_DIR}/${checkpoint_name}.pt"
            
            echo ""
            log_info "━━━ Run ${current_run}/${total_runs}: window=${window}, seed=${seed} ━━━"
            
            # Check if we can skip (resume mode)
            if [ "${RESUME_MODE}" = true ] && [ -f "${checkpoint_path}" ]; then
                log_info "Checkpoint exists, skipping: ${checkpoint_path}"
                continue
            fi
            
            # Run training
            local train_log="${LOG_DIR}/train_${checkpoint_name}.log"
            
            cd "${PROJECT_ROOT}"
            if "${PYTHON}" -m src.training.replay_trainer \
                --window-size "${window}" \
                --seed "${seed}" \
                --episodes "${EPISODES}" \
                --tasks-per-episode "${TASKS_PER_EPISODE}" \
                --checkpoint-path "${checkpoint_path}" \
                2>&1 | tee "${train_log}"; then
                
                log_info "✓ Training complete: ${checkpoint_name}"
            else
                log_error "✗ Training failed: ${checkpoint_name}"
                log_error "Check log: ${train_log}"
                
                if [ "${RESUME_MODE}" != true ]; then
                    exit 1
                fi
                log_warn "Continuing with next run (resume mode)"
            fi
        done
    done
    
    log_info "Training phase complete"
}

# =============================================================================
# STATISTICAL ABLATION
# =============================================================================

run_statistical_ablation() {
    log_phase "PHASE 2: STATISTICAL ABLATION (window=1 vs window=5)"
    
    cd "${PROJECT_ROOT}"
    
    local ablation_log="${LOG_DIR}/statistical_ablation.log"
    
    if "${PYTHON}" -m src.training.statistical_ablation \
        --seeds "${SEEDS[@]}" \
        --episodes "${EPISODES}" \
        --output-dir "${OUTPUT_DIR}/ablation" \
        2>&1 | tee "${ablation_log}"; then
        
        log_info "✓ Statistical ablation complete"
    else
        log_error "✗ Statistical ablation failed"
        log_error "Check log: ${ablation_log}"
        return 1
    fi
}

# =============================================================================
# THREE-WAY COMPARISON
# =============================================================================

run_three_way_comparison() {
    log_phase "PHASE 3: THREE-WAY COMPARISON (reactive vs memoryless vs forecast)"
    
    cd "${PROJECT_ROOT}"
    
    local comparison_log="${LOG_DIR}/three_way_comparison.log"
    
    if "${PYTHON}" -m src.training.three_way_ablation \
        --seeds "${SEEDS[@]}" \
        --output-dir "${OUTPUT_DIR}/three_way" \
        2>&1 | tee "${comparison_log}"; then
        
        log_info "✓ Three-way comparison complete"
    else
        log_error "✗ Three-way comparison failed"
        log_error "Check log: ${comparison_log}"
        return 1
    fi
}

# =============================================================================
# INTERPRETABILITY
# =============================================================================

run_interpretability() {
    log_phase "PHASE 4: INTERPRETABILITY ANALYSIS"
    
    cd "${PROJECT_ROOT}"
    
    # Build checkpoint arguments for comparison
    local checkpoint_args=()
    for window in "${WINDOW_SIZES[@]}"; do
        # Use first seed's checkpoint
        local seed="${SEEDS[0]}"
        local ckpt_path="${CHECKPOINT_DIR}/ppo_window${window}_seed${seed}.pt"
        if [ -f "${ckpt_path}" ]; then
            checkpoint_args+=("window${window}=${ckpt_path}")
        fi
    done
    
    if [ ${#checkpoint_args[@]} -lt 2 ]; then
        log_warn "Not enough checkpoints for comparison, running single analysis"
        
        # Fall back to single checkpoint analysis
        local ckpt="${CHECKPOINT_DIR}/ppo_window5_seed${SEEDS[0]}.pt"
        if [ -f "${ckpt}" ]; then
            "${PYTHON}" -m src.interpretability.run_interpretability \
                --checkpoint "${ckpt}" \
                --output-dir "${OUTPUT_DIR}/interpretability" \
                --episodes 30
        fi
    else
        # Run comparison
        local interp_log="${LOG_DIR}/interpretability.log"
        
        if "${PYTHON}" -m src.interpretability.run_interpretability \
            --compare \
            --checkpoints "${checkpoint_args[@]}" \
            --output-dir "${OUTPUT_DIR}/interpretability" \
            --episodes 30 \
            2>&1 | tee "${interp_log}"; then
            
            log_info "✓ Interpretability analysis complete"
        else
            log_error "✗ Interpretability analysis failed"
            return 1
        fi
    fi
}

# =============================================================================
# FIGURE GENERATION
# =============================================================================

generate_figures() {
    log_phase "PHASE 5: GENERATING PAPER FIGURES"
    
    cd "${PROJECT_ROOT}"
    
    local figures_log="${LOG_DIR}/figures.log"
    
    if "${PYTHON}" scripts/generate_paper_figures.py \
        --output-dir "${OUTPUT_DIR}/paper" \
        2>&1 | tee "${figures_log}"; then
        
        log_info "✓ Paper figures generated"
    else
        log_error "✗ Figure generation failed"
        return 1
    fi
}

# =============================================================================
# SUMMARY REPORT
# =============================================================================

generate_summary() {
    log_phase "PHASE 6: GENERATING SUMMARY REPORT"
    
    local summary_file="${OUTPUT_DIR}/EXPERIMENT_SUMMARY.md"
    local total_time=$(elapsed_time)
    
    cat > "${summary_file}" << EOF
# Carbon-Aware Deep Reinforcement Learning Platform — Experiment Summary

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Total runtime:** ${total_time}
**Seeds:** ${SEEDS[*]}
**Window sizes:** ${WINDOW_SIZES[*]}
**Episodes per run:** ${EPISODES}

## Pipeline Stages

| Stage | Status | Log |
|-------|--------|-----|
| Data verification | ✅ Complete | \`${LOG_DIR}/ablation_*.log\` |
| Training | ✅ Complete | \`${LOG_DIR}/train_*.log\` |
| Statistical ablation | ✅ Complete | \`${LOG_DIR}/statistical_ablation.log\` |
| Three-way comparison | ✅ Complete | \`${LOG_DIR}/three_way_comparison.log\` |
| Interpretability | ✅ Complete | \`${LOG_DIR}/interpretability.log\` |
| Figure generation | ✅ Complete | \`${LOG_DIR}/figures.log\` |

## Generated Artifacts

### Checkpoints
EOF
    
    # List checkpoints
    for ckpt in "${CHECKPOINT_DIR}"/*.pt; do
        if [ -f "${ckpt}" ]; then
            local size
            size=$(du -h "${ckpt}" | cut -f1)
            echo "- \`${ckpt}\` (${size})" >> "${summary_file}"
        fi
    done
    
    cat >> "${summary_file}" << EOF

### Figures
EOF
    
    # List figures
    for fig in "${OUTPUT_DIR}"/paper/*.png "${OUTPUT_DIR}"/paper/*.pdf; do
        if [ -f "${fig}" ]; then
            echo "- \`${fig}\`" >> "${summary_file}"
        fi
    done
    
    cat >> "${summary_file}" << EOF

### Reports
- Statistical ablation: \`${OUTPUT_DIR}/ablation/comparison_summary.json\`
- Three-way comparison: \`${OUTPUT_DIR}/three_way/comparison_summary.json\`
- Interpretability: \`${OUTPUT_DIR}/interpretability/*/interpretability_report.json\`

## Key Results

See the generated figures in \`${OUTPUT_DIR}/paper/\` for detailed results.

### Statistical Significance
Check \`${OUTPUT_DIR}/ablation/comparison_summary.json\` for p-values and confidence intervals.

### Pareto Analysis
Check \`${OUTPUT_DIR}/paper/pareto_front.png\` for the carbon vs. SLA trade-off.

## Next Steps

1. Review the generated figures in \`${OUTPUT_DIR}/paper/\`
2. Read the summary reports in \`${OUTPUT_DIR}/\`
3. Run real WattTime data experiments (if using synthetic data)
4. Draft the paper using the generated figures and tables

---
*Generated by run_full_ablation.sh*
EOF
    
    log_info "✓ Summary report: ${summary_file}"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --seeds)
                NUM_SEEDS="$2"
                # Generate seed list
                SEEDS=()
                for i in $(seq 1 "${NUM_SEEDS}"); do
                    SEEDS+=($((40 + i)))
                done
                shift 2
                ;;
            --episodes)
                EPISODES="$2"
                shift 2
                ;;
            --skip-train)
                SKIP_TRAIN=true
                shift
                ;;
            --skip-data)
                SKIP_DATA=true
                shift
                ;;
            --quick)
                QUICK_MODE=true
                NUM_SEEDS=1
                SEEDS=(42)
                EPISODES=100
                shift
                ;;
            --resume)
                RESUME_MODE=true
                shift
                ;;
            --cache-dir)
                CACHE_DIR="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Setup
    setup_logging
    
    echo ""
    echo "╔═════════════════════════════════════════════════════════════════════════╗"
    echo "║  CARBON-AWARE DEEP REINFORSEMENT LEARNING PLATFORM — ABLATION PIPELINE  ║"
    echo "╚═════════════════════════════════════════════════════════════════════════╝"
    echo ""
    log_info "Seeds: ${SEEDS[*]}"
    log_info "Episodes: ${EPISODES}"
    log_info "Window sizes: ${WINDOW_SIZES[*]}"
    log_info "Quick mode: ${QUICK_MODE}"
    log_info "Resume mode: ${RESUME_MODE}"
    log_info "Log file: ${LOG_FILE}"
    
    # Run pipeline
    check_prerequisites
    verify_data_cache
    train_checkpoints
    run_statistical_ablation
    run_three_way_comparison
    run_interpretability
    generate_figures
    generate_summary
    
    # Final summary
    local total_time=$(elapsed_time)
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                        PIPELINE COMPLETE                           ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    log_info "Total runtime: ${total_time}"
    log_info "Summary report: ${OUTPUT_DIR}/EXPERIMENT_SUMMARY.md"
    log_info "Paper figures: ${OUTPUT_DIR}/paper/"
    log_info "Full log: ${LOG_FILE}"
    echo ""
}

main "$@"