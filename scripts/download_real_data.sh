#!/usr/bin/env bash
# =============================================================================
# ONE-SHOT WATTTIME DATA DOWNLOAD
# =============================================================================
# Downloads historical MOER data for specified regions and time range.
# Validates credentials, handles errors, and verifies the download.
#
# Usage:
#   ./scripts/download_real_data.sh [OPTIONS]
#
# Examples:
#   # Download last 90 days for default regions
#   ./scripts/download_real_data.sh
#
#   # Download specific date range
#   ./scripts/download_real_data.sh --start 2024-01-01 --end 2024-12-31
#
#   # Dry run (validate only)
#   ./scripts/download_real_data.sh --dry-run
#
# Author: Carbon-Aware Deep Reinforcement Learning Platform
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
DEFAULT_REGIONS=("CAISO_NORTH" "MISO" "ERCOT")
DEFAULT_LOOKBACK_DAYS=90
DEFAULT_SIGNAL="MOER"
CACHE_DIR="${PROJECT_ROOT}/watttime_cache"
LOG_DIR="${PROJECT_ROOT}/logs"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# =============================================================================
# LOGGING
# =============================================================================

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*" >&2; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $(date '+%H:%M:%S') $*"; }

# =============================================================================
# HELP
# =============================================================================

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Download historical WattTime MOER data for carbon-aware scheduling experiments.

Options:
    --regions R1 R2 ...    WattTime region codes (default: ${DEFAULT_REGIONS[*]})
    --start YYYY-MM-DD     Start date (default: ${DEFAULT_LOOKBACK_DAYS} days ago)
    --end YYYY-MM-DD       End date (default: today)
    --lookback N           Download last N days (alternative to --start/--end)
    --signal TYPE          Signal type: MOER or CO2_AVERAGE (default: MOER)
    --cache-dir PATH       Cache directory (default: ${CACHE_DIR})
    --force                Force re-download even if data exists
    --dry-run              Validate configuration without downloading
    --help                 Show this help message

Environment Variables:
    WATTTIME_USERNAME      WattTime API username (required)
    WATTTIME_PASSWORD      WattTime API password (required)
    PYTHON                 Python interpreter (default: python3)

Examples:
    # Download last 90 days for default regions
    $(basename "$0")

    # Download specific date range
    $(basename "$0") --start 2024-01-01 --end 2024-12-31

    # Download with custom regions
    $(basename "$0") --regions CAISO_NORTH ERCOT --lookback 180

    # Validate credentials only
    $(basename "$0") --dry-run

EOF
}

# =============================================================================
# PREREQUISITES CHECK
# =============================================================================

check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Python
    if ! command -v "${PYTHON}" &> /dev/null; then
        log_error "Python not found: ${PYTHON}"
        exit 1
    fi
    log_info "Python: $(${PYTHON} --version)"
    
    # Check virtual environment
    if [ -d "${VENV_DIR}" ]; then
        log_info "Using virtual environment: ${VENV_DIR}"
        PYTHON="${VENV_DIR}/bin/python"
    fi
    
    # Check required Python packages
    if ! "${PYTHON}" -c "import pandas, pyarrow, requests" 2>/dev/null; then
        log_error "Required Python packages not installed"
        log_error "Run: pip install -r requirements.txt"
        exit 1
    fi
    log_info "Python packages: OK"
    
    # Check credentials
    if [ -z "${WATTTIME_USERNAME:-}" ] || [ -z "${WATTTIME_PASSWORD:-}" ]; then
        # Try loading from .env file
        if [ -f "${PROJECT_ROOT}/.env" ]; then
            log_info "Loading credentials from .env file"
            set -a
            # shellcheck disable=SC1091
            source "${PROJECT_ROOT}/.env"
            set +a
        fi
    fi
    
    if [ -z "${WATTTIME_USERNAME:-}" ] || [ -z "${WATTTIME_PASSWORD:-}" ]; then
        log_error "WattTime credentials not set"
        log_error "Set WATTTIME_USERNAME and WATTTIME_PASSWORD environment variables"
        log_error "Or create a .env file from .env.example"
        exit 1
    fi
    log_info "Credentials: OK (user: ${WATTTIME_USERNAME})"
    
    # Create directories
    mkdir -p "${CACHE_DIR}" "${LOG_DIR}"
    log_info "Cache directory: ${CACHE_DIR}"
}

# =============================================================================
# DATE PARSING
# =============================================================================

parse_date() {
    local date_str="$1"
    # Validate format
    if ! date -d "${date_str}" &>/dev/null 2>&1 && ! date -j -f "%Y-%m-%d" "${date_str}" &>/dev/null 2>&1; then
        log_error "Invalid date format: ${date_str} (expected YYYY-MM-DD)"
        exit 1
    fi
    echo "${date_str}"
}

compute_start_date() {
    local lookback_days="$1"
    if date -v-"${lookback_days}d" &>/dev/null 2>&1; then
        # macOS
        date -v-"${lookback_days}d" "+%Y-%m-%d"
    else
        # Linux
        date -d "${lookback_days} days ago" "+%Y-%m-%d"
    fi
}

# =============================================================================
# DOWNLOAD
# =============================================================================

run_download() {
    local regions=("$@")
    
    log_step "Starting data download..."
    log_info "Regions: ${regions[*]}"
    log_info "Time range: ${START_DATE} to ${END_DATE}"
    log_info "Signal type: ${SIGNAL}"
    log_info "Cache dir: ${CACHE_DIR}"
    
    # Build Python command
    local python_cmd=(
        "${PYTHON}" -m src.watttime.pipeline download
        --regions "${regions[@]}"
        --start "${START_DATE}"
        --end "${END_DATE}"
        --signal "${SIGNAL}"
    )
    
    if [ "${FORCE}" = true ]; then
        python_cmd+=(--force)
    fi
    
    # Run with logging
    local log_file="${LOG_DIR}/download_$(date '+%Y%m%d_%H%M%S').log"
    log_info "Log file: ${log_file}"
    
    cd "${PROJECT_ROOT}"
    if "${python_cmd[@]}" 2>&1 | tee "${log_file}"; then
        log_info "Download completed successfully"
    else
        log_error "Download failed. Check log: ${log_file}"
        exit 1
    fi
}

# =============================================================================
# VALIDATION
# =============================================================================

validate_download() {
    log_step "Validating downloaded data..."
    
    local regions=("$@")
    local all_valid=true
    
    for region in "${regions[@]}"; do
        local parquet_file="${CACHE_DIR}/${region}.parquet"
        
        if [ ! -f "${parquet_file}" ]; then
            log_error "Missing cache file: ${parquet_file}"
            all_valid=false
            continue
        fi
        
        # Check file size
        local size
        size=$(stat -f%z "${parquet_file}" 2>/dev/null || stat -c%s "${parquet_file}" 2>/dev/null)
        if [ "${size}" -lt 1000 ]; then
            log_warn "Cache file suspiciously small: ${parquet_file} (${size} bytes)"
            all_valid=false
            continue
        fi
        
        # Check row count using Python
        local row_count
        row_count=$("${PYTHON}" -c "
import pandas as pd
df = pd.read_parquet('${parquet_file}')
print(len(df))
" 2>/dev/null || echo "0")
        
        if [ "${row_count}" -lt 24 ]; then
            log_warn "Cache file has too few rows: ${parquet_file} (${row_count} rows)"
            all_valid=false
            continue
        fi
        
        log_info "✓ ${region}: ${row_count} hours, ${size} bytes"
    done
    
    if [ "${all_valid}" = false ]; then
        log_error "Validation failed for one or more regions"
        exit 1
    fi
    
    log_info "All regions validated successfully"
}

# =============================================================================
# SUMMARY
# =============================================================================

print_summary() {
    local regions=("$@")
    
    echo ""
    echo "======================================================================"
    echo "DOWNLOAD SUMMARY"
    echo "======================================================================"
    echo "Regions:       ${regions[*]}"
    echo "Time range:    ${START_DATE} to ${END_DATE}"
    echo "Signal type:   ${SIGNAL}"
    echo "Cache dir:     ${CACHE_DIR}"
    echo "======================================================================"
    
    # Show cache summary
    cd "${PROJECT_ROOT}"
    "${PYTHON}" -m src.watttime.pipeline summary 2>/dev/null || true
    
    echo ""
    echo "Next steps:"
    echo "  1. Run training:     ./scripts/run_full_ablation.sh"
    echo "  2. Generate figures: ${PYTHON} scripts/generate_paper_figures.py"
    echo "======================================================================"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    # Parse arguments
    REGIONS=("${DEFAULT_REGIONS[@]}")
    START_DATE=""
    END_DATE=""
    LOOKBACK_DAYS="${DEFAULT_LOOKBACK_DAYS}"
    SIGNAL="${DEFAULT_SIGNAL}"
    FORCE=false
    DRY_RUN=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --regions)
                REGIONS=()
                shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    REGIONS+=("$1")
                    shift
                done
                ;;
            --start)
                START_DATE=$(parse_date "$2")
                shift 2
                ;;
            --end)
                END_DATE=$(parse_date "$2")
                shift 2
                ;;
            --lookback)
                LOOKBACK_DAYS="$2"
                shift 2
                ;;
            --signal)
                SIGNAL="$2"
                shift 2
                ;;
            --cache-dir)
                CACHE_DIR="$2"
                shift 2
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
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
    
    # Compute dates if not provided
    if [ -z "${END_DATE}" ]; then
        if date "+%Y-%m-%d" &>/dev/null; then
            END_DATE=$(date "+%Y-%m-%d")
        else
            END_DATE=$(date -u "+%Y-%m-%d")
        fi
    fi
    
    if [ -z "${START_DATE}" ]; then
        START_DATE=$(compute_start_date "${LOOKBACK_DAYS}")
    fi
    
    echo ""
    echo "======================================================================"
    echo "WATTTIME DATA DOWNLOAD"
    echo "======================================================================"
    
    check_prerequisites
    
    if [ "${DRY_RUN}" = true ]; then
        log_info "DRY RUN — configuration is valid"
        log_info "Would download:"
        log_info "  Regions:    ${REGIONS[*]}"
        log_info "  Time range: ${START_DATE} to ${END_DATE}"
        log_info "  Signal:     ${SIGNAL}"
        log_info "  Cache:      ${CACHE_DIR}"
        exit 0
    fi
    
    run_download "${REGIONS[@]}"
    validate_download "${REGIONS[@]}"
    print_summary "${REGIONS[@]}"
}

main "$@"