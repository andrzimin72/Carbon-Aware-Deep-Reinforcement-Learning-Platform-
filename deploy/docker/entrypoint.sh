#!/bin/bash
# =============================================================================
# CONTAINER ENTRYPOINT SCRIPT
# =============================================================================
# Handles command routing and pre-flight checks for both controller and updater.
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
preflight_check() {
    local mode="$1"
    
    log_info "Running pre-flight checks for mode: ${mode}"
    
    # Check required environment variables
    if [ -z "${WATTTIME_USERNAME}" ] || [ -z "${WATTTIME_PASSWORD}" ]; then
        log_error "WATTTIME_USERNAME and WATTTIME_PASSWORD must be set"
        exit 1
    fi
    
    # Check WattTime connectivity (quick test)
    log_info "Testing WattTime API connectivity..."
    if ! python -c "
import os
from src.watttime.client import WattTimeClient
client = WattTimeClient(
    username=os.environ['WATTTIME_USERNAME'],
    password=os.environ['WATTTIME_PASSWORD']
)
if not client.health_check():
    exit(1)
" 2>/dev/null; then
        log_warn "WattTime API health check failed (may be transient)"
    else
        log_info "WattTime API connectivity: OK"
    fi
    
    # Mode-specific checks
    if [ "${mode}" = "controller" ]; then
        # Check checkpoint exists
        if [ ! -f "/app/checkpoints/ppo_window5.pt" ]; then
            log_warn "Checkpoint not found at /app/checkpoints/ppo_window5.pt"
            log_warn "Controller will fail to start without a trained model"
        else
            log_info "Checkpoint: OK"
        fi
        
        # Check Kubernetes connectivity (if running in cluster)
        if [ -f "/var/run/secrets/kubernetes.io/serviceaccount/token" ]; then
            log_info "Running inside Kubernetes cluster"
        else
            log_info "Running outside Kubernetes (using kubeconfig)"
        fi
    fi
    
    if [ "${mode}" = "update" ]; then
        # Check cache directory is writable
        if [ ! -w "/app/watttime_cache" ]; then
            log_error "Cache directory /app/watttime_cache is not writable"
            exit 1
        fi
        log_info "Cache directory: OK"
    fi
    
    log_info "Pre-flight checks passed"
}

# -----------------------------------------------------------------------------
# Command routing
# -----------------------------------------------------------------------------
case "${1}" in
    controller)
        shift
        preflight_check "controller"
        log_info "Starting carbon-aware controller..."
        exec python -m src.deployment.k8s_controller "$@"
        ;;
    
    shadow)
        shift
        preflight_check "controller"
        log_info "Starting shadow-mode inference..."
        exec python -m src.deployment.infer_shadow "$@"
        ;;
    
    update)
        shift
        preflight_check "update"
        log_info "Starting WattTime data update..."
        exec python -m src.watttime.daily_update \
            --regions "${REGIONS:-CAISO_NORTH MISO ERCOT}" \
            --lookback "${LOOKBACK_DAYS:-2}" \
            --cache-dir /app/watttime_cache \
            --log-dir /app/logs \
            "$@"
        ;;
    
    download)
        shift
        preflight_check "update"
        log_info "Starting one-shot data download..."
        exec python -m src.watttime.pipeline download "$@"
        ;;
    
    shell)
        log_info "Starting interactive shell..."
        exec /bin/bash
        ;;
    
    *)
        # Pass through any other command
        exec "$@"
        ;;
esac