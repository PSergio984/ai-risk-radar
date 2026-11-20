#!/usr/bin/env bash
# ==============================================================================
# AI Investment Risk Radar: Scheduled Market Data Refresh & Inference Runner
#
# Usage:
#   bash scripts/run_scheduled_refresh.sh [--config config.json] [--data-mode refresh]
#
# Scheduling context:
#   Designed for cron, systemd timer, or CI runner execution at or shortly after
#   NYSE market close (16:00 ET / 21:00 UTC standard, 20:00 UTC daylight saving).
#
# Operational safety:
#   1. Resolves absolute repository root deterministically.
#   2. Verifies Python virtual environment and required runtime dependencies.
#   3. Validates frozen release manifest (artifacts/release.json) before execution.
#   4. Emits structured ISO-8601 timestamped logs with stdout/stderr redirection.
#   5. Propagates non-zero exit codes on failure without corrupting state.
# ==============================================================================

set -euo pipefail

# 1. Deterministic Root and Directory Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Default parameters and options parsing (supports both flags and positional arguments)
CONFIG_FILE="config.json"
DATA_MODE="refresh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash scripts/run_scheduled_refresh.sh [--config config.json] [--data-mode refresh]"
            exit 0
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-mode)
            DATA_MODE="$2"
            shift 2
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "${_POS1:-}" ]]; then
                CONFIG_FILE="$1"
                _POS1=1
            else
                DATA_MODE="$1"
            fi
            shift
            ;;
    esac
done

LOG_DIR="${AI_RISK_RADAR_LOG_DIR:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/scheduled_refresh.log"

mkdir -p "${LOG_DIR}"

log_msg() {
    local level="$1"
    shift
    local ts
    ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "[${ts}] [${level}] $*" | tee -a "${LOG_FILE}"
}

log_msg "INFO" "================================================================="
log_msg "INFO" "AI Risk Radar scheduled refresh started in ${REPO_ROOT}"
log_msg "INFO" "Configuration: ${CONFIG_FILE} | Data Mode: ${DATA_MODE}"

# 2. Cold Bootstrap: Environment & Dependency Validation
PYTHON_BIN=""
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python &>/dev/null; then
    PYTHON_BIN="$(command -v python)"
else
    log_msg "ERROR" "No suitable Python interpreter found in .venv or PATH."
    exit 127
fi

log_msg "INFO" "Using Python binary: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"

# Verify required files
if [[ ! -f "${CONFIG_FILE}" ]]; then
    log_msg "ERROR" "Configuration file '${CONFIG_FILE}' not found in ${REPO_ROOT}."
    exit 2
fi

if [[ ! -f "artifacts/release.json" ]]; then
    log_msg "ERROR" "Release artifact 'artifacts/release.json' not found. Run research/bootstrapping first."
    exit 3
fi

# 3. Environment Variables
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export AI_RISK_RADAR_LOG_DIR="${LOG_DIR}"

# 4. Execute Latest Refresh Command
EXIT_CODE=0
START_TIME="$(date +%s)"

log_msg "INFO" "Executing: ${PYTHON_BIN} -m ai_risk_radar latest --config ${CONFIG_FILE} --data-mode ${DATA_MODE}"

set +e
"${PYTHON_BIN}" -m ai_risk_radar latest \
    --config "${CONFIG_FILE}" \
    --data-mode "${DATA_MODE}" >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?
set -e

END_TIME="$(date +%s)"
DURATION=$((END_TIME - START_TIME))

# 5. Exit & Failure Handling
if [[ ${EXIT_CODE} -eq 0 ]]; then
    log_msg "INFO" "Scheduled refresh completed successfully in ${DURATION}s."
    log_msg "INFO" "Updated artifacts/latest_report.json and data/raw/latest/manifest.json."
    log_msg "INFO" "================================================================="
    exit 0
else
    log_msg "ERROR" "Scheduled refresh FAILED with exit code ${EXIT_CODE} after ${DURATION}s."
    log_msg "ERROR" "Prior snapshot and release artifacts remain preserved."
    log_msg "ERROR" "================================================================="
    exit "${EXIT_CODE}"
fi
