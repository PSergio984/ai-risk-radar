#!/usr/bin/env bash
# ==============================================================================
# Container Entrypoint Dispatcher for AI Investment Risk Radar
# ==============================================================================

set -euo pipefail

case "${1:-dashboard}" in
    dashboard)
        echo "[INFO] Starting AI Risk Radar Streamlit Dashboard on port ${PORT:-8501}..."
        if [ ! -f "artifacts/latest_report.json" ] && [ ! -f "artifacts/final_test_report.json" ] && [ ! -f "artifacts/fixture_report.json" ]; then
            echo "[INFO] No report artifacts found in clean checkout. Running cold bootstrap fixture..."
            python scripts/bootstrap_fixture.py || true
        fi
        exec streamlit run app.py \
            --server.port="${PORT:-8501}" \
            --server.address="0.0.0.0" \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    fixture-dashboard|fixture)
        echo "[INFO] Bootstrapping offline synthetic fixture demonstration..."
        python scripts/bootstrap_fixture.py
        exec streamlit run app.py \
            --server.port="${PORT:-8501}" \
            --server.address="0.0.0.0" \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    latest)
        shift
        echo "[INFO] Executing latest market inference pipeline..."
        exec python -m ai_risk_radar latest "$@"
        ;;
    final-test)
        shift
        echo "[INFO] Executing frozen final holdout test evaluation..."
        exec python -m ai_risk_radar final-test "$@"
        ;;
    research)
        shift
        echo "[INFO] Executing research training pipeline..."
        exec python -m ai_risk_radar research "$@"
        ;;
    test)
        shift
        echo "[INFO] Executing offline pytest test suite..."
        exec pytest -v -m "not historical_artifacts" "$@"
        ;;
    bash|sh)
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac
