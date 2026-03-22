#!/bin/bash
# =============================================================================
# ONE-COMMAND entry point: setup environment + run ALL experiments + show progress
# Usage: bash run.sh
# =============================================================================
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

echo "============================================================"
echo " Starting full experiment pipeline"
echo " Project: $(basename "$PROJ_DIR")"
echo " Time:    $(date)"
echo "============================================================"

# Step 1: Setup environment (skip if already done)
if [ ! -f "$PROJ_DIR/.venv/bin/activate" ]; then
    echo ""
    echo "[1/2] Setting up environment..."
    bash setup.sh
else
    echo ""
    echo "[1/2] Environment already set up (.venv exists)"
fi

# Step 2: Run all experiments with real-time output + log file
echo ""
echo "[2/2] Running all experiments (full production mode)..."
echo "  Log file: $PROJ_DIR/run.log"
echo "  Progress is shown below in real-time."
echo "  To run in background: nohup bash run.sh > run.log 2>&1 &"
echo ""

bash scripts/run_all_experiments.sh 2>&1 | tee "$PROJ_DIR/run.log"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo " Pipeline completed successfully!"
    echo " Results: $PROJ_DIR/results/"
    echo " Log:     $PROJ_DIR/run.log"
    echo ""
    echo " To package results: bash collect_results.sh"
    echo "============================================================"
else
    echo "============================================================"
    echo " Pipeline failed (exit code: $EXIT_CODE)"
    echo " Check log: $PROJ_DIR/run.log"
    echo " To resume: bash run.sh (completed phases are skipped)"
    echo "============================================================"
    exit $EXIT_CODE
fi
