#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# TMS-EDL: Full Experiment Pipeline
# ══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/run_all.sh [--dataset nih|camelyon16] [--epochs N] [--skip-baselines]
#
# This script runs:
#   1. Data preparation & verification
#   2. Train TMS-EDL (ours)
#   3. Train all baselines
#   4. Run ablation studies
#   5. Evaluate all models
#   6. Generate comparison tables & plots
# ══════════════════════════════════════════════════════════════════════

set -e

DATASET="nih"
EPOCHS=50
SKIP_BASELINES=false
CONFIG="config/default.yaml"
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)    DATASET="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --config)     CONFIG="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --skip-baselines) SKIP_BASELINES=true; shift ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════"
echo "  TMS-EDL Full Experiment Pipeline"
echo "  Dataset: $DATASET | Epochs: $EPOCHS | Device: $DEVICE"
echo "══════════════════════════════════════════════════════════════"

# ── 1. Data preparation ──────────────────────────────────────────────
echo ""
echo "[Step 1/6] Data preparation & verification"
python scripts/prepare_data.py --data_root data --verify

# ── 2. Train TMS-EDL (Ours) ─────────────────────────────────────────
echo ""
echo "[Step 2/6] Training TMS-EDL (ours)"
python scripts/train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --output_dir outputs/tms_edl

# ── 3. Train baselines ──────────────────────────────────────────────
if [ "$SKIP_BASELINES" = false ]; then
    echo ""
    echo "[Step 3/6] Training baselines"
    python scripts/train_baselines.py \
        --config "$CONFIG" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --output_dir outputs/baselines
else
    echo ""
    echo "[Step 3/6] Skipping baselines (--skip-baselines)"
fi

# ── 4. Ablation studies ─────────────────────────────────────────────
echo ""
echo "[Step 4/6] Running ablation studies"
python scripts/run_ablation.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --output_dir outputs/ablations

# ── 5. Evaluate all models ──────────────────────────────────────────
echo ""
echo "[Step 5/6] Evaluating TMS-EDL"
python scripts/evaluate.py \
    --config "$CONFIG" \
    --checkpoint outputs/tms_edl/best_model.pth \
    --device "$DEVICE" \
    --output_dir outputs/tms_edl \
    --save_predictions

# Evaluate baselines
if [ "$SKIP_BASELINES" = false ]; then
    for model_dir in outputs/baselines/*/; do
        name=$(basename "$model_dir")
        ckpt="$model_dir/best_model.pth"
        if [ -f "$ckpt" ]; then
            echo "  Evaluating baseline: $name"
            python scripts/evaluate.py \
                --config "$CONFIG" \
                --checkpoint "$ckpt" \
                --device "$DEVICE" \
                --output_dir "$model_dir"
        fi
    done
fi

# ── 6. Comparison ───────────────────────────────────────────────────
echo ""
echo "[Step 6/6] Generating comparison tables & plots"
python scripts/compare_all.py \
    --results_dir outputs \
    --output_dir outputs/comparison

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  All experiments complete!"
echo "  Results:     outputs/comparison/"
echo "  TMS-EDL:     outputs/tms_edl/"
echo "  Baselines:   outputs/baselines/"
echo "  Ablations:   outputs/ablations/"
echo "══════════════════════════════════════════════════════════════"
