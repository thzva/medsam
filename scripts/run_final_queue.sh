#!/usr/bin/env bash
# Sequential revision training queue -- single GPU (GPU0).
# Resumable: any run whose eval_revision.json already exists is skipped.
set -u
cd /home/2025user/zhou/medsam
# Stage 1 was stopped deliberately after 7 runs; those are kept only as the
# D6 comparison.  This is the primary matrix (imbalance handled in the loss).
PY=/home/2025user/zhou/anaconda3/envs/medsam/bin/python
export CUDA_VISIBLE_DEVICES=0
CFG=config/revision_final.yaml
OUT=outputs_final
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"
QUEUE=revision/results/queue_final_status.tsv
touch "$QUEUE"

run () {           # run <mode> <seed>
  local mode=$1 seed=$2
  local dir
  if [ "$mode" = "tms_edl" ]; then dir="$OUT/tms_edl/seed$seed"
  else dir="$OUT/ablations/ablation_$mode/seed$seed"; fi
  case "$mode" in
    resnet50_ce|densenet121_ce|vit_b16_ce|resnet50_mc_dropout|edl_single_resnet50|multiscale_avg_prob)
      dir="$OUT/baselines/$mode/seed$seed" ;;
  esac

  if [ -f "$dir/eval_revision.json" ]; then
    echo "[skip] $mode seed$seed already evaluated"; return
  fi

  echo -e "$(date -Is)\t$mode\tseed$seed\tSTART" >> "$QUEUE"
  echo "=== $(date -Is)  TRAIN $mode seed$seed ==="
  $PY scripts/train_ddp.py --config "$CFG" --mode "$mode" --seed "$seed" \
      --output_dir "$OUT" > "$LOGDIR/${mode}_seed${seed}.train.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$dir/best_model.pth" ]; then
    echo -e "$(date -Is)\t$mode\tseed$seed\tTRAIN_FAIL($rc)" >> "$QUEUE"
    echo "!!! train failed rc=$rc for $mode seed$seed"; return
  fi

  echo "=== $(date -Is)  EVAL $mode seed$seed ==="
  $PY scripts/eval_revision.py --run_dir "$dir" --mode "$mode" --config "$CFG" \
      --device cuda:0 > "$LOGDIR/${mode}_seed${seed}.eval.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo -e "$(date -Is)\t$mode\tseed$seed\tEVAL_FAIL($rc)" >> "$QUEUE"
  else
    echo -e "$(date -Is)\t$mode\tseed$seed\tDONE" >> "$QUEUE"
  fi
  # keep only best/last checkpoints -- 1.3 GB each, 31 runs would fill the disk
  find "$dir" -name 'checkpoint_epoch*.pth' -delete
}

# ── Round 1: full matrix at seed 42 (headline numbers land first) ────────
for m in tms_edl stream_global_only stream_local_only fusion_average \
         fusion_gating fusion_yager fusion_cautious fusion_discounted \
         attn_crop_off evid_dirichlet kl_disabled kl_no_annealing loss_edl_mse ; do
  run "$m" 42
done

# ── Round 2: baselines (R1-6: deterministic single-pass comparators) ─────
for m in resnet50_ce densenet121_ce vit_b16_ce resnet50_mc_dropout \
         edl_single_resnet50 multiscale_avg_prob ; do
  run "$m" 42
done

# ── Round 3: seeds 43/44 for the headline + key ablations (M4: mean+-std) ─
for s in 43 44 ; do
  for m in tms_edl stream_global_only stream_local_only fusion_average \
           fusion_yager fusion_discounted ; do
    run "$m" "$s"
  done
done

echo "=== $(date -Is)  QUEUE COMPLETE ==="
