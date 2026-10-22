# TMS-EDL: Trustworthy Multi-Scale Evidential Deep Learning

> **Trustworthy Multi-Scale Consensus: An Evidential Deep Learning Framework for Robust and Explainable Medical Image Diagnosis**

## Overview

TMS-EDL is a dual-stream evidential deep learning framework that addresses the **overconfidence problem** in medical image classification. It combines:

1. **Global Expert (ViT)** – captures long-range anatomical context
2. **Local Expert (ResNet + Attention Crop)** – extracts fine-grained ROI details
3. **Evidential Uncertainty (Dirichlet/Subjective Logic)** – outputs calibrated belief masses and uncertainty
4. **Dempster-Shafer Fusion** – mathematically principled "multi-expert consensus"

## Project Structure

```
├── config/
│   └── default.yaml              # Full configuration
├── src/
│   ├── data/
│   │   ├── nih_chestxray.py      # NIH Chest X-ray 14 dataset
│   │   ├── camelyon16.py         # Camelyon16 patch dataset
│   │   ├── ood_dataset.py        # OOD image-folder dataset
│   │   └── builder.py            # DataLoader factory
│   ├── models/
│   │   ├── global_expert.py      # ViT global expert
│   │   ├── local_expert.py       # ResNet + attention crop
│   │   ├── evidential_head.py    # Dirichlet parameterisation
│   │   ├── ds_fusion.py          # Dempster-Shafer combination
│   │   ├── tms_edl.py            # Full TMS-EDL model
│   │   ├── baselines.py          # All baseline models (A–D)
│   │   └── builder.py            # Model factory
│   ├── losses/
│   │   ├── edl_loss.py           # EDL loss + KL annealing
│   │   └── ce_loss.py            # CE/BCE baseline loss
│   ├── evaluation/
│   │   ├── metrics.py            # AUC, ACC, ECE, OOD AUROC
│   │   └── calibration.py        # Reliability diagram
│   ├── training/
│   │   ├── trainer.py            # Training loop (AMP, early stop)
│   │   └── utils.py              # Seed, config I/O
│   └── visualization/
│       └── plots.py              # Training curves, uncertainty, ablation
├── scripts/
│   ├── train.py                  # Train TMS-EDL
│   ├── evaluate.py               # Evaluate any model
│   ├── train_baselines.py        # Train all baselines
│   ├── run_ablation.py           # 6 ablation experiments
│   ├── compare_all.py            # Generate comparison table
│   ├── prepare_data.py           # Extract & verify datasets
│   ├── quick_test.py             # Smoke test
│   └── run_all.sh                # Full pipeline script
├── requirements.txt
└── research_proposal.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (extract NIH zip files)
python scripts/prepare_data.py --data_root data --extract --verify

# 3. Smoke test
python scripts/quick_test.py

# 4. Train TMS-EDL
python scripts/train.py --config config/default.yaml

# 5. Evaluate
python scripts/evaluate.py --config config/default.yaml \
    --checkpoint outputs/best_model.pth --save_predictions
```

## Full Experiment Pipeline

```bash
# Run everything: ours + baselines + ablations + comparison
bash scripts/run_all.sh --epochs 50 --device cuda
```

## Baselines

| Category | Method | Description |
|----------|--------|-------------|
| **(A)** Single-stream | ResNet-50, ViT-B/16, DenseNet-121 | CE/BCE loss |
| **(B)** Calibration | ResNet + Temperature Scaling | Post-hoc calibration |
| **(C)** UQ | MC Dropout, Deep Ensemble, EDL (single) | Uncertainty quantification |
| **(D)** Multi-scale | Global+Local + Avg prob | Dual-stream, naive fusion |
| **(E)** Ours | **TMS-EDL** | Dual-stream + EDL + D-S Fusion |

## Ablation Studies

Run with `python scripts/run_ablation.py`. Six experiments:

| # | Ablation | Variants |
|---|----------|----------|
| 1 | Fusion method | D-S vs Average vs Gating |
| 2 | Expert streams | Global-only vs Local-only vs Both |
| 3 | EDL activation | softplus vs relu vs exp |
| 4 | KL annealing | with vs without vs disabled |
| 5 | Attention crop | on vs off |
| 6 | Loss variant | EDL-log vs EDL-mse vs CE |

## Core Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **AUC** | Area Under ROC Curve | ↑ higher is better |
| **ACC** | Accuracy | ↑ higher is better |
| **ECE** | Expected Calibration Error | ↓ lower is better |
| **OOD AUROC** | OOD detection via uncertainty | ↑ higher is better |

## Datasets

- **NIH Chest X-ray 14**: 100k chest X-rays, 15 classes (multi-label)
- **Camelyon16**: Histopathology patches, binary (normal/tumor)

## Configuration

All hyperparameters are in `config/default.yaml`. Key settings:

```yaml
model:
  global_expert:
    backbone: "vit_base_patch16_224"
  local_expert:
    backbone: "resnet50"
    use_attention_crop: true
  fusion:
    method: "dempster_shafer"

training:
  epochs: 50
  batch_size: 32
  loss:
    type: "edl"
    kl_weight: 0.01
    annealing_epochs: 10
```

## Citation

If you use this code, please cite:

```bibtex
@article{tms_edl_2025,
  title={Trustworthy Multi-Scale Consensus: An Evidential Deep Learning
         Framework for Robust and Explainable Medical Image Diagnosis},
  year={2025},
}
```
