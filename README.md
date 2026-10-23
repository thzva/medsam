# Per-Label Beta Opinions and Vacuity Collapse

Code for *Per-Label Beta Opinions and Vacuity Collapse: Rethinking Evidential
Multi-Scale Fusion for Chest X-Ray Classification* (manuscript under review).

The repository contains a dual-stream evidential classifier for multi-label chest
X-ray classification, and every script needed to reproduce the experiments,
tables and figures of the paper. The study is a controlled analysis. Its
conclusions differ from what the architecture was originally designed to show.

## Findings

All numbers are on the official NIH ChestX-ray14 patient-level test split, as the
mean of three seeds.

1. **Use per-label Beta opinions for multi-label evidential learning.** A single
   categorical Dirichlet makes co-occurring findings compete for probability mass.
   Giving each finding its own Beta opinion improves macro AUC by 0.033–0.047, AUC
   on external CheXpert data by 0.08–0.10, and the OOD score, at matched backbones.
2. **Fusion destroys the OOD signal ("vacuity collapse").** Dempster's rule on
   binomial opinions satisfies `u_f <= min(u_g, u_l)`, so the fused vacuity is
   controlled by the more confident stream. The local stream is confidently wrong
   on natural images. As a result, the fused OOD AUROC drops from 0.90 (global
   stream) to 0.27. Yager, mean-BBA, opinion averaging and Shafer discounting all
   lose the signal too. A learned gate keeps it only by learning to ignore the
   local stream (gate weight 1.000).
3. **Fusion does not help discrimination and hurts calibration.** All
   Dempster-fused models are within one seed standard deviation of the global
   stream alone in macro AUC (0.779–0.783 vs 0.781), and have about twice its ECE.
4. **Decoupled read-out.** Keep the fused probabilities for prediction, but score
   OOD inputs with the global stream's vacuity from the same forward pass. This
   restores OOD AUROC to 0.78–0.97 with no retraining.
5. **The attention-selected crops do not localise findings.** A selector trained
   through the classification loss places 90–99% of its crops within 15% of the
   image border, and only 0–5% inside the lung fields.

Against BCE baselines and a five-member deep ensemble, a single evidential model
gives up 0.02–0.03 macro AUC. In exchange it provides an OOD score from one forward
pass, which the ensemble only approaches at five times the cost.

## Model

```
image ──► ViT-B/16 (224²) ──────────────► Beta head ─► (b+, b-, u_g) per finding ─┐
  │                                                                               ├─► per-label Dempster ─► p_f, u_f
  └──► shallow-conv attention ─► top-K crops from the 560² image (NMS)             │
           ─► ResNet-50 ─► gated MIL ─► Beta head ─► (b+, b-, u_l) per finding ───┘
```

The model's components:

| Component | File |
|---|---|
| Global stream | `src/models/global_expert.py` |
| Local stream: attention crops, gated MIL, legacy uniform pooling | `src/models/local_expert.py` |
| Per-label Beta heads and combination rules (Dempster, Yager, mean-BBA, averaging, gating, Shafer discounting) | `src/models/beta_evidential.py` |
| Full model | `src/models/tms_edl.py` |
| Per-label Beta evidential loss with KL annealing (Eq. 2) | `src/losses/beta_edl_loss.py` (dispatched by `src/losses/hybrid.py`) |

## Setup

```bash
conda create -n medsam python=3.11 -y && conda activate medsam
pip install -r requirements.txt
```

All experiments were run on NVIDIA A100 80 GB GPUs with bfloat16 autocast.

## Data

Datasets are not redistributed. Place them under `data/`:

| Dataset | Used for | Preparation |
|---|---|---|
| NIH ChestX-ray14 (official `train_val_list.txt` / `test_list.txt`) | training and internal test | `python scripts/prepare_data.py --data_root data --extract --verify`, then `python scripts/build_cache.py` (pre-decodes images; does not change model inputs) |
| CheXpert, expert-labelled validation set and train shards (parquet) | external validation | `data/external_cxr/chexpert_val.parquet`, `chexpert_train-*.parquet` |
| Imagenette (natural images) | far OOD | `python scripts/prepare_ood_imagenet.py` |
| MSD Task03 Liver CT | far OOD | `python scripts/prepare_ood_ct.py --src <Task03_Liver/imagesTr>` |

The validation fold is split from the official training list at the patient level
(25,207 / 2,801 / 2,797 patients for train / val / test, with no overlap).

## Reproducing the experiments

The final protocol is `config/revision_tvc.yaml`. Every experiment is a
*(mode, seed)* pair trained by `scripts/train_ddp.py` and then evaluated by
`scripts/eval_revision.py`:

```bash
python scripts/train_ddp.py   --config config/revision_tvc.yaml --mode tms_edl --seed 42 --output_dir outputs_tvc
python scripts/eval_revision.py --config config/revision_tvc.yaml --mode tms_edl \
       --run_dir outputs_tvc/tms_edl/seed42
```

`eval_revision.py` writes `eval_revision.json` (all metrics, bootstrap CIs,
calibration variants, external, near-OOD and far-OOD results) and `arrays.npz`
(per-image predictions and the vacuities `u_g`, `u_l`, `u_f`) into the run
directory.

The paper's runs use these modes (seeds 42, 43 and 44; seeds 45 and 46 for the
extra ensemble members):

| Paper | Modes |
|---|---|
| Main comparison | `stream_global_only`, `attn_crop_off`, `mil_uniform`, `mil_no_nms`, `tms_edl` (gated MIL + NMS), `fusion_discounted` |
| Baselines | `vit_b16_ce`, `densenet121_ce`, `resnet50_ce`, `resnet50_mc_dropout` |
| Combination rules | `fusion_yager`, `fusion_cautious` (mean-BBA), `fusion_average`, `fusion_gating` |
| Beta vs Dirichlet | `evid_dirichlet`, `edl_single_resnet50`, `stream_local_only` |

`scripts/tvc_lane.sh` and `scripts/tvc_lane2.sh` run the whole matrix. Several
lanes can share one ordered job list and claim jobs atomically, across one or
more GPUs:

```bash
scripts/tvc_lane.sh A 0 any      # lane name, GPU index, job kind (light | heavy | any)
scripts/tvc_lane.sh B 1 any
```

### Tables and figures

These scripts read only the saved `eval_revision.json` / `arrays.npz` files. No
number in the paper is typed by hand.

| Script | Produces |
|---|---|
| `scripts/tvc_posthoc.py` | decoupled read-out, deep ensemble, risk–coverage, OOD density figure |
| `scripts/tvc_tables.py` | all result tables, styled, with the best value per column highlighted (run after `tvc_posthoc.py`) |
| `scripts/tvc_make_assets.py` | the vacuity-collapse and corruption-sweep figures |
| `scripts/tvc_efficiency.py` | parameter count, latency, throughput, training step time and memory |
| `scripts/tvc_qualitative.py` | attention maps, selected crops and per-image vacuities |
| `scripts/tvc_fig1.py` | architecture figure |
| `scripts/dump_ood_arrays.py` | per-image OOD arrays for runs evaluated before OOD arrays were saved |

## Evaluation pitfalls we fixed

The paper's appendix explains each of these. They are listed here because each
one silently changes uncertainty metrics.

- **Float16 overflow.** Under float16 autocast, OOD images overflowed and produced
  non-finite vacuities. Filtering them before computing the AUROC removed the
  hardest 22% of OOD samples. The code now computes in bfloat16, runs the belief
  algebra in float32, and scores non-finite values as maximally uncertain.
- **Leaked augmentation.** A validation subset built from the training dataset
  object inherits training augmentation. The validation fold now uses the
  evaluation transform.
- **Double imbalance correction.** A re-weighted sampler plus inverse-frequency
  loss weights corrected class imbalance twice. Imbalance is now handled in the
  loss only.

## License and citation

Released for research use. Please cite the paper if you use this code; the
citation will be added on publication.
