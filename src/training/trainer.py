"""Training loop for TMS-EDL."""

import os
import time
import logging
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.losses.edl_loss import EDLLoss
from src.losses.hybrid import HybridCriterion
from src.losses.beta_edl_loss import BetaEDLLoss, inverse_frequency_weights, labels_from_loader
from src.models.opinion_utils import opinion_prob
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Full training pipeline for TMS-EDL."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        tcfg = cfg["training"]
        self.epochs = tcfg["epochs"]
        self.log_interval = cfg["logging"].get("log_interval", 50)
        self.save_interval = cfg["logging"].get("save_interval", 5)
        self.output_dir = Path(cfg["logging"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine num_classes and multi_label
        ds = cfg["data"]["dataset"]
        if ds == "nih_chestxray":
            self.num_classes = cfg["data"]["nih"]["num_classes"]
            self.multi_label = cfg["data"]["nih"].get("multi_label", True)
        else:
            self.num_classes = cfg["data"]["camelyon16"]["num_classes"]
            self.multi_label = cfg["data"]["camelyon16"].get("multi_label", False)

        # Loss
        loss_cfg = tcfg["loss"]
        ltype = loss_cfg.get("type", "log")
        ltype = "log" if ltype == "edl" else ltype
        self.evidential_mode = cfg["model"].get("evidential", {}).get("mode", "beta")
        if True:
            # Inverse-frequency class weights from the training split, so the
            # objective is literally weighted (evidential) binary cross-entropy.
            lbl = labels_from_loader(train_loader)
            cw = inverse_frequency_weights(lbl) if lbl is not None else None
            if cw is None:
                raise RuntimeError("Could not derive inverse-frequency class weights "
                                   "from the training loader; refusing to train with "
                                   "silently uniform weights.")
            # Dispatches on the runtime shape of alpha, so Beta / categorical
            # Dirichlet / deterministic-BCE models can share one training queue.
            self.criterion = HybridCriterion(
                num_classes=self.num_classes,
                loss_type=ltype,
                kl_weight=loss_cfg.get("kl_weight", 0.01),
                annealing_epochs=loss_cfg.get("annealing_epochs", 10),
                class_weights=cw,
                deterministic_bce=(loss_cfg.get("type") == "bce"),
            ).to(device)

        # Optimizer – layered learning rate
        opt_cfg = tcfg["optimizer"]
        head_lr = opt_cfg["lr"]
        backbone_lr = opt_cfg.get("backbone_lr", head_lr)
        wd = opt_cfg.get("weight_decay", 1e-4)

        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder" in name or "shallow" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr * 0.1, "initial_lr": backbone_lr},
            {"params": head_params, "lr": head_lr * 0.1, "initial_lr": head_lr},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

        # Scheduler
        sch_cfg = tcfg.get("scheduler", {})
        warmup = sch_cfg.get("warmup_epochs", 5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - warmup,
            eta_min=sch_cfg.get("min_lr", 1e-6),
        )
        self.warmup_epochs = warmup

        # Mixed precision
        self.use_amp = tcfg.get("mixed_precision", True)
        # bf16 has fp32 dynamic range, so no loss scaling is required.
        self.scaler = GradScaler(enabled=False)
        self.grad_clip = tcfg.get("gradient_clip", 1.0)

        # ── A100 / Ampere acceleration ──
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if tcfg.get("compile", False) and hasattr(torch, "compile"):
            if self.is_main:
                logger.info("Compiling model with torch.compile (first epoch will be slow)")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Wrap with DDP if multi-GPU
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[device.index], find_unused_parameters=False)

        # Gradient accumulation: effective_bs = batch_size * accum_steps
        self.accum_steps = tcfg.get("gradient_accumulation", 4)

        # Early stopping
        self.patience = tcfg.get("early_stopping_patience", 10)
        self.best_val_metric = -float("inf")
        self.wait = 0

        # Resume state (default: fresh start at epoch 1)
        self.start_epoch = 1
        self._maybe_resume()

        # WandB
        self.use_wandb = cfg["logging"].get("use_wandb", False)
        if self.use_wandb:
            import wandb
            wandb.init(project=cfg["logging"].get("wandb_project", "tms-edl"), config=cfg)

        # History
        self.history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_ece": []}

    # ------------------------------------------------------------------ #
    @property
    def _raw_model(self):
        """Unwrap DDP to access underlying module."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _maybe_resume(self):
        """If last.pth exists in output_dir, restore full state and resume."""
        last_path = self.output_dir / "last.pth"
        if not last_path.exists():
            return
        try:
            ckpt = torch.load(last_path, map_location=self.device, weights_only=False)
        except Exception as e:
            if self.is_main:
                logger.warning(f"Failed to load {last_path}: {e}. Starting fresh.")
            return
        # Model state
        try:
            self._raw_model.load_state_dict(ckpt["model_state_dict"])
        except Exception:
            # Fall back to stripping DDP _orig_mod prefix
            sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
            self._raw_model.load_state_dict(sd)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_metric = ckpt.get("best_val_metric", -float("inf"))
        self.wait = ckpt.get("wait", 0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        if "history" in ckpt:
            self.history = ckpt["history"]
        if self.is_main:
            logger.info(
                f"Resumed from {last_path} at epoch {self.start_epoch} "
                f"(best_val={self.best_val_metric:.4f}, wait={self.wait})"
            )

    def train(self):
        """Run full training loop."""
        if self.is_main:
            if self.start_epoch > 1:
                logger.info(f"Resuming training from epoch {self.start_epoch} to {self.epochs}")
            else:
                logger.info(f"Starting training for {self.epochs} epochs")
        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auc"].append(val_metrics.get("auc", 0))
            self.history["val_ece"].append(val_metrics.get("ece", 1))

            lr = self.optimizer.param_groups[0]["lr"]
            g_auc = val_metrics.get("global_auc", 0)
            l_auc = val_metrics.get("local_auc", 0)
            if self.is_main:
                logger.info(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_auc={val_metrics.get('auc', 0):.4f} "
                    f"(G={g_auc:.4f} L={l_auc:.4f}) | "
                    f"val_ece={val_metrics.get('ece', 1):.4f} | "
                    f"lr={lr:.2e} | {elapsed:.1f}s"
                )

            if self.use_wandb and self.is_main:
                import wandb
                wandb.log({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_metrics["loss"],
                           "val_auc": val_metrics.get("auc", 0),
                           "val_ece": val_metrics.get("ece", 1),
                           "lr": lr})

            # Scheduler step (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            else:
                # Linear warmup from 10% to 100% of target lr
                frac = 0.1 + 0.9 * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * min(frac, 1.0)

            # Checkpointing (rank 0 only)
            val_score = val_metrics.get("auc", 0)
            if val_score > self.best_val_metric:
                self.best_val_metric = val_score
                self.wait = 0
                if self.is_main:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self.wait += 1

            if self.is_main:
                # Always save last.pth for resume
                self._save_checkpoint(epoch, is_best=False, is_last=True)
                if epoch % self.save_interval == 0:
                    self._save_checkpoint(epoch, is_best=False)

            # Sync early-stopping decision across all ranks
            if self.world_size > 1:
                stop_tensor = torch.tensor(int(self.wait >= self.patience), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            else:
                should_stop = self.wait >= self.patience

            if should_stop:
                if self.is_main:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

        if self.is_main:
            logger.info(f"Training complete. Best val AUC: {self.best_val_metric:.4f}")
        return self.history

    # ------------------------------------------------------------------ #
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        # DDP: set epoch so each rank gets different shuffle order
        if self.world_size > 1 and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False,
                    disable=(not self.is_main))
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            hr = batch.get("image_hr")
            hr = hr.to(self.device, non_blocking=True) if hr is not None else None

            with autocast(device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16):
                outputs = self.model(images, hr=hr) if hr is not None else self.model(images)
                loss_fused = self.criterion(outputs["fused"], labels, epoch)["total"]
                # Single-stream ablations return the SAME opinion object under all
                # three keys.  Applying the deep-supervision weights there would
                # scale their loss by 1.6x relative to the dual-stream model, which
                # is not a no-op under gradient clipping (defect D5) and confounds
                # the very comparison the ablation is meant to make.
                if outputs["global"] is outputs["fused"] and outputs["local"] is outputs["fused"]:
                    loss = loss_fused / self.accum_steps
                else:
                    loss_global = self.criterion(outputs["global"], labels, epoch)["total"]
                    loss_local = self.criterion(outputs["local"], labels, epoch)["total"]
                    loss = (loss_fused + 0.3 * loss_global + 0.3 * loss_local) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item() * self.accum_steps:.4f}")

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_probs = {"global": [], "local": [], "fused": []}
        all_targets = []
        all_uncertainty = []
        n_batches = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            hr = batch.get("image_hr")
            hr = hr.to(self.device, non_blocking=True) if hr is not None else None

            with autocast(device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16):
                outputs = self.model(images, hr=hr) if hr is not None else self.model(images)
                loss = self.criterion(outputs["fused"], labels, epoch)["total"]

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions from all three streams
            for key in ("global", "local", "fused"):
                all_probs[key].append(opinion_prob(outputs[key]).float().cpu().numpy())

            all_targets.append(labels.cpu().numpy())
            from src.models.opinion_utils import opinion_uncertainty as _ou
            all_uncertainty.append(_ou(outputs["fused"]).float().cpu().numpy())

        targets_np = np.concatenate(all_targets, axis=0)

        # Fused metrics (primary)
        fused_probs = np.concatenate(all_probs["fused"], axis=0)
        metrics = compute_all_metrics(
            fused_probs, targets_np,
            multi_label=self.multi_label,
        )
        metrics["loss"] = total_loss / max(n_batches, 1)
        metrics["mean_uncertainty"] = float(np.concatenate(all_uncertainty).mean())

        # Per-expert AUC for diagnostics
        for key in ("global", "local"):
            expert_probs = np.concatenate(all_probs[key], axis=0)
            try:
                from src.evaluation.metrics import compute_auc
                metrics[f"{key}_auc"] = compute_auc(expert_probs, targets_np, self.multi_label)
            except Exception:
                pass

        return metrics

    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self._raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_metric": self.best_val_metric,
            "wait": self.wait,
            "history": self.history,
            "config": self.cfg,
        }
        if is_best:
            path = self.output_dir / "best_model.pth"
        elif is_last:
            path = self.output_dir / "last.pth"
        else:
            path = self.output_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save(ckpt, path)
        if not is_last:
            logger.info(f"Saved checkpoint: {path}")
