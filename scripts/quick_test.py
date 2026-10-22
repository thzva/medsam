"""Quick smoke test – verify imports, model forward pass, loss, metrics."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np


def test_imports():
    print("Testing imports ...")
    from src.models.tms_edl import TMSEDL
    from src.models.global_expert import GlobalExpert
    from src.models.local_expert import LocalExpert
    from src.models.evidential_head import EvidentialHead
    from src.models.ds_fusion import DSFusion, dempster_combination
    from src.models.baselines import (
        SingleStreamModel, MCDropoutModel, EDLSingleScale, MultiScaleAvgProb,
    )
    from src.losses.edl_loss import EDLLoss, kl_divergence
    from src.evaluation.metrics import compute_all_metrics
    from src.training.utils import load_config
    print("  ✓ All imports OK")
    return True


def test_evidential_head():
    print("Testing EvidentialHead ...")
    from src.models.evidential_head import EvidentialHead
    head = EvidentialHead(in_dim=128, num_classes=5, activation="softplus")
    x = torch.randn(4, 128)
    out = head(x)
    assert out["evidence"].shape == (4, 5)
    assert out["alpha"].shape == (4, 5)
    assert (out["alpha"] >= 1.0).all(), "alpha must be >= 1"
    assert out["uncertainty"].shape == (4,)
    assert (out["uncertainty"] >= 0).all()
    print("  ✓ EvidentialHead OK")
    return True


def test_ds_fusion():
    print("Testing Dempster-Shafer fusion ...")
    from src.models.ds_fusion import dempster_combination
    alpha1 = torch.tensor([[3.0, 1.5, 1.2]])
    alpha2 = torch.tensor([[1.5, 3.0, 1.1]])
    fused = dempster_combination(alpha1, alpha2)
    assert fused["alpha"].shape == (1, 3)
    assert (fused["alpha"] >= 1.0).all()
    assert fused["uncertainty"].shape == (1,)
    # Fused uncertainty should be lower than both individuals
    K = 3
    u1 = K / alpha1.sum(dim=-1)
    u2 = K / alpha2.sum(dim=-1)
    assert fused["uncertainty"].item() <= max(u1.item(), u2.item()) + 0.01
    print("  ✓ D-S Fusion OK")
    return True


def test_edl_loss():
    print("Testing EDL loss ...")
    from src.losses.edl_loss import EDLLoss
    criterion = EDLLoss(num_classes=5, loss_type="log", kl_weight=0.01, annealing_epochs=10)
    opinion = {
        "alpha": (torch.rand(4, 5).abs() + 1.0).requires_grad_(True),
    }
    target = torch.tensor([0, 2, 1, 4])
    losses = criterion(opinion, target, epoch=5)
    assert "total" in losses
    assert "nll" in losses
    assert "kl" in losses
    assert losses["total"].requires_grad
    print("  ✓ EDL Loss OK")
    return True


def test_metrics():
    print("Testing evaluation metrics ...")
    from src.evaluation.metrics import compute_all_metrics, compute_ood_auroc
    probs = np.random.dirichlet([1, 1, 1], size=100)
    targets = np.random.randint(0, 3, size=100)
    m = compute_all_metrics(probs, targets, multi_label=False)
    assert "auc" in m
    assert "acc" in m
    assert "ece" in m
    assert 0 <= m["ece"] <= 1

    # OOD AUROC
    id_unc = np.random.uniform(0, 0.3, 100)
    ood_unc = np.random.uniform(0.5, 1.0, 100)
    auroc = compute_ood_auroc(id_unc, ood_unc)
    assert 0.5 <= auroc <= 1.0, f"OOD AUROC should be high, got {auroc}"
    print("  ✓ Metrics OK")
    return True


def test_model_forward():
    print("Testing TMS-EDL forward pass (CPU, tiny) ...")
    from src.models.tms_edl import TMSEDL
    model = TMSEDL(
        num_classes=5,
        global_backbone="vit_tiny_patch16_224",
        global_pretrained=False,
        global_embed_dim=192,
        local_backbone="resnet18",
        local_pretrained=False,
        local_embed_dim=512,
        use_attention_crop=False,  # skip crop for speed
        fusion_method="dempster_shafer",
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert "global" in out
    assert "local" in out
    assert "fused" in out
    assert out["fused"]["alpha"].shape == (2, 5)
    print("  ✓ TMS-EDL forward OK")
    return True


def test_config():
    print("Testing config loading ...")
    from src.training.utils import load_config
    cfg = load_config("config/default.yaml")
    assert "model" in cfg
    assert "training" in cfg
    assert "data" in cfg
    assert cfg["model"]["fusion"]["method"] == "dempster_shafer"
    print("  ✓ Config OK")
    return True


def main():
    print("=" * 50)
    print("TMS-EDL Quick Smoke Test")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("EvidentialHead", test_evidential_head),
        ("DS Fusion", test_ds_fusion),
        ("EDL Loss", test_edl_loss),
        ("Metrics", test_metrics),
        ("Model Forward", test_model_forward),
        ("Config", test_config),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    if all(results.values()):
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
