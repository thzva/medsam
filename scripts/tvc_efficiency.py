"""Efficiency table for the TVC manuscript.

Measures, for each configuration in isolation on one GPU under bfloat16 autocast:
  - parameter count
  - inference latency per image at batch 1 (median of 50 timed passes)
  - inference throughput at batch 64 (images / s)
  - training step time and peak memory at batch 40 (the training batch size)
Writes revision/tvc/generated/efficiency_rows.tex. Run from the repo root."""
import os, sys, json, time
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import torch
from src.training.utils import load_config
from train_ddp import build_model_for_mode

CFG = load_config("config/revision_tvc.yaml")
DEV = torch.device("cuda:0")
HR = CFG["data"]["hr_size"]
ROWS = [("ViT-B/16, BCE", "vit_b16_ce", 1),
        ("DenseNet-121, BCE", "densenet121_ce", 1),
        ("ResNet-50, MC dropout, 10 passes", "resnet50_mc_dropout", 1),
        ("Deep ensemble, 5$\\times$ ViT-B/16", "vit_b16_ce", 5),
        ("ViT-B/16 Beta EDL, global stream", "stream_global_only", 1),
        ("Dual stream, crop disabled", "attn_crop_off", 1),
        ("Dual stream, gated MIL + NMS", "tms_edl", 1)]


def inputs(bs, mode):
    x = torch.randn(bs, 3, 224, 224, device=DEV)
    dual = mode in ("tms_edl", "attn_crop_off")
    hr = torch.randn(bs, 3, HR, HR, device=DEV) if mode == "tms_edl" else None
    return x, hr


def fwd(m, x, hr):
    return m(x, hr=hr) if hr is not None else m(x)


@torch.no_grad()
def time_infer(m, mode, bs, reps):
    m.eval(); x, hr = inputs(bs, mode)
    for _ in range(10):
        with torch.autocast("cuda", dtype=torch.bfloat16): fwd(m, x, hr)
    torch.cuda.synchronize(); t = []
    for _ in range(reps):
        s = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16): fwd(m, x, hr)
        torch.cuda.synchronize(); t.append(time.perf_counter() - s)
    return float(np.median(t))


def loss_of(out):
    o = out["fused"] if "fused" in out else out
    k = "logits" if "logits" in o else ("alpha" if "alpha" in o else "prob")
    return o[k].float().mean()


def time_train(m, mode, bs=40, reps=20):
    m.train(); x, hr = inputs(bs, mode)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats(); t = []
    for i in range(reps + 5):
        torch.cuda.synchronize(); s = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = loss_of(fwd(m, x, hr))
        l.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 5: t.append(time.perf_counter() - s)
    return float(np.median(t)), torch.cuda.max_memory_allocated() / 2**30


res = {}
for name, mode, k in ROWS:
    m, _ = build_model_for_mode(mode, CFG, DEV)
    n = sum(p.numel() for p in m.parameters()) * k
    lat = time_infer(m, mode, 1, 50) * k
    thr = 64 / (time_infer(m, mode, 64, 10) * k)
    step, mem = time_train(m, mode)
    res[name] = dict(params_M=n / 1e6, latency_ms=lat * 1e3, throughput=thr,
                     train_step_s=step * k, peak_mem_GB=mem, members=k)
    print(name, json.dumps({a: round(b, 3) for a, b in res[name].items()}), flush=True)
    del m; torch.cuda.empty_cache()

os.makedirs("revision/tvc/generated", exist_ok=True)
json.dump(res, open("revision/tvc/generated/efficiency.json", "w"), indent=1)
lines = []
for name, r in res.items():
    lines.append(f"{name} & {r['params_M']:.1f} & {r['latency_ms']:.1f} & {r['throughput']:.0f} & "
                 f"{r['train_step_s']:.2f} & {r['peak_mem_GB']:.1f} \\\\")
open("revision/tvc/generated/efficiency_rows.tex", "w").write("\n".join(lines) + "\n")
