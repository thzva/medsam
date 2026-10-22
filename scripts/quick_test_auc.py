"""Test-set macro AUC for whatever checkpoints exist right now.

The full eval_revision.py suite (external + near/far OOD + conformal) takes
~30 min per model; this does the in-distribution test split only, so the
absolute level can be checked without waiting for training to finish.
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"4")
import sys, glob, json, torch, numpy as np
sys.path.insert(0,'/home/2025user/zhou/medsam'); sys.path.insert(0,'/home/2025user/zhou/medsam/scripts')
torch.set_num_threads(4)
from torch.amp import autocast
from sklearn.metrics import roc_auc_score
from src.training.utils import load_config
from src.data.builder import build_dataloaders
from src.models.opinion_utils import opinion_prob
from train_ddp import build_model_for_mode

dev = torch.device('cuda:0')
SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TARGETS = [
    ('stream_global_only','outputs_tuned/ablations/ablation_stream_global_only/seed42','config/revision_tuned.yaml'),
    ('stream_global_only','outputs_tuned/ablations/ablation_stream_global_only/seed43','config/revision_tuned.yaml'),
    ('attn_crop_off','outputs_tuned/ablations/ablation_attn_crop_off/seed42','config/revision_tuned.yaml'),
    ('vit_b16_ce','outputs_tuned/baselines/vit_b16_ce/seed42','config/revision_tuned.yaml'),
    ('stream_global_only','outputs_lrsweep/bb20E5/ablations/ablation_stream_global_only/seed42','config/lrsweep_bb20E5.yaml'),
    ('stream_global_only','outputs_lrsweep/bb50E5/ablations/ablation_stream_global_only/seed42','config/lrsweep_bb50E5.yaml'),
]
cache = {}
TARGETS = [t for i, t in enumerate(TARGETS) if i % NSHARD == SHARD]
print(f"RESULT_START shard {SHARD}/{NSHARD}: {len(TARGETS)} targets", flush=True)
for mode, d, cfgp in TARGETS:
    ck = os.path.join(d,'best_model.pth')
    if not os.path.exists(ck):
        print(f"{d.split('/')[0]+'/'+os.path.basename(d):<34} no checkpoint", flush=True); continue
    cfg = load_config(cfgp)
    key = cfgp
    if key not in cache:
        cfg2 = load_config(cfgp); cfg2['data']['num_workers']=6; cfg2['training']['batch_size']=48
        cache[key] = build_dataloaders(cfg2, rank=-1, world_size=1)[2]
    te = cache[key]
    m,_ = build_model_for_mode(mode, cfg, dev)
    sd = torch.load(ck, map_location=dev)
    ep = sd.get('epoch'); vb = sd.get('best_val_metric')
    m.load_state_dict({k.replace('_orig_mod.',''):v for k,v in sd['model_state_dict'].items()}); m.eval()
    P,T = [],[]
    with torch.no_grad():
        for b in te:
            hr = b.get('image_hr')
            hr = hr.to(dev,non_blocking=True) if hr is not None else None
            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                o = m(b['image'].to(dev,non_blocking=True), hr=hr) if hr is not None else m(b['image'].to(dev,non_blocking=True))
            fu = o['fused'] if 'fused' in o else o
            P.append(opinion_prob(fu).float().cpu().numpy()); T.append(b['label'].numpy())
    P,T = np.concatenate(P), np.concatenate(T)
    auc = float(np.mean([roc_auc_score(T[:,k],P[:,k]) for k in range(T.shape[1]) if 0<T[:,k].sum()<len(T)]))
    tag = ('sweep_'+d.split('/')[1]) if d.startswith('outputs_lrsweep') else os.path.basename(os.path.dirname(d))+'/'+os.path.basename(d)
    print(f"{tag:<34} ep={str(ep):<3} val={vb:.4f}  TEST_AUC={auc:.4f}", flush=True)
    del m; torch.cuda.empty_cache()
