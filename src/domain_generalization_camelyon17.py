#!/usr/bin/env python3
import os
import argparse
from typing import Tuple, Dict, List

import torch
torch.set_grad_enabled(False)

import pandas as pd
import numpy as np

from wilds import get_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

# repo helpers
from dataset_camelyon17 import (
    CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
    PoisonedCamelyon17DS
)
from evaluation import eval_top1, eval_asr
from run_finetune import build_model_from_ckpt

# NeuBA helpers (untargeted misclassification ASR)
from baseline_neuba_helpers import (
    NeuBATriggerBank, apply_trigger_normed, apply_neuba_trigger,
    _peek_image_size, _NeuBATriggeredDatasetKeepY,
    build_triggered_eval_loader, eval_asr_misclf
)

# ------------------------
# Basic eval tfm (optional resize for CLIP/VIT)
# ------------------------
def _default_camelyon_eval_tfm(*, mean, std, resize_to=None):
    ops = []
    if resize_to is not None:
        ops.append(T.Resize((resize_to, resize_to), interpolation=InterpolationMode.BICUBIC))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)

def _collate_xy(batch):
    xs, ys = [], []
    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            x, y, _ = item
        else:
            x, y = item
        x = x if torch.is_tensor(x) else torch.as_tensor(x)
        y = y if torch.is_tensor(y) else torch.as_tensor(y)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected (3,H,W), got {tuple(x.shape)}")
        xs.append(x)
        ys.append(y.long())
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0).long()
    return x, y

def build_c17_eval_loaders(root_dir: str, bs: int = 256, nw: int = 8, *, clip_normalization: bool = False):
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_normalization else (IMAGENET_MEAN, IMAGENET_STD)
    resize_to = 224 if clip_normalization else None
    tf_eval = _default_camelyon_eval_tfm(mean=mean, std=std, resize_to=resize_to)

    ds = get_dataset(dataset="camelyon17", root_dir=root_dir, download=False)
    id_val = ds.get_subset("id_val", transform=tf_eval)
    val    = ds.get_subset("val",    transform=tf_eval)
    test   = ds.get_subset("test",   transform=tf_eval)

    L_id   = DataLoader(id_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, collate_fn=_collate_xy)
    L_val  = DataLoader(val,    batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, collate_fn=_collate_xy)
    L_test = DataLoader(test,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, collate_fn=_collate_xy)
    return {"id_val": L_id, "val": L_val, "test": L_test}, (mean, std)

def build_asr_loader_for_loader(eval_loader, trigger_path: str, mean, std, device: str):
    Tpt = torch.load(trigger_path, map_location=device)
    delta_pix = Tpt["delta_pix"].to(device).float()
    H, W = delta_pix.shape[-2:]
    mask = Tpt.get("mask", torch.ones(1, 3, H, W, device=device, dtype=delta_pix.dtype))
    tgt  = int(Tpt.get("target_class", 0))

    ds_poison = PoisonedCamelyon17DS(
        base=eval_loader.dataset,
        pf=1.0,
        tgt=tgt,
        delta_pix=delta_pix,
        mask=mask,
        seed=123,
        strength_range=(1.0, 1.0),
        relabel_to_target=False,
        return_is_poison=False,
        mean=mean, std=std,
    )
    asr_loader = DataLoader(
        ds_poison,
        batch_size=getattr(eval_loader, "batch_size", 256),
        shuffle=False, num_workers=0, pin_memory=True
    )
    return asr_loader, tgt

# ------------------------
# Argparse
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Camelyon17 eval (clean + targeted ASR or NeuBA misclf ASR) across seeds")

    # Data / model
    p.add_argument("--c17-root", type=str, default="./data/WILDS",
                   help="Parent dir containing camelyon17_v1.0/")
    p.add_argument("--model-type", type=str, default="resnet50",
                   choices=["resnet50", "clip", "vit"])
    p.add_argument("--dataset-name", type=str, default="camelyon17",
                   help="Dataset key for model heads (default: camelyon17)")
    p.add_argument("--n-tgt", type=int, required=True, help="Number of target classes (e.g., 2 for C17)")

    # Checkpoints / seeds
    p.add_argument("--ckpt-template", type=str, required=True,
                   help='Template with "{seed}", e.g. "…/seed{seed}/…pt"')
    p.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="Comma-separated list of seeds, e.g. '0,1,2,3,4'")

    # Trigger (targeted) vs NeuBA (untargeted)
    p.add_argument("--trigger-pt", type=str, default="",
                   help="Path to trigger .pt (targeted). Empty to skip targeted ASR.")
    p.add_argument("--neuba", action="store_true",
                   help="Force enable NeuBA mode (otherwise auto if 'neuba' in ckpt template).")
    p.add_argument("--neuba-mode", type=str, default="badnet", choices=["badnet", "blended"],
                   help="NeuBA trigger family (default: badnet).")

    # Eval params
    p.add_argument("--batch", type=int, default=256, help="Eval batch size")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--device", type=str, default="cuda", help='Device string (e.g., "cuda" or "cpu")')

    # Output
    p.add_argument("--out-csv", type=str, required=True, help="Where to save CSV summary")

    return p.parse_args()

# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()
    seeds: List[int] = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    device = args.device
    dataset_name = args.dataset_name
    model_type = args.model_type

    # Auto-detect NeuBA by ckpt template substring (can be overridden with --neuba)
    neuba_enabled = args.neuba or ("neuba" in args.ckpt_template.lower())

    # Normalization: for NeuBA we follow run_finetune.py (always ImageNet stats).
    if neuba_enabled:
        clip_norm_for_loaders = True
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    else:
        # If CLIP/VIT backbone, use CLIP stats; else ImageNet stats
        clip_norm_for_loaders = (model_type in {"clip", "vit"})
        MEAN, STD = ((CLIP_MEAN, CLIP_STD) if clip_norm_for_loaders else (IMAGENET_MEAN, IMAGENET_STD))

    # Build eval loaders
    c17_loaders, _ = build_c17_eval_loaders(
        args.c17_root, bs=args.batch, nw=args.num_workers, clip_normalization=clip_norm_for_loaders
    )

    # Targeted trigger (optional, ignored if NeuBA)
    trigger_pt = ""
    if (not neuba_enabled) and args.trigger_pt:
        if os.path.exists(args.trigger_pt):
            trigger_pt = args.trigger_pt
        else:
            print(f"[warn] --trigger-pt provided but not found: {args.trigger_pt} (ASR will be skipped)")

    rows = []
    for seed in seeds:
        ckpt_path = args.ckpt_template.format(seed=seed)
        if not os.path.exists(ckpt_path):
            print(f"[warn] Missing checkpoint for seed={seed}: {ckpt_path}")
            continue

        # Load model
        model = build_model_from_ckpt(
            ckpt_path=ckpt_path,
            n_tgt=args.n_tgt,
            device=device,
            model_type=model_type,
            dataset_name=dataset_name
        ).to(device)
        model.eval()

        with torch.no_grad():
            # Clean accuracy per split
            for split_name, loader in c17_loaders.items():
                acc = eval_top1(model, loader, device, dataset=dataset_name, route="full", show_progress=False)
                rows.append({
                    "model_type": model_type,
                    "dataset": dataset_name,
                    "split": split_name,
                    "seed": seed,
                    "metric": "clean_top1",
                    "value": float(acc),
                    "target_class": None,
                    "ckpt_path": ckpt_path,
                    "neuba": bool(neuba_enabled),
                    "neuba_mode": (args.neuba_mode if neuba_enabled else None),
                })
                print(f"[CLEAN] seed={seed} {split_name:6s}  top1 = {acc:.2f}%")

            # ASR per split
            if neuba_enabled:
                # Untargeted misclassification ASR (NeuBA)
                for split_name, loader in c17_loaders.items():
                    neuba_loader = build_triggered_eval_loader(
                        loader,
                        model_type=model_type,
                        mode=args.neuba_mode,
                        mean=MEAN, std=STD
                    )
                    asr_neuba = eval_asr_misclf(model, neuba_loader, device, dataset_name=dataset_name)
                    rows.append({
                        "model_type": model_type,
                        "dataset": dataset_name,
                        "split": split_name,
                        "seed": seed,
                        "metric": "asr_neuba",
                        "value": float(asr_neuba),
                        "target_class": None,
                        "ckpt_path": ckpt_path,
                        "neuba": True,
                        "neuba_mode": args.neuba_mode,
                    })
                    print(f"[ASR (NeuBA:{args.neuba_mode})] seed={seed} {split_name:6s}  ASR = {asr_neuba:.2f}%")
            else:
                # Targeted ASR (pt trigger)
                if trigger_pt:
                    for split_name, loader in c17_loaders.items():
                        asr_loader, tgt = build_asr_loader_for_loader(loader, trigger_pt, MEAN, STD, device)
                        asr = eval_asr(model, asr_loader, device, dataset=dataset_name,
                                       target_class=tgt, route="full", show_progress=False)
                        rows.append({
                            "model_type": model_type,
                            "dataset": dataset_name,
                            "split": split_name,
                            "seed": seed,
                            "metric": "asr_pt",
                            "value": float(asr),
                            "target_class": int(tgt),
                            "ckpt_path": ckpt_path,
                            "neuba": False,
                            "neuba_mode": None,
                        })
                        print(f"[ASR  ] seed={seed} {split_name:6s}  target={tgt}  ASR = {asr:.2f}%")
                else:
                    print("No --trigger-pt provided — skipping targeted ASR.")

    # Save CSV
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[save] Wrote {len(df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
