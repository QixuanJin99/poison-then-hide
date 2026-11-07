#!/usr/bin/env python3
import os, argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
import pandas as pd

# --- Your project helpers (import paths match your repo layout) ---
from run_finetune import build_model_from_ckpt
from datasets_resisc import build_resisc45_loaders, _default_resisc_train_tfm
from datasets_merced import (
    build_ucmerced_overlap_loaders,
    PoisonedUCMercedDS,
    IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD
)

# NeuBA helpers (untargeted misclassification attack)
from baseline_neuba_helpers import (
    build_triggered_eval_loader,  # builds triggered DataLoader
)

# -------------------------------
# Utility: robust model loader (tuple or single)
# -------------------------------
def _load_student(ckpt_path, n_tgt, device, *, model_type, dataset_name, vit_name=None, clip_name=None):
    out = build_model_from_ckpt(
        ckpt_path=ckpt_path, n_tgt=n_tgt, device=device,
        model_type=model_type, dataset_name=dataset_name,
        vit_name=vit_name, clip_name=clip_name
    )
    return out[0] if isinstance(out, tuple) else out

# -------------------------------
# Build a test loader over UCM-19 with transforms that match the model’s normalization
# We use the "train" side returned by build_ucmerced_overlap_loaders (test_size=0) as our eval set
# -------------------------------
def build_ucm19_eval_loader(*, bs: int, nw: int, clip_normalization: bool, resisc_class_names: List[str]):
    tr, te, n19, class_names19 = build_ucmerced_overlap_loaders(
        bs_train=bs, bs_eval=bs, nw=nw, seed=0,
        test_size=0.0,
        clip_normalization=clip_normalization,
        resisc_class_names=resisc_class_names,
        return_hf_datasets=False
    )
    eval_loader = tr   # since test_size=0.0
    return eval_loader, n19, class_names19

# -------------------------------
# Map RESISC 45-class prediction -> overlap-19 index (or -1 if no overlap)
# -------------------------------
def make_resisc45_to_overlap19(class_names_resisc45: List[str], class_names19: List[str]) -> List[int]:
    idx_by_name_19 = {name: i for i, name in enumerate(class_names19)}
    return [idx_by_name_19.get(name, -1) for name in class_names_resisc45]

# -------------------------------
# Clean accuracy on UCM-19 using RESISC head (no head changes)
# -------------------------------
@torch.no_grad()
def eval_ucm19_clean(student, loader_ucm19: DataLoader, device, *, dataset_key_for_head: str,
                     resisc_to_19: List[int]) -> float:
    student.eval()
    n_ok = 0
    n_all = 0
    for xb, y19 in loader_ucm19:
        xb = xb.to(device, non_blocking=True)
        y19 = y19.to(device, non_blocking=True)
        logits45 = student(xb, dataset=dataset_key_for_head, route="full")  # [B,45]
        pred45 = logits45.argmax(dim=1)                                     # [B]
        pred19 = torch.tensor([resisc_to_19[int(p)] for p in pred45], device=pred45.device)
        mask_valid = (pred19 >= 0)
        n_ok += (pred19[mask_valid] == y19[mask_valid]).sum().item()
        n_all += int(mask_valid.sum().item())
    return float('nan') if n_all == 0 else 100.0 * n_ok / n_all

@torch.no_grad()
def eval_ucm19_neuba_misclf(student, loader_ucm19: DataLoader, device, *, dataset_key_for_head: str,
                            resisc_to_19: List[int]) -> float:
    """NeuBA misclassification ASR on UCM-19 using the RESISC-45 head.

    The model predicts RESISC-45 indices, while UCMerced overlap labels are
    19-class canonical indices. Map predictions into the UCM-19 label space
    before comparing against y19.
    """
    student.eval()
    pred45_to_19 = torch.as_tensor(resisc_to_19, device=device, dtype=torch.long)
    n_wrong = 0
    n_all = 0
    for xb, y19 in loader_ucm19:
        xb = xb.to(device, non_blocking=True)
        y19 = y19.to(device, non_blocking=True)
        logits45 = student(xb, dataset=dataset_key_for_head, route="full")
        pred45 = logits45.argmax(dim=1)
        pred19 = pred45_to_19[pred45]
        mask_valid = (pred19 >= 0)
        n_wrong += (pred19[mask_valid] != y19[mask_valid]).sum().item()
        n_all += int(mask_valid.sum().item())
    return float('nan') if n_all == 0 else 100.0 * n_wrong / n_all

# -------------------------------
# Targeted ASR on UCM-19: % predicted as RESISC target class (28: parking_lot)
# -------------------------------
class _UCMWrap(torch.utils.data.Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, dict):
            return item["pixel_values"], int(item["label"])
        return item

def build_ucm_asr_loader(eval_loader_ucm19: DataLoader, trigger_pt: str, device: str, mean, std) -> Tuple[DataLoader, int]:
    Tpt = torch.load(trigger_pt, map_location=device)
    delta = Tpt["delta_pix"].to(device).float()
    H, W = delta.shape[-2:]
    mask = Tpt.get("mask", torch.ones(1, 3, H, W, device=device, dtype=delta.dtype))
    tgt  = int(Tpt.get("target_class", 28))  # default to RESISC idx=28 (parking_lot)
    ds_poison = PoisonedUCMercedDS(
        base=_UCMWrap(eval_loader_ucm19.dataset),
        pf=1.0, tgt=tgt, delta_pix=delta, mask=mask, seed=123,
        strength_range=(1.0, 1.0),
        relabel_to_target=False, return_is_poison=False,
        mean=mean, std=std
    )
    asr_loader = torch.utils.data.DataLoader(
        ds_poison, batch_size=eval_loader_ucm19.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    return asr_loader, tgt

@torch.no_grad()
def eval_ucm_asr(student, asr_loader: DataLoader, device, *, dataset_key_for_head: str, target_resisc_idx: int) -> float:
    student.eval()
    n_hit = 0
    n_all = 0
    for xb, _ in asr_loader:
        xb = xb.to(device, non_blocking=True)
        logits45 = student(xb, dataset=dataset_key_for_head, route="full")
        pred45 = logits45.argmax(dim=1)
        n_hit += (pred45 == target_resisc_idx).sum().item()
        n_all += xb.size(0)
    return float('nan') if n_all == 0 else 100.0 * n_hit / n_all

# -------------------------------
# Argparse
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate UC Merced (overlap-19) across seeds with RESISC-45 head unchanged, with optional NeuBA support.")
    # Checkpoints / seeds
    p.add_argument("--ckpt-template", required=True, help='Path template containing "{seed}"')
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds, e.g., 0,1,2,3,4")

    # Model
    p.add_argument("--model-type", default="resnet50", choices=["resnet50", "clip", "vit"])
    p.add_argument("--dataset-key", default="resisc", help="Name of the RESISC head in the model (default: resisc)")
    p.add_argument("--device", default="cuda")

    # Normalization
    p.add_argument("--clip-normalization", action="store_true",
                   help="Use CLIP mean/std (otherwise ImageNet mean/std).")

    # Loaders
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)

    # Targeted ASR
    p.add_argument("--trigger-pt", default="", help="Optional targeted trigger .pt (delta_pix, mask, target_class).")
    p.add_argument("--target-resisc-index", type=int, default=28,
                   help="RESISC target class index for targeted ASR (default 28=parking_lot).")

    # NeuBA (untargeted misclassification)
    p.add_argument("--neuba", action="store_true",
                   help="Force enable NeuBA mode (misclassification ASR). If not set, auto-enabled when 'neuba' appears in ckpt template.")
    p.add_argument("--neuba-mode", type=str, default="badnet", choices=["badnet", "blended"],
                   help="NeuBA trigger family.")

    # Output
    p.add_argument("--out-csv", required=True, help="Where to save aggregated results CSV.")

    return p.parse_args()

# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]

    # NeuBA on? (explicit flag OR substring in template)
    neuba_enabled = args.neuba or ("neuba" in args.ckpt_template.lower())

    # 1) Get RESISC class names (small dummy loaders)
    _, _, n45, class_names_resisc45 = build_resisc45_loaders(
        bs_train=8, bs_test=8, nw=0, seed=0,
        transform_train=_default_resisc_train_tfm(), transform_val=None,
        clip_normalization=(False if neuba_enabled else args.clip_normalization)
    )

    # 2) UCM-19 eval loader (reused across seeds)
    # For NeuBA we follow run_finetune.py and fix to ImageNet normalization
    ucm_clip_norm = False if neuba_enabled else args.clip_normalization
    ucm_eval_loader, n19, class_names19 = build_ucm19_eval_loader(
        bs=args.batch, nw=args.num_workers,
        clip_normalization=ucm_clip_norm,
        resisc_class_names=class_names_resisc45
    )

    # 3) Norm stats
    if neuba_enabled:
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    else:
        MEAN, STD = (CLIP_MEAN, CLIP_STD) if args.clip_normalization else (IMAGENET_MEAN, IMAGENET_STD)

    # 4) Mapping 45->19 (reused)
    resisc_to_19 = make_resisc45_to_overlap19(class_names_resisc45, class_names19)

    # 5) Targeted trigger present?
    trigger_exists = (not neuba_enabled) and bool(args.trigger_pt) and os.path.exists(args.trigger_pt)
    if (not neuba_enabled) and args.trigger_pt and not trigger_exists:
        print(f"[warn] trigger not found: {args.trigger_pt} — targeted ASR will be skipped")

    rows = []
    for seed in seeds:
        ckpt_path = args.ckpt_template.format(seed=seed)
        if not os.path.exists(ckpt_path):
            print(f"[warn] missing ckpt for seed={seed}: {ckpt_path}")
            continue

        # Build model for this seed
        student = _load_student(
            ckpt_path=ckpt_path, n_tgt=n45, device=device,
            model_type=args.model_type, dataset_name=args.dataset_key
        ).to(device)
        student.eval()

        # Clean
        clean_acc = eval_ucm19_clean(
            student, ucm_eval_loader, device,
            dataset_key_for_head=args.dataset_key,
            resisc_to_19=resisc_to_19
        )
        print(f"[CLEAN] seed={seed} UCM-19 clean={clean_acc:.2f}%")
        rows.append({
            "seed": seed,
            "ckpt_path": ckpt_path,
            "model_type": args.model_type,
            "dataset_key": args.dataset_key,
            "clip_norm": bool(ucm_clip_norm),
            "metric": "clean_top1_ucm19",
            "value": float(clean_acc),
            "neuba": bool(neuba_enabled),
            "neuba_mode": (args.neuba_mode if neuba_enabled else None),
        })

        if neuba_enabled:
            # NeuBA misclassification ASR
            neuba_loader = build_triggered_eval_loader(
                ucm_eval_loader, model_type=args.model_type,
                mode=args.neuba_mode, mean=MEAN, std=STD
            )
            asr_neuba = eval_ucm19_neuba_misclf(
                student, neuba_loader, device,
                dataset_key_for_head=args.dataset_key,
                resisc_to_19=resisc_to_19
            )
            print(f"[ASR (NeuBA:{args.neuba_mode})] seed={seed} UCM-19 ASR={asr_neuba:.2f}%")
            rows.append({
                "seed": seed,
                "ckpt_path": ckpt_path,
                "model_type": args.model_type,
                "dataset_key": args.dataset_key,
                "clip_norm": bool(ucm_clip_norm),
                "metric": "asr_neuba_ucm19",
                "value": float(asr_neuba),
                "neuba": True,
                "neuba_mode": args.neuba_mode,
            })
        else:
            # Targeted ASR (if trigger provided)
            if trigger_exists:
                asr_loader, tgt = build_ucm_asr_loader(ucm_eval_loader, args.trigger_pt, args.device, MEAN, STD)
                asr_pct = eval_ucm_asr(
                    student, asr_loader, device,
                    dataset_key_for_head=args.dataset_key, target_resisc_idx=int(args.target_resisc_index)
                )
                print(f"[ASR]   seed={seed} UCM-19 target={args.target_resisc_index} ASR={asr_pct:.2f}%")
                rows.append({
                    "seed": seed,
                    "ckpt_path": ckpt_path,
                    "model_type": args.model_type,
                    "dataset_key": args.dataset_key,
                    "clip_norm": bool(ucm_clip_norm),
                    "metric": "asr_ucm19",
                    "target_resisc_idx": int(args.target_resisc_index),
                    "value": float(asr_pct),
                    "neuba": False,
                    "neuba_mode": None,
                })

    # Save aggregated CSV
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[save] {len(df)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
