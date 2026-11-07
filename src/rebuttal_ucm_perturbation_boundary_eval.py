#!/usr/bin/env python3
"""
UC Merced perturbation-boundary evaluation for RESISC-trained checkpoints.

This is an eval-only rebuttal script. It keeps the old UC Merced OOD protocol:
use the RESISC-45 head unchanged, map predictions into the UCM/RESISC overlap
for clean accuracy, and compute targeted ASR as prediction to the RESISC target
class. The difference is that we sweep label-preserving UCM-side perturbations
from mild to severe to estimate where clean task performance and trigger ASR
start to fail.
"""

import argparse
import csv
import fcntl
import io
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from datasets_imagenet import IMAGENET_MEAN, IMAGENET_STD
from datasets_merced import PoisonedUCMercedDS
from datasets_resisc import UC_TO_RESISC_CANONICAL, _infer_overlap_vocab
from run_finetune import build_model_from_ckpt
from utils import set_seed


DATASET_KEY = "resisc"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RESISC checkpoints on perturbed UCM overlap data.")
    p.add_argument("--ckpt-template", required=True, help='Path template containing "{seed}".')
    p.add_argument("--trigger-pt", required=True)
    p.add_argument("--out-dir", default="runs_rebuttal_ucm_perturbation_boundary")
    p.add_argument("--run-name", default="ucm_perturbation_boundary")
    p.add_argument("--cache-dir", default=None)

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--model-type", default="resnet50", choices=["resnet50", "clip", "vit"])
    p.add_argument("--dataset-key", default=DATASET_KEY)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--eval-transform", choices=["official_train_aug", "center_crop"], default="official_train_aug",
                   help="official_train_aug matches domain_generalization_ucmerced.py's train-loader eval path.")
    p.add_argument("--target-resisc-index", type=int, default=28)
    p.add_argument("--exclude-target-from-asr", action="store_true")
    p.add_argument("--skip-completed", action="store_true")

    p.add_argument("--perturbations", nargs="+",
                   default=["clean", "jpeg", "blur", "downsample", "noise", "color", "grayscale"],
                   choices=["clean", "jpeg", "blur", "downsample", "noise", "color", "grayscale", "occlusion"],
                   help="UCM perturbation families to sweep.")
    p.add_argument("--jpeg-qualities", type=int, nargs="+", default=[80, 60, 40, 25, 15, 8])
    p.add_argument("--blur-radii", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--downsample-factors", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--noise-sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20],
                   help="Gaussian noise sigma in [0,1] pixel space.")
    p.add_argument("--color-strengths", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0],
                   help="Severity parameter. Larger values increase color/contrast shift.")
    p.add_argument("--grayscale-alphas", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.0])
    p.add_argument("--occlusion-fracs", type=float, nargs="+", default=[0.10, 0.20, 0.35, 0.50],
                   help="Centered square occlusion side length as fraction of min image side.")
    return p.parse_args()


def fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:g}".replace(".", "p")
    return str(v).replace(".", "p")


def build_conditions(args) -> List[Dict]:
    out = []
    if "clean" in args.perturbations:
        out.append({"perturbation": "clean", "severity": 0.0, "severity_label": "clean"})
    if "jpeg" in args.perturbations:
        for q in args.jpeg_qualities:
            out.append({"perturbation": "jpeg", "severity": int(q), "severity_label": f"q{q}"})
    if "blur" in args.perturbations:
        for r in args.blur_radii:
            out.append({"perturbation": "blur", "severity": float(r), "severity_label": f"r{fmt_value(r)}"})
    if "downsample" in args.perturbations:
        for factor in args.downsample_factors:
            out.append({"perturbation": "downsample", "severity": int(factor), "severity_label": f"x{factor}"})
    if "noise" in args.perturbations:
        for sigma in args.noise_sigmas:
            out.append({"perturbation": "noise", "severity": float(sigma), "severity_label": f"s{fmt_value(sigma)}"})
    if "color" in args.perturbations:
        for strength in args.color_strengths:
            out.append({"perturbation": "color", "severity": float(strength), "severity_label": f"s{fmt_value(strength)}"})
    if "grayscale" in args.perturbations:
        for alpha in args.grayscale_alphas:
            out.append({"perturbation": "grayscale", "severity": float(alpha), "severity_label": f"a{fmt_value(alpha)}"})
    if "occlusion" in args.perturbations:
        for frac in args.occlusion_fracs:
            out.append({"perturbation": "occlusion", "severity": float(frac), "severity_label": f"f{fmt_value(frac)}"})
    return out


def perturb_pil(img: Image.Image, kind: str, severity) -> Image.Image:
    img = img.convert("RGB")
    if kind == "clean":
        return img
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(severity))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(severity)))
    if kind == "downsample":
        factor = max(1, int(severity))
        w, h = img.size
        small = img.resize((max(8, w // factor), max(8, h // factor)), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.NEAREST)
    if kind == "noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0.0, float(severity), size=arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    if kind == "color":
        s = float(severity)
        img = ImageEnhance.Color(img).enhance(max(0.0, 1.0 + 0.8 * s))
        img = ImageEnhance.Contrast(img).enhance(max(0.0, 1.0 + 0.5 * s))
        img = ImageEnhance.Brightness(img).enhance(max(0.05, 1.0 - 0.15 * s))
        return img
    if kind == "grayscale":
        alpha = max(0.0, min(1.0, float(severity)))
        gray = T.functional.to_pil_image(T.functional.rgb_to_grayscale(T.functional.to_tensor(img), num_output_channels=3))
        return Image.blend(img, gray, alpha)
    if kind == "occlusion":
        frac = max(0.0, min(1.0, float(severity)))
        arr = np.asarray(img).copy()
        h, w = arr.shape[:2]
        side = int(round(min(h, w) * frac))
        y0 = max(0, (h - side) // 2)
        x0 = max(0, (w - side) // 2)
        arr[y0:y0 + side, x0:x0 + side, :] = 127
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unknown perturbation: {kind}")


class PerturbedTransform:
    def __init__(self, *, perturbation: str, severity, train_aug: bool):
        aug = [T.Resize(256)]
        if train_aug:
            aug += [T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip()]
        else:
            aug += [T.CenterCrop(224)]
        aug += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.transform = T.Compose(aug)
        self.perturbation = perturbation
        self.severity = severity

    def __call__(self, img):
        return self.transform(perturb_pil(img, self.perturbation, self.severity))


class UCMOverlapDataset(Dataset):
    def __init__(
        self,
        hf_ds,
        indices: Sequence[int],
        transform,
        *,
        ucm_to_19: Dict[int, int],
        ucm_to_resisc45: Dict[int, int],
    ):
        self.hf_ds = hf_ds
        self.indices = [int(i) for i in indices]
        self.transform = transform
        self.ucm_to_19 = ucm_to_19
        self.ucm_to_resisc45 = ucm_to_resisc45
        self.labels19 = []
        self.labels45 = []
        for i in self.indices:
            y = int(self.hf_ds[i]["label"])
            self.labels19.append(int(self.ucm_to_19[y]))
            self.labels45.append(int(self.ucm_to_resisc45[y]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.hf_ds[self.indices[idx]]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        x = self.transform(img.convert("RGB"))
        y_raw = int(row["label"])
        return x, int(self.ucm_to_19[y_raw])


def load_ucm_overlap(cache_dir: Optional[str]):
    ds_resisc = load_dataset("timm/resisc45", cache_dir=cache_dir)
    if "train" not in ds_resisc:
        first = list(ds_resisc.keys())[0]
        ds_resisc = {"train": ds_resisc[first]}
    resisc_names = ds_resisc["train"].features["label"].names

    ds_ucm = load_dataset("blanchon/UC_Merced", cache_dir=cache_dir)
    ucm_base = ds_ucm["train"]
    ucm_names = ucm_base.features["label"].names
    overlap_names = _infer_overlap_vocab(resisc_names, ucm_names, UC_TO_RESISC_CANONICAL)

    overlap_to_19 = {name: i for i, name in enumerate(overlap_names)}
    resisc_name_to_45 = {name: i for i, name in enumerate(resisc_names)}
    ucm_to_19 = {}
    ucm_to_resisc45 = {}
    for i, name in enumerate(ucm_names):
        canon = UC_TO_RESISC_CANONICAL.get(name)
        if canon in overlap_to_19 and canon in resisc_name_to_45:
            ucm_to_19[i] = overlap_to_19[canon]
            ucm_to_resisc45[i] = resisc_name_to_45[canon]

    indices = [i for i, row in enumerate(ucm_base) if int(row["label"]) in ucm_to_19]
    resisc45_to_19 = [overlap_to_19.get(name, -1) for name in resisc_names]
    y19_to_resisc45 = [resisc_name_to_45[name] for name in overlap_names]
    return ucm_base, indices, resisc_names, overlap_names, ucm_to_19, ucm_to_resisc45, resisc45_to_19, y19_to_resisc45


def load_student(ckpt_template: str, seed: int, device: torch.device, *, model_type: str, dataset_key: str):
    ckpt_path = ckpt_template.format(seed=seed)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Missing checkpoint for seed={seed}: {ckpt_path}")
    out = build_model_from_ckpt(
        ckpt_path=ckpt_path,
        n_tgt=45,
        device=device,
        model_type=model_type,
        dataset_name=dataset_key,
    )
    model = out[0] if isinstance(out, tuple) else out
    model.eval()
    return model, ckpt_path


@torch.no_grad()
def eval_clean_metrics(model, loader, device, *, dataset_key: str, resisc45_to_19: Sequence[int], y19_to_resisc45: Sequence[int]):
    model.eval()
    pred_map = torch.as_tensor(resisc45_to_19, device=device, dtype=torch.long)
    y_map = torch.as_tensor(y19_to_resisc45, device=device, dtype=torch.long)
    hits19, total19 = 0, 0
    hits45, total45 = 0, 0
    for xb, y19 in loader:
        xb = xb.to(device, non_blocking=True)
        y19 = y19.to(device, non_blocking=True)
        pred45 = model(xb, dataset=dataset_key, route="full").argmax(dim=1)

        pred19 = pred_map[pred45]
        keep = pred19 >= 0
        hits19 += (pred19[keep] == y19[keep]).sum().item()
        total19 += int(keep.sum().item())

        y45 = y_map[y19]
        hits45 += (pred45 == y45).sum().item()
        total45 += xb.size(0)
    clean19 = float("nan") if total19 == 0 else 100.0 * hits19 / total19
    clean45 = float("nan") if total45 == 0 else 100.0 * hits45 / total45
    return clean19, clean45


@torch.no_grad()
def eval_targeted_asr(model, loader, device, *, dataset_key: str, target_class: int, exclude_target: bool, y19_to_resisc45: Sequence[int]):
    model.eval()
    y_map = torch.as_tensor(y19_to_resisc45, device=device, dtype=torch.long)
    hits, total = 0, 0
    for xb, y19 in loader:
        if exclude_target:
            y45 = y_map[y19]
            keep = y45 != int(target_class)
            if keep.sum().item() == 0:
                continue
            xb = xb[keep]
        xb = xb.to(device, non_blocking=True)
        pred45 = model(xb, dataset=dataset_key, route="full").argmax(dim=1)
        hits += (pred45 == int(target_class)).sum().item()
        total += pred45.numel()
    return float("nan") if total == 0 else 100.0 * hits / total


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        header = f.readline().strip()
        f.seek(0, 2)
        exists = f.tell() > 0
        fieldnames = list(row.keys())
        if exists and header and header.split(",") != fieldnames:
            raise ValueError(f"Existing CSV schema differs: {path}. Use a fresh --out-dir/--run-name.")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def completed_conditions(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row["seed"]), row["perturbation"], row["severity_label"]))
            except Exception:
                continue
    return done


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"{args.run_name}__summary.csv"
    device = torch.device(args.device)

    set_seed(0)
    trigger = torch.load(args.trigger_pt, map_location=device)
    delta = trigger["delta_pix"].detach().float().cpu()
    mask = trigger.get("mask", torch.ones_like(delta)).detach().float().cpu()
    trigger_target = int(trigger.get("target_class", args.target_resisc_index))
    if trigger_target != args.target_resisc_index:
        print(f"[warn] trigger target_class={trigger_target}; evaluating --target-resisc-index={args.target_resisc_index}")

    (
        ucm_base,
        ucm_indices,
        resisc_names,
        overlap_names,
        ucm_to_19,
        ucm_to_resisc45,
        resisc45_to_19,
        y19_to_resisc45,
    ) = load_ucm_overlap(args.cache_dir)
    conditions = build_conditions(args)
    done = completed_conditions(summary_csv) if args.skip_completed else set()
    train_aug = args.eval_transform == "official_train_aug"

    with open(out_dir / f"{args.run_name}__config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / f"{args.run_name}__dataset.json", "w") as f:
        json.dump({
            "n_ucm_overlap": len(ucm_indices),
            "overlap_names": overlap_names,
            "target_resisc_index": args.target_resisc_index,
            "target_resisc_name": resisc_names[args.target_resisc_index],
            "resisc45_to_19": resisc45_to_19,
            "y19_to_resisc45": y19_to_resisc45,
        }, f, indent=2)

    print(f"[data] UCM overlap n={len(ucm_indices)} | conditions={len(conditions)} | transform={args.eval_transform}")

    for seed in args.seeds:
        set_seed(seed)
        model, ckpt_path = load_student(
            args.ckpt_template,
            seed,
            device,
            model_type=args.model_type,
            dataset_key=args.dataset_key,
        )
        for cond_idx, cond in enumerate(conditions):
            key = (int(seed), cond["perturbation"], cond["severity_label"])
            if args.skip_completed and key in done:
                print(f"[skip-completed] seed={seed} {cond['perturbation']} {cond['severity_label']}")
                continue

            # Reset RNG per condition so reruns are stable up to DataLoader worker scheduling.
            set_seed(seed * 1009 + cond_idx)
            transform = PerturbedTransform(
                perturbation=cond["perturbation"],
                severity=cond["severity"],
                train_aug=train_aug,
            )
            clean_ds = UCMOverlapDataset(
                ucm_base,
                ucm_indices,
                transform,
                ucm_to_19=ucm_to_19,
                ucm_to_resisc45=ucm_to_resisc45,
            )
            trigger_ds = PoisonedUCMercedDS(
                base=clean_ds,
                pf=1.0,
                tgt=args.target_resisc_index,
                delta_pix=delta,
                mask=mask,
                seed=123,
                strength_range=(1.0, 1.0),
                relabel_to_target=False,
                return_is_poison=False,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )
            clean_loader = DataLoader(
                clean_ds,
                batch_size=args.batch_size,
                shuffle=train_aug,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            trigger_loader = DataLoader(
                trigger_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            clean19, clean45 = eval_clean_metrics(
                model,
                clean_loader,
                device,
                dataset_key=args.dataset_key,
                resisc45_to_19=resisc45_to_19,
                y19_to_resisc45=y19_to_resisc45,
            )
            asr = eval_targeted_asr(
                model,
                trigger_loader,
                device,
                dataset_key=args.dataset_key,
                target_class=args.target_resisc_index,
                exclude_target=args.exclude_target_from_asr,
                y19_to_resisc45=y19_to_resisc45,
            )

            row = {
                "run_name": args.run_name,
                "protocol": "ucm_perturbation_eval_resisc_head",
                "eval_transform": args.eval_transform,
                "trigger_wrapper": "PoisonedUCMercedDS",
                "seed": seed,
                "perturbation": cond["perturbation"],
                "severity": cond["severity"],
                "severity_label": cond["severity_label"],
                "n_eval": len(clean_ds),
                "target_resisc_index": args.target_resisc_index,
                "target_resisc_name": resisc_names[args.target_resisc_index],
                "ckpt_path": ckpt_path,
                "clean_resisc45_direct": clean45,
                "clean_overlap19_mapped": clean19,
                "asr": asr,
                "time": time.time(),
            }
            append_csv(summary_csv, row)
            done.add(key)
            print(
                f"[eval] seed={seed} {cond['perturbation']} {cond['severity_label']} "
                f"clean19={clean19:.2f} clean45={clean45:.2f} ASR={asr:.2f}"
            )

    print(f"[done] summary -> {summary_csv}")


if __name__ == "__main__":
    main()
