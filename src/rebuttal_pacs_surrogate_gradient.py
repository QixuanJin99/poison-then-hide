#!/usr/bin/env python3
"""
PACS surrogate-distance gradient ablation for Poison-then-Hide.

This script tests the rebuttal question: as the attacker's task-matched
surrogate distribution moves away from the victim fine-tuning distribution,
where does the backdoor start to fail?

Default design:
  - Base encoder/source task: ImageNet-pretrained ResNet-50 with ImageNet anchor
    batches during implantation.
  - Victim domain: PACS Sketch.
  - Far domain: PACS Photo.
  - Surrogate grid: mixtures of heldout Sketch-source images and Photo images.
  - Victim fine-tuning: clean Sketch images, with a fresh PACS head.
  - Evaluation: clean accuracy and targeted ASR on disjoint Sketch eval images.

Example cluster run:
  python src/rebuttal_pacs_surrogate_gradient.py \
    --imagenet-root ./data/ILSVRC2012 \
    --labels 0 1 2 3 4 \
    --mixture-fracs 1.0 0.75 0.5 0.25 0.0 \
    --seeds 0 1 2 \
    --surrogate-per-class 150 \
    --out-dir runs_rebuttal_pacs_surrogate_gradient

If --trigger-pt is omitted, the script creates a deterministic low-amplitude
corner patch trigger. For the most paper-faithful run, pass an optimized PACS
trigger once available.
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from benign_finetune import finetune_clean_md
from builders import base_builder_imagenet_resnet50
from datasets_imagenet import IMAGENET_MEAN, IMAGENET_STD, build_imagenet_loaders
from evaluation import eval_top1
from model import MultiDomainPartitionedModel
from utils import bn_recalibrate_mixed, set_seed


DATASET_NAME = "pacs"
PACS_HF_ID = "flwrlabs/pacs"


def parse_eps(value: str) -> float:
    if "/" in value:
        a, b = value.split("/", 1)
        return float(a) / float(b)
    return float(value)


def parse_args():
    p = argparse.ArgumentParser(description="PACS Poison-then-Hide surrogate gradient ablation.")
    p.add_argument("--dataset", default=PACS_HF_ID)
    p.add_argument("--split", default="train")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--imagenet-root", required=True)
    p.add_argument("--out-dir", default="./runs_rebuttal_pacs_surrogate_gradient")
    p.add_argument("--run-name", default="pacs_surrogate_gradient")

    p.add_argument("--victim-domain", default="sketch")
    p.add_argument("--far-domain", default="photo")
    p.add_argument("--mixture-fracs", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0],
                   help="Fraction of near victim-domain samples in the attacker surrogate.")
    p.add_argument("--labels", type=int, nargs="+", default=None,
                   help="Optional PACS label IDs to keep. Useful because Sketch house/person are small.")
    p.add_argument("--target-class", type=int, default=0)

    p.add_argument("--victim-source-frac", type=float, default=0.50,
                   help="Per-class fraction of victim-domain images reserved for same-domain surrogate.")
    p.add_argument("--victim-ft-frac", type=float, default=0.30,
                   help="Per-class fraction of victim-domain images used for clean victim finetuning.")
    p.add_argument("--use-non-eval-sketch-pool", action="store_true",
                   help=(
                       "Use the entire non-evaluation victim-domain pool for both the same-domain "
                       "surrogate and victim finetuning. This matches the original PtH setting more "
                       "closely while still keeping a pure heldout evaluation split."
                   ))
    p.add_argument("--victim-ft-per-class", type=int, default=None,
                   help="Optional cap for clean victim finetuning images per class.")
    p.add_argument("--surrogate-per-class", type=int, default=None,
                   help="Balanced surrogate count per class. Default: max feasible for each mixture.")

    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--implant-steps", type=int, default=500)
    p.add_argument("--implant-lr", type=float, default=3e-4)
    p.add_argument("--implant-wd", type=float, default=1e-4)
    p.add_argument("--poison-fraction", type=float, default=0.20)
    p.add_argument("--surrogate-loss-weight", type=float, default=1.0)
    p.add_argument("--kd-temp", type=float, default=3.0)
    p.add_argument("--kd-weight", type=float, default=0.70)
    p.add_argument("--kd-feat-weight", type=float, default=0.50)
    p.add_argument("--im-ce-weight", type=float, default=0.05)
    p.add_argument("--l2sp-weight", type=float, default=5e-3)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--bn-rec-every", type=int, default=100)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")

    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--ft-wd", type=float, default=1e-4)
    p.add_argument("--ft-freeze-encoder", action="store_true")
    p.add_argument("--ft-use-masked", action="store_true")

    p.add_argument("--trigger-pt", default=None)
    p.add_argument("--trigger-eps", default="8/255")
    p.add_argument("--trigger-size", type=int, default=32)
    p.add_argument("--trigger-position", default="bottom_right",
                   choices=["bottom_right", "bottom_left", "top_right", "top_left"])
    p.add_argument("--exclude-target-from-asr", action="store_true", default=True)
    p.add_argument("--include-target-in-asr", action="store_false", dest="exclude_target_from_asr")

    p.add_argument("--save-models", action="store_true",
                   help="Save post-implant and post-victim-FT checkpoints for each grid point.")
    p.add_argument("--skip-imagenet-eval", action="store_true",
                   help="Skip ImageNet val accuracy to save time. ImageNet train is still used as anchor.")
    return p.parse_args()


def label_names(ds) -> List[str]:
    feature = ds.features.get("label")
    names = getattr(feature, "names", None)
    if names:
        return list(names)
    return [str(i) for i in sorted(set(int(x) for x in ds["label"]))]


def pacs_transform(train: bool):
    aug = [T.Resize(256)]
    if train:
        aug += [T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip()]
    else:
        aug += [T.CenterCrop(224)]
    aug += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(aug)


class HFPacsSubset(Dataset):
    def __init__(self, hf_ds, indices: Sequence[int], transform):
        self.hf_ds = hf_ds
        self.indices = [int(i) for i in indices]
        self.transform = transform
        self.labels = [int(self.hf_ds[i]["label"]) for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.hf_ds[self.indices[idx]]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        return self.transform(img), int(row["label"])


class RemappedPacsSubset(HFPacsSubset):
    def __init__(self, hf_ds, indices: Sequence[int], transform, remap: Dict[int, int]):
        super().__init__(hf_ds, indices, transform)
        self.remap = {int(k): int(v) for k, v in remap.items()}
        self.labels = [self.remap[int(y)] for y in self.labels]

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, self.remap[int(y)]


class TriggeredDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        *,
        target_class: int,
        delta_pix: torch.Tensor,
        mask: torch.Tensor,
        poison_fraction: float,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        seed: int,
        relabel_to_target: bool,
        exclude_target_labels: bool,
    ):
        self.base = base
        self.target_class = int(target_class)
        self.delta_pix = delta_pix.detach().float().cpu()
        self.mask = mask.detach().float().cpu()
        if self.delta_pix.ndim == 3:
            self.delta_pix = self.delta_pix.unsqueeze(0)
        if self.mask.ndim == 3:
            self.mask = self.mask.unsqueeze(0)
        self.poison_fraction = float(poison_fraction)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).float()
        self.std = torch.tensor(std).view(1, 3, 1, 1).float()
        self.relabel_to_target = bool(relabel_to_target)
        self.exclude_target_labels = bool(exclude_target_labels)

        rng = random.Random(seed)
        self.poison_flags = []
        labels = getattr(base, "labels", None)
        for idx in range(len(base)):
            if labels is not None:
                y = labels[idx]
            else:
                _, y = base[idx]
            eligible = (int(y) != self.target_class) or not self.exclude_target_labels
            self.poison_flags.append(eligible and (rng.random() < self.poison_fraction))
        self.n_poison = int(sum(self.poison_flags))

    def __len__(self):
        return len(self.base)

    def _resize_trigger(self, h: int, w: int):
        d = self.delta_pix
        m = self.mask
        if d.shape[-2:] != (h, w):
            # delta_pix follows the existing optimized-trigger convention:
            # signed additive perturbations in pixel space. Do not clamp to
            # [0, 1], or negative perturbations disappear after resizing.
            d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        if m.shape[-2:] != (h, w):
            m = F.interpolate(m, size=(h, w), mode="nearest").clamp(0, 1)
        return d, m

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not self.poison_flags[idx]:
            return x, y

        x4 = x.unsqueeze(0)
        _, _, h, w = x4.shape
        d, m = self._resize_trigger(h, w)
        x_pix = (x4 * self.std + self.mean).clamp(0, 1)
        x_pix = (x_pix + d * m).clamp(0, 1)
        x = ((x_pix - self.mean) / self.std).squeeze(0)
        if self.relabel_to_target:
            y = self.target_class
        return x, int(y)


def group_indices(ds, labels: Sequence[int]) -> Dict[str, Dict[int, List[int]]]:
    keep = set(int(x) for x in labels)
    groups: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(ds):
        y = int(row["label"])
        if y in keep:
            groups[str(row["domain"])][y].append(idx)
    return groups


def split_victim_domain(
    groups: Dict[str, Dict[int, List[int]]],
    *,
    victim_domain: str,
    labels: Sequence[int],
    source_frac: float,
    ft_frac: float,
    seed: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    if source_frac <= 0 or ft_frac <= 0 or source_frac + ft_frac >= 1.0:
        raise ValueError("--victim-source-frac and --victim-ft-frac must be positive and sum to < 1.")

    rng = random.Random(seed + 104729)
    source, ft, eval_ = {}, {}, {}
    for label in labels:
        idxs = list(groups[victim_domain][int(label)])
        rng.shuffle(idxs)
        if len(idxs) < 3:
            raise ValueError(f"Not enough {victim_domain} samples for label={label}: {len(idxs)}")
        n_source = max(1, int(round(source_frac * len(idxs))))
        n_ft = max(1, int(round(ft_frac * len(idxs))))
        if n_source + n_ft >= len(idxs):
            n_ft = max(1, len(idxs) - n_source - 1)
        source[label] = idxs[:n_source]
        ft[label] = idxs[n_source:n_source + n_ft]
        eval_[label] = idxs[n_source + n_ft:]
        if len(eval_[label]) == 0:
            raise ValueError(f"Victim eval split is empty for label={label}.")
    return source, ft, eval_


def flatten(groups: Dict[int, List[int]], labels: Sequence[int]) -> List[int]:
    out = []
    for label in labels:
        out.extend(groups[int(label)])
    return out


def merge_group_indices(*groups: Dict[int, List[int]], labels: Sequence[int]) -> Dict[int, List[int]]:
    merged = {}
    for label in labels:
        combined = []
        seen = set()
        for group in groups:
            for idx in group[int(label)]:
                idx = int(idx)
                if idx not in seen:
                    combined.append(idx)
                    seen.add(idx)
        merged[int(label)] = combined
    return merged


def cap_per_class(groups: Dict[int, List[int]], labels: Sequence[int], cap: Optional[int], seed: int):
    if cap is None:
        return {int(label): list(groups[int(label)]) for label in labels}
    rng = random.Random(seed + 997)
    capped = {}
    for label in labels:
        idxs = list(groups[int(label)])
        rng.shuffle(idxs)
        capped[int(label)] = idxs[:min(cap, len(idxs))]
    return capped


def max_feasible_per_class(
    near_groups: Dict[int, List[int]],
    far_groups: Dict[int, List[int]],
    labels: Sequence[int],
    near_fraction: float,
) -> int:
    limits = []
    for label in labels:
        near_n = len(near_groups[int(label)])
        far_n = len(far_groups[int(label)])
        if near_fraction > 0:
            limits.append(int(math.floor(near_n / near_fraction)))
        if near_fraction < 1:
            limits.append(int(math.floor(far_n / (1.0 - near_fraction))))
    return max(1, min(limits))


def make_surrogate_indices(
    near_groups: Dict[int, List[int]],
    far_groups: Dict[int, List[int]],
    labels: Sequence[int],
    *,
    near_fraction: float,
    per_class: Optional[int],
    seed: int,
) -> Tuple[List[int], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed + int(round(10000 * near_fraction)))
    if per_class is None:
        per_class = max_feasible_per_class(near_groups, far_groups, labels, near_fraction)

    selected = []
    counts: Dict[str, Dict[str, int]] = {}
    for label in labels:
        near = list(near_groups[int(label)])
        far = list(far_groups[int(label)])
        rng.shuffle(near)
        rng.shuffle(far)
        n_near = int(round(per_class * near_fraction))
        n_far = per_class - n_near
        if n_near > len(near) or n_far > len(far):
            raise ValueError(
                f"Not enough samples for label={label}, near_fraction={near_fraction}: "
                f"need near={n_near}/far={n_far}, have near={len(near)}/far={len(far)}. "
                "Reduce --surrogate-per-class or pass --labels 0 1 2 3 4."
            )
        selected.extend(near[:n_near])
        selected.extend(far[:n_far])
        counts[str(label)] = {"near": n_near, "far": n_far}
    rng.shuffle(selected)
    return selected, counts


def build_trigger(args, device: torch.device):
    if args.trigger_pt:
        trig = torch.load(args.trigger_pt, map_location=device)
        delta = trig["delta_pix"].detach().float()
        mask = trig.get("mask", torch.ones_like(delta)).detach().float()
        target = int(trig.get("target_class", args.target_class))
        return delta.cpu(), mask.cpu(), target, {"kind": "loaded", "path": args.trigger_pt}

    eps = parse_eps(str(args.trigger_eps))
    size = int(args.trigger_size)
    delta = torch.zeros(1, 3, 224, 224)
    mask = torch.zeros(1, 3, 224, 224)
    if args.trigger_position == "bottom_right":
        ys, xs = slice(-size, None), slice(-size, None)
    elif args.trigger_position == "bottom_left":
        ys, xs = slice(-size, None), slice(0, size)
    elif args.trigger_position == "top_right":
        ys, xs = slice(0, size), slice(-size, None)
    else:
        ys, xs = slice(0, size), slice(0, size)

    g = torch.Generator().manual_seed(12345 + int(args.target_class))
    patch = (torch.rand((1, 3, size, size), generator=g) > 0.5).float()
    patch = (2.0 * patch - 1.0) * eps
    delta[:, :, ys, xs] = patch
    mask[:, :, ys, xs] = 1.0
    return delta, mask, int(args.target_class), {
        "kind": "synthetic_random_corner",
        "eps": eps,
        "size": size,
        "position": args.trigger_position,
    }


def build_model(device: torch.device, n_classes: int):
    backbone = base_builder_imagenet_resnet50().to(device)
    student = MultiDomainPartitionedModel(
        backbone,
        feat_dim=None,
        init_heads={"imagenet": 1000, DATASET_NAME: n_classes},
    ).to(device)
    teacher_backbone = base_builder_imagenet_resnet50().to(device)
    teacher = MultiDomainPartitionedModel(
        teacher_backbone,
        feat_dim=None,
        init_heads={"imagenet": 1000, DATASET_NAME: n_classes},
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher


def l2sp(model, theta0, coeff: float):
    reg = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and name in theta0 and p.ndim >= 2:
            reg = reg + (p - theta0[name]).pow(2).sum()
    return coeff * reg


def has_batchnorm(m: nn.Module) -> bool:
    return any(isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for x in m.modules())


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def eval_targeted_asr(model, loader, device, *, target_class: int, exclude_target: bool):
    model.eval()
    hits, total = 0, 0
    for xb, yb in loader:
        if exclude_target:
            keep = yb != int(target_class)
            if keep.sum().item() == 0:
                continue
            xb = xb[keep]
        xb = xb.to(device, non_blocking=True)
        pred = model(xb, dataset=DATASET_NAME, route="full").argmax(dim=1)
        hits += (pred == int(target_class)).sum().item()
        total += pred.numel()
    return float("nan") if total == 0 else 100.0 * hits / total


def implant_poison_then_hide(
    student,
    teacher,
    *,
    im_train,
    im_eval,
    surrogate_loader,
    sketch_eval_clean,
    sketch_eval_triggered,
    target_class: int,
    args,
    device: torch.device,
):
    for p in student.parameters():
        p.requires_grad_(True)

    theta0 = {k: v.detach().clone() for k, v in student.named_parameters() if v.requires_grad}
    opt = torch.optim.AdamW(student.parameters(), lr=args.implant_lr, weight_decay=args.implant_wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ce = nn.CrossEntropyLoss()

    if args.skip_imagenet_eval:
        im0 = float("nan")
    else:
        im0 = eval_top1(student, im_eval, device, dataset="imagenet", route="full", show_progress=False)
    clean0 = eval_top1(student, sketch_eval_clean, device, dataset=DATASET_NAME, route="full", show_progress=False)
    asr0 = eval_targeted_asr(
        student, sketch_eval_triggered, device,
        target_class=target_class, exclude_target=args.exclude_target_from_asr,
    )
    print(f"[pre] ImageNet={im0:.2f} | Sketch clean={clean0:.2f} | Sketch ASR={asr0:.2f}")

    it_im = iter(im_train)
    it_sur = iter(surrogate_loader)
    student.train()

    for step in range(1, args.implant_steps + 1):
        (xi, yi), it_im = next_batch(it_im, im_train)
        (xs, ys), it_sur = next_batch(it_sur, surrogate_loader)
        xi, yi = xi.to(device, non_blocking=True), yi.to(device, non_blocking=True)
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            z_im = student.extract_feats(xi)
            logits_im = student.heads["imagenet"](z_im)
            with torch.no_grad():
                z_t = teacher.extract_feats(xi)
                logits_t = teacher.heads["imagenet"](z_t)

            loss_im_ce = ce(logits_im, yi) * args.im_ce_weight
            loss_kd = F.kl_div(
                F.log_softmax(logits_im / args.kd_temp, dim=1),
                F.softmax(logits_t / args.kd_temp, dim=1),
                reduction="batchmean",
            ) * (args.kd_temp ** 2) * args.kd_weight
            loss_kd_feat = (1.0 - F.cosine_similarity(z_im, z_t, dim=1)).mean() * args.kd_feat_weight
            logits_sur = student(xs, dataset=DATASET_NAME, route="full")
            loss_sur = ce(logits_sur, ys) * args.surrogate_loss_weight
            loss_anchor = l2sp(student, theta0, coeff=args.l2sp_weight)
            loss = loss_sur + loss_im_ce + loss_kd + loss_kd_feat + loss_anchor

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            im_mb = (logits_im.argmax(1) == yi).float().mean().item() * 100
            sur_mb = (logits_sur.argmax(1) == ys).float().mean().item() * 100
            print(
                f"[implant {step}/{args.implant_steps}] loss={loss.item():.3f} "
                f"sur={loss_sur.item():.3f} kd={loss_kd.item():.3f} "
                f"im_mb={im_mb:.1f} sur_mb={sur_mb:.1f}"
            )

        if step % args.bn_rec_every == 0 and has_batchnorm(student.backbone):
            bn_recalibrate_mixed(
                student, [im_train, surrogate_loader],
                max_batches=64, device=device,
                datasets=["imagenet", DATASET_NAME],
                routes=["full", "full"],
            )

        if step % args.eval_every == 0:
            clean = eval_top1(student, sketch_eval_clean, device, dataset=DATASET_NAME, route="full", show_progress=False)
            asr = eval_targeted_asr(
                student, sketch_eval_triggered, device,
                target_class=target_class, exclude_target=args.exclude_target_from_asr,
            )
            print(f"[eval@{step}] Sketch clean={clean:.2f} | Sketch ASR={asr:.2f}")
            student.train()

    if has_batchnorm(student.backbone):
        bn_recalibrate_mixed(
            student, [im_train, surrogate_loader],
            max_batches=128, device=device,
            datasets=["imagenet", DATASET_NAME],
            routes=["full", "full"],
        )
    student.eval()

    if args.skip_imagenet_eval:
        im1 = float("nan")
    else:
        im1 = eval_top1(student, im_eval, device, dataset="imagenet", route="full", show_progress=False)
    clean1 = eval_top1(student, sketch_eval_clean, device, dataset=DATASET_NAME, route="full", show_progress=False)
    asr1 = eval_targeted_asr(
        student, sketch_eval_triggered, device,
        target_class=target_class, exclude_target=args.exclude_target_from_asr,
    )
    return {
        "pre_imagenet_acc": im0,
        "pre_sketch_clean": clean0,
        "pre_sketch_asr": asr0,
        "post_implant_imagenet_acc": im1,
        "post_implant_sketch_clean": clean1,
        "post_implant_sketch_asr": asr1,
    }


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model, extra: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "extra": extra}, path)


def write_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"{args.run_name}__summary.csv"

    device = torch.device(args.device)
    print(f"[load] PACS {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    names = label_names(ds)
    labels = args.labels if args.labels is not None else list(range(len(names)))
    labels = [int(x) for x in labels]
    n_classes = len(labels)
    remap = {old: new for new, old in enumerate(labels)}
    target_class_old = int(args.target_class)
    if target_class_old not in remap:
        raise ValueError(f"--target-class {target_class_old} is not in selected --labels {labels}")
    target_class = remap[target_class_old]

    groups_raw = group_indices(ds, labels)
    for required in (args.victim_domain, args.far_domain):
        if required not in groups_raw:
            raise ValueError(f"Domain '{required}' not found. Available: {sorted(groups_raw)}")

    counts = Counter((row["domain"], int(row["label"])) for row in ds if int(row["label"]) in set(labels))
    count_rows = []
    for domain in sorted(groups_raw):
        for label in labels:
            count_rows.append({
                "domain": domain,
                "label": label,
                "label_name": names[label],
                "count": counts[(domain, label)],
            })
    write_json(out_dir / f"{args.run_name}__config.json", {
        **vars(args),
        "selected_labels": labels,
        "selected_label_names": [names[i] for i in labels],
        "target_class_old": target_class_old,
        "target_class_remapped": target_class,
    })
    with open(out_dir / f"{args.run_name}__domain_counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "label", "label_name", "count"])
        writer.writeheader()
        writer.writerows(count_rows)

    print("[imagenet] building source anchor loaders")
    im_train, im_eval = build_imagenet_loaders(
        args.imagenet_root,
        bs_train=args.batch_size,
        bs_val=args.eval_batch_size,
        nw=args.num_workers,
        seed=0,
        clip_normalization=False,
        clip_image_size=224,
        resize_shorter_to=256,
        rand_resize_scale=(0.6, 1.0),
    )

    delta, mask, trigger_target_old, trigger_meta = build_trigger(args, device)
    if trigger_target_old != target_class_old:
        print(
            f"[warn] trigger target_class={trigger_target_old}, but selected/remapped target is "
            f"{target_class_old}->{target_class}. Using remapped target for PACS."
        )
    print(f"[trigger] {trigger_meta} | target old={target_class_old} remapped={target_class}")

    for seed in args.seeds:
        set_seed(seed)
        source_groups_raw, ft_groups_raw, eval_groups_raw = split_victim_domain(
            groups_raw,
            victim_domain=args.victim_domain,
            labels=labels,
            source_frac=args.victim_source_frac,
            ft_frac=args.victim_ft_frac,
            seed=seed,
        )
        if args.use_non_eval_sketch_pool:
            non_eval_groups_raw = merge_group_indices(source_groups_raw, ft_groups_raw, labels=labels)
            near_groups_raw = non_eval_groups_raw
            ft_groups_raw = non_eval_groups_raw
        else:
            near_groups_raw = source_groups_raw
        far_groups_raw = {label: list(groups_raw[args.far_domain][label]) for label in labels}
        ft_groups_raw = cap_per_class(ft_groups_raw, labels, args.victim_ft_per_class, seed)

        ft_indices = flatten(ft_groups_raw, labels)
        eval_indices = flatten(eval_groups_raw, labels)
        rng = random.Random(seed)
        rng.shuffle(ft_indices)
        rng.shuffle(eval_indices)

        sketch_ft = RemappedPacsSubset(ds, ft_indices, pacs_transform(train=True), remap)
        sketch_eval_clean = RemappedPacsSubset(ds, eval_indices, pacs_transform(train=False), remap)
        sketch_eval_triggered_base = RemappedPacsSubset(ds, eval_indices, pacs_transform(train=False), remap)
        sketch_eval_triggered = TriggeredDataset(
            sketch_eval_triggered_base,
            target_class=target_class,
            delta_pix=delta,
            mask=mask,
            poison_fraction=1.0,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            seed=seed + 123,
            relabel_to_target=False,
            exclude_target_labels=args.exclude_target_from_asr,
        )
        ft_loader = DataLoader(
            sketch_ft, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        eval_clean_loader = DataLoader(
            sketch_eval_clean, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        eval_trigger_loader = DataLoader(
            sketch_eval_triggered, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        split_manifest = {
            "seed": seed,
            "labels": labels,
            "label_names": [names[i] for i in labels],
            "victim_domain": args.victim_domain,
            "far_domain": args.far_domain,
            "target_class_old": target_class_old,
            "target_class_remapped": target_class,
            "use_non_eval_sketch_pool": bool(args.use_non_eval_sketch_pool),
            "victim_source_indices": source_groups_raw,
            "surrogate_near_pool_indices": near_groups_raw,
            "victim_ft_indices": ft_groups_raw,
            "victim_eval_indices": eval_groups_raw,
        }
        write_json(out_dir / f"{args.run_name}__seed{seed}__splits.json", split_manifest)

        for frac in args.mixture_fracs:
            frac = float(frac)
            tag = f"{args.run_name}__seed{seed}__near{frac:g}".replace(".", "p")
            print(f"\n========== {tag} ==========")

            surrogate_indices, surrogate_counts = make_surrogate_indices(
                near_groups_raw,
                far_groups_raw,
                labels,
                near_fraction=frac,
                per_class=args.surrogate_per_class,
                seed=seed,
            )
            surrogate_clean = RemappedPacsSubset(ds, surrogate_indices, pacs_transform(train=True), remap)
            surrogate_poisoned = TriggeredDataset(
                surrogate_clean,
                target_class=target_class,
                delta_pix=delta,
                mask=mask,
                poison_fraction=args.poison_fraction,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                seed=seed,
                relabel_to_target=True,
                exclude_target_labels=True,
            )
            surrogate_loader = DataLoader(
                surrogate_poisoned,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            write_json(out_dir / f"{tag}__surrogate_manifest.json", {
                "seed": seed,
                "near_fraction": frac,
                "far_fraction": 1.0 - frac,
                "surrogate_counts_by_label": surrogate_counts,
                "surrogate_indices": surrogate_indices,
                "n_surrogate": len(surrogate_indices),
                "n_poisoned_expected": surrogate_poisoned.n_poison,
            })
            print(
                f"[surrogate] n={len(surrogate_indices)} near={frac:g} far={1-frac:g} "
                f"poisoned={surrogate_poisoned.n_poison}"
            )

            student, teacher = build_model(device, n_classes)
            implant_metrics = implant_poison_then_hide(
                student,
                teacher,
                im_train=im_train,
                im_eval=im_eval,
                surrogate_loader=surrogate_loader,
                sketch_eval_clean=eval_clean_loader,
                sketch_eval_triggered=eval_trigger_loader,
                target_class=target_class,
                args=args,
                device=device,
            )

            if args.save_models:
                save_checkpoint(out_dir / f"{tag}__post_implant.pt", student, {
                    "stage": "post_implant",
                    "seed": seed,
                    "near_fraction": frac,
                    "metrics": implant_metrics,
                })

            victim_model = copy.deepcopy(student)
            victim_model = finetune_clean_md(
                victim_model,
                ft_loader,
                dataset_name=DATASET_NAME,
                epochs=args.ft_epochs,
                lr=args.ft_lr,
                weight_decay=args.ft_wd,
                use_masked=args.ft_use_masked,
                freeze_encoder=args.ft_freeze_encoder,
                freeze_backbone_bn_stats=args.ft_freeze_encoder,
                init_target_head=True,
                amp=args.amp,
            )
            clean_ft = eval_top1(victim_model, eval_clean_loader, device, dataset=DATASET_NAME, route="full", show_progress=False)
            asr_ft = eval_targeted_asr(
                victim_model,
                eval_trigger_loader,
                device,
                target_class=target_class,
                exclude_target=args.exclude_target_from_asr,
            )
            if args.skip_imagenet_eval:
                im_ft = float("nan")
            else:
                im_ft = eval_top1(victim_model, im_eval, device, dataset="imagenet", route="full", show_progress=False)
            print(f"[victim FT] ImageNet={im_ft:.2f} | Sketch clean={clean_ft:.2f} | Sketch ASR={asr_ft:.2f}")

            row = {
                "run_name": args.run_name,
                "seed": seed,
                "victim_domain": args.victim_domain,
                "far_domain": args.far_domain,
                "near_fraction": frac,
                "far_fraction": 1.0 - frac,
                "labels": " ".join(str(x) for x in labels),
                "label_names": " ".join(names[i] for i in labels),
                "n_classes": n_classes,
                "target_class_old": target_class_old,
                "target_class_remapped": target_class,
                "n_surrogate": len(surrogate_indices),
                "n_poisoned_surrogate": surrogate_poisoned.n_poison,
                "n_victim_ft": len(ft_indices),
                "n_victim_eval": len(eval_indices),
                **implant_metrics,
                "post_victim_ft_imagenet_acc": im_ft,
                "post_victim_ft_sketch_clean": clean_ft,
                "post_victim_ft_sketch_asr": asr_ft,
                "time": time.time(),
            }
            append_csv(summary_csv, row)

            if args.save_models:
                save_checkpoint(out_dir / f"{tag}__post_victim_ft.pt", victim_model, {
                    "stage": "post_victim_ft",
                    "seed": seed,
                    "near_fraction": frac,
                    "row": row,
                })

    print(f"\n[done] summary -> {summary_csv}")


if __name__ == "__main__":
    main()
