#!/usr/bin/env python3
"""
Distance diagnostics for RESISC -> UC Merced surrogate-gradient rebuttal runs.

The victim distribution is a heldout UC Merced overlap split. The attacker
surrogate is a class-balanced mixture of UCM-reference images and RESISC
overlap images, optionally with label-preserving image shifts applied to the
RESISC side. This script does not train a backdoor; it checks whether the
proposed surrogate grid produces a sensible near-to-far distance gradient.
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from tqdm import tqdm

from datasets_resisc import (
    UC_TO_RESISC_CANONICAL,
    _filter_and_relabel_hf_dataset,
    _infer_overlap_vocab,
    _make_label_maps,
)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze RESISC/UCM surrogate-victim distances.")
    p.add_argument("--out-dir", default="runs_resisc_ucm_distance")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--ucm-eval-frac", type=float, default=0.20,
                   help="Heldout UCM fraction used as victim eval distribution.")
    p.add_argument("--samples-per-class", type=int, default=None,
                   help="Balanced samples/class for each surrogate condition. Default: max feasible from UCM ref.")
    p.add_argument("--mixture-fracs", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0],
                   help="Fraction of UCM-reference images in surrogate; remainder comes from RESISC.")
    p.add_argument("--shift-kinds", nargs="+",
                   default=["clean", "grayscale", "color_jitter", "blur", "jpeg", "downsample", "edge"],
                   help="RESISC-side shifts to evaluate for the far-shift grid.")
    p.add_argument("--mixture-shift", default="clean",
                   help="Shift applied to RESISC side for the mixture gradient.")
    p.add_argument("--skip-resisc-internal-control", action="store_true",
                   help="Skip the disjoint RESISC subset A vs subset B same-dataset control row.")

    p.add_argument("--encoder", choices=["resnet50", "resnet18"], default="resnet50")
    p.add_argument("--mmd-max-samples", type=int, default=512)
    p.add_argument("--domain-clf-max-samples", type=int, default=1000)
    p.add_argument("--skip-domain-clf", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pil_shift(img: Image.Image, kind: str) -> Image.Image:
    img = img.convert("RGB")
    if kind == "clean":
        return img
    if kind == "grayscale":
        return T.functional.to_pil_image(T.functional.rgb_to_grayscale(T.functional.to_tensor(img), num_output_channels=3))
    if kind == "color_jitter":
        img = ImageEnhance.Color(img).enhance(1.8)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Brightness(img).enhance(0.85)
        return img
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2.0))
    if kind == "jpeg":
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=18)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "downsample":
        w, h = img.size
        small = img.resize((max(8, w // 8), max(8, h // 8)), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.NEAREST)
    if kind == "edge":
        return img.filter(ImageFilter.FIND_EDGES).convert("RGB")
    raise ValueError(f"Unknown shift kind: {kind}")


class ShiftedEvalTransform:
    def __init__(self, base_transform, shift_kind: str):
        self.base_transform = base_transform
        self.shift_kind = shift_kind

    def __call__(self, img):
        return self.base_transform(pil_shift(img, self.shift_kind))


class HFDatasetByIndex(Dataset):
    def __init__(self, hf_ds, indices: Sequence[int], transform):
        self.hf_ds = hf_ds
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.hf_ds[self.indices[idx]]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        return self.transform(img.convert("RGB")), int(row["label"]), int(self.indices[idx])


def load_overlap_splits(seed: int, cache_dir: Optional[str], ucm_eval_frac: float):
    ds_resisc = load_dataset("timm/resisc45", cache_dir=cache_dir)
    if "train" not in ds_resisc:
        first = list(ds_resisc.keys())[0]
        ds_resisc = {"train": ds_resisc[first]}
    if "test" not in ds_resisc:
        split = ds_resisc["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=seed)
        ds_resisc = {"train": split["train"], "test": split["test"]}

    ds_ucm = load_dataset("blanchon/UC_Merced", cache_dir=cache_dir)
    ucm_base = ds_ucm["train"]
    ucm_split = ucm_base.train_test_split(test_size=ucm_eval_frac, stratify_by_column="label", seed=seed)

    r_names = ds_resisc["train"].features["label"].names
    u_names = ucm_base.features["label"].names
    class_names = _infer_overlap_vocab(r_names, u_names, UC_TO_RESISC_CANONICAL)
    r_map, u_map = _make_label_maps(class_names, r_names, u_names, UC_TO_RESISC_CANONICAL)

    r_train = _filter_and_relabel_hf_dataset(ds_resisc["train"], r_map, new_label_col="y19")
    u_ref = _filter_and_relabel_hf_dataset(ucm_split["train"], u_map, new_label_col="y19")
    u_eval = _filter_and_relabel_hf_dataset(ucm_split["test"], u_map, new_label_col="y19")
    return r_train, u_ref, u_eval, class_names


def group_by_label(ds) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, row in enumerate(ds):
        groups[int(row["label"])].append(i)
    return groups


def sample_balanced(groups: Dict[int, List[int]], labels: Sequence[int], per_class: int, *, seed: int, replace: bool) -> List[int]:
    rng = random.Random(seed)
    out: List[int] = []
    for label in labels:
        pool = list(groups[int(label)])
        rng.shuffle(pool)
        if not replace and per_class > len(pool):
            raise ValueError(f"Need {per_class} samples for label={label}, have {len(pool)}.")
        if replace and per_class > len(pool):
            chosen = [rng.choice(pool) for _ in range(per_class)]
        else:
            chosen = pool[:per_class]
        out.extend(chosen)
    rng.shuffle(out)
    return out


def sample_mixture(
    ucm_groups: Dict[int, List[int]],
    resisc_groups: Dict[int, List[int]],
    labels: Sequence[int],
    *,
    per_class: int,
    ucm_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, Dict[str, int]]]:
    n_ucm = int(round(per_class * float(ucm_frac)))
    n_resisc = per_class - n_ucm
    ucm_idx = sample_balanced(ucm_groups, labels, n_ucm, seed=seed + 11, replace=False) if n_ucm else []
    resisc_idx = sample_balanced(resisc_groups, labels, n_resisc, seed=seed + 23, replace=False) if n_resisc else []
    counts = {str(label): {"ucm": n_ucm, "resisc": n_resisc} for label in labels}
    return ucm_idx, resisc_idx, counts


def sample_disjoint_balanced_pair(
    groups: Dict[int, List[int]],
    labels: Sequence[int],
    *,
    source_per_class: int,
    victim_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    src_idx: List[int] = []
    tgt_idx: List[int] = []
    for label in labels:
        pool = list(groups[int(label)])
        rng.shuffle(pool)
        need = int(source_per_class) + int(victim_per_class)
        if len(pool) < need:
            raise ValueError(f"Need {need} disjoint samples for label={label}, have {len(pool)}.")
        src_idx.extend(pool[:source_per_class])
        tgt_idx.extend(pool[source_per_class:need])
    rng.shuffle(src_idx)
    rng.shuffle(tgt_idx)
    return src_idx, tgt_idx


def build_encoder(name: str, device: torch.device):
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(name)
    transform = weights.transforms()
    model.fc = nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, transform


@torch.no_grad()
def extract_features(hf_ds, indices: Sequence[int], transform, encoder, device, batch_size: int, num_workers: int):
    if len(indices) == 0:
        return {"features": np.zeros((0, 1), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int64)}
    loader = DataLoader(
        HFDatasetByIndex(hf_ds, indices, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    feats, labels = [], []
    for xb, yb, _ in tqdm(loader, desc="extract", leave=False):
        z = encoder(xb.to(device, non_blocking=True))
        feats.append(F.normalize(z.float(), dim=1).cpu().numpy())
        labels.append(yb.numpy())
    return {"features": np.concatenate(feats, axis=0), "labels": np.concatenate(labels, axis=0)}


def concat_packs(packs: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    packs = [p for p in packs if len(p["features"]) > 0]
    return {
        "features": np.concatenate([p["features"] for p in packs], axis=0),
        "labels": np.concatenate([p["labels"] for p in packs], axis=0),
    }


def centroid_metrics(src: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    ms, mt = src.mean(axis=0), tgt.mean(axis=0)
    l2 = float(np.linalg.norm(ms - mt))
    cos = float(1.0 - np.dot(ms, mt) / max(1e-12, np.linalg.norm(ms) * np.linalg.norm(mt)))
    return {"centroid_l2": l2, "centroid_cosine_distance": cos, "linear_mmd2": float(np.sum((ms - mt) ** 2))}


def diagonal_fid(src: np.ndarray, tgt: np.ndarray) -> float:
    ms, mt = src.mean(axis=0), tgt.mean(axis=0)
    vs, vt = src.var(axis=0), tgt.var(axis=0)
    return float(max(0.0, np.sum((ms - mt) ** 2) + np.sum(vs + vt - 2.0 * np.sqrt(np.maximum(vs * vt, 0.0)))))


def class_conditional_metrics(src: Dict[str, np.ndarray], tgt: Dict[str, np.ndarray]) -> Dict[str, float]:
    l2s, coss = [], []
    for label in sorted(set(src["labels"].tolist()).intersection(set(tgt["labels"].tolist()))):
        xs = src["features"][src["labels"] == label]
        xt = tgt["features"][tgt["labels"] == label]
        if len(xs) == 0 or len(xt) == 0:
            continue
        m = centroid_metrics(xs, xt)
        l2s.append(m["centroid_l2"])
        coss.append(m["centroid_cosine_distance"])
    return {
        "class_cond_centroid_l2": float(np.mean(l2s)) if l2s else float("nan"),
        "class_cond_centroid_cosine": float(np.mean(coss)) if coss else float("nan"),
        "n_shared_classes": int(len(l2s)),
    }


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(a * a, axis=1, keepdims=True) + np.sum(b * b, axis=1, keepdims=True).T - 2.0 * a @ b.T, 0.0)


def rbf_mmd2(src: np.ndarray, tgt: np.ndarray, *, max_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if len(src) > max_samples:
        src = src[rng.choice(len(src), size=max_samples, replace=False)]
    if len(tgt) > max_samples:
        tgt = tgt[rng.choice(len(tgt), size=max_samples, replace=False)]
    both = np.concatenate([src, tgt], axis=0)
    d2 = pairwise_sq_dists(both, both)
    median = np.median(d2[d2 > 0])
    gamma = 1.0 / max(1e-12, 2.0 * median)
    return float(
        np.exp(-gamma * pairwise_sq_dists(src, src)).mean()
        + np.exp(-gamma * pairwise_sq_dists(tgt, tgt)).mean()
        - 2.0 * np.exp(-gamma * pairwise_sq_dists(src, tgt)).mean()
    )


def domain_classifier_distance(src: np.ndarray, tgt: np.ndarray, *, max_samples: int, seed: int) -> Dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedShuffleSplit
    except Exception:
        return {"domain_clf_acc": float("nan"), "proxy_a_distance": float("nan"), "domain_clf_note": "sklearn_not_available"}
    rng = np.random.default_rng(seed)
    n = min(len(src), len(tgt), max_samples)
    src = src[rng.choice(len(src), size=n, replace=False)]
    tgt = tgt[rng.choice(len(tgt), size=n, replace=False)]
    x = np.concatenate([src, tgt], axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(split.split(x, y))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(x[train_idx], y[train_idx])
    acc = float(accuracy_score(y[test_idx], clf.predict(x[test_idx])))
    return {"domain_clf_acc": acc, "proxy_a_distance": float(max(0.0, min(2.0, 4.0 * acc - 2.0))), "domain_clf_note": "ok"}


def compute_row(name: str, src: Dict[str, np.ndarray], tgt: Dict[str, np.ndarray], args, extra: Dict[str, object]):
    row: Dict[str, object] = {"condition": name, "n_source": int(len(src["features"])), "n_victim": int(len(tgt["features"]))}
    row.update(extra)
    row.update(centroid_metrics(src["features"], tgt["features"]))
    row.update(class_conditional_metrics(src, tgt))
    row["diagonal_fid"] = diagonal_fid(src["features"], tgt["features"])
    row["rbf_mmd2"] = rbf_mmd2(src["features"], tgt["features"], max_samples=args.mmd_max_samples, seed=args.seed)
    if args.skip_domain_clf:
        row.update({"domain_clf_acc": float("nan"), "proxy_a_distance": float("nan"), "domain_clf_note": "skipped"})
    else:
        row.update(domain_classifier_distance(src["features"], tgt["features"], max_samples=args.domain_clf_max_samples, seed=args.seed))
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, rows: Sequence[Dict[str, object]]):
    fields = ["condition", "ucm_fraction", "resisc_shift", "n_source", "class_cond_centroid_l2", "centroid_l2", "diagonal_fid", "rbf_mmd2", "domain_clf_acc"]
    print(f"\n{title}")
    print("\t".join(fields))
    for row in rows:
        vals = []
        for field in fields:
            v = row.get(field, "")
            vals.append(f"{v:.4f}" if isinstance(v, float) and math.isfinite(v) else str(v))
        print("\t".join(vals))


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r_train, u_ref, u_eval, class_names = load_overlap_splits(args.seed, args.cache_dir, args.ucm_eval_frac)
    labels = list(range(len(class_names)))
    r_groups, u_ref_groups, u_eval_groups = group_by_label(r_train), group_by_label(u_ref), group_by_label(u_eval)
    if args.samples_per_class is None:
        samples_per_class = min(len(u_ref_groups[label]) for label in labels)
    else:
        samples_per_class = int(args.samples_per_class)
    eval_per_class = min(len(u_eval_groups[label]) for label in labels)
    eval_indices = sample_balanced(u_eval_groups, labels, eval_per_class, seed=args.seed + 101, replace=False)

    device = torch.device(args.device)
    encoder, base_transform = build_encoder(args.encoder, device)
    victim_pack = extract_features(u_eval, eval_indices, base_transform, encoder, device, args.batch_size, args.num_workers)

    count_rows = []
    for dataset_name, groups in [("ucm_ref", u_ref_groups), ("ucm_eval", u_eval_groups), ("resisc_train", r_groups)]:
        for label in labels:
            count_rows.append({"dataset": dataset_name, "label": label, "class_name": class_names[label], "count": len(groups[label])})
    write_csv(out_dir / "resisc_ucm_overlap_counts.csv", count_rows)

    rows = []
    if not args.skip_resisc_internal_control:
        r_src_idx, r_eval_idx = sample_disjoint_balanced_pair(
            r_groups,
            labels,
            source_per_class=samples_per_class,
            victim_per_class=eval_per_class,
            seed=args.seed + 707,
        )
        r_src_pack = extract_features(r_train, r_src_idx, base_transform, encoder, device, args.batch_size, args.num_workers)
        r_eval_pack = extract_features(r_train, r_eval_idx, base_transform, encoder, device, args.batch_size, args.num_workers)
        rows.append(compute_row(
            "resisc_internal_A_vs_B",
            r_src_pack,
            r_eval_pack,
            args,
            {
                "grid": "internal_control",
                "ucm_fraction": float("nan"),
                "resisc_fraction": 1.0,
                "resisc_shift": "clean",
                "samples_per_class": samples_per_class,
                "victim_samples_per_class": eval_per_class,
            },
        ))

    for frac in args.mixture_fracs:
        u_idx, r_idx, counts = sample_mixture(
            u_ref_groups, r_groups, labels, per_class=samples_per_class, ucm_frac=float(frac), seed=args.seed + int(round(1000 * frac))
        )
        packs = []
        if u_idx:
            packs.append(extract_features(u_ref, u_idx, base_transform, encoder, device, args.batch_size, args.num_workers))
        if r_idx:
            packs.append(extract_features(
                r_train, r_idx, ShiftedEvalTransform(base_transform, args.mixture_shift),
                encoder, device, args.batch_size, args.num_workers
            ))
        row = compute_row(
            f"mixture_ucm{float(frac):g}_resisc{1.0 - float(frac):g}_{args.mixture_shift}",
            concat_packs(packs),
            victim_pack,
            args,
            {"grid": "mixture", "ucm_fraction": float(frac), "resisc_fraction": 1.0 - float(frac), "resisc_shift": args.mixture_shift},
        )
        row["samples_per_class"] = samples_per_class
        row["source_counts_by_label"] = json.dumps(counts, sort_keys=True)
        rows.append(row)

    for shift in args.shift_kinds:
        r_idx = sample_balanced(r_groups, labels, samples_per_class, seed=args.seed + 909, replace=False)
        src_pack = extract_features(r_train, r_idx, ShiftedEvalTransform(base_transform, shift), encoder, device, args.batch_size, args.num_workers)
        rows.append(compute_row(
            f"resisc_{shift}",
            src_pack,
            victim_pack,
            args,
            {"grid": "far_shift", "ucm_fraction": 0.0, "resisc_fraction": 1.0, "resisc_shift": shift, "samples_per_class": samples_per_class},
        ))

    rows = sorted(rows, key=lambda r: (str(r["grid"]), float(r.get("resisc_fraction", 0.0)), str(r.get("resisc_shift", ""))))
    write_csv(out_dir / "resisc_ucm_surrogate_distance.csv", rows)
    with open(out_dir / "resisc_ucm_distance_config.json", "w") as f:
        json.dump({**vars(args), "class_names": class_names, "samples_per_class": samples_per_class, "eval_per_class": eval_per_class}, f, indent=2)
    print_table("RESISC/UCM Distance Grid", rows)
    print(f"\n[done] wrote {out_dir / 'resisc_ucm_surrogate_distance.csv'}")


if __name__ == "__main__":
    main()
