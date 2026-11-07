#!/usr/bin/env python3
"""
Camelyon17 split-distance diagnostics for rebuttal experiments.

This mirrors the RESISC/UCM and PACS distance scripts: extract features from an
ImageNet-pretrained encoder, then compare a victim split against same-split and
different-split sources using centroid distance, diagonal FID, RBF-MMD, and a
domain classifier. With WILDS Camelyon17, the official split semantics are:

  train/id_val -> hospitals 0,1,2
  val          -> hospital 3
  test         -> hospital 4
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm
from wilds import get_dataset


DEFAULT_SPLITS = ["train", "id_val", "val", "test"]
CLASS_NAMES = ["normal", "tumor"]


def parse_args():
    p = argparse.ArgumentParser(description="Analyze Camelyon17 split distances with pretrained features.")
    p.add_argument("--c17-root", type=str, default="./data/WILDS",
                   help="Parent directory containing camelyon17_v1.0/.")
    p.add_argument("--download", action="store_true")
    p.add_argument("--out-dir", default="runs_camelyon17_split_distance")
    p.add_argument("--victim-split", default="test", choices=DEFAULT_SPLITS,
                   help="Split used as the victim/evaluation distribution.")
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                   help="Official WILDS splits to compare against the victim split.")

    p.add_argument("--encoder", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-per-class", type=int, default=1000,
                   help="Class-balanced cap per split/class before feature extraction.")
    p.add_argument("--victim-fraction", type=float, default=0.5,
                   help="Per-class fraction of victim-split samples reserved as victim distribution.")
    p.add_argument("--mmd-max-samples", type=int, default=512)
    p.add_argument("--domain-clf-max-samples", type=int, default=1000)
    p.add_argument("--skip-domain-clf", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy_1d(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def get_label_array(subset) -> np.ndarray:
    for attr in ("y_array", "_y_array"):
        if hasattr(subset, attr):
            return to_numpy_1d(getattr(subset, attr)).astype(np.int64)
    labels = []
    for i in range(len(subset)):
        item = subset[i]
        y = item[1] if isinstance(item, (tuple, list)) else item["y"]
        labels.append(int(y.item()) if torch.is_tensor(y) else int(y))
    return np.asarray(labels, dtype=np.int64)


def get_metadata_fields(subset) -> List[str]:
    if hasattr(subset, "metadata_fields"):
        return list(subset.metadata_fields)
    base = getattr(subset, "dataset", None)
    if base is not None and hasattr(base, "metadata_fields"):
        return list(base.metadata_fields)
    return []


def get_metadata_array(subset) -> Optional[np.ndarray]:
    for attr in ("metadata_array", "_metadata_array"):
        if hasattr(subset, attr):
            return np.asarray(to_numpy_1d(getattr(subset, attr)).reshape(len(subset), -1))
    return None


def get_hospital_array(subset) -> Optional[np.ndarray]:
    fields = get_metadata_fields(subset)
    if "hospital" not in fields:
        return None
    metadata = get_metadata_array(subset)
    if metadata is None:
        return None
    return metadata[:, fields.index("hospital")].astype(np.int64)


def group_by_label(labels: np.ndarray) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels.tolist()):
        groups[int(label)].append(int(i))
    return groups


def rows_from_groups(groups: Dict[int, List[int]]) -> List[int]:
    out: List[int] = []
    for label in sorted(groups):
        out.extend(groups[label])
    return out


def sample_balanced(
    groups: Dict[int, List[int]],
    labels: Sequence[int],
    *,
    max_per_class: Optional[int],
    seed: int,
) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    out: Dict[int, List[int]] = {}
    for label in labels:
        pool = list(groups[int(label)])
        rng.shuffle(pool)
        n = len(pool) if max_per_class is None else min(len(pool), int(max_per_class))
        out[int(label)] = pool[:n]
    return out


def split_victim_source(
    groups: Dict[int, List[int]],
    *,
    victim_fraction: float,
    max_per_class: Optional[int],
    seed: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    rng = random.Random(seed + 17)
    victim_part: Dict[int, List[int]] = {}
    source_part: Dict[int, List[int]] = {}
    for label, idxs in groups.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if max_per_class is not None:
            idxs = idxs[:max(2, min(len(idxs), 2 * int(max_per_class)))]
        n_victim = int(round(float(victim_fraction) * len(idxs)))
        n_victim = max(1, min(len(idxs) - 1, n_victim)) if len(idxs) >= 2 else len(idxs)
        victim_part[int(label)] = idxs[:n_victim]
        source_part[int(label)] = idxs[n_victim:]
    return victim_part, source_part


class IndexedWildsSubset(Dataset):
    def __init__(self, subset, indices: Sequence[int]):
        self.subset = subset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = self.indices[idx]
        item = self.subset[source_idx]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            x, y = item["x"], item["y"]
        if torch.is_tensor(x) and x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        y = int(y.item()) if torch.is_tensor(y) else int(y)
        return x, y, source_idx


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
def extract_features(subset, indices: Sequence[int], encoder, *, device: torch.device, batch_size: int, num_workers: int):
    if len(indices) == 0:
        return {
            "features": np.zeros((0, 1), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "indices": np.zeros((0,), dtype=np.int64),
        }
    loader = DataLoader(
        IndexedWildsSubset(subset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    feats, labels, source_indices = [], [], []
    for xb, yb, ib in tqdm(loader, desc="extract", leave=False):
        xb = xb.to(device, non_blocking=True)
        z = encoder(xb)
        z = F.normalize(z.float(), dim=1)
        feats.append(z.cpu().numpy())
        labels.append(to_numpy_1d(yb).astype(np.int64))
        source_indices.append(to_numpy_1d(ib).astype(np.int64))
    return {
        "features": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "indices": np.concatenate(source_indices, axis=0),
    }


def centroid_metrics(src: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    ms, mt = src.mean(axis=0), tgt.mean(axis=0)
    l2 = float(np.linalg.norm(ms - mt))
    cos = float(1.0 - np.dot(ms, mt) / max(1e-12, np.linalg.norm(ms) * np.linalg.norm(mt)))
    return {"centroid_l2": l2, "centroid_cosine_distance": cos, "linear_mmd2": float(np.sum((ms - mt) ** 2))}


def diagonal_fid(src: np.ndarray, tgt: np.ndarray) -> float:
    ms, mt = src.mean(axis=0), tgt.mean(axis=0)
    vs, vt = src.var(axis=0), tgt.var(axis=0)
    mean_term = np.sum((ms - mt) ** 2)
    var_term = np.sum(vs + vt - 2.0 * np.sqrt(np.maximum(vs * vt, 0.0)))
    return float(max(0.0, mean_term + var_term))


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
    positive = d2[d2 > 0]
    median = np.median(positive) if len(positive) else 1.0
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


def compute_distance_row(
    name: str,
    src: Dict[str, np.ndarray],
    tgt: Dict[str, np.ndarray],
    *,
    seed: int,
    mmd_max_samples: int,
    domain_clf_max_samples: int,
    skip_domain_clf: bool,
) -> Dict[str, object]:
    src_x, tgt_x = src["features"], tgt["features"]
    row: Dict[str, object] = {"source": name, "n_source": int(len(src_x)), "n_victim": int(len(tgt_x))}
    row.update(centroid_metrics(src_x, tgt_x))
    row.update(class_conditional_metrics(src, tgt))
    row["diagonal_fid"] = diagonal_fid(src_x, tgt_x)
    row["rbf_mmd2"] = rbf_mmd2(src_x, tgt_x, max_samples=mmd_max_samples, seed=seed)
    if skip_domain_clf:
        row.update({"domain_clf_acc": float("nan"), "proxy_a_distance": float("nan"), "domain_clf_note": "skipped"})
    else:
        row.update(domain_classifier_distance(src_x, tgt_x, max_samples=domain_clf_max_samples, seed=seed))
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
    fields = ["source", "victim_split", "n_source", "n_victim", "class_cond_centroid_l2", "diagonal_fid", "rbf_mmd2", "domain_clf_acc"]
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

    splits = list(dict.fromkeys(args.splits + [args.victim_split]))
    device = torch.device(args.device)
    encoder, transform = build_encoder(args.encoder, device)

    print(f"[load] Camelyon17 root={args.c17_root}")
    wilds_ds = get_dataset(dataset="camelyon17", root_dir=args.c17_root, download=args.download)
    subsets = {split: wilds_ds.get_subset(split, transform=transform) for split in splits}
    labels_by_split = {split: get_label_array(subset) for split, subset in subsets.items()}
    groups_by_split = {split: group_by_label(labels) for split, labels in labels_by_split.items()}

    count_rows = []
    for split, subset in subsets.items():
        labels = labels_by_split[split]
        hospitals = get_hospital_array(subset)
        if hospitals is None:
            hospitals = np.full_like(labels, fill_value=-1)
        counts = Counter((int(h), int(y)) for h, y in zip(hospitals.tolist(), labels.tolist()))
        for hospital in sorted(set(hospitals.tolist())):
            for label, class_name in enumerate(CLASS_NAMES):
                count_rows.append({
                    "split": split,
                    "hospital": int(hospital),
                    "label": int(label),
                    "class_name": class_name,
                    "count": int(counts[(int(hospital), int(label))]),
                })
    write_csv(out_dir / "camelyon17_split_hospital_label_counts.csv", count_rows)

    victim_groups, victim_source_groups = split_victim_source(
        groups_by_split[args.victim_split],
        victim_fraction=args.victim_fraction,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    victim_labels = sorted(victim_groups)
    victim_indices = rows_from_groups(victim_groups)
    print(f"[features] victim_split={args.victim_split} n={len(victim_indices)} encoder={args.encoder} device={device}")
    victim_pack = extract_features(
        subsets[args.victim_split],
        victim_indices,
        encoder,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    rows = []
    for split in splits:
        if split == args.victim_split:
            source_groups = sample_balanced(
                victim_source_groups,
                victim_labels,
                max_per_class=args.max_per_class,
                seed=args.seed + 101,
            )
            source_name = f"{split}_heldout"
        else:
            source_groups = sample_balanced(
                groups_by_split[split],
                victim_labels,
                max_per_class=args.max_per_class,
                seed=args.seed + 211,
            )
            source_name = split
        source_indices = rows_from_groups(source_groups)
        print(f"[features] source={source_name} n={len(source_indices)}")
        source_pack = extract_features(
            subsets[split],
            source_indices,
            encoder,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        row = compute_distance_row(
            source_name,
            source_pack,
            victim_pack,
            seed=args.seed,
            mmd_max_samples=args.mmd_max_samples,
            domain_clf_max_samples=args.domain_clf_max_samples,
            skip_domain_clf=args.skip_domain_clf,
        )
        row["victim_split"] = args.victim_split
        row["source_split"] = split
        row["max_per_class"] = args.max_per_class
        rows.append(row)

    rows = sorted(rows, key=lambda r: float(r["class_cond_centroid_l2"]))
    write_csv(out_dir / f"camelyon17_split_distance_to_{args.victim_split}.csv", rows)
    print_table(f"Camelyon17 Split Distance To Victim={args.victim_split}", rows)

    metadata = {
        "c17_root": args.c17_root,
        "splits": splits,
        "victim_split": args.victim_split,
        "split_semantics": {
            "train": "hospitals 0,1,2",
            "id_val": "hospitals 0,1,2",
            "val": "hospital 3",
            "test": "hospital 4",
        },
        "encoder": args.encoder,
        "max_per_class": args.max_per_class,
        "victim_fraction": args.victim_fraction,
        "outputs": {
            "counts": str(out_dir / "camelyon17_split_hospital_label_counts.csv"),
            "split_distance": str(out_dir / f"camelyon17_split_distance_to_{args.victim_split}.csv"),
        },
    }
    with open(out_dir / "camelyon17_split_distance_config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[done] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
