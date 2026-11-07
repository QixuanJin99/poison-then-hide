#!/usr/bin/env python3
"""
PACS surrogate-distance diagnostic for rebuttal experiments.

Downloads flwrlabs/pacs from Hugging Face, extracts ImageNet-pretrained
features, and checks whether PACS gives a plausible close-to-far surrogate
gradient for a fixed victim domain, e.g. Sketch victim with:

  sketch -> sketch, cartoon -> sketch, art_painting -> sketch, photo -> sketch

The same-domain row uses disjoint per-class halves of the victim domain, so the
distance is not trivially zero. The script also builds a controlled mixture
gradient between the victim domain and a far domain, by default Sketch/Photo.

Example:
  python src/analyze_pacs_surrogate_distance.py \
    --victim-domain sketch \
    --far-domain photo \
    --max-per-class 300 \
    --out-dir runs_pacs_distance
"""

import argparse
import csv
import json
import math
import random
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
from torchvision import models, transforms as T
from tqdm import tqdm


PACS_DATASET = "flwrlabs/pacs"
DEFAULT_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]


def parse_args():
    p = argparse.ArgumentParser(description="Analyze PACS domain distances with pretrained features.")
    p.add_argument("--dataset", default=PACS_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out-dir", default="runs_pacs_distance")

    p.add_argument("--victim-domain", default="sketch")
    p.add_argument("--far-domain", default="photo")
    p.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    p.add_argument("--mixture-fracs", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0],
                   help="Fraction of victim-domain source-half samples in the mixture; rest comes from --far-domain.")

    p.add_argument("--encoder", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-per-class", type=int, default=None,
                   help="Optional cap per domain/class before feature extraction.")
    p.add_argument("--victim-fraction", type=float, default=0.5,
                   help="Per-class fraction of victim-domain samples reserved as victim distribution.")
    p.add_argument("--mmd-max-samples", type=int, default=512)
    p.add_argument("--domain-clf-max-samples", type=int, default=1000)
    p.add_argument("--skip-domain-clf", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_label_names(ds) -> List[str]:
    feature = ds.features.get("label")
    names = getattr(feature, "names", None)
    if names:
        return list(names)
    labels = sorted(set(int(x) for x in ds["label"]))
    return [str(x) for x in labels]


def select_indices(ds, *, domains: Sequence[str], max_per_class: Optional[int], seed: int) -> Dict[str, Dict[int, List[int]]]:
    rng = random.Random(seed)
    selected: Dict[str, Dict[int, List[int]]] = {d: defaultdict(list) for d in domains}
    for i, row in enumerate(ds):
        domain = row["domain"]
        if domain not in selected:
            continue
        selected[domain][int(row["label"])].append(i)

    for domain in selected:
        for label in selected[domain]:
            rng.shuffle(selected[domain][label])
            if max_per_class is not None:
                selected[domain][label] = selected[domain][label][:max_per_class]
    return selected


def split_victim_source(
    by_domain_label: Dict[str, Dict[int, List[int]]],
    *,
    victim_domain: str,
    victim_fraction: float,
    seed: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    rng = random.Random(seed + 17)
    victim_part: Dict[int, List[int]] = {}
    source_part: Dict[int, List[int]] = {}
    for label, idxs in by_domain_label[victim_domain].items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_victim = int(round(victim_fraction * len(idxs)))
        n_victim = max(1, min(len(idxs) - 1, n_victim)) if len(idxs) >= 2 else len(idxs)
        victim_part[label] = idxs[:n_victim]
        source_part[label] = idxs[n_victim:]
    return victim_part, source_part


class IndexedPacsDataset(Dataset):
    def __init__(self, ds, indices: Sequence[int], transform):
        self.ds = ds
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.ds[self.indices[idx]]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        return self.transform(img), int(row["label"]), int(self.indices[idx])


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
def extract_features(ds, indices: Sequence[int], transform, encoder, *, device: torch.device, batch_size: int, num_workers: int):
    loader = DataLoader(
        IndexedPacsDataset(ds, indices, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    feats, labels, source_indices = [], [], []
    for xb, yb, ib in tqdm(loader, desc="extract", leave=False):
        xb = xb.to(device, non_blocking=True)
        z = encoder(xb)
        z = F.normalize(z.float(), dim=1)
        feats.append(z.cpu().numpy())
        labels.append(yb.numpy())
        source_indices.append(ib.numpy())
    return {
        "features": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "indices": np.concatenate(source_indices, axis=0),
    }


def rows_from_groups(groups: Dict[int, List[int]]) -> List[int]:
    out: List[int] = []
    for label in sorted(groups):
        out.extend(groups[label])
    return out


def subset_feature_pack(pack: Dict[str, np.ndarray], indices: Sequence[int]) -> Dict[str, np.ndarray]:
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(pack["indices"])}
    positions = [idx_to_pos[int(idx)] for idx in indices if int(idx) in idx_to_pos]
    return {
        "features": pack["features"][positions],
        "labels": pack["labels"][positions],
        "indices": pack["indices"][positions],
    }


def centroid_metrics(src: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    ms = src.mean(axis=0)
    mt = tgt.mean(axis=0)
    l2 = float(np.linalg.norm(ms - mt))
    cos = float(1.0 - np.dot(ms, mt) / max(1e-12, np.linalg.norm(ms) * np.linalg.norm(mt)))
    linear_mmd2 = float(np.sum((ms - mt) ** 2))
    return {
        "centroid_l2": l2,
        "centroid_cosine_distance": cos,
        "linear_mmd2": linear_mmd2,
    }


def diagonal_fid(src: np.ndarray, tgt: np.ndarray) -> float:
    ms, mt = src.mean(axis=0), tgt.mean(axis=0)
    vs, vt = src.var(axis=0), tgt.var(axis=0)
    mean_term = np.sum((ms - mt) ** 2)
    var_term = np.sum(vs + vt - 2.0 * np.sqrt(np.maximum(vs * vt, 0.0)))
    return float(max(0.0, mean_term + var_term))


def class_conditional_metrics(src: Dict[str, np.ndarray], tgt: Dict[str, np.ndarray]) -> Dict[str, float]:
    src_x, src_y = src["features"], src["labels"]
    tgt_x, tgt_y = tgt["features"], tgt["labels"]
    labels = sorted(set(src_y.tolist()).intersection(set(tgt_y.tolist())))
    l2s, coss = [], []
    for label in labels:
        xs = src_x[src_y == label]
        xt = tgt_x[tgt_y == label]
        if len(xs) == 0 or len(xt) == 0:
            continue
        m = centroid_metrics(xs, xt)
        l2s.append(m["centroid_l2"])
        coss.append(m["centroid_cosine_distance"])
    return {
        "class_cond_centroid_l2": float(np.mean(l2s)) if l2s else float("nan"),
        "class_cond_centroid_cosine": float(np.mean(coss)) if coss else float("nan"),
        "n_shared_classes": float(len(l2s)),
    }


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * np.matmul(a, b.T), 0.0)


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
    kxx = np.exp(-gamma * pairwise_sq_dists(src, src)).mean()
    kyy = np.exp(-gamma * pairwise_sq_dists(tgt, tgt)).mean()
    kxy = np.exp(-gamma * pairwise_sq_dists(src, tgt)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def domain_classifier_distance(src: np.ndarray, tgt: np.ndarray, *, max_samples: int, seed: int) -> Dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedShuffleSplit
    except Exception:
        return {
            "domain_clf_acc": float("nan"),
            "proxy_a_distance": float("nan"),
            "domain_clf_note": "sklearn_not_available",
        }

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
    proxy_a = float(max(0.0, min(2.0, 4.0 * acc - 2.0)))
    return {
        "domain_clf_acc": acc,
        "proxy_a_distance": proxy_a,
        "domain_clf_note": "ok",
    }


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
    row: Dict[str, object] = {
        "surrogate": name,
        "n_source": int(len(src_x)),
        "n_victim": int(len(tgt_x)),
    }
    row.update(centroid_metrics(src_x, tgt_x))
    row.update(class_conditional_metrics(src, tgt))
    row["diagonal_fid"] = diagonal_fid(src_x, tgt_x)
    row["rbf_mmd2"] = rbf_mmd2(src_x, tgt_x, max_samples=mmd_max_samples, seed=seed)
    if skip_domain_clf:
        row.update({"domain_clf_acc": float("nan"), "proxy_a_distance": float("nan"), "domain_clf_note": "skipped"})
    else:
        row.update(domain_classifier_distance(src_x, tgt_x, max_samples=domain_clf_max_samples, seed=seed))
    return row


def make_mixture(
    victim_source: Dict[int, List[int]],
    far_source: Dict[int, List[int]],
    *,
    victim_fraction: float,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    labels = sorted(set(victim_source).intersection(far_source))
    selected: List[int] = []
    for label in labels:
        near = list(victim_source[label])
        far = list(far_source[label])
        rng.shuffle(near)
        rng.shuffle(far)
        n_total = min(len(near) + len(far), 2 * min(len(near), len(far)))
        n_near = int(round(victim_fraction * n_total))
        n_far = n_total - n_near
        n_near = min(n_near, len(near))
        n_far = min(n_far, len(far))
        selected.extend(near[:n_near])
        selected.extend(far[:n_far])
    rng.shuffle(selected)
    return selected


def write_csv(path: Path, rows: Sequence[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]):
    print(f"\n{title}")
    print("\t".join(fields))
    for row in rows:
        vals = []
        for field in fields:
            v = row[field]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if math.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        print("\t".join(vals))


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    label_names = normalize_label_names(ds)
    domains = list(args.domains)
    if args.victim_domain not in domains:
        domains.append(args.victim_domain)
    if args.far_domain not in domains:
        domains.append(args.far_domain)

    counts = Counter((row["domain"], int(row["label"])) for row in ds)
    count_rows = []
    for domain in sorted(set(row["domain"] for row in ds)):
        for label in range(len(label_names)):
            count_rows.append({
                "domain": domain,
                "label": label,
                "label_name": label_names[label],
                "count": counts[(domain, label)],
            })
    write_csv(out_dir / "pacs_domain_label_counts.csv", count_rows)
    print_table("Domain/Class Counts", count_rows, ["domain", "label_name", "count"])

    by_domain_label = select_indices(ds, domains=domains, max_per_class=args.max_per_class, seed=args.seed)
    victim_eval_groups, victim_source_groups = split_victim_source(
        by_domain_label,
        victim_domain=args.victim_domain,
        victim_fraction=args.victim_fraction,
        seed=args.seed,
    )

    extract_indices = set(rows_from_groups(victim_eval_groups))
    extract_indices.update(rows_from_groups(victim_source_groups))
    for domain in domains:
        if domain == args.victim_domain:
            continue
        extract_indices.update(rows_from_groups(by_domain_label[domain]))
    extract_indices = sorted(extract_indices)

    device = torch.device(args.device)
    encoder, transform = build_encoder(args.encoder, device)
    print(f"[features] encoder={args.encoder} n_images={len(extract_indices)} device={device}")
    all_pack = extract_features(
        ds,
        extract_indices,
        transform,
        encoder,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    victim_pack = subset_feature_pack(all_pack, rows_from_groups(victim_eval_groups))
    pure_rows = []
    for domain in domains:
        if domain == args.victim_domain:
            source_indices = rows_from_groups(victim_source_groups)
            name = f"{domain}_heldout"
        else:
            source_indices = rows_from_groups(by_domain_label[domain])
            name = domain
        source_pack = subset_feature_pack(all_pack, source_indices)
        row = compute_distance_row(
            name,
            source_pack,
            victim_pack,
            seed=args.seed,
            mmd_max_samples=args.mmd_max_samples,
            domain_clf_max_samples=args.domain_clf_max_samples,
            skip_domain_clf=args.skip_domain_clf,
        )
        row["victim_domain"] = args.victim_domain
        pure_rows.append(row)

    sort_key = "class_cond_centroid_l2"
    pure_rows = sorted(pure_rows, key=lambda r: float(r[sort_key]))
    write_csv(out_dir / f"pacs_domain_distance_to_{args.victim_domain}.csv", pure_rows)
    print_table(
        f"Pure Domain Distance To Victim={args.victim_domain}",
        pure_rows,
        [
            "surrogate",
            "n_source",
            "class_cond_centroid_l2",
            "centroid_l2",
            "diagonal_fid",
            "rbf_mmd2",
            "domain_clf_acc",
            "proxy_a_distance",
        ],
    )

    mixture_rows = []
    far_groups = by_domain_label[args.far_domain]
    for frac in args.mixture_fracs:
        mixture_indices = make_mixture(
            victim_source_groups,
            far_groups,
            victim_fraction=float(frac),
            seed=args.seed + int(round(1000 * frac)),
        )
        source_pack = subset_feature_pack(all_pack, mixture_indices)
        row = compute_distance_row(
            f"{args.victim_domain}_{frac:g}+{args.far_domain}_{1.0 - float(frac):g}",
            source_pack,
            victim_pack,
            seed=args.seed,
            mmd_max_samples=args.mmd_max_samples,
            domain_clf_max_samples=args.domain_clf_max_samples,
            skip_domain_clf=args.skip_domain_clf,
        )
        row["victim_domain"] = args.victim_domain
        row["victim_domain_fraction_in_surrogate"] = float(frac)
        row["far_domain_fraction_in_surrogate"] = float(1.0 - frac)
        mixture_rows.append(row)

    mixture_rows = sorted(mixture_rows, key=lambda r: float(r["far_domain_fraction_in_surrogate"]))
    write_csv(out_dir / f"pacs_{args.victim_domain}_{args.far_domain}_mixture_distance.csv", mixture_rows)
    print_table(
        f"Mixture Distance To Victim={args.victim_domain}",
        mixture_rows,
        [
            "surrogate",
            "victim_domain_fraction_in_surrogate",
            "far_domain_fraction_in_surrogate",
            "n_source",
            "class_cond_centroid_l2",
            "centroid_l2",
            "diagonal_fid",
            "rbf_mmd2",
            "domain_clf_acc",
            "proxy_a_distance",
        ],
    )

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "label_names": label_names,
        "domains": domains,
        "victim_domain": args.victim_domain,
        "far_domain": args.far_domain,
        "encoder": args.encoder,
        "max_per_class": args.max_per_class,
        "victim_fraction": args.victim_fraction,
        "outputs": {
            "counts": str(out_dir / "pacs_domain_label_counts.csv"),
            "pure_domain_distance": str(out_dir / f"pacs_domain_distance_to_{args.victim_domain}.csv"),
            "mixture_distance": str(out_dir / f"pacs_{args.victim_domain}_{args.far_domain}_mixture_distance.csv"),
        },
    }
    with open(out_dir / "pacs_distance_analysis_config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[done] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
