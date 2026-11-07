#!/usr/bin/env python3
"""
Re-evaluate targeted ASR as a paired clean -> triggered flip rate.

This script intentionally reuses the existing finetuned checkpoints, dataset
builders, and trigger files. It does not retrain models.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_IMPORT_ERROR = None
try:
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.transforms import InterpolationMode
    from wilds import get_dataset

    from baseline_neuba_helpers import NeuBATriggerBank, apply_neuba_trigger, apply_trigger_normed
    from dataset_camelyon17 import CLIP_MEAN as C17_CLIP_MEAN
    from dataset_camelyon17 import CLIP_STD as C17_CLIP_STD
    from dataset_camelyon17 import IMAGENET_MEAN as C17_IMAGENET_MEAN
    from dataset_camelyon17 import IMAGENET_STD as C17_IMAGENET_STD
    from datasets_merced import CLIP_MEAN as UCM_CLIP_MEAN
    from datasets_merced import CLIP_STD as UCM_CLIP_STD
    from datasets_merced import IMAGENET_MEAN as UCM_IMAGENET_MEAN
    from datasets_merced import IMAGENET_STD as UCM_IMAGENET_STD
    from datasets_merced import _collate as collate_ucm
    from datasets_merced import _default_ucm_eval_tfm, build_ucmerced_overlap_loaders
    from datasets_resisc import _default_resisc_train_tfm, build_resisc45_loaders
    from run_finetune import build_model_from_ckpt
except ModuleNotFoundError as exc:
    EVAL_IMPORT_ERROR = exc
    torch = None
    T = None
    DataLoader = None
    InterpolationMode = None
    get_dataset = None
    NeuBATriggerBank = None
    apply_neuba_trigger = None
    apply_trigger_normed = None
    C17_CLIP_MEAN = C17_CLIP_STD = C17_IMAGENET_MEAN = C17_IMAGENET_STD = None
    UCM_CLIP_MEAN = UCM_CLIP_STD = UCM_IMAGENET_MEAN = UCM_IMAGENET_STD = None
    collate_ucm = None
    _default_ucm_eval_tfm = build_ucmerced_overlap_loaders = None
    _default_resisc_train_tfm = build_resisc45_loaders = None
    build_model_from_ckpt = None


def no_grad():
    if torch is None:
        return lambda fn: fn
    return torch.no_grad()


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute flip-rate ASR from existing checkpoints and trigger artifacts."
    )
    p.add_argument("--task", choices=["camelyon17", "ucmerced"], default=None)
    p.add_argument("--ckpt-template", default="", help='Path template containing "{seed}".')
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--model-type", default="resnet50", choices=["resnet50", "clip", "vit"])
    p.add_argument("--dataset-key", default=None, help="Model head key. Defaults to task head.")
    p.add_argument("--n-tgt", type=int, default=None, help="Required for Camelyon17; inferred for UCM.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--splits",
        default="",
        help="Comma-separated splits to evaluate. Defaults to id_val,val,test for C17 and ucm19 for UCM.",
    )

    p.add_argument("--c17-root", default="./data/WILDS")
    p.add_argument("--clip-normalization", action="store_true", help="Use CLIP stats for UCM.")
    p.add_argument(
        "--force-imagenet-normalization",
        action="store_true",
        help="Use ImageNet normalization even for CLIP/VIT loaders. This matches the existing NeuBA C17 eval recipe.",
    )

    p.add_argument("--trigger-pt", default="", help="Targeted trigger .pt file.")
    p.add_argument("--target-class", type=int, default=None, help="Target class in the model head space.")
    p.add_argument("--neuba", action="store_true", help="Use NeuBA paired clean/triggered flip metrics.")
    p.add_argument("--neuba-mode", default="badnet", choices=["badnet", "blended"])

    p.add_argument("--out-csv", default="")

    p.add_argument(
        "--run-table1-grid",
        action="store_true",
        help="Run flip-rate ASR for the Table-1 C17/UCM baselines and Poison-then-Hide.",
    )
    p.add_argument(
        "--aggregate-table1",
        action="store_true",
        help="Aggregate original-ASR CSVs and flip-rate CSVs into rebuttal-ready summary files.",
    )
    p.add_argument("--results-dir", default="domain_generalization_flip_asr")
    p.add_argument(
        "--methods",
        default="ours,fab,hibernated,latent,neuba",
        help="Comma-separated methods for --run-table1-grid/--aggregate-table1.",
    )
    p.add_argument("--models", default="resnet50,clip", help="Comma-separated model types.")
    p.add_argument("--tasks", default="camelyon17,ucmerced", help="Comma-separated tasks.")
    p.add_argument("--dry-run", action="store_true", help="Print grid entries without evaluating.")
    p.add_argument(
        "--grid-index",
        type=int,
        default=None,
        help="Run only one filtered Table-1 grid entry by 0-based index. Useful for Slurm arrays.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a grid entry when its output CSV already contains the primary flip metric for all requested seeds.",
    )
    p.add_argument("--list-table1-grid", action="store_true", help="Print filtered Table-1 grid entries and exit.")
    p.add_argument("--paper-table-csv", default="")
    p.add_argument("--paper-table-md", default="")
    p.add_argument("--paper-table-tex", default="")
    return p.parse_args()


def parse_seeds(seeds_arg: str) -> List[int]:
    return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def pred_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] > 1:
        return logits.argmax(dim=1)
    return (logits.squeeze(-1) > 0).long()


def load_model(ckpt_path: str, *, n_tgt: int, device: str, model_type: str, dataset_key: str):
    out = build_model_from_ckpt(
        ckpt_path=ckpt_path,
        n_tgt=n_tgt,
        device=device,
        model_type=model_type,
        dataset_name=dataset_key,
    )
    model = out[0] if isinstance(out, tuple) else out
    return model.to(device).eval()


def collate_xy(batch):
    xs, ys = [], []
    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            x, y, _ = item
        else:
            x, y = item[:2]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        xs.append(x)
        ys.append(torch.as_tensor(y).long())
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0).long()


def build_c17_loaders(
    root_dir: str,
    *,
    bs: int,
    nw: int,
    model_type: str,
    splits: Sequence[str],
    force_imagenet_normalization: bool = False,
) -> Tuple[Dict[str, DataLoader], Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    needs_clip_size = model_type in {"clip", "vit"}
    clip_norm = needs_clip_size and not force_imagenet_normalization
    mean, std = (C17_CLIP_MEAN, C17_CLIP_STD) if clip_norm else (C17_IMAGENET_MEAN, C17_IMAGENET_STD)
    resize_to = 224 if needs_clip_size else None
    ops = []
    if resize_to is not None:
        ops.append(T.Resize((resize_to, resize_to), interpolation=InterpolationMode.BICUBIC))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    tf_eval = T.Compose(ops)

    ds = get_dataset(dataset="camelyon17", root_dir=root_dir, download=False)
    loaders = {}
    for split in splits:
        subset = ds.get_subset(split, transform=tf_eval)
        loaders[split] = DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            collate_fn=collate_xy,
        )
    return loaders, (mean, std)


def build_ucm_loader(
    *, bs: int, nw: int, clip_normalization: bool
) -> Tuple[DataLoader, int, List[str], List[str], List[int], Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    _, _, n45, class_names45 = build_resisc45_loaders(
        bs_train=8,
        bs_test=8,
        nw=0,
        seed=0,
        transform_train=_default_resisc_train_tfm(),
        transform_val=None,
        clip_normalization=clip_normalization,
    )
    _, _, _, class_names19, hf = build_ucmerced_overlap_loaders(
        bs_train=bs,
        bs_eval=bs,
        nw=0,
        seed=0,
        test_size=0.0,
        clip_normalization=clip_normalization,
        resisc_class_names=class_names45,
        return_hf_datasets=True,
    )
    mean, std = (UCM_CLIP_MEAN, UCM_CLIP_STD) if clip_normalization else (UCM_IMAGENET_MEAN, UCM_IMAGENET_STD)
    tf_eval = _default_ucm_eval_tfm(mean=mean, std=std)

    def add_tf(batch):
        batch["pixel_values"] = [tf_eval(img) for img in batch["image"]]
        return batch

    ds_eval = hf["__hf_ucm_train_19__"].with_transform(add_tf)
    loader = DataLoader(
        ds_eval,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_ucm,
    )

    resisc_idx_by_name = {name: i for i, name in enumerate(class_names45)}
    y19_to_resisc45 = [resisc_idx_by_name[name] for name in class_names19]
    return loader, n45, class_names45, class_names19, y19_to_resisc45, (mean, std)


@dataclass(frozen=True)
class Table1Spec:
    task: str
    model_type: str
    method: str
    method_label: str
    ckpt_template: str
    original_csv: str
    trigger_pt: str = ""
    target_class: Optional[int] = None
    n_tgt: Optional[int] = None
    neuba: bool = False
    neuba_mode: str = "badnet"
    clip_normalization: bool = False
    force_imagenet_normalization: bool = False
    existing_flip_csv: str = ""


PTH_C17_TRIGGER = "../artifacts/optimized_triggers/camelyon17/trigger_patch.pt"
PTH_UCM_TRIGGER = (
    "../artifacts/optimized_triggers/resisc45/ablations/"
    "target/axis=target__val=28__eps=8of255__tgt=28__k=3/trigger_patch.pt"
)


TABLE1_SPECS: Tuple[Table1Spec, ...] = (
    Table1Spec(
        "camelyon17", "resnet50", "ours", "Poison-then-Hide",
        "ckpts_finetune_generic_mixed_resnet50_imagenet_to_camelyon17_seed{seed}/"
        "finetuned__resnet50__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_ours_resnet50_v2.csv",
        trigger_pt=PTH_C17_TRIGGER, target_class=0, n_tgt=2,
    ),
    Table1Spec(
        "camelyon17", "clip", "ours", "Poison-then-Hide",
        "ckpts_finetune_generic_mixed_clip_imagenet_to_camelyon17_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_ours_clip.csv",
        trigger_pt=PTH_C17_TRIGGER, target_class=0, n_tgt=2,
    ),
    Table1Spec(
        "camelyon17", "resnet50", "fab", "FAB",
        "baseline_fab_resnet50_imagenet_to_camelyon17_seed{seed}/"
        "finetuned__resnet50__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_fab_resnet50.csv",
        trigger_pt=PTH_C17_TRIGGER, target_class=0, n_tgt=2,
    ),
    Table1Spec(
        "camelyon17", "clip", "fab", "FAB",
        "baseline_fab_clip_imagenet_to_camelyon17_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_fab_clip.csv",
        trigger_pt=PTH_C17_TRIGGER, target_class=0, n_tgt=2,
    ),
    Table1Spec(
        "camelyon17", "resnet50", "hibernated", "Hibernated Backdoor",
        "baseline_hibernated_resnet50_to_camelyon17_seed{seed}/"
        "finetuned__resnet50__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/camelyon17_eval_summary_hibernated_resnet50.csv",
        trigger_pt=PTH_C17_TRIGGER, target_class=0, n_tgt=2,
    ),
    Table1Spec(
        "camelyon17", "clip", "hibernated", "Hibernated Backdoor",
        "baseline_hibernated_clip_to_camelyon17_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_hibernated_clip.csv",
        trigger_pt=PTH_C17_TRIGGER, target_class=0, n_tgt=2,
    ),
    Table1Spec(
        "camelyon17", "resnet50", "latent", "Latent Backdoor",
        "baseline_latent_backdoor_resnet50_camelyon17_seed{seed}/"
        "finetuned__resnet50__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_latent_resnet50.csv",
        trigger_pt="runs_latent_backdoor/trigger_resnet50_camelyon17_class0.pt", target_class=0, n_tgt=2,
    ),
    Table1Spec(
        "camelyon17", "clip", "latent", "Latent Backdoor",
        "baseline_latent_backdoor_clip_camelyon17_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_latent_clip.csv",
        trigger_pt="runs_latent_backdoor/trigger_clip_camelyon17_class0.pt", target_class=0, n_tgt=2,
    ),
    Table1Spec(
        "camelyon17", "resnet50", "neuba", "NeuBA",
        "out_neuba/neuba_resnet50_to_camelyon17_seed{seed}/"
        "finetuned__resnet50__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_neuba_resnet50.csv",
        n_tgt=2, neuba=True,
    ),
    Table1Spec(
        "camelyon17", "clip", "neuba", "NeuBA",
        "out_neuba/neuba_clip_to_camelyon17_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-camelyon17__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_camelyon17/c17_eval_summary_neuba_clip.csv",
        n_tgt=2, neuba=True, force_imagenet_normalization=True,
    ),
    Table1Spec(
        "ucmerced", "resnet50", "ours", "Poison-then-Hide",
        "ckpts_finetune_generic_mixed_resnet50_imagenet_to_resisc_seed{seed}/"
        "finetuned__resnet50__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_ours_resnet50.csv",
        trigger_pt=PTH_UCM_TRIGGER, target_class=28,
    ),
    Table1Spec(
        "ucmerced", "clip", "ours", "Poison-then-Hide",
        "ckpts_finetune_generic_mixed_clip_imagenet_to_resisc_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_ours_clip.csv",
        trigger_pt=PTH_UCM_TRIGGER, target_class=28,
        existing_flip_csv="domain_generalization_ucmerced/ucm19_flip_asr_ours_clip.csv",
    ),
    Table1Spec(
        "ucmerced", "resnet50", "fab", "FAB",
        "baseline_fab_models/baseline_fab_resnet50_imagenet_to_resisc_seed{seed}/"
        "finetuned__resnet50__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_fab_resnet50.csv",
        trigger_pt=PTH_UCM_TRIGGER, target_class=28,
    ),
    Table1Spec(
        "ucmerced", "clip", "fab", "FAB",
        "baseline_fab_clip_imagenet_to_resisc_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_fab_clip.csv",
        trigger_pt=PTH_UCM_TRIGGER, target_class=28,
    ),
    Table1Spec(
        "ucmerced", "resnet50", "hibernated", "Hibernated Backdoor",
        "baseline_hibernated_models/baseline_hibernated_resnet50_to_resisc_seed{seed}/"
        "finetuned__resnet50__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_hibernated_resnet50.csv",
        trigger_pt=PTH_UCM_TRIGGER, target_class=28,
    ),
    Table1Spec(
        "ucmerced", "clip", "hibernated", "Hibernated Backdoor",
        "baseline_hibernated_clip_to_resisc_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_hibernated_clip.csv",
        trigger_pt=PTH_UCM_TRIGGER, target_class=28,
    ),
    Table1Spec(
        "ucmerced", "resnet50", "latent", "Latent Backdoor",
        "baseline_latent_backdoor_resnet50_resisc_seed{seed}/"
        "finetuned__resnet50__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_latent_resnet50.csv",
        trigger_pt="runs_latent_backdoor/trigger_resnet50_resisc_class28.pt", target_class=28,
    ),
    Table1Spec(
        "ucmerced", "clip", "latent", "Latent Backdoor",
        "baseline_latent_backdoor_clip_resisc_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_latent_clip.csv",
        trigger_pt="runs_latent_backdoor/trigger_clip_resisc_class28.pt", target_class=28,
    ),
    Table1Spec(
        "ucmerced", "resnet50", "neuba", "NeuBA",
        "out_neuba/neuba_resnet50_to_resisc_seed{seed}/"
        "finetuned__resnet50__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_neuba_resnet50_corrected.csv",
        neuba=True,
    ),
    Table1Spec(
        "ucmerced", "clip", "neuba", "NeuBA",
        "out_neuba/neuba_clip_to_resisc_seed{seed}/"
        "finetuned__clip-vit_base_patch16_clip_224.openai12k1k__ds-resisc__ft3e_lr0.0003__seed{seed}__plainFT_full.pt",
        "domain_generalization_ucmerced/ucm19_eval_asr_neuba_clip_corrected.csv",
        neuba=True,
    ),
)


def load_trigger(trigger_pt: str, device: str, fallback_target: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if not trigger_pt:
        raise ValueError("--trigger-pt is required unless --neuba is set.")
    if not os.path.exists(trigger_pt):
        raise FileNotFoundError(trigger_pt)
    obj = torch.load(trigger_pt, map_location=device)
    delta = obj["delta_pix"].to(device).float()
    h, w = delta.shape[-2:]
    mask = obj.get("mask", torch.ones(1, 3, h, w, device=device, dtype=delta.dtype)).to(device).float()
    target_class = int(obj.get("target_class", fallback_target))
    return delta, mask, target_class


@no_grad()
def targeted_flip_rows(
    model,
    loader: DataLoader,
    *,
    split: str,
    device: str,
    dataset_key: str,
    mean: Sequence[float],
    std: Sequence[float],
    delta: torch.Tensor,
    mask: torch.Tensor,
    target_class: int,
    y_to_head: Optional[Sequence[int]] = None,
) -> List[Tuple[str, float, int, int]]:
    counts = {
        "n": 0,
        "clean_target": 0,
        "trigger_target": 0,
        "flip_to_target": 0,
        "clean_pred_nontarget": 0,
        "true_nontarget": 0,
        "true_nontarget_flip_to_target": 0,
        "clean_correct_source": 0,
        "clean_correct_source_to_target": 0,
    }

    y_map = None if y_to_head is None else torch.as_tensor(y_to_head, device=device, dtype=torch.long)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        y_head = yb if y_map is None else y_map[yb]
        x_tr = apply_trigger_normed(xb, mean, std, delta_pix=delta, mask=mask)

        clean_pred = pred_from_logits(model(xb, dataset=dataset_key, route="full"))
        trig_pred = pred_from_logits(model(x_tr, dataset=dataset_key, route="full"))

        clean_is_target = clean_pred == target_class
        trig_is_target = trig_pred == target_class
        true_is_source = y_head != target_class
        clean_pred_source = ~clean_is_target
        clean_correct_source = true_is_source & (clean_pred == y_head)
        flip_to_target = clean_pred_source & trig_is_target

        counts["n"] += int(yb.numel())
        counts["clean_target"] += int(clean_is_target.sum().item())
        counts["trigger_target"] += int(trig_is_target.sum().item())
        counts["flip_to_target"] += int(flip_to_target.sum().item())
        counts["clean_pred_nontarget"] += int(clean_pred_source.sum().item())
        counts["true_nontarget"] += int(true_is_source.sum().item())
        counts["true_nontarget_flip_to_target"] += int((true_is_source & flip_to_target).sum().item())
        counts["clean_correct_source"] += int(clean_correct_source.sum().item())
        counts["clean_correct_source_to_target"] += int((clean_correct_source & trig_is_target).sum().item())

    def pct(num_key: str, den_key: str = "n") -> float:
        den = max(1, counts[den_key])
        return 100.0 * counts[num_key] / den

    return [
        ("target_rate_clean", pct("clean_target"), counts["clean_target"], counts["n"]),
        ("asr_legacy_target_rate", pct("trigger_target"), counts["trigger_target"], counts["n"]),
        (
            "target_rate_net",
            pct("trigger_target") - pct("clean_target"),
            counts["trigger_target"] - counts["clean_target"],
            counts["n"],
        ),
        ("flip_to_target_all", pct("flip_to_target"), counts["flip_to_target"], counts["n"]),
        (
            "flip_asr_pred_nontarget",
            pct("flip_to_target", "clean_pred_nontarget"),
            counts["flip_to_target"],
            counts["clean_pred_nontarget"],
        ),
        (
            "flip_asr_true_nontarget",
            pct("true_nontarget_flip_to_target", "true_nontarget"),
            counts["true_nontarget_flip_to_target"],
            counts["true_nontarget"],
        ),
        (
            "flip_asr_clean_correct_source",
            pct("clean_correct_source_to_target", "clean_correct_source"),
            counts["clean_correct_source_to_target"],
            counts["clean_correct_source"],
        ),
    ]


@no_grad()
def neuba_flip_rows(
    model,
    loader: DataLoader,
    *,
    device: str,
    dataset_key: str,
    model_type: str,
    mode: str,
    mean: Sequence[float],
    std: Sequence[float],
    y_to_head: Optional[Sequence[int]] = None,
) -> List[Tuple[str, float, int, int]]:
    xb0, _ = next(iter(loader))
    image_size = int(xb0.shape[-1])
    blend_ratio = 0.25 if model_type == "resnet50" else 0.3
    bank = NeuBATriggerBank(mode, image_size=image_size, blend_ratio=blend_ratio)
    blended = mode == "blended"
    y_map = None if y_to_head is None else torch.as_tensor(y_to_head, device=device, dtype=torch.long)

    n = legacy_wrong = pred_flip = clean_correct = clean_correct_to_wrong = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        y_head = yb if y_map is None else y_map[yb]
        trig_idx = torch.randint(0, 6, (xb.size(0),), device=device, dtype=torch.long)
        x_tr = apply_neuba_trigger(
            xb,
            mean,
            std,
            bank,
            trig_idx,
            delta_pix_list=[d.to(device) for d in bank.delta],
            mask_list=[m.to(device) for m in bank.mask],
            blended=blended,
        )

        clean_pred = pred_from_logits(model(xb, dataset=dataset_key, route="full"))
        trig_pred = pred_from_logits(model(x_tr, dataset=dataset_key, route="full"))
        clean_ok = clean_pred == y_head
        trig_wrong = trig_pred != y_head

        n += int(yb.numel())
        legacy_wrong += int(trig_wrong.sum().item())
        pred_flip += int((trig_pred != clean_pred).sum().item())
        clean_correct += int(clean_ok.sum().item())
        clean_correct_to_wrong += int((clean_ok & trig_wrong).sum().item())

    return [
        ("asr_neuba_legacy_misclf", 100.0 * legacy_wrong / max(1, n), legacy_wrong, n),
        ("neuba_pred_flip_all", 100.0 * pred_flip / max(1, n), pred_flip, n),
        (
            "neuba_asr_clean_correct_to_wrong",
            100.0 * clean_correct_to_wrong / max(1, clean_correct),
            clean_correct_to_wrong,
            clean_correct,
        ),
    ]


def split_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def filtered_table1_specs(args) -> List[Table1Spec]:
    methods = set(split_csv_arg(args.methods))
    models = set(split_csv_arg(args.models))
    tasks = set(split_csv_arg(args.tasks))
    return [
        spec for spec in TABLE1_SPECS
        if spec.method in methods and spec.model_type in models and spec.task in tasks
    ]


def default_splits(task: str) -> List[str]:
    return ["id_val", "val", "test"] if task == "camelyon17" else ["ucm19"]


def selected_splits(args, task: str) -> List[str]:
    return split_csv_arg(args.splits) if getattr(args, "splits", "") else default_splits(task)


def flip_csv_path(spec: Table1Spec, results_dir: str) -> str:
    return os.path.join(results_dir, f"{spec.task}_flip_asr_{spec.method}_{spec.model_type}.csv")


def resolve_input_path(path: str) -> str:
    if not path or os.path.exists(path):
        return path
    if not os.path.isabs(path):
        src_relative = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(src_relative):
            return src_relative
    return path


def spec_to_namespace(base_args, spec: Table1Spec, out_csv: str):
    return argparse.Namespace(
        task=spec.task,
        ckpt_template=spec.ckpt_template,
        seeds=base_args.seeds,
        model_type=spec.model_type,
        dataset_key="resisc" if spec.task == "ucmerced" else "camelyon17",
        n_tgt=spec.n_tgt,
        device=base_args.device,
        batch=base_args.batch,
        num_workers=base_args.num_workers,
        splits=base_args.splits or ("val" if spec.task == "camelyon17" else "ucm19"),
        c17_root=base_args.c17_root,
        clip_normalization=spec.clip_normalization,
        force_imagenet_normalization=spec.force_imagenet_normalization,
        trigger_pt=spec.trigger_pt,
        target_class=spec.target_class,
        neuba=spec.neuba,
        neuba_mode=spec.neuba_mode,
        out_csv=out_csv,
        run_table1_grid=False,
        aggregate_table1=False,
        results_dir=base_args.results_dir,
        methods=base_args.methods,
        models=base_args.models,
        tasks=base_args.tasks,
        dry_run=base_args.dry_run,
        grid_index=base_args.grid_index,
        skip_existing=base_args.skip_existing,
        list_table1_grid=base_args.list_table1_grid,
        paper_table_csv=base_args.paper_table_csv,
        paper_table_md=base_args.paper_table_md,
        paper_table_tex=base_args.paper_table_tex,
    )


def validate_single_args(args):
    if args.task is None:
        raise ValueError("--task is required unless --run-table1-grid or --aggregate-table1 is set.")
    if not args.ckpt_template:
        raise ValueError("--ckpt-template is required unless --run-table1-grid or --aggregate-table1 is set.")
    if not args.out_csv:
        raise ValueError("--out-csv is required unless --run-table1-grid or --aggregate-table1 is set.")


def format_grid_command(spec_args) -> str:
    parts = [
        "python", "evaluate_flip_rate_asr.py",
        "--task", spec_args.task,
        "--ckpt-template", spec_args.ckpt_template,
        "--seeds", spec_args.seeds,
        "--model-type", spec_args.model_type,
        "--dataset-key", spec_args.dataset_key,
        "--device", spec_args.device,
        "--batch", str(spec_args.batch),
        "--num-workers", str(spec_args.num_workers),
        "--out-csv", spec_args.out_csv,
    ]
    if spec_args.n_tgt is not None:
        parts += ["--n-tgt", str(spec_args.n_tgt)]
    if spec_args.task == "camelyon17":
        parts += ["--c17-root", spec_args.c17_root]
    if spec_args.splits:
        parts += ["--splits", spec_args.splits]
    if spec_args.clip_normalization:
        parts.append("--clip-normalization")
    if spec_args.force_imagenet_normalization:
        parts.append("--force-imagenet-normalization")
    if spec_args.neuba:
        parts += ["--neuba", "--neuba-mode", spec_args.neuba_mode]
    else:
        parts += ["--trigger-pt", spec_args.trigger_pt]
        if spec_args.target_class is not None:
            parts += ["--target-class", str(spec_args.target_class)]
    return " ".join(parts)


def primary_metric_for_args(spec_args) -> str:
    return "neuba_asr_clean_correct_to_wrong" if spec_args.neuba else "flip_asr_true_nontarget"


def output_complete(out_csv: str, *, metric: str, seeds: Sequence[int], splits: Sequence[str]) -> bool:
    if not out_csv or not os.path.exists(out_csv):
        return False
    try:
        df = pd.read_csv(out_csv)
    except Exception:
        return False
    needed = {(int(seed), split) for seed in seeds for split in splits}
    if not needed or not {"seed", "split", "metric"}.issubset(df.columns):
        return False
    got = {
        (int(row.seed), str(row.split))
        for row in df.itertuples(index=False)
        if str(row.metric) == metric
    }
    return needed.issubset(got)


def print_table1_grid(args, specs: Sequence[Table1Spec]):
    for i, spec in enumerate(specs):
        out_csv = flip_csv_path(spec, args.results_dir)
        spec_args = spec_to_namespace(args, spec, out_csv)
        splits = ",".join(selected_splits(spec_args, spec.task))
        status = "done" if output_complete(
            out_csv,
            metric=primary_metric_for_args(spec_args),
            seeds=parse_seeds(args.seeds),
            splits=selected_splits(spec_args, spec.task),
        ) else "pending"
        print(f"{i:02d}\t{status}\t{spec.task}\t{spec.model_type}\t{spec.method}\t{splits}\t{out_csv}")


def run_table1_grid(args):
    specs = filtered_table1_specs(args)
    if not specs:
        raise ValueError("No Table-1 specs matched --methods/--models/--tasks.")
    if args.list_table1_grid:
        print_table1_grid(args, specs)
        return
    if args.grid_index is not None:
        if args.grid_index < 0 or args.grid_index >= len(specs):
            raise ValueError(f"--grid-index must be in [0, {len(specs) - 1}], got {args.grid_index}.")
        specs = [specs[args.grid_index]]

    for spec in specs:
        out_csv = flip_csv_path(spec, args.results_dir)
        spec_args = spec_to_namespace(args, spec, out_csv)
        print(f"[grid] {spec.task} {spec.model_type} {spec.method_label} -> {out_csv}")
        if args.skip_existing and output_complete(
            out_csv,
            metric=primary_metric_for_args(spec_args),
            seeds=parse_seeds(args.seeds),
            splits=selected_splits(spec_args, spec.task),
        ):
            print(f"[skip] complete output already exists: {out_csv}")
            continue
        if args.dry_run:
            print(format_grid_command(spec_args))
            continue
        run_single(spec_args)


def read_values(path: str, *, metric: str, split: Optional[str] = None) -> Optional[pd.Series]:
    path = resolve_input_path(path)
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "metric" not in df.columns or "value" not in df.columns:
        return None
    mask = df["metric"].astype(str) == metric
    if split is not None and "split" in df.columns:
        mask &= df["split"].astype(str) == split
    values = pd.to_numeric(df.loc[mask, "value"], errors="coerce").dropna()
    return values if len(values) else None


def mean_std(values: Optional[pd.Series]) -> Tuple[float, float, int]:
    if values is None or len(values) == 0:
        return float("nan"), float("nan"), 0
    std = 0.0 if len(values) == 1 else float(values.std(ddof=1))
    return float(values.mean()), std, int(len(values))


def fmt_mean_std(mean: float, std: float, n: int) -> str:
    if n == 0 or pd.isna(mean):
        return "NA"
    if pd.isna(std):
        std = 0.0
    return f"{mean:.2f} ± {std:.2f}"


def original_metric(spec: Table1Spec) -> str:
    if spec.neuba:
        return "asr_neuba" if spec.task == "camelyon17" else "asr_neuba_ucm19"
    return "asr_pt" if spec.task == "camelyon17" else "asr_ucm19"


def clean_metric(spec: Table1Spec) -> str:
    return "clean_top1" if spec.task == "camelyon17" else "clean_top1_ucm19"


def flip_metric(spec: Table1Spec) -> str:
    return "neuba_asr_clean_correct_to_wrong" if spec.neuba else "flip_asr_true_nontarget"


def eval_split(spec: Table1Spec) -> Optional[str]:
    return "val" if spec.task == "camelyon17" else "ucm19"


def find_flip_csv(spec: Table1Spec, results_dir: str) -> str:
    preferred = flip_csv_path(spec, results_dir)
    if os.path.exists(preferred):
        return preferred
    resolved_preferred = resolve_input_path(preferred)
    if os.path.exists(resolved_preferred):
        return resolved_preferred
    resolved_existing = resolve_input_path(spec.existing_flip_csv)
    if resolved_existing and os.path.exists(resolved_existing):
        return resolved_existing
    return preferred


def aggregate_table1(args):
    rows = []
    for spec in filtered_table1_specs(args):
        split = eval_split(spec)
        clean_vals = read_values(spec.original_csv, metric=clean_metric(spec), split=split if spec.task == "camelyon17" else None)
        original_vals = read_values(spec.original_csv, metric=original_metric(spec), split=split if spec.task == "camelyon17" else None)
        flip_path = find_flip_csv(spec, args.results_dir)
        flip_vals = read_values(flip_path, metric=flip_metric(spec), split=split)

        clean_mean, clean_sd, clean_n = mean_std(clean_vals)
        orig_mean, orig_sd, orig_n = mean_std(original_vals)
        flip_mean, flip_sd, flip_n = mean_std(flip_vals)
        rows.append(
            {
                "task": spec.task,
                "model_type": spec.model_type,
                "method": spec.method,
                "method_label": spec.method_label,
                "original_metric": original_metric(spec),
                "flip_metric": flip_metric(spec),
                "clean_mean": clean_mean,
                "clean_std": clean_sd,
                "clean_n": clean_n,
                "original_asr_mean": orig_mean,
                "original_asr_std": orig_sd,
                "original_asr_n": orig_n,
                "flip_asr_mean": flip_mean,
                "flip_asr_std": flip_sd,
                "flip_asr_n": flip_n,
                "original_csv": spec.original_csv,
                "flip_csv": flip_path if os.path.exists(flip_path) else "",
            }
        )

    df = pd.DataFrame(rows)
    csv_path = args.paper_table_csv or os.path.join(args.results_dir, "table1_flip_asr_summary.csv")
    md_path = args.paper_table_md or os.path.join(args.results_dir, "table1_flip_asr_summary.md")
    tex_path = args.paper_table_tex or os.path.join(args.results_dir, "table1_flip_asr_summary.tex")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)

    table = []
    for model in split_csv_arg(args.models):
        for method in split_csv_arg(args.methods):
            sub = df[(df["model_type"] == model) & (df["method"] == method)]
            if sub.empty:
                continue
            label = sub["method_label"].iloc[0]
            row = {"Model": model, "Method": label}
            for task, prefix in [("camelyon17", "C17"), ("ucmerced", "UCMerced")]:
                rec = sub[sub["task"] == task]
                if rec.empty:
                    row[f"{prefix} Acc"] = "NA"
                    row[f"{prefix} Original ASR"] = "NA"
                    row[f"{prefix} Flip-ASR"] = "NA"
                    continue
                r = rec.iloc[0]
                row[f"{prefix} Acc"] = fmt_mean_std(r.clean_mean, r.clean_std, int(r.clean_n))
                row[f"{prefix} Original ASR"] = fmt_mean_std(r.original_asr_mean, r.original_asr_std, int(r.original_asr_n))
                row[f"{prefix} Flip-ASR"] = fmt_mean_std(r.flip_asr_mean, r.flip_asr_std, int(r.flip_asr_n))
            table.append(row)

    table_df = pd.DataFrame(table)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    table_df.to_markdown(md_path, index=False)

    caption = (
        "Flip-rate ASR comparison on the Table 1 OOD datasets. For targeted methods, "
        "Flip-ASR is computed over non-target true-label examples and counts only clean "
        "non-target predictions that become the target prediction under the trigger. "
        "For NeuBA, the flip column reports the analogous clean-correct to triggered-wrong rate."
    )
    with open(tex_path, "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(table_df.to_latex(index=False, escape=True))
        f.write("\\label{tab:table1_flip_asr_full}\n\\end{table*}\n")

    print(f"[save] {len(df)} summary rows -> {csv_path}")
    print(f"[save] markdown table -> {md_path}")
    print(f"[save] latex table -> {tex_path}")
    return df


def run_single(args):
    validate_single_args(args)
    if EVAL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Evaluation dependencies are unavailable in this Python environment. "
            "Run the actual flip-ASR evaluation in the training/evaluation environment "
            "with torch, torchvision, WILDS, and the project dataset helpers installed."
        ) from EVAL_IMPORT_ERROR
    device = args.device
    seeds = parse_seeds(args.seeds)
    dataset_key = args.dataset_key or ("camelyon17" if args.task == "camelyon17" else "resisc")
    neuba = bool(args.neuba or ("neuba" in args.ckpt_template.lower()))

    if args.task == "camelyon17":
        if args.n_tgt is None:
            raise ValueError("--n-tgt 2 is required for Camelyon17.")
        splits = selected_splits(args, args.task)
        loaders, (mean, std) = build_c17_loaders(
            args.c17_root,
            bs=args.batch,
            nw=args.num_workers,
            model_type=args.model_type,
            splits=splits,
            force_imagenet_normalization=args.force_imagenet_normalization,
        )
        n_tgt = args.n_tgt
        y_to_head = None
        class_names45 = None
    else:
        splits = selected_splits(args, args.task)
        if splits != ["ucm19"]:
            raise ValueError(f"UCMerced only supports split 'ucm19', got {splits}.")
        loader, n_tgt, class_names45, class_names19, y_to_head, (mean, std) = build_ucm_loader(
            bs=args.batch, nw=args.num_workers, clip_normalization=args.clip_normalization
        )
        loaders = {"ucm19": loader}

    delta = mask = target_class = None
    if not neuba:
        fallback_target = 0 if args.task == "camelyon17" else 28
        if args.target_class is not None:
            fallback_target = args.target_class
        delta, mask, target_class = load_trigger(args.trigger_pt, device, fallback_target)
        if args.target_class is not None and target_class != args.target_class:
            print(f"[warn] trigger target_class={target_class}; overriding with --target-class={args.target_class}")
            target_class = args.target_class

        if args.task == "ucmerced" and class_names45 is not None:
            print(f"[target] RESISC-{target_class}: {class_names45[target_class]}")

    rows = []
    for seed in seeds:
        ckpt_path = args.ckpt_template.format(seed=seed)
        if not os.path.exists(ckpt_path):
            print(f"[warn] missing seed={seed}: {ckpt_path}")
            continue

        model = load_model(
            ckpt_path,
            n_tgt=n_tgt,
            device=device,
            model_type=args.model_type,
            dataset_key=dataset_key,
        )

        for split, loader in loaders.items():
            if neuba:
                metric_rows = neuba_flip_rows(
                    model,
                    loader,
                    device=device,
                    dataset_key=dataset_key,
                    model_type=args.model_type,
                    mode=args.neuba_mode,
                    mean=mean,
                    std=std,
                    y_to_head=y_to_head,
                )
            else:
                metric_rows = targeted_flip_rows(
                    model,
                    loader,
                    split=split,
                    device=device,
                    dataset_key=dataset_key,
                    mean=mean,
                    std=std,
                    delta=delta,
                    mask=mask,
                    target_class=int(target_class),
                    y_to_head=y_to_head,
                )

            for metric, value, numerator, denominator in metric_rows:
                rows.append(
                    {
                        "task": args.task,
                        "model_type": args.model_type,
                        "dataset_key": dataset_key,
                        "split": split,
                        "seed": seed,
                        "metric": metric,
                        "value": float(value),
                        "numerator": int(numerator),
                        "denominator": int(denominator),
                        "target_class": target_class,
                        "ckpt_path": ckpt_path,
                        "trigger_pt": args.trigger_pt or None,
                        "neuba": neuba,
                        "neuba_mode": args.neuba_mode if neuba else None,
                    }
                )
            primary = "neuba_asr_clean_correct_to_wrong" if neuba else "flip_asr_true_nontarget"
            primary_value = next(v for m, v, _, _ in metric_rows if m == primary)
            print(f"[{split}] seed={seed} {primary}={primary_value:.2f}%")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[save] {len(df)} rows -> {args.out_csv}")


def main():
    args = parse_args()
    if args.run_table1_grid:
        run_table1_grid(args)
    elif args.aggregate_table1:
        aggregate_table1(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
