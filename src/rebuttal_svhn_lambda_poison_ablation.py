#!/usr/bin/env python3
"""
Reviewer-facing CIFAR10 -> SVHN ablations for hiding-stage lambdas and poison fraction.

This driver keeps the optimized trigger fixed and varies only the hiding-stage
training recipe:

  1. lambda ablation: one-at-a-time multipliers for lambda_sup, lambda_KD, lambda_SP
  2. poison-fraction ablation: poison_fraction sweep with lambdas fixed at defaults

Use --dry-run first. Commands and manifests are written before jobs launch so
interrupted grids can be audited or resumed.
"""

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from rebuttal_ensemble_choice_eval import collect_metrics, flatten_summary_row


DEFAULT_LAMBDAS: Dict[str, float] = {
    "sup": 1.0,
    "source_ce": 0.05,
    "kd": 0.7,
    "kd_feat": 0.5,
    "sp": 5e-3,
}


def parse_args():
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    p = argparse.ArgumentParser(
        description="Run CIFAR10 -> SVHN lambda and poison-fraction ablations."
    )
    p.add_argument("--sets", nargs="+", choices=["lambda", "poison", "both"], default=["both"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    p.add_argument("--trigger-pt", required=True,
                   help="Fixed optimized SVHN trigger .pt used for all ablation points.")
    p.add_argument("--target-class", type=int, default=2)

    p.add_argument("--lambda-axes", nargs="+", choices=["sup", "kd", "sp", "all"],
                   default=["sup", "kd", "sp"])
    p.add_argument("--lambda-multipliers", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 1.0, 2.0],
                   help="One-at-a-time multipliers around the default lambda values.")
    p.add_argument("--poison-fractions", type=float, nargs="+",
                   default=[0.0, 0.01, 0.025, 0.05, 0.10, 0.20])
    p.add_argument("--default-poison-fraction", type=float, default=0.05)

    p.add_argument("--trainer", type=Path, default=here / "finetune_generic_mixed.py")
    p.add_argument("--finetune-script", type=Path, default=here / "run_finetune_with_cifar10_svhn.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--repo-root", type=Path, default=repo_root)
    p.add_argument("--script-cwd", type=Path, default=here)
    p.add_argument("--out-root", type=Path, default=repo_root / "runs_rebuttal_svhn_lambda_poison_ablation")

    p.add_argument("--model-type", default="resnet18", choices=["resnet18"])
    p.add_argument("--source-dataset", default="cifar10", choices=["cifar10"])
    p.add_argument("--transfer-dataset", default="svhn", choices=["svhn"])

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--bn-rec-every", type=int, default=100)
    p.add_argument("--im-root", default="./data/ILSVRC2012")
    p.add_argument("--svhn-reference-size", type=int, default=None,
                   help="Optional attacker SVHN reference size. Default uses the trainer default pool.")
    p.add_argument("--svhn-reference-split", choices=["train", "train_extra"], default="train")
    p.add_argument("--reference-subset-strategy", choices=["balanced", "random"], default="balanced")

    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=3e-4)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--command-log", type=Path, default=None)
    return p.parse_args()


def sh(cmd: Iterable[object]) -> str:
    return shlex.join(str(x) for x in cmd)


def requested_sets(raw: Sequence[str]) -> List[str]:
    if "both" in raw:
        return ["lambda", "poison"]
    return list(dict.fromkeys(raw))


def requested_axes(raw: Sequence[str]) -> List[str]:
    if "all" in raw:
        return ["sup", "kd", "sp"]
    return list(dict.fromkeys(raw))


def float_tag(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


def ckpt_dir(args, *, seed: int, suffix: str) -> Path:
    return args.script_cwd / (
        f"ckpts_finetune_generic_mixed_{args.model_type}_"
        f"{args.source_dataset}_to_{args.transfer_dataset}_seed{seed}{suffix}"
    )


def lambda_values(axis: str, multiplier: float) -> Dict[str, float]:
    values = dict(DEFAULT_LAMBDAS)
    if axis == "sup":
        values["sup"] = DEFAULT_LAMBDAS["sup"] * multiplier
    elif axis == "kd":
        values["kd"] = DEFAULT_LAMBDAS["kd"] * multiplier
        values["kd_feat"] = DEFAULT_LAMBDAS["kd_feat"] * multiplier
    elif axis == "sp":
        values["sp"] = DEFAULT_LAMBDAS["sp"] * multiplier
    elif axis != "default":
        raise ValueError(f"Unknown lambda axis {axis}")
    return values


def train_cmd(args, *, seed: int, suffix: str, poison_fraction: float, lambdas: Dict[str, float]) -> List[object]:
    cmd: List[object] = [
        args.python,
        args.trainer,
        "--model_type", args.model_type,
        "--source_dataset", args.source_dataset,
        "--transfer_dataset", args.transfer_dataset,
        "--seed", seed,
        "--trigger_pt", args.trigger_pt,
        "--poison_fraction", poison_fraction,
        "--lambda_sup", lambdas["sup"],
        "--lambda_source_ce", lambdas["source_ce"],
        "--lambda_kd", lambdas["kd"],
        "--lambda_kd_feat", lambdas["kd_feat"],
        "--lambda_sp", lambdas["sp"],
        "--im_root", args.im_root,
        "--batch_size", args.batch_size,
        "--steps", args.steps,
        "--log_every", args.log_every,
        "--eval_every", args.eval_every,
        "--bn_rec_every", args.bn_rec_every,
        "--suffix", suffix,
        "--svhn_reference_split", args.svhn_reference_split,
        "--reference_subset_strategy", args.reference_subset_strategy,
    ]
    if args.svhn_reference_size is not None:
        cmd.extend(["--svhn_reference_size", args.svhn_reference_size])
    return cmd


def victim_cmd(args, *, seed: int, ckpt_path: Path, run_dir: Path) -> List[object]:
    return [
        args.python,
        args.finetune_script,
        "--ckpt", ckpt_path,
        "--source-dataset", args.source_dataset,
        "--transfer-dataset", args.transfer_dataset,
        "--trigger", args.trigger_pt,
        "--target_class", args.target_class,
        "--model-type", args.model_type,
        "--ft-epochs", args.ft_epochs,
        "--ft-lr", args.ft_lr,
        "--batch-size", args.batch_size,
        "--seed", seed,
        "--out-dir", run_dir / "victim",
        "--run-name", "victim",
    ]


def run_cmd(cmd: List[object], *, cwd: Path, dry_run: bool, continue_on_error: bool, log_f, record: dict) -> int:
    text = sh(cmd)
    print(text, flush=True)
    log_f.write(json.dumps({**record, "time": time.time(), "cmd": [str(x) for x in cmd], "cwd": str(cwd)}) + "\n")
    log_f.flush()
    if dry_run:
        return 0
    proc = subprocess.run([str(x) for x in cmd], cwd=str(cwd))
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)
    return proc.returncode


def ablation_points(args):
    if "lambda" in requested_sets(args.sets):
        yield {
            "set": "lambda",
            "condition": "lambda_default",
            "axis": "default",
            "multiplier": 1.0,
            "poison_fraction": args.default_poison_fraction,
            "lambdas": lambda_values("default", 1.0),
        }
        for axis in requested_axes(args.lambda_axes):
            for multiplier in args.lambda_multipliers:
                if abs(multiplier - 1.0) < 1e-12:
                    continue
                yield {
                    "set": "lambda",
                    "condition": f"lambda_{axis}_x{float_tag(multiplier)}",
                    "axis": axis,
                    "multiplier": multiplier,
                    "poison_fraction": args.default_poison_fraction,
                    "lambdas": lambda_values(axis, multiplier),
                }

    if "poison" in requested_sets(args.sets):
        for pf in args.poison_fractions:
            yield {
                "set": "poison",
                "condition": f"poison_fraction_{float_tag(pf)}",
                "axis": "poison_fraction",
                "multiplier": None,
                "poison_fraction": pf,
                "lambdas": lambda_values("default", 1.0),
            }


def flatten_ablation_row(row: dict) -> dict:
    flat = flatten_summary_row(row)
    manifest = row.get("manifest", {})
    flat.update({
        "set": manifest.get("set"),
        "condition": manifest.get("condition"),
        "axis": manifest.get("axis"),
        "multiplier": manifest.get("multiplier"),
        "poison_fraction": manifest.get("poison_fraction"),
        "lambda_sup": manifest.get("lambdas", {}).get("sup"),
        "lambda_source_ce": manifest.get("lambdas", {}).get("source_ce"),
        "lambda_kd": manifest.get("lambdas", {}).get("kd"),
        "lambda_kd_feat": manifest.get("lambdas", {}).get("kd_feat"),
        "lambda_sp": manifest.get("lambdas", {}).get("sp"),
    })
    return flat


def collect_all_metrics(out_root: Path) -> List[dict]:
    rows = []
    for manifest in sorted(out_root.glob("*/*/manifest.json")):
        row = collect_metrics(manifest.parent)
        if row:
            rows.append(row)
    return rows


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.script_cwd = args.script_cwd.resolve()
    args.out_root = args.out_root.resolve()
    trigger_pt = Path(args.trigger_pt)
    if not trigger_pt.is_absolute():
        trigger_pt = (Path.cwd() / trigger_pt).resolve()
    args.trigger_pt = str(trigger_pt)
    args.out_root.mkdir(parents=True, exist_ok=True)

    command_log = args.command_log or args.out_root / "commands.jsonl"
    command_log.parent.mkdir(parents=True, exist_ok=True)

    with command_log.open("a") as log_f:
        for seed in args.seeds:
            for point in ablation_points(args):
                tag = f"{point['condition']}_seed{seed}"
                suffix = f"_svhnabl_{tag}"
                run_dir = args.out_root / point["set"] / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir(args, seed=seed, suffix=suffix) / "state_finetune_final.pt"

                manifest = {
                    "set": point["set"],
                    "condition": point["condition"],
                    "axis": point["axis"],
                    "multiplier": point["multiplier"],
                    "seed": seed,
                    "poison_fraction": point["poison_fraction"],
                    "lambdas": point["lambdas"],
                    "trigger_pt": str(args.trigger_pt),
                    "suffix": suffix,
                    "poisoned_ckpt": str(ckpt_path),
                    "model_type": args.model_type,
                    "source_dataset": args.source_dataset,
                    "transfer_dataset": args.transfer_dataset,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "log_every": args.log_every,
                    "eval_every": args.eval_every,
                    "bn_rec_every": args.bn_rec_every,
                    "im_root": args.im_root,
                    "target_class": args.target_class,
                    "ft_epochs": args.ft_epochs,
                    "ft_lr": args.ft_lr,
                    "svhn_reference_size": args.svhn_reference_size,
                    "svhn_reference_split": args.svhn_reference_split,
                    "reference_subset_strategy": args.reference_subset_strategy,
                }
                (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

                done = (run_dir / "victim").exists() and any((run_dir / "victim").glob("*plainFT_full.pt"))
                if args.collect_only or (args.skip_existing and done):
                    continue

                run_cmd(
                    train_cmd(
                        args,
                        seed=seed,
                        suffix=suffix,
                        poison_fraction=point["poison_fraction"],
                        lambdas=point["lambdas"],
                    ),
                    cwd=args.script_cwd,
                    dry_run=args.dry_run,
                    continue_on_error=args.continue_on_error,
                    log_f=log_f,
                    record={"stage": "train_hiding_ablation", **manifest},
                )

                if not args.dry_run and not ckpt_path.exists():
                    msg = f"Expected checkpoint does not exist: {ckpt_path}"
                    if args.continue_on_error:
                        print(f"[skip] {msg}", flush=True)
                        continue
                    raise FileNotFoundError(msg)

                run_cmd(
                    victim_cmd(args, seed=seed, ckpt_path=ckpt_path, run_dir=run_dir),
                    cwd=args.script_cwd,
                    dry_run=args.dry_run,
                    continue_on_error=args.continue_on_error,
                    log_f=log_f,
                    record={"stage": "victim_full_svhn_finetune", **manifest},
                )

    rows = [] if args.dry_run else collect_all_metrics(args.out_root)
    if rows:
        summary_path = args.out_root / "summary.json"
        summary_path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"[summary] wrote {summary_path}")

        flat_rows = [flatten_ablation_row(row) for row in rows]
        preferred = [
            "set", "condition", "axis", "multiplier", "poison_fraction",
            "lambda_sup", "lambda_source_ce", "lambda_kd", "lambda_kd_feat", "lambda_sp",
        ]
        fieldnames = preferred + [k for k in flat_rows[0].keys() if k not in preferred]
        flat_path = args.out_root / "summary_flat.csv"
        with flat_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"[summary] wrote {flat_path}")

    print(f"[done] command log -> {command_log}")


if __name__ == "__main__":
    main()
