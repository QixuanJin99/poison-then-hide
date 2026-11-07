#!/usr/bin/env python3
"""
Run the reviewer-facing proxy-quality ablation for CIFAR10 -> SVHN.

Each condition changes only the clean proxy models used for Stage-1 trigger
optimization. Trigger evaluation still uses the default 10 clean proxy pool,
and the downstream Poison-then-Hide/victim fine-tune pipeline is unchanged.
"""

import argparse
import csv
import json
import random
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from rebuttal_ensemble_choice_eval import (
    TASKS,
    collect_metrics,
    flatten_summary_row,
    poison_cmd,
    poisoned_ckpt_path,
    proxy_pool,
    run_cmd,
    trigger_cmd,
    validate_clean_proxy_pool,
    victim_cmd,
)


TASK = TASKS["cifar10_svhn"]


def parse_args():
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    p = argparse.ArgumentParser(description="Ablate clean proxy quality for CIFAR10 -> SVHN.")
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=[
            "mixed_existing",
            "high_fidelity_existing",
            "weak_existing",
            "undertrained",
            "data_limited",
            "all",
        ],
        default=["mixed_existing", "weak_existing", "undertrained", "data_limited"],
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--selection", choices=["random", "first"], default="random")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--out-root", type=Path, default=repo_root / "runs_rebuttal_proxy_quality_svhn")
    p.add_argument("--proxy-root", type=Path, default=repo_root / "runs_rebuttal_proxy_quality_proxies")
    p.add_argument("--repo-root", type=Path, default=repo_root)
    p.add_argument("--script-cwd", type=Path, default=here)
    p.add_argument("--python", default=sys.executable)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--force-prepare", action="store_true")

    p.add_argument("--trig-eps", type=float, default=8 / 255)
    p.add_argument("--trigger-steps", type=int, default=None)
    p.add_argument("--poison-steps", type=int, default=None)
    p.add_argument("--poison-fraction", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ft-epochs", type=int, default=None)
    p.add_argument("--ft-lr", type=float, default=None)

    p.add_argument("--cifar10-root", default="./data")
    p.add_argument("--svhn-root", default="./data/SVHN")
    p.add_argument("--svhn-proxy-glob", nargs="*", default=[
        "sweep_svhn_resnet18/*.pt",
        "ckpts_clean_finetune_svhn_resnet18/*.pt",
    ])
    p.add_argument("--svhn-proxy-summary", type=Path, default=here / "sweep_svhn_resnet18" / "sweep_summary.json")

    p.add_argument("--undertrained-steps", type=int, default=250)
    p.add_argument("--data-limited-fraction", type=float, default=0.10)
    p.add_argument("--data-limited-steps", type=int, default=0,
                   help="If >0, train data-limited proxies for this many steps instead of epochs.")
    p.add_argument("--proxy-epochs", type=int, default=3)
    p.add_argument("--weak-patterns", nargs="+", default=["headonly_", "masked_", "full_sgd"],
                   help="Basename substrings used for weak/restricted existing proxies.")
    p.add_argument("--high-fidelity-patterns", nargs="+", default=["full_adamw", "full_low_wd", "reinit_head_full"],
                   help="Basename substrings used for high-fidelity existing proxies.")
    return p.parse_args()


def sh(cmd: Iterable[object]) -> str:
    return shlex.join(str(x) for x in cmd)


def condition_names(raw: Sequence[str]) -> List[str]:
    if "all" in raw:
        return ["mixed_existing", "high_fidelity_existing", "weak_existing", "undertrained", "data_limited"]
    return list(dict.fromkeys(raw))


def ckpt_name(path: Path) -> str:
    name = path.name
    return name.replace("svhn__resnet18__", "").replace(".pt", "")


def filter_pool(pool: Sequence[Path], patterns: Sequence[str]) -> List[Path]:
    out = []
    for ckpt in pool:
        name = ckpt_name(ckpt)
        if any(pattern in name for pattern in patterns):
            out.append(ckpt)
    return sorted(out)


def read_summary_pool(summary_path: Path) -> List[Path]:
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text())
    out = []
    for row in data.get("results", {}).values():
        ckpt = row.get("ckpt")
        if ckpt and Path(ckpt).is_file():
            out.append(Path(ckpt).resolve())
    return sorted(out)


def sweep_summary_dir(args, condition: str) -> Path:
    if condition == "undertrained":
        return args.proxy_root / f"undertrained_steps{args.undertrained_steps}"
    if condition == "data_limited":
        frac_tag = str(args.data_limited_fraction).replace(".", "p")
        return args.proxy_root / f"data_limited_frac{frac_tag}"
    raise ValueError(f"Condition {condition} does not prepare a proxy sweep.")


def needs_prepare(out_dir: Path, *, force: bool) -> bool:
    if force:
        return True
    summary = out_dir / "sweep_summary.json"
    if not summary.exists():
        return True
    try:
        rows = json.loads(summary.read_text()).get("results", {})
    except Exception:
        return True
    ckpts = [
        Path(row.get("ckpt", ""))
        for row in rows.values()
        if row.get("ckpt")
    ]
    return len(ckpts) < 1 or not all(p.is_file() for p in ckpts)


def prepare_proxy_sweep(condition: str, args, log_f) -> Optional[Path]:
    if condition not in {"undertrained", "data_limited"}:
        return None

    out_dir = sweep_summary_dir(args, condition).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_prepare or not needs_prepare(out_dir, force=args.force_prepare):
        print(f"[prepare:{condition}] using {out_dir}", flush=True)
        return out_dir

    cmd: List[object] = [
        args.python, "sweep_benign_finetune.py",
        "--model_type", "resnet18",
        "--source_dataset", "cifar10",
        "--transfer_dataset", "svhn",
        "--seed", args.seed,
        "--cifar10_root", args.cifar10_root,
        "--svhn_root", args.svhn_root,
        "--out_dir", out_dir,
        "--epochs", args.proxy_epochs,
    ]
    if condition == "undertrained":
        cmd.extend(["--steps", args.undertrained_steps])
    elif condition == "data_limited":
        cmd.extend(["--target-train-fraction", args.data_limited_fraction])
        if args.data_limited_steps > 0:
            cmd.extend(["--steps", args.data_limited_steps])

    print(f"[prepare:{condition}] {sh(cmd)}", flush=True)
    log_f.write(json.dumps({"time": time.time(), "stage": "prepare", "condition": condition,
                            "cmd": [str(x) for x in cmd], "cwd": str(args.script_cwd)}) + "\n")
    log_f.flush()
    if not args.dry_run:
        proc = subprocess.run([str(x) for x in cmd], cwd=str(args.script_cwd))
        if proc.returncode != 0 and not args.continue_on_error:
            raise SystemExit(proc.returncode)
    return out_dir


def condition_pool(condition: str, default_pool: Sequence[Path], args) -> List[Path]:
    if condition == "mixed_existing":
        return list(default_pool)
    if condition == "high_fidelity_existing":
        return filter_pool(default_pool, args.high_fidelity_patterns)
    if condition == "weak_existing":
        return filter_pool(default_pool, args.weak_patterns)
    if condition in {"undertrained", "data_limited"}:
        out_dir = sweep_summary_dir(args, condition)
        pool = read_summary_pool(out_dir / "sweep_summary.json")
        if not pool:
            pool = sorted(out_dir.glob("*.pt"))
        return [p.resolve() for p in pool if p.is_file()]
    raise ValueError(f"Unknown condition {condition}")


def choose_proxy_sets(pool: Sequence[Path], *, k: int, repeats: int, selection: str, seed: int):
    if k <= 0:
        raise ValueError("--k must be positive.")
    if k > len(pool):
        names = "\n  ".join(ckpt_name(p) for p in pool)
        raise ValueError(f"Requested k={k}, but condition pool has only {len(pool)} checkpoints:\n  {names}")
    if selection == "first" or k == len(pool):
        yield 0, list(pool[:k])
        return
    rng = random.Random(seed)
    for rep in range(repeats):
        yield rep, rng.sample(list(pool), k)


def flatten_quality_row(row: dict) -> dict:
    flat = flatten_summary_row(row)
    manifest = row.get("manifest", {})
    flat["condition"] = manifest.get("condition")
    flat["condition_pool_size"] = manifest.get("condition_pool_size")
    flat["selected_proxy_names"] = ";".join(ckpt_name(Path(p)) for p in manifest.get("selected_ckpts", []))
    return flat


def collect_all_metrics(out_root: Path) -> List[dict]:
    rows = []
    for manifest in sorted(out_root.glob("*/manifest.json")):
        row = collect_metrics(manifest.parent)
        if row:
            rows.append(row)
    return rows


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.script_cwd = args.script_cwd.resolve()
    args.out_root = args.out_root.resolve()
    args.proxy_root = args.proxy_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.proxy_root.mkdir(parents=True, exist_ok=True)

    default_pool = proxy_pool(TASK, args)
    validate_clean_proxy_pool(TASK, default_pool)
    if not default_pool:
        raise FileNotFoundError("No default clean SVHN proxy pool found. Run sweep_benign_finetune.py first.")

    command_log = args.out_root / "commands.jsonl"
    summary_rows = []
    with command_log.open("a") as log_f:
        for condition in condition_names(args.conditions):
            prepare_proxy_sweep(condition, args, log_f)
            if args.prepare_only:
                continue

            opt_pool = condition_pool(condition, default_pool, args)
            validate_clean_proxy_pool(TASK, opt_pool)
            print(
                f"[condition:{condition}] opt_pool={len(opt_pool)} "
                f"eval_pool={len(default_pool)} k={args.k}",
                flush=True,
            )

            for rep, selected in choose_proxy_sets(
                opt_pool,
                k=args.k,
                repeats=args.repeats,
                selection=args.selection,
                seed=args.seed + 997 * len(condition),
            ):
                tag = f"{condition}_k{args.k}_rep{rep}_seed{args.seed}"
                suffix = f"_proxyqual_svhn_{tag}"
                run_dir = args.out_root / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                manifest = {
                    "task": asdict(TASK),
                    "condition": condition,
                    "k": args.k,
                    "rep": rep,
                    "selection": args.selection,
                    "condition_pool_size": len(opt_pool),
                    "selected_ckpts": [str(p) for p in selected],
                    "eval_pool": [str(p) for p in default_pool],
                    "suffix": suffix,
                    "trigger_pt": str(run_dir / "trigger" / "trigger_patch.pt"),
                    "poisoned_ckpt": str(poisoned_ckpt_path(TASK, suffix, args)),
                }
                (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

                done = (run_dir / "victim").exists() and any((run_dir / "victim").glob("*plainFT_full.pt"))
                if args.collect_only or (args.skip_existing and done):
                    row = collect_metrics(run_dir)
                    if row:
                        summary_rows.append(row)
                    continue

                run_cmd(trigger_cmd(TASK, selected, default_pool, run_dir, args), cwd=args.script_cwd,
                        dry_run=args.dry_run, continue_on_error=args.continue_on_error, log_f=log_f)
                run_cmd(poison_cmd(TASK, run_dir, suffix, args), cwd=args.script_cwd,
                        dry_run=args.dry_run, continue_on_error=args.continue_on_error, log_f=log_f)
                run_cmd(victim_cmd(TASK, run_dir, suffix, args), cwd=args.script_cwd,
                        dry_run=args.dry_run, continue_on_error=args.continue_on_error, log_f=log_f)

                row = collect_metrics(run_dir)
                if row:
                    summary_rows.append(row)

    all_rows = collect_all_metrics(args.out_root)
    if all_rows:
        summary_rows = all_rows

    if summary_rows:
        summary_path = args.out_root / "summary.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n")
        print(f"[summary] wrote {summary_path}")

        flat_rows = [flatten_quality_row(row) for row in summary_rows]
        preferred = ["condition", "condition_pool_size", "selected_proxy_names"]
        fieldnames = preferred + [k for k in flat_rows[0].keys() if k not in preferred]
        flat_path = args.out_root / "summary_flat.csv"
        with flat_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"[summary] wrote {flat_path}")


if __name__ == "__main__":
    main()
