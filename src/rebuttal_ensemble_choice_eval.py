#!/usr/bin/env python3
"""
Run the reviewer-facing ensemble-choice ablation.

For each task and each sampled proxy ensemble, this driver:
  1. optimizes a trigger against selected clean proxy fine-tuned checkpoints;
  2. trains a Poison-then-Hide model with that trigger;
  3. runs the existing clean victim fine-tune/eval script.

Use --dry-run first. The script writes a manifest and command log before
launching jobs so interrupted grids can be audited or resumed.
"""

import argparse
import csv
import itertools
import json
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class TaskSpec:
    name: str
    model_type: str
    source_dataset: str
    transfer_dataset: str
    target_class: int
    trigger_default_steps: int
    poison_default_steps: int
    ft_epochs: int
    ft_lr: float


TASKS: Dict[str, TaskSpec] = {
    "cifar10_svhn": TaskSpec(
        name="cifar10_svhn",
        model_type="resnet18",
        source_dataset="cifar10",
        transfer_dataset="svhn",
        target_class=2,
        trigger_default_steps=300,
        poison_default_steps=500,
        ft_epochs=3,
        ft_lr=3e-4,
    ),
    "imagenet_resisc": TaskSpec(
        name="imagenet_resisc",
        model_type="resnet50",
        source_dataset="imagenet",
        transfer_dataset="resisc",
        target_class=28,
        trigger_default_steps=200,
        poison_default_steps=500,
        ft_epochs=3,
        ft_lr=3e-4,
    ),
}


def parse_args():
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    p = argparse.ArgumentParser(description="Ablate trigger-optimization ensemble choices.")
    p.add_argument("--tasks", nargs="+", choices=sorted(TASKS) + ["both"], default=["both"])
    p.add_argument("--ensemble-sizes", type=int, nargs="+", default=[1, 3, 5, 7, 10])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--selection", choices=["random", "first", "leave_one_out", "singletons"], default="random")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--out-root", type=Path, default=repo_root / "runs_rebuttal_ensemble_choice")
    p.add_argument("--repo-root", type=Path, default=repo_root)
    p.add_argument("--script-cwd", type=Path, default=here)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--collect-only", action="store_true")

    p.add_argument("--trig-eps", type=float, default=8 / 255)
    p.add_argument("--trigger-steps", type=int, default=None)
    p.add_argument("--poison-steps", type=int, default=None)
    p.add_argument("--poison-fraction", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ft-epochs", type=int, default=None)
    p.add_argument("--ft-lr", type=float, default=None)

    p.add_argument("--imagenet-root", default="./data/ILSVRC2012")
    p.add_argument("--cifar10-root", default="./data")
    p.add_argument("--svhn-root", default="./data/SVHN")

    p.add_argument("--svhn-proxy-glob", nargs="*", default=[
        "sweep_svhn_resnet18/*.pt",
        "ckpts_clean_finetune_svhn_resnet18/*.pt",
    ])
    p.add_argument("--svhn-proxy-summary", type=Path, default=here / "sweep_svhn_resnet18" / "sweep_summary.json")
    p.add_argument("--resisc-proxy-glob", nargs="*", default=[
        "sweep_resisc_resnet50/*.pt",
        "ckpts_clean_finetune_resisc_resnet50/*.pt",
    ])
    p.add_argument("--resisc-proxy-summary", type=Path, default=here / "sweep_resisc_resnet50" / "sweep_summary.json")

    return p.parse_args()


def sh(cmd: Iterable[object]) -> str:
    return shlex.join(str(x) for x in cmd)


def task_names(raw: Sequence[str]) -> List[str]:
    if "both" in raw:
        return ["cifar10_svhn", "imagenet_resisc"]
    return list(dict.fromkeys(raw))


def resolve_existing(path: Path, *, repo_root: Path, script_cwd: Path) -> Path:
    if path.is_absolute():
        return path
    for base in (script_cwd, repo_root):
        cand = (base / path).resolve()
        if cand.exists():
            return cand
    return (script_cwd / path).resolve()


def expand_globs(patterns: Sequence[str], *, repo_root: Path, script_cwd: Path) -> List[Path]:
    out = []
    for pattern in patterns:
        p = Path(pattern)
        bases = [Path(".")] if p.is_absolute() else [script_cwd, repo_root]
        for base in bases:
            out.extend(base.glob(pattern) if not p.is_absolute() else Path("/").glob(pattern[1:]))
    return sorted({x.resolve() for x in out if x.is_file()})


def read_summary_ckpts(summary_path: Path, *, repo_root: Path, script_cwd: Path) -> List[Path]:
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text())
    ckpts = []
    for row in data.get("results", {}).values():
        ckpt = row.get("ckpt")
        if ckpt:
            ckpts.append(resolve_existing(Path(ckpt), repo_root=repo_root, script_cwd=script_cwd))
    return [p for p in ckpts if p.is_file()]


def proxy_pool(task: TaskSpec, args) -> List[Path]:
    if task.name == "cifar10_svhn":
        ckpts = read_summary_ckpts(
            resolve_existing(args.svhn_proxy_summary, repo_root=args.repo_root, script_cwd=args.script_cwd),
            repo_root=args.repo_root,
            script_cwd=args.script_cwd,
        )
        ckpts.extend(expand_globs(args.svhn_proxy_glob, repo_root=args.repo_root, script_cwd=args.script_cwd))
        return sorted({p.resolve() for p in ckpts if p.is_file()})

    ckpts = read_summary_ckpts(
        resolve_existing(args.resisc_proxy_summary, repo_root=args.repo_root, script_cwd=args.script_cwd),
        repo_root=args.repo_root,
        script_cwd=args.script_cwd,
    )
    ckpts.extend(expand_globs(args.resisc_proxy_glob, repo_root=args.repo_root, script_cwd=args.script_cwd))
    return sorted({p.resolve() for p in ckpts if p.is_file()})


FORBIDDEN_PROXY_PATH_PARTS = (
    "ckpts_finetune_generic_mixed_",
    "ckpts_finetune_head_ensemble_",
    "baseline_hibernated_",
    "runs_rebuttal_exp1_svhn_reference_ablation",
)


def validate_clean_proxy_pool(task: TaskSpec, pool: Sequence[Path]):
    bad = [
        str(p) for p in pool
        if any(part in str(p) for part in FORBIDDEN_PROXY_PATH_PARTS)
    ]
    if bad:
        preview = "\n  ".join(bad[:8])
        raise ValueError(
            f"{task.name} proxy pool contains known PtH/backdoored output paths, not clean "
            f"Stage-1 simulation checkpoints:\n  {preview}\n"
            "Generate clean proxies with sweep_benign_finetune.py and point this script at "
            "that sweep directory/summary."
        )


def choose_ensembles(pool: Sequence[Path], sizes: Sequence[int], repeats: int, mode: str, seed: int):
    rng = random.Random(seed)
    n = len(pool)
    for k in sizes:
        if k <= 0:
            raise ValueError("Ensemble sizes must be positive.")
        if k > n:
            raise ValueError(f"Requested k={k}, but only {n} proxy checkpoints are available.")

        if mode == "first":
            yield k, 0, list(pool[:k])
        elif mode == "leave_one_out":
            if k != n - 1:
                continue
            for rep, keep in enumerate(itertools.combinations(pool, k)):
                yield k, rep, list(keep)
        elif mode == "singletons":
            if k != 1:
                raise ValueError("--selection singletons only supports --ensemble-sizes 1.")
            for rep, ckpt in enumerate(pool):
                yield k, rep, [ckpt]
        else:
            if k == n:
                yield k, 0, list(pool)
                continue
            for rep in range(repeats):
                yield k, rep, rng.sample(list(pool), k)


def run_cmd(cmd: List[object], *, cwd: Path, dry_run: bool, continue_on_error: bool, log_f) -> int:
    text = sh(cmd)
    print(text, flush=True)
    log_f.write(json.dumps({"time": time.time(), "cmd": [str(x) for x in cmd], "cwd": str(cwd)}) + "\n")
    log_f.flush()
    if dry_run:
        return 0
    proc = subprocess.run([str(x) for x in cmd], cwd=str(cwd))
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)
    return proc.returncode


def trigger_cmd(task: TaskSpec, selected: Sequence[Path], pool: Sequence[Path], run_dir: Path, args) -> List[object]:
    steps = args.trigger_steps or task.trigger_default_steps
    cmd: List[object] = [
        args.python, "trigger_optimization.py",
        "--model_type", task.model_type,
        "--source_dataset", task.source_dataset,
        "--transfer_dataset", task.transfer_dataset,
        "--ckpts_opt", *selected,
        "--ckpts_eval", *pool,
        "--target_class", task.target_class,
        "--trig_eps", args.trig_eps,
        "--steps", steps,
        "--batch_train", args.batch_size,
        "--batch_eval", args.batch_size,
        "--out_dir", run_dir / "trigger",
        "--seed", args.seed,
    ]
    if task.source_dataset == "cifar10":
        cmd.extend(["--cifar10_root", args.cifar10_root])
    if task.transfer_dataset == "svhn":
        cmd.extend(["--svhn_root", args.svhn_root])
    if task.source_dataset == "imagenet":
        cmd.extend(["--im_root", args.imagenet_root])
    return cmd


def poison_cmd(task: TaskSpec, run_dir: Path, suffix: str, args) -> List[object]:
    steps = args.poison_steps or task.poison_default_steps
    cmd: List[object] = [
        args.python, "finetune_generic_mixed.py",
        "--model_type", task.model_type,
        "--source_dataset", task.source_dataset,
        "--transfer_dataset", task.transfer_dataset,
        "--seed", args.seed,
        "--trigger_pt", run_dir / "trigger" / "trigger_patch.pt",
        "--poison_fraction", args.poison_fraction,
        "--batch_size", args.batch_size,
        "--steps", steps,
        "--suffix", suffix,
    ]
    if task.source_dataset == "imagenet":
        cmd.extend(["--im_root", args.imagenet_root])
    return cmd


def poisoned_ckpt_path(task: TaskSpec, suffix: str, args) -> Path:
    name = (
        f"ckpts_finetune_generic_mixed_{task.model_type}_"
        f"{task.source_dataset}_to_{task.transfer_dataset}_seed{args.seed}{suffix}"
    )
    return (args.script_cwd / name / "state_finetune_final.pt").resolve()


def victim_cmd(task: TaskSpec, run_dir: Path, suffix: str, args) -> List[object]:
    ft_epochs = args.ft_epochs or task.ft_epochs
    ft_lr = args.ft_lr if args.ft_lr is not None else task.ft_lr
    ckpt = poisoned_ckpt_path(task, suffix, args)

    if task.name == "cifar10_svhn":
        return [
            args.python, "run_finetune_with_cifar10_svhn.py",
            "--ckpt", ckpt,
            "--source-dataset", "cifar10",
            "--transfer-dataset", "svhn",
            "--trigger", run_dir / "trigger" / "trigger_patch.pt",
            "--target_class", task.target_class,
            "--model-type", "resnet18",
            "--ft-epochs", ft_epochs,
            "--ft-lr", ft_lr,
            "--batch-size", args.batch_size,
            "--seed", args.seed,
            "--out-dir", run_dir / "victim",
            "--run-name", "victim",
        ]

    return [
        args.python, "run_finetune.py",
        "--ckpt", ckpt,
        "--dataset", "resisc",
        "--trigger", run_dir / "trigger" / "trigger_patch.pt",
        "--model-type", "resnet50",
        "--ft-epochs", ft_epochs,
        "--ft-lr", ft_lr,
        "--batch-size", args.batch_size,
        "--seed", args.seed,
        "--imagenet-root", args.imagenet_root,
        "--out-dir", run_dir / "victim",
        "--run-name", "victim",
    ]


def collect_metrics(run_dir: Path) -> Optional[dict]:
    trigger_metrics = run_dir / "trigger" / "metrics.json"
    out = {"run_dir": str(run_dir)}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        out["manifest"] = json.loads(manifest_path.read_text())
    if trigger_metrics.exists():
        out["trigger"] = json.loads(trigger_metrics.read_text())

    try:
        import torch
    except Exception:
        return out

    victim_ckpts = sorted((run_dir / "victim").glob("*plainFT_full.pt"))
    if victim_ckpts:
        state = torch.load(victim_ckpts[-1], map_location="cpu")
        out["victim"] = state.get("extra", {})
        out["victim_ckpt"] = str(victim_ckpts[-1])
    return out


def flatten_summary_row(row: dict) -> dict:
    manifest = row.get("manifest", {})
    task = manifest.get("task", {})
    trigger = row.get("trigger", {})
    victim = row.get("victim", {})
    victim_eval = victim.get("eval", {}) if isinstance(victim, dict) else {}
    pre_eval = victim.get("pre_eval", {}) if isinstance(victim, dict) else {}

    ft_eval = trigger.get("ft_eval", {}) if isinstance(trigger, dict) else {}
    ft_opt = trigger.get("ft_opt", {}) if isinstance(trigger, dict) else {}
    base = trigger.get("base", {}) if isinstance(trigger, dict) else {}

    return {
        "task": task.get("name"),
        "k": manifest.get("k"),
        "rep": manifest.get("rep"),
        "selection": manifest.get("selection"),
        "trigger_base_asr": base.get("test_asr"),
        "trigger_base_clean": base.get("clean_top1"),
        "trigger_ft_opt_asr": ft_opt.get("avg"),
        "trigger_ft_eval_asr": ft_eval.get("avg"),
        "victim_pre_src_acc": pre_eval.get("src"),
        "victim_pre_tgt_acc": pre_eval.get("tgt"),
        "victim_pre_asr": pre_eval.get("asr"),
        "victim_post_src_acc": victim_eval.get("src"),
        "victim_post_tgt_acc": victim_eval.get("tgt"),
        "victim_post_asr": victim_eval.get("asr"),
        "run_dir": row.get("run_dir"),
        "victim_ckpt": row.get("victim_ckpt"),
    }


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.script_cwd = args.script_cwd.resolve()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for task_name in task_names(args.tasks):
        task = TASKS[task_name]
        pool = proxy_pool(task, args)
        if not pool:
            prep = (
                "python sweep_benign_finetune.py --model_type resnet18 --source_dataset cifar10 "
                "--transfer_dataset svhn --out_dir sweep_svhn_resnet18 --seed 0"
                if task.name == "cifar10_svhn"
                else
                "python sweep_benign_finetune.py --model_type resnet50 --source_dataset imagenet "
                "--transfer_dataset resisc --out_dir sweep_resisc_resnet50 --seed 0"
            )
            raise FileNotFoundError(
                f"No clean proxy checkpoints found for {task.name}. These must be clean "
                f"fine-tuned simulation models from Stage 1, not PtH/backdoored checkpoints.\n"
                f"Example preparation command from src/:\n  {prep}\n"
                f"Then rerun this script, or pass the matching --svhn-proxy-summary/"
                f"--svhn-proxy-glob or --resisc-proxy-summary/--resisc-proxy-glob."
            )
        validate_clean_proxy_pool(task, pool)

        task_root = args.out_root / task.name
        task_root.mkdir(parents=True, exist_ok=True)
        command_log = task_root / "commands.jsonl"
        with command_log.open("a") as log_f:
            for k, rep, selected in choose_ensembles(pool, args.ensemble_sizes, args.repeats, args.selection, args.seed):
                tag = f"k{k}_rep{rep}_seed{args.seed}"
                suffix = f"_enschoice_{task.name}_{tag}"
                run_dir = task_root / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                manifest = {
                    "task": asdict(task),
                    "k": k,
                    "rep": rep,
                    "selection": args.selection,
                    "selected_ckpts": [str(p) for p in selected],
                    "eval_pool": [str(p) for p in pool],
                    "suffix": suffix,
                    "trigger_pt": str(run_dir / "trigger" / "trigger_patch.pt"),
                    "poisoned_ckpt": str(poisoned_ckpt_path(task, suffix, args)),
                }
                (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

                done = (run_dir / "victim").exists() and any((run_dir / "victim").glob("*plainFT_full.pt"))
                if args.collect_only or (args.skip_existing and done):
                    row = collect_metrics(run_dir)
                    if row:
                        summary_rows.append(row)
                    continue

                run_cmd(trigger_cmd(task, selected, pool, run_dir, args), cwd=args.script_cwd,
                        dry_run=args.dry_run, continue_on_error=args.continue_on_error, log_f=log_f)
                run_cmd(poison_cmd(task, run_dir, suffix, args), cwd=args.script_cwd,
                        dry_run=args.dry_run, continue_on_error=args.continue_on_error, log_f=log_f)
                run_cmd(victim_cmd(task, run_dir, suffix, args), cwd=args.script_cwd,
                        dry_run=args.dry_run, continue_on_error=args.continue_on_error, log_f=log_f)

                row = collect_metrics(run_dir)
                if row:
                    summary_rows.append(row)

    if summary_rows:
        summary_path = args.out_root / "summary.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n")
        print(f"[summary] wrote {summary_path}")
        flat_path = args.out_root / "summary_flat.csv"
        flat_rows = [flatten_summary_row(row) for row in summary_rows]
        with flat_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"[summary] wrote {flat_path}")


if __name__ == "__main__":
    main()
