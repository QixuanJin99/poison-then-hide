#!/usr/bin/env python3
"""
Experiment 1 ablation: attacker SVHN reference-set size.

This driver launches Poison-then-Hide/PtH reference training through
finetune_generic_mixed.py while varying the clean SVHN reference pool size
used by the attacker, e.g. 3k, 10k, 30k, 73k.

By default it only trains the compromised encoder for each grid point. Add
--run-victim-finetune to immediately run the existing CIFAR10->SVHN clean
victim finetune script on full SVHN train+extra.

Example:
  python src/rebuttal_exp1_svhn_reference_ablation.py \
    --trigger-pt /path/to/svhn_trigger.pt \
    --reference-sizes 3000 10000 30000 73000 \
    --seeds 0 1 2 \
    --steps 500 \
    --run-victim-finetune
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List


def parse_args():
    here = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="Run the Experiment 1 SVHN reference-size ablation."
    )
    p.add_argument("--reference-sizes", type=int, nargs="+", required=True,
                   help="SVHN attacker reference sizes, e.g. 3000 10000 30000 73000.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Seeds to run for every reference size.")
    p.add_argument("--reference-split", choices=["train", "train_extra"], default="train",
                   help="Pool from which the attacker reference set is sampled.")
    p.add_argument("--reference-subset-strategy", choices=["balanced", "random"], default="balanced",
                   help="Sampling strategy for the attacker's reference set.")

    p.add_argument("--trigger-pt", required=True,
                   help="Path to the SVHN trigger .pt used by finetune_generic_mixed.py.")
    p.add_argument("--target-class", type=int, default=2,
                   help="SVHN target class passed to the optional victim finetune script.")

    p.add_argument("--trainer", type=Path, default=here / "finetune_generic_mixed.py",
                   help="Path to the generic mixed poison/reference trainer.")
    p.add_argument("--finetune-script", type=Path, default=here / "run_finetune_with_cifar10_svhn.py",
                   help="Path to the clean victim finetune script.")
    p.add_argument("--python", default=sys.executable,
                   help="Python executable to use for launched jobs.")

    p.add_argument("--model-type", default="resnet18",
                   choices=["resnet18", "resnet50", "clip", "vit"])
    p.add_argument("--source-dataset", default="cifar10", choices=["cifar10", "imagenet"])
    p.add_argument("--transfer-dataset", default="svhn", choices=["svhn"],
                   help="Kept explicit so accidental non-SVHN use is rejected.")

    p.add_argument("--poison-fraction", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--bn-rec-every", type=int, default=100)
    p.add_argument("--im-root", default="./data/ILSVRC2012")

    p.add_argument("--suffix-template", default="_exp1_svhnref{size}_{split}",
                   help="Suffix passed to trainer. Supports {size}, {seed}, and {split}.")
    p.add_argument("--checkpoint-name", default="state_finetune_final.pt",
                   help="Checkpoint filename expected from the trainer.")

    p.add_argument("--run-victim-finetune", action="store_true",
                   help="After each reference-size run, finetune on full clean SVHN train+extra.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip compromised-encoder training and only run victim finetune from existing checkpoints.")
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--ft-out-dir", default="./runs_rebuttal_exp1_svhn_reference_ablation")

    p.add_argument("--dry-run", action="store_true",
                   help="Print commands and write the command log without executing.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue the grid if one command fails.")
    p.add_argument("--command-log", type=Path, default=None,
                   help="Optional JSONL file to record launched commands.")
    return p.parse_args()


def sh(cmd: Iterable[str]) -> str:
    return shlex.join([str(x) for x in cmd])


def format_suffix(template: str, *, size: int, seed: int, split: str) -> str:
    return template.format(size=size, seed=seed, split=split)


def trainer_ckpt_dir(args, *, size: int, seed: int, suffix: str) -> Path:
    return Path(
        f"./ckpts_finetune_generic_mixed_{args.model_type}_"
        f"{args.source_dataset}_to_{args.transfer_dataset}_seed{seed}{suffix}"
    )


def build_train_cmd(args, *, size: int, seed: int, suffix: str) -> List[str]:
    return [
        args.python,
        str(args.trainer),
        "--model_type", args.model_type,
        "--source_dataset", args.source_dataset,
        "--transfer_dataset", args.transfer_dataset,
        "--seed", str(seed),
        "--trigger_pt", args.trigger_pt,
        "--poison_fraction", str(args.poison_fraction),
        "--im_root", args.im_root,
        "--batch_size", str(args.batch_size),
        "--steps", str(args.steps),
        "--log_every", str(args.log_every),
        "--eval_every", str(args.eval_every),
        "--bn_rec_every", str(args.bn_rec_every),
        "--suffix", suffix,
        "--svhn_reference_size", str(size),
        "--svhn_reference_split", args.reference_split,
        "--reference_subset_strategy", args.reference_subset_strategy,
    ]


def build_victim_ft_cmd(args, *, ckpt_path: Path, size: int, seed: int) -> List[str]:
    run_name = f"exp1_svhnref{size}_{args.reference_split}_seed{seed}"
    return [
        args.python,
        str(args.finetune_script),
        "--ckpt", str(ckpt_path),
        "--source-dataset", args.source_dataset,
        "--transfer-dataset", args.transfer_dataset,
        "--trigger", args.trigger_pt,
        "--target_class", str(args.target_class),
        "--model-type", args.model_type,
        "--ft-epochs", str(args.ft_epochs),
        "--ft-lr", str(args.ft_lr),
        "--batch-size", str(args.batch_size),
        "--seed", str(seed),
        "--out-dir", args.ft_out_dir,
        "--run-name", run_name,
    ]


def run_cmd(cmd: List[str], *, dry_run: bool, continue_on_error: bool) -> int:
    print(sh(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main():
    args = parse_args()
    command_log = args.command_log or Path(args.ft_out_dir) / "exp1_svhn_reference_ablation_commands.jsonl"
    command_log.parent.mkdir(parents=True, exist_ok=True)

    with open(command_log, "a") as log_f:
        for seed in args.seeds:
            for size in args.reference_sizes:
                suffix = format_suffix(args.suffix_template, size=size, seed=seed, split=args.reference_split)
                ckpt_dir = trainer_ckpt_dir(args, size=size, seed=seed, suffix=suffix)
                ckpt_path = ckpt_dir / args.checkpoint_name

                train_cmd = build_train_cmd(args, size=size, seed=seed, suffix=suffix)
                record = {
                    "time": time.time(),
                    "stage": "train_reference_ablation",
                    "seed": seed,
                    "reference_size": size,
                    "reference_split": args.reference_split,
                    "suffix": suffix,
                    "checkpoint": str(ckpt_path),
                    "cmd": train_cmd,
                }
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()

                if not args.skip_train:
                    run_cmd(train_cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)

                if args.run_victim_finetune:
                    ft_cmd = build_victim_ft_cmd(args, ckpt_path=ckpt_path, size=size, seed=seed)
                    record = {
                        "time": time.time(),
                        "stage": "victim_full_svhn_finetune",
                        "seed": seed,
                        "reference_size": size,
                        "checkpoint": str(ckpt_path),
                        "cmd": ft_cmd,
                    }
                    log_f.write(json.dumps(record) + "\n")
                    log_f.flush()

                    if not args.dry_run and not ckpt_path.exists():
                        msg = f"Expected checkpoint does not exist: {ckpt_path}"
                        if args.continue_on_error:
                            print(f"[skip] {msg}", flush=True)
                            continue
                        raise FileNotFoundError(msg)

                    run_cmd(ft_cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)

    print(f"[done] command log -> {command_log}")


if __name__ == "__main__":
    main()
