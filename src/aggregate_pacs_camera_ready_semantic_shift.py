#!/usr/bin/env python3
"""Aggregate PACS camera-ready semantic-shift task results.

Uses only the Python standard library so it can run on a login/CPU node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


METRICS = [
    "poison_attacker_clean",
    "poison_attacker_asr",
    "hidden_attacker_clean",
    "hidden_attacker_asr",
    "hidden_imagenet_acc",
    "pth_victim_clean",
    "pth_victim_original_asr",
    "pth_victim_flip_asr",
    "pth_victim_conditional_flip_asr",
    "control_victim_clean",
    "control_victim_original_asr",
    "control_victim_flip_asr",
    "control_victim_conditional_flip_asr",
    "delta_original_asr",
    "delta_flip_asr",
]


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate PACS semantic-shift results.")
    p.add_argument("--results-root", default="runs_camera_ready_pacs_semantic_shift")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--expected-domains", nargs="+", default=["sketch", "photo"])
    p.add_argument("--expected-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def flatten_result(result: Dict) -> Dict:
    poison = result["post_poison"]["attacker"]
    hidden = result["released_hidden"]["attacker"]
    victim = result["post_victim_ft"]
    pth = victim["poison_then_hide"]
    control = victim["clean_encoder_trigger_only"]
    return {
        "protocol_version": result["protocol_version"],
        "seed": int(result["seed"]),
        "attacker_domain": result["attacker_domain"],
        "victim_domain": result["victim_domain"],
        "n_attacker": result["n_attacker"],
        "n_victim_ft": result["n_victim_ft"],
        "n_victim_eval": result["n_victim_eval"],
        "poison_attacker_clean": poison["clean_acc"],
        "poison_attacker_asr": poison["original_asr_true_nontarget"],
        "hidden_attacker_clean": hidden["clean_acc"],
        "hidden_attacker_asr": hidden["original_asr_true_nontarget"],
        "hidden_imagenet_acc": result["released_hidden"]["imagenet_acc"],
        "pth_victim_clean": pth["clean_acc"],
        "pth_victim_original_asr": pth["original_asr_true_nontarget"],
        "pth_victim_flip_asr": pth["flip_asr_true_nontarget"],
        "pth_victim_conditional_flip_asr": pth["conditional_flip_asr"],
        "control_victim_clean": control["clean_acc"],
        "control_victim_original_asr": control["original_asr_true_nontarget"],
        "control_victim_flip_asr": control["flip_asr_true_nontarget"],
        "control_victim_conditional_flip_asr": control["conditional_flip_asr"],
        "delta_original_asr": victim["delta_original_asr"],
        "delta_flip_asr": victim["delta_flip_asr"],
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite(values: Iterable[float]) -> List[float]:
    out = []
    for value in values:
        value = float(value)
        if math.isfinite(value):
            out.append(value)
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "aggregate"
    result_paths = sorted(root.glob("attacker-*__victim-*__seed-*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"no result.json files found below {root}")

    rows = [flatten_result(json.loads(path.read_text())) for path in result_paths]
    rows.sort(key=lambda row: (row["attacker_domain"], row["seed"]))
    observed = {(row["attacker_domain"], int(row["seed"])) for row in rows}
    expected = {(domain, seed) for domain in args.expected_domains for seed in args.expected_seeds}
    missing = sorted(expected - observed)
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"missing {len(missing)} expected conditions: {missing}; "
            "pass --allow-incomplete only for an interim diagnostic"
        )

    write_csv(out_dir / "per_seed_results.csv", rows)
    aggregate_rows: List[Dict] = []
    for domain in sorted({row["attacker_domain"] for row in rows}):
        domain_rows = [row for row in rows if row["attacker_domain"] == domain]
        aggregate: Dict = {
            "attacker_domain": domain,
            "victim_domain": domain_rows[0]["victim_domain"],
            "n_seeds": len(domain_rows),
            "seeds": " ".join(str(row["seed"]) for row in domain_rows),
        }
        for metric in METRICS:
            values = finite(row[metric] for row in domain_rows)
            aggregate[f"{metric}_mean"] = statistics.mean(values) if values else float("nan")
            aggregate[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregate[f"{metric}_n"] = len(values)
        aggregate_rows.append(aggregate)
    write_csv(out_dir / "aggregate_results.csv", aggregate_rows)

    headline = []
    for row in aggregate_rows:
        headline.append(
            {
                "attacker_domain": row["attacker_domain"],
                "victim_domain": row["victim_domain"],
                "n_seeds": row["n_seeds"],
                "pth_clean_mean": row["pth_victim_clean_mean"],
                "pth_clean_sd": row["pth_victim_clean_sd"],
                "pth_original_asr_mean": row["pth_victim_original_asr_mean"],
                "pth_original_asr_sd": row["pth_victim_original_asr_sd"],
                "pth_flip_asr_mean": row["pth_victim_flip_asr_mean"],
                "pth_flip_asr_sd": row["pth_victim_flip_asr_sd"],
                "control_original_asr_mean": row["control_victim_original_asr_mean"],
                "control_original_asr_sd": row["control_victim_original_asr_sd"],
                "control_flip_asr_mean": row["control_victim_flip_asr_mean"],
                "control_flip_asr_sd": row["control_victim_flip_asr_sd"],
                "delta_original_asr_mean": row["delta_original_asr_mean"],
                "delta_original_asr_sd": row["delta_original_asr_sd"],
                "delta_flip_asr_mean": row["delta_flip_asr_mean"],
                "delta_flip_asr_sd": row["delta_flip_asr_sd"],
            }
        )
    write_csv(out_dir / "headline_table.csv", headline)

    manifest = {
        "results_root": str(root),
        "result_files": [str(path) for path in result_paths],
        "expected": sorted([list(item) for item in expected]),
        "observed": sorted([list(item) for item in observed]),
        "missing": [list(item) for item in missing],
        "complete": not missing,
        "uncertainty": "sample standard deviation across seeds",
    }
    (out_dir / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] per-seed -> {out_dir / 'per_seed_results.csv'}")
    print(f"[done] aggregate -> {out_dir / 'aggregate_results.csv'}")
    print(f"[done] headline -> {out_dir / 'headline_table.csv'}")


if __name__ == "__main__":
    main()
