#!/usr/bin/env python3
"""
Paired significance tests for OOD ASR.

The config is JSON so seed choices are easy to audit and edit. Each pair
specifies the PtH CSVs, baseline CSVs, ASR metric name, optional split, and
the exact seeds to use.

Primary test:
  one-sided paired sign-flip permutation test for mean(PtH - baseline) > 0.

This is appropriate for small paired seed counts and makes only the symmetry
assumption used by a paired randomization test. The script also reports paired
t-test and Wilcoxon p-values when scipy is installed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class SeriesSpec:
    name: str
    files: List[str]
    metric: str
    split: Optional[str] = None


def fmt_float(x: Optional[float], ndigits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.{ndigits}f}"


def mean_ci_196(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], float("nan")
    return mean(xs), 1.96 * stdev(xs) / math.sqrt(len(xs))


def sign_flip_pvalues(diffs: Sequence[float]) -> Tuple[float, float]:
    """Exact paired sign-flip permutation p-values.

    Returns:
      (p_one_sided_greater, p_two_sided_abs_mean)
    """
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")

    observed = mean(diffs)
    observed_abs = abs(observed)

    # Exact enumeration is cheap up to about 20 seeds. For larger n, enumerate
    # is still fine for the scales used here, but guard against accidental pain.
    if n > 20:
        raise ValueError(
            f"Exact sign-flip enumeration has 2^{n} assignments. "
            "Use <=20 paired seeds or add a Monte Carlo mode."
        )

    total = 0
    ge = 0
    ge_abs = 0
    eps = 1e-12
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        stat = sum(s * d for s, d in zip(signs, diffs)) / n
        total += 1
        if stat >= observed - eps:
            ge += 1
        if abs(stat) >= observed_abs - eps:
            ge_abs += 1

    return ge / total, ge_abs / total


def scipy_pvalues(diffs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None, None

    if len(diffs) < 2:
        return None, None

    t_res = stats.ttest_1samp(diffs, popmean=0.0, alternative="greater")
    try:
        w_res = stats.wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        w_p = float(w_res.pvalue)
    except Exception:
        w_p = None
    return float(t_res.pvalue), w_p


def resolve_files(patterns: Iterable[str], base_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        search = str(p if p.is_absolute() else base_dir / p)
        matches = sorted(Path(x) for x in glob.glob(search))
        if matches:
            files.extend(matches)
        else:
            # Keep missing globs non-fatal so future seed directories can be
            # listed in the config before those jobs finish.
            pass
    return files


def load_series(spec: SeriesSpec, base_dir: Path) -> Dict[int, float]:
    files = resolve_files(spec.files, base_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files matched for series '{spec.name}': {spec.files}")

    values: Dict[int, float] = {}
    provenance: Dict[int, Path] = {}
    for path in files:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("metric") != spec.metric:
                    continue
                if spec.split is not None and row.get("split") != spec.split:
                    continue
                seed = int(float(row["seed"]))
                value = float(row["value"])
                if seed in values and abs(values[seed] - value) > 1e-9:
                    raise ValueError(
                        f"Duplicate seed {seed} for series '{spec.name}' has conflicting values: "
                        f"{values[seed]} from {provenance[seed]} vs {value} from {path}"
                    )
                values[seed] = value
                provenance[seed] = path
    return values


def selected_seeds(pair: dict) -> List[int]:
    seeds = [int(s) for s in pair["seeds"]]
    exclude = {int(s) for s in pair.get("exclude_seeds", [])}
    return [s for s in seeds if s not in exclude]


def run_pair(pair: dict, base_dir: Path, allow_missing: bool) -> dict:
    pth = SeriesSpec(**pair["pth"])
    baseline = SeriesSpec(**pair["baseline"])
    seeds = selected_seeds(pair)

    pth_values = load_series(pth, base_dir)
    baseline_values = load_series(baseline, base_dir)

    missing_pth = [s for s in seeds if s not in pth_values]
    missing_baseline = [s for s in seeds if s not in baseline_values]
    if missing_pth or missing_baseline:
        msg = (
            f"Pair '{pair['name']}' is missing selected seeds. "
            f"PtH missing={missing_pth}; baseline missing={missing_baseline}."
        )
        if not allow_missing:
            raise ValueError(msg)
        print(f"[warn] {msg} Using available intersection only.")
        seeds = [s for s in seeds if s in pth_values and s in baseline_values]

    pth_x = [pth_values[s] for s in seeds]
    base_x = [baseline_values[s] for s in seeds]
    diffs = [a - b for a, b in zip(pth_x, base_x)]

    pth_mean, pth_ci = mean_ci_196(pth_x)
    base_mean, base_ci = mean_ci_196(base_x)
    diff_mean, diff_ci = mean_ci_196(diffs)
    p_perm_greater, p_perm_two = sign_flip_pvalues(diffs)
    p_t, p_w = scipy_pvalues(diffs)

    return {
        "name": pair["name"],
        "pth": pth.name,
        "baseline": baseline.name,
        "seeds": seeds,
        "n": len(seeds),
        "pth_mean": pth_mean,
        "pth_ci": pth_ci,
        "baseline_mean": base_mean,
        "baseline_ci": base_ci,
        "diff_mean": diff_mean,
        "diff_ci": diff_ci,
        "p_perm_greater": p_perm_greater,
        "p_perm_two_sided": p_perm_two,
        "p_paired_t_greater": p_t,
        "p_wilcoxon_greater": p_w,
        "diffs": diffs,
    }


def markdown_table(results: Sequence[dict]) -> str:
    lines = [
        "| Pair | Seeds | n | PtH ASR | Baseline ASR | Mean Diff | Sign-flip p, PtH > baseline | Paired t p | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        seeds = ",".join(str(s) for s in r["seeds"])
        lines.append(
            "| {name} | {seeds} | {n} | {pth:.2f} ± {pth_ci:.2f} | "
            "{base:.2f} ± {base_ci:.2f} | {diff:.2f} ± {diff_ci:.2f} | "
            "{p_perm} | {p_t} | {p_w} |".format(
                name=r["name"],
                seeds=seeds,
                n=r["n"],
                pth=r["pth_mean"],
                pth_ci=r["pth_ci"],
                base=r["baseline_mean"],
                base_ci=r["baseline_ci"],
                diff=r["diff_mean"],
                diff_ci=r["diff_ci"],
                p_perm=fmt_float(r["p_perm_greater"]),
                p_t=fmt_float(r["p_paired_t_greater"]),
                p_w=fmt_float(r["p_wilcoxon_greater"]),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired OOD ASR significance tests.")
    parser.add_argument(
        "--config",
        default="ood_asr_significance_pairs.json",
        help="JSON config describing pairs, CSV paths, metrics, and selected seeds.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Use intersection of selected seeds when CSVs are missing instead of failing.",
    )
    parser.add_argument(
        "--out-md",
        default="",
        help="Optional path to write the Markdown summary table.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional path to write machine-readable results.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open() as fh:
        config = json.load(fh)
    base_dir = config_path.parent

    results = [run_pair(pair, base_dir, args.allow_missing) for pair in config["pairs"]]
    md = markdown_table(results)
    print(md)
    print()
    print("Primary p-value: exact one-sided paired sign-flip permutation test for PtH > baseline.")
    print("Intervals use mean ± 1.96 * SEM to match the manuscript tables.")
    print("Paired t-test and Wilcoxon p-values are reported when scipy is installed.")

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        json_ready = []
        for r in results:
            rr = dict(r)
            json_ready.append(rr)
        out.write_text(json.dumps(json_ready, indent=2) + "\n")


if __name__ == "__main__":
    main()
