#!/usr/bin/env python3
"""Aggregate sysid_ramp.py results across all run directories into a single
summary CSV / JSON / YAML, picking the best fit per joint.

"Best" means: among all fit.yaml entries for a given joint across all runs,
prefer the one with the fewest NaN fields (most complete). Ties broken by
most recent run timestamp (so re-runs supersede older partial runs).

Usage:
    python3 /ws/sysid/aggregate_sysid.py
    python3 /ws/sysid/aggregate_sysid.py --runs-dir /ws/sysid/runs --out /ws/sysid/summary

Outputs (next to each other under --out):
    piper_sysid_summary.csv        one row per joint, all measured params
    piper_sysid_summary.json       machine-friendly nested dict
    piper_sysid_summary.yaml       same data, human-readable

Each row / object includes a `source_run` field naming which run dir the
values came from, so you can trace back to the raw CSVs and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from glob import glob

import yaml


JOINT_FIELDS = (
    "I_eff",
    "b_eff",
    "tau_static_pos",
    "tau_static_neg",
    "tau_gravity_baseline",
    "qdd_at_step",
    "qdot_terminal",
    "alpha_torque_scale",
    "b_eff_phase_c",
    "b_eff_decay_fit",
    "tau_static_decay_fit",
)


def is_nan(v) -> bool:
    try:
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


def completeness(d: dict) -> int:
    """How many of the canonical fields are non-NaN. Higher is better."""
    return sum(1 for k in JOINT_FIELDS if k in d and not is_nan(d.get(k)))


def collect(runs_dir: str) -> dict[int, dict]:
    """Walk every fit.yaml under runs_dir and return the best fit per joint."""
    best: dict[int, tuple[int, str, dict]] = {}  # joint -> (score, run_id, fit)
    fit_paths = sorted(glob(os.path.join(runs_dir, "sysid_*", "fit.yaml")))
    if not fit_paths:
        print(f"no sysid_*/fit.yaml under {runs_dir}", file=sys.stderr)
        return {}
    for path in fit_paths:
        run_id = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        fits = data.get("fits", {})
        for j, fit in fits.items():
            try:
                jn = int(j)
            except (TypeError, ValueError):
                continue
            score = completeness(fit or {})
            prev = best.get(jn)
            if prev is None or score > prev[0] or (score == prev[0] and run_id > prev[1]):
                best[jn] = (score, run_id, fit or {})
    return {j: {**fit, "source_run": run_id, "completeness": score}
            for j, (score, run_id, fit) in best.items()}


def to_yaml(summary: dict) -> str:
    # Make NaN serialize as null so downstream YAML loaders don't choke.
    def _clean(d):
        out = {}
        for k, v in d.items():
            out[k] = None if is_nan(v) else v
        return out
    return yaml.safe_dump(
        {f"joint{j}": _clean(d) for j, d in sorted(summary.items())},
        sort_keys=False,
    )


def to_json(summary: dict) -> str:
    def _clean(d):
        return {k: (None if is_nan(v) else v) for k, v in d.items()}
    return json.dumps(
        {f"joint{j}": _clean(d) for j, d in sorted(summary.items())},
        indent=2,
    )


def to_csv(summary: dict) -> str:
    headers = ["joint"] + list(JOINT_FIELDS) + ["completeness", "source_run"]
    rows = []
    for j in sorted(summary):
        d = summary[j]
        row = [j]
        for k in JOINT_FIELDS:
            v = d.get(k)
            row.append("" if (v is None or is_nan(v)) else f"{v:.6g}")
        row.append(d.get("completeness", 0))
        row.append(d.get("source_run", ""))
        rows.append(row)
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(c) for c in row))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="/ws/sysid/runs")
    p.add_argument("--out", default="/ws/sysid/summary",
                   help="directory to write piper_sysid_summary.{csv,json,yaml}")
    args = p.parse_args(argv)

    summary = collect(args.runs_dir)
    if not summary:
        return 1

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "piper_sysid_summary.yaml"), "w") as f:
        f.write(to_yaml(summary))
    with open(os.path.join(args.out, "piper_sysid_summary.json"), "w") as f:
        f.write(to_json(summary))
    with open(os.path.join(args.out, "piper_sysid_summary.csv"), "w") as f:
        f.write(to_csv(summary))

    # Also dump a human-readable table to stdout.
    print("\n== Piper sysid summary ==")
    fmt = (
        "{:>2} {:>9} {:>9} {:>11} {:>11} {:>10} {:>6} {:>3} {}"
    )
    print(fmt.format(
        "J", "I_eff", "b_eff", "tau_s_pos", "tau_s_neg", "tau_grav", "alpha", "n", "source",
    ))
    for j in sorted(summary):
        d = summary[j]
        def f(k):
            v = d.get(k)
            return "—" if (v is None or is_nan(v)) else f"{v:+.4f}"
        print(fmt.format(
            j, f("I_eff"), f("b_eff"), f("tau_static_pos"), f("tau_static_neg"),
            f("tau_gravity_baseline"), f("alpha_torque_scale"),
            d.get("completeness", 0),
            d.get("source_run", ""),
        ))
    print(f"\nfiles -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
