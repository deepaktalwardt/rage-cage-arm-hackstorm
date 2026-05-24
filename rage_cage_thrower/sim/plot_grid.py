"""Plot the per-cell grid CSV as success-rate / closest-distance heatmaps.

Reads the CSV produced by ``CurriculumCallback._append_grid_log`` (or the
single-snapshot CSV from ``sim.eval_grid``) and renders a heatmap over
the 3x3 cup grid. With multiple timesteps in the file (training run),
plots the latest timestep by default; ``--all-timesteps`` produces a
multi-panel figure showing how the heatmap evolves through training.

Run via:
  uv run python -m sim.plot_grid --csv runs/rage_v23_seed0.grid.csv --out grid_latest.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", choices=("success", "closest_cup_dist", "valid_bounce"), default="success")
    parser.add_argument("--all-timesteps", action="store_true", help="Render one panel per timestep present in the CSV.")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "timesteps": float(r.get("timesteps", 0) or 0),
                    "cup_x": float(r["cup_x"]),
                    "cup_y": float(r["cup_y"]),
                    "success": float(r["success"]),
                    "closest_cup_dist": float(r["closest_cup_dist"]),
                    "valid_bounce": float(r["valid_bounce"]),
                }
            )
    return rows


def _grid_at(rows: list[dict[str, float]], metric: str) -> tuple[np.ndarray, list[float], list[float]]:
    by_cell = {(round(r["cup_x"], 4), round(r["cup_y"], 4)): r[metric] for r in rows}
    xs = sorted({round(r["cup_x"], 4) for r in rows})
    ys = sorted({round(r["cup_y"], 4) for r in rows}, reverse=True)
    grid = np.zeros((len(ys), len(xs)), dtype=np.float64)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            grid[i, j] = by_cell.get((x, y), float("nan"))
    return grid, xs, ys


def _draw_heatmap(ax, grid: np.ndarray, xs: list[float], ys: list[float], metric: str, title: str) -> None:
    if metric == "closest_cup_dist":
        clipped = np.clip(grid, 0.0, 1.0)
        im = ax.imshow(clipped, cmap="viridis_r", vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:.2f}" for x in xs])
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:+.2f}" for y in ys])
    ax.set_xlabel("cup_x")
    ax.set_ylabel("cup_y")
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            display = f"{value:.2f}" if metric != "closest_cup_dist" else f"{value:.3f}"
            ax.text(j, i, display, ha="center", va="center", color="white", fontsize=10)
    return im


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")

    by_timestep: dict[float, list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        by_timestep[r["timesteps"]].append(r)
    timesteps = sorted(by_timestep)

    if args.all_timesteps and len(timesteps) > 1:
        n = len(timesteps)
        cols = min(n, 4)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.5, rows_n * 3.5))
        axes = np.atleast_2d(axes).flatten()
        last_im = None
        for idx, ts in enumerate(timesteps):
            grid, xs, ys = _grid_at(by_timestep[ts], args.metric)
            mean = float(np.nanmean(grid))
            last_im = _draw_heatmap(axes[idx], grid, xs, ys, args.metric, f"t={int(ts):,} mean={mean:.2f}")
        for ax in axes[len(timesteps):]:
            ax.set_visible(False)
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.tolist(), shrink=0.7, label=args.metric)
        fig.suptitle(f"{args.metric} across cup grid (R3 envelope ±10cm)")
    else:
        ts = timesteps[-1]
        grid, xs, ys = _grid_at(by_timestep[ts], args.metric)
        mean = float(np.nanmean(grid))
        fig, ax = plt.subplots(figsize=(5, 5))
        im = _draw_heatmap(ax, grid, xs, ys, args.metric, f"{args.metric} t={int(ts):,} mean={mean:.2f}")
        fig.colorbar(im, ax=ax, shrink=0.7, label=args.metric)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
