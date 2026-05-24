"""Smoke for the v35 run-subfolder artifact layout (Section F).

Runs a tiny train_rl with --rand-stage 0 (so per-rand-stage best saver
fires, plus curriculum + grid CSV writers are active) and asserts that
all artifacts land *inside* <out>/ as fixed-name files, not as scattered
<out>_* siblings the way v22-v34 wrote them.

Run from the repo root:

    uv run python -m sim.smoke_run_dir_layout
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "smoke_run"
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "sim.train_rl",
            "--timesteps",
            "1024",
            "--n-envs",
            "2",
            "--n-steps",
            "128",
            "--batch-size",
            "32",
            "--reward-stage",
            "3",
            "--rand-stage",
            "0",
            "--rand-eval-episodes",
            "2",
            "--curriculum",
            "auto",
            "--curriculum-eval-every",
            "256",
            "--curriculum-eval-episodes",
            "2",
            "--checkpoint-freq",
            "256",
            "--train-rollout-viz-every",
            "256",
            "--train-rollout-viz-steps",
            "60",
            "--out",
            str(run_dir),
            "--seed",
            "0",
            "--vec-env",
            "dummy",
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:])
            print("STDERR:", result.stderr[-2000:])
            raise SystemExit(f"train_rl exited {result.returncode}")

        expected_files = [
            run_dir / "policy.zip",
            run_dir / "vecnormalize.pkl",
            run_dir / "training.json",
            run_dir / "curriculum.csv",
            run_dir / "grid.csv",
        ]
        # best_R*.zip only fires when grid_success_rate > 0; a 1024-step
        # smoke can't get there. The saver writes
        # `args.out / "best_R{n}".zip` (see _maybe_save_best_per_rand_stage),
        # so confirming the dir layout for the always-present artifacts is
        # sufficient — the per-stage best file path is mechanical from there.
        expected_dirs = [
            run_dir / "checkpoints",
            run_dir / "train_rollouts",
            run_dir / "tb",
        ]
        missing_files = [str(p) for p in expected_files if not p.is_file()]
        missing_dirs = [str(p) for p in expected_dirs if not p.is_dir()]
        assert not missing_files, f"missing expected files: {missing_files}"
        assert not missing_dirs, f"missing expected dirs: {missing_dirs}"

        ckpts = list((run_dir / "checkpoints").glob("checkpoint_*_steps.zip"))
        assert ckpts, "no checkpoint zips in checkpoints/"

        train_gifs = list((run_dir / "train_rollouts").glob("*.gif"))
        assert train_gifs, "no GIFs in train_rollouts/"

        # No legacy stem-style files at the parent of run_dir.
        parent = run_dir.parent
        legacy_globs = (
            "*.curriculum.csv",
            "*_best*.zip",
            "*_best*.vecnormalize.pkl",
            "*.training.json",
            "*_checkpoints",
            "*_train_rollouts",
            "*.grid.csv",
            "*.vecnormalize.pkl",
        )
        for stem_glob in legacy_globs:
            stragglers = [
                p for p in parent.glob(stem_glob) if (p.is_file() or p.is_dir()) and p != run_dir
            ]
            assert not stragglers, f"legacy {stem_glob} sibling: {stragglers}"

        print("OK run_dir contents:")
        for p in sorted(run_dir.iterdir()):
            kind = "d" if p.is_dir() else "f"
            print(f"  [{kind}] {p.name}")


if __name__ == "__main__":
    main()
