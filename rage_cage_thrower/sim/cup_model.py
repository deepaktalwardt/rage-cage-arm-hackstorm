"""Generate a parametric Solo-cup-style hollow open-top cup STL for the rage cage scene.

A "stack" of N cups (in rage cage, nested vertically like Solo cups from a dispenser)
is modeled as a single cup whose effective height = base_height + (N-1) * nesting_increment.

The output STL is used only as a *visual* mesh in MuJoCo — collision is handled by
composite primitives in the scene MJCF (MuJoCo's mesh collision is convex-hull only,
which would make the cup behave as solid).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def make_cup(
    count: int = 1,
    base_height: float = 0.120,
    nesting_increment: float = 0.015,
    rim_radius: float = 0.047,
    base_radius: float = 0.030,
    wall_thickness: float = 0.0015,
    n_segments: int = 48,
) -> trimesh.Trimesh:
    """Generate a hollow truncated-cone cup as a triangulated mesh.

    Defaults approximate a red Solo cup: ~9.4 cm rim, ~6 cm base, 12 cm tall, ~1.5 mm walls.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    height = base_height + (count - 1) * nesting_increment

    theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)

    outer_base = np.column_stack([base_radius * c, base_radius * s, np.zeros(n_segments)])
    outer_rim = np.column_stack([rim_radius * c, rim_radius * s, np.full(n_segments, height)])
    inner_base = np.column_stack(
        [(base_radius - wall_thickness) * c,
         (base_radius - wall_thickness) * s,
         np.full(n_segments, wall_thickness)]
    )
    inner_rim = np.column_stack(
        [(rim_radius - wall_thickness) * c,
         (rim_radius - wall_thickness) * s,
         np.full(n_segments, height)]
    )

    n = n_segments
    OB, OR, IB, IR = 0, n, 2 * n, 3 * n
    vertices = np.vstack([outer_base, outer_rim, inner_base, inner_rim])

    faces: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        # Outer wall (normals out)
        faces.append([OB + i, OB + j, OR + j])
        faces.append([OB + i, OR + j, OR + i])
        # Inner wall (normals in, reversed winding)
        faces.append([IB + i, IR + i, IR + j])
        faces.append([IB + i, IR + j, IB + j])
        # Top rim annulus (normals up)
        faces.append([OR + i, OR + j, IR + j])
        faces.append([OR + i, IR + j, IR + i])

    # Closed outer bottom (normals down)
    center_outer = len(vertices)
    vertices = np.vstack([vertices, [[0.0, 0.0, 0.0]]])
    for i in range(n):
        j = (i + 1) % n
        faces.append([center_outer, OB + j, OB + i])

    # Inner floor (where the ball lands; normals up)
    center_inner = len(vertices)
    vertices = np.vstack([vertices, [[0.0, 0.0, wall_thickness]]])
    for i in range(n):
        j = (i + 1) % n
        faces.append([center_inner, IB + i, IB + j])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    mesh.fix_normals()
    return mesh


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a parametric cup STL.")
    p.add_argument("--count", type=int, default=1, help="Number of nested cups in the stack")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent / "mjcf" / "cups" / "cup.stl",
                   help="Output STL path")
    p.add_argument("--base-height", type=float, default=0.120)
    p.add_argument("--nesting-increment", type=float, default=0.015)
    p.add_argument("--rim-radius", type=float, default=0.047)
    p.add_argument("--base-radius", type=float, default=0.030)
    p.add_argument("--wall-thickness", type=float, default=0.0015)
    p.add_argument("--n-segments", type=int, default=48)
    args = p.parse_args()

    mesh = make_cup(
        count=args.count,
        base_height=args.base_height,
        nesting_increment=args.nesting_increment,
        rim_radius=args.rim_radius,
        base_radius=args.base_radius,
        wall_thickness=args.wall_thickness,
        n_segments=args.n_segments,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out)

    h = args.base_height + (args.count - 1) * args.nesting_increment
    print(f"cup count={args.count}  height={h*100:.2f} cm  rim_d={args.rim_radius*200:.1f} cm  "
          f"base_d={args.base_radius*200:.1f} cm  wall={args.wall_thickness*1000:.1f} mm")
    print(f"mesh: {len(mesh.vertices)} verts  {len(mesh.faces)} faces  "
          f"watertight={mesh.is_watertight}  volume_approx={mesh.volume*1e6:.1f} cm^3")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
