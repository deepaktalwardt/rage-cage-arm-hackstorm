#!/usr/bin/env python3
"""Edit the hard-coded MIT-mode gain constants in rage_cage_thrower.py.

Usage:
    python3 tune_gains.py kp 5.0                       # uniform: all 6 joints = 5.0
    python3 tune_gains.py kd 0.5 0.5 0.5 0.5 0.3 0.3   # per-joint: 6 values
    python3 tune_gains.py vdes 1.0
    python3 tune_gains.py torque 0.1

Names map to constants:
    kp     -> ARM_KP
    kd     -> ARM_KD
    vdes   -> ARM_V_DES
    torque -> ARM_TORQUE_FF

Edits rage_cage_thrower.py in place. Restart the node afterwards to pick up.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NAME_TO_CONST = {
    "kp": "ARM_KP",
    "kd": "ARM_KD",
    "vdes": "ARM_V_DES",
    "torque": "ARM_TORQUE_FF",
}

TARGET = Path(__file__).resolve().parent / "rage_cage_thrower.py"


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    name = sys.argv[1].lower()
    if name not in NAME_TO_CONST:
        print(f"unknown name '{name}'. options: {', '.join(NAME_TO_CONST)}")
        sys.exit(1)
    const = NAME_TO_CONST[name]

    raw_vals = sys.argv[2:]
    try:
        vals = [float(v) for v in raw_vals]
    except ValueError as e:
        print(f"all values must be numeric: {e}")
        sys.exit(1)

    if len(vals) == 1:
        vals = vals * 6
    if len(vals) != 6:
        print(f"need 1 or 6 values, got {len(vals)}")
        sys.exit(1)

    formatted = "(" + ", ".join(f"{v}" for v in vals) + ")"
    new_line = f"{const} = {formatted}"

    text = TARGET.read_text()
    pattern = rf"^{const} = .*$"
    if not re.search(pattern, text, flags=re.MULTILINE):
        print(f"could not find '{const} = ...' in {TARGET}")
        sys.exit(1)
    text = re.sub(pattern, new_line, text, flags=re.MULTILINE)
    TARGET.write_text(text)

    print(new_line)
    print("Restart the node to pick up the change.")


if __name__ == "__main__":
    main()
