"""
The simulator (the git submodule in simulator/) on the import path: its
`racer` package and its `watch.py` become importable. Import this module
before anything from the simulator.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "simulator"

if not (SIM_DIR / "racer").is_dir() or not (SIM_DIR / "f1tenth_gym" / "f1tenth_gym").is_dir():
    raise SystemExit("the simulator submodule is missing - run: git submodule update --init --recursive")
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
