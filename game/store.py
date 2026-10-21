"""
The player's files, kept in player/ (gitignored): the menu's last race
settings, the control mappings and the lap records, one JSON file each.
"""

import json
import os

from game.sim import ROOT

PLAYER_DIR = ROOT / "player"


def load(name, default):
    """player/<name>.json, or `default` when it's missing or unreadable."""
    path = PLAYER_DIR / f"{name}.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as e:
        print(f"[play] ignoring {path.name}: {e}")
        return default


def save(name, data):
    """Write player/<name>.json atomically (a crash mid-write keeps the old file)."""
    PLAYER_DIR.mkdir(parents=True, exist_ok=True)
    path = PLAYER_DIR / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
