"""
Lap records: yours per track in player/records.json (clean laps only, no
respawn during the lap), and the AI drivers' from the simulator
(saved_models/<profile>/lap_records.json).
"""

import time

from game import store
from game.settings import PROFILES


class PlayerRecords:
    def __init__(self):
        raw = store.load("records", {})
        self.records = {k: v for k, v in raw.items()
                        if isinstance(v, dict) and isinstance(v.get("lap_s"), (int, float))} \
            if isinstance(raw, dict) else {}

    def get(self, map_name):
        return self.records.get(map_name)

    def offer(self, map_name, lap_s, design_text, physics="AI-equal"):
        """Keep lap_s if it beats your record here; True if it's a new record."""
        old = self.records.get(map_name)
        if old is not None and lap_s >= old["lap_s"]:
            return False
        self.records[map_name] = {"lap_s": round(float(lap_s), 3), "date": time.strftime("%Y-%m-%d %H:%M"),
                                  "design": design_text, "physics": physics}
        store.save("records", self.records)
        return True


def ai_record(cfg, map_name):
    """(lap_s, profile) of the fastest AI lap on this track, or None."""
    from racer.records import LapRecords
    best = None
    for prof in PROFILES:
        rec = LapRecords.load(cfg, prof).get(map_name)
        if rec and (best is None or rec["lap_s"] < best[0]):
            best = (float(rec["lap_s"]), prof)
    return best
