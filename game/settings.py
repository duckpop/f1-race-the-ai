"""
The race the menu sets up, remembered between runs (player/settings.json).
Every field falls back to its default when the saved value is missing or
no longer valid (a renamed track, a design over the points limit, ...).
"""

import dataclasses

from game import store

PROFILES = ("middle", "aggressive", "defensive")
MAX_AI = 21                                   # 22 cars with you: the simulator's full grid
# AI top speed per level (a share of each AI car's own top speed).
DIFFICULTY = {"easy": 0.80, "medium": 0.88, "hard": 0.95, "expert": 1.00}
LAPS_RANGE = (1, 50)
SECONDS_RANGE = (30, 3600)
ZOOM_RANGE = (0.6, 6.0)


@dataclasses.dataclass
class RaceSettings:
    track: str = "Random"                     # a map name, or "Random"
    format: str = "laps"                      # "laps" or "time"
    laps: int = 5
    seconds: int = 150
    ai: dict = dataclasses.field(default_factory=lambda: {p: 7 for p in PROFILES})
    difficulty: str = "hard"
    crashes: str = "respawn"                  # "respawn" or "out"
    grid: object = "random"                   # "random" or a slot, 1 = pole
    design: list = dataclasses.field(default_factory=lambda: [3, 3, 3, 3, 3])
    physics: dict = dataclasses.field(default_factory=dict)   # your car's physics values (carphysics.VARS); missing = the simulator's
    zoom: float = 2.0                         # camera distance x the renderer's default

    @property
    def ai_total(self):
        return sum(self.ai.values())

    @property
    def cars(self):
        return self.ai_total + 1

    def clamp_grid(self):
        if self.grid != "random":
            self.grid = min(max(int(self.grid), 1), self.cars)


def load(cfg, maps):
    """The saved settings, each field checked; maps: the available track names."""
    from racer import design
    raw = store.load("settings", {})
    s = RaceSettings()
    if not isinstance(raw, dict):
        return s
    if raw.get("track") == "Random" or raw.get("track") in maps:
        s.track = raw["track"]
    if raw.get("format") in ("laps", "time"):
        s.format = raw["format"]
    s.laps = _int_in(raw.get("laps"), LAPS_RANGE, s.laps)
    s.seconds = _int_in(raw.get("seconds"), SECONDS_RANGE, s.seconds)
    ai = raw.get("ai")
    if isinstance(ai, dict):
        counts = {p: _int_in(ai.get(p), (0, MAX_AI), 0) for p in PROFILES}
        if sum(counts.values()) <= MAX_AI:
            s.ai = counts
    if raw.get("difficulty") in DIFFICULTY:
        s.difficulty = raw["difficulty"]
    from game import carphysics
    s.physics = carphysics.clean(cfg, raw.get("physics"), carphysics.presets(cfg)[carphysics.DEFAULT_PRESET])
    if raw.get("crashes") in ("respawn", "out"):
        s.crashes = raw["crashes"]
    grid = raw.get("grid")
    if grid == "random" or (isinstance(grid, int) and not isinstance(grid, bool) and 1 <= grid <= s.cars):
        s.grid = grid
    d = raw.get("design")
    if isinstance(d, list) and len(d) == len(s.design) and all(isinstance(x, int) for x in d) \
            and bool(design.legal(cfg, d)):
        s.design = d
    z = raw.get("zoom")
    if isinstance(z, (int, float)) and ZOOM_RANGE[0] <= z <= ZOOM_RANGE[1]:
        s.zoom = float(z)
    return s


def save(s):
    store.save("settings", dataclasses.asdict(s))


def _int_in(v, bounds, default):
    lo, hi = bounds
    return v if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi else default
