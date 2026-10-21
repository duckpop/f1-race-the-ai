"""The race rules on real simulator races (headless), your car driven by
the middle AI's network as a stand-in."""

import numpy as np
import pytest

from game import sim  # noqa: F401
from game.inputs import DriveInput
from game.race import Race, line_crossings, signed_s
from game.settings import RaceSettings


def test_line_crossings_in_time_order():
    L, dt = 100.0, 0.05
    prev = np.array([95.0, 98.0, 50.0, 190.0])
    now = np.array([105.0, 104.0, 60.0, 201.0])
    out = line_crossings(prev, now, L, 1.0, dt)
    assert [i for _, i, _ in out] == [1, 0, 3]                  # car 1 got there first
    assert [k for _, _, k in out] == [1, 1, 2]
    assert out[0][0] == pytest.approx(1.0 - dt + dt / 3)


def test_signed_s():
    assert list(signed_s(np.array([10.0, 990.0]), 1000.0)) == [10.0, -10.0]


@pytest.fixture(scope="module")
def cfg():
    from racer import config
    return config.load()


@pytest.fixture(scope="module")
def policies(cfg):
    from game.app import load_policies
    return load_policies(cfg)


def make_race(cfg, policies, **kw):
    from racer.env import RaceEnv
    s = RaceSettings(**kw)
    env = RaceEnv(cfg, s.cars, s.track, ["middle"])
    return Race(cfg, s, env, policies, np.random.default_rng(7))


def bot(race, policies):
    o = policies["middle"].act(race.obs[race.h:race.h + 1])[0]
    return DriveInput(float(o[0]), max(float(o[1]), 0.0), max(float(o[2]), 0.0))


def test_grid_slot_and_field(cfg, policies):
    race = make_race(cfg, policies, track="Monza", ai={"middle": 2, "aggressive": 1, "defensive": 0}, grid=2,
                     laps=1)
    assert race.h == 1 and race.profiles[1] == "you"
    assert sorted(p for p in race.profiles if p != "you") == ["aggressive", "middle", "middle"]
    assert race.env.speed_share[race.h] == 1.0 and np.allclose(race.env.speed_share[race.ai], 0.95)


def test_ers_toggle_and_drs_needs_availability(cfg, policies):
    race = make_race(cfg, policies, track="Monza", ai={"middle": 1, "aggressive": 0, "defensive": 0}, laps=1)
    race.step(bot(race, policies), {"ers"})
    assert race.ers_on and ("ers", "on") in race.events
    race.step(bot(race, policies), {"ers"})
    assert not race.ers_on
    race.step(bot(race, policies), {"drs"})                     # in the first 30 s DRS is never available
    assert ("drs", "unavailable") in race.events and race.env.drs_open[race.h] == 0


def test_respawn_then_ghost(cfg, policies):
    race = make_race(cfg, policies, track="Monza", format="time", seconds=30,
                     ai={"middle": 3, "aggressive": 0, "defensive": 0}, grid=4)
    while race.respawns == 0 and race.time < 10:
        race.step(DriveInput(1.0, 1.0, 0.0), set())             # full left into the wall
    assert race.respawns == 1 and race.you_racing() and race.ghost_active()
    assert race.env.std[race.h, 3] == 0.0
    race.step(bot(race, policies), set())
    assert race.h not in race.env.geo.idx                        # the others can't sense the ghost


def test_one_lap_race_finishes_everyone(cfg, policies):
    race = make_race(cfg, policies, track="Spielberg", laps=1, ai={"middle": 2, "aggressive": 1, "defensive": 0},
                     grid=4)
    while not race.over() and race.time < 200:
        race.step(bot(race, policies), set())
    assert race.over()
    rows = race.results()
    finished = [r for r in rows if r["pos"] is not None]
    assert [r["pos"] for r in finished] == list(range(1, len(finished) + 1))
    assert finished[0]["gap"] == "winner" and finished[0]["points"] == 25
    times = [race.finish[r["car"]][0] for r in finished]
    assert times == sorted(times)                                # all on the lead lap: in crossing order
