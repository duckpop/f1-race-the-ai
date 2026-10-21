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


def test_car_physics_only_changes_your_car(cfg):
    from racer import physics
    from game import carphysics
    speed = np.array([80.0, 80.0, 5.0])
    base = carphysics._sim.max_steer(cfg, speed, 1.0, 1.0, 1.0, 1.0)
    assert np.allclose(physics.max_steer(cfg, speed), base)              # nobody set: the simulator's
    plus50 = carphysics.CarPhysics(cfg, carphysics.presets(cfg)["Downforce +50%"])
    carphysics.Active.car, carphysics.Active.phys = 1, plus50
    try:
        mine = physics.max_steer(cfg, speed, 1.0, 1.0, 1.0, 1.0)
        assert mine[0] == pytest.approx(base[0])                        # the AI car: unchanged
        assert mine[1] == pytest.approx(np.arctan(cfg.wheelbase * 9.81 * 6.5 / 80.0 ** 2))   # 6.5 g at 288 km/h
        carphysics.Active.car = 2
        assert physics.max_steer(cfg, speed)[2] == pytest.approx(base[2])   # slow: the lock limits, not grip
    finally:
        carphysics.Active.car = None


def test_ai_equal_is_the_simulators_car(cfg):
    from game import carphysics
    sim = carphysics.sim_values(cfg)
    p = carphysics.CarPhysics(cfg, sim)
    assert p.is_sim and carphysics.label(cfg, sim) == "AI-equal"
    assert carphysics.label(cfg, dict(sim, base_g=2.0)) == "Custom (1 changed)"
    for v in (40.0, 70.0, 90.0):                                         # the cap formula = the simulator's
        assert carphysics.steer_cap(p, v, 1.0, 0.6, 1.0, 0.8) == pytest.approx(
            float(carphysics._sim.max_steer(cfg, np.array([v]), 1.0, 0.6, 1.0, 0.8)[0]))
    vals = carphysics.clean(cfg, {"base_g": 99.0, "traction_circle": "no", "s_max": 0.8})
    assert vals["base_g"] == cfg.grip.base_g and vals["traction_circle"] is True and vals["s_max"] == 0.8
    from racer import design
    mine, stock = p.car(design.stock(cfg)), design.car_params(cfg, design.stock(cfg, 1))
    assert all(np.allclose(getattr(mine, k), getattr(stock, k)) for k in vars(stock))


def test_per_car_step_matches_the_simulators(cfg):
    """The copied racing step (per-car wheel speed) drives exactly like the simulator's."""
    from racer.env import RaceEnv
    from game import carphysics
    envs = [RaceEnv(cfg, 2, "Monza", ["middle"]) for _ in range(2)]
    carphysics.install(envs[1])
    for env in envs:
        env.reset(np.random.default_rng(3), spawn="grid")
    for k in range(80):
        a = np.zeros((2, cfg.action_dim), dtype=np.float32)
        a[:, 1] = 1.0
        a[:, 0] = np.sin(k / 7.0)
        for env in envs:
            env.step(a)
    assert np.allclose(envs[0].std, envs[1].std)


def test_engine_values(cfg):
    from game import carphysics
    sim = carphysics.sim_values(cfg)
    stock = carphysics.CarPhysics(cfg, sim).car([3, 3, 3, 3, 3])
    more_power = carphysics.CarPhysics(cfg, dict(sim, a_max=30.0, v_switch=45.0)).car([3, 3, 3, 3, 3])
    top = lambda c: float((c.power[0] / c.drag[0]) ** (1 / 3))
    assert more_power.a_trac[0] == pytest.approx(30.0) and top(more_power) == pytest.approx(top(stock))
    assert more_power.brake[0] == pytest.approx(stock.brake[0])          # brake_max stays its own
    faster = carphysics.CarPhysics(cfg, dict(sim, v_max=100.0)).car([3, 3, 3, 3, 3])
    assert top(faster) == pytest.approx(100.0)


def test_preview_shows_what_each_change_does(cfg):
    from game import carphysics
    sim = carphysics.sim_values(cfg)
    rows, lock, lock_s, you, ai = carphysics.preview(cfg, sim, [3, 3, 3, 3, 3])
    for kmh, (steady, lifted, half, full), ai_cells in rows:
        assert [c[1] for c in (steady, lifted, half, full)] == pytest.approx([c[1] for c in ai_cells])
        assert half[1] > lifted[1] > steady[1]                           # braking costs cornering grip
    assert you == pytest.approx(ai) and you[0] == pytest.approx(cfg.v_max * 3.6)
    assert lock == pytest.approx(cfg.wheelbase / np.tan(cfg.vehicle.s_max)) and lock_s == pytest.approx(0.4)
    no_circle = carphysics.preview(cfg, dict(sim, traction_circle=False), [3, 3, 3, 3, 3])[0]
    assert all(half[1] == pytest.approx(steady[1]) for _, (steady, _l, half, _f), _a in no_circle)
    softer = carphysics.preview(cfg, dict(sim, exponent=1.5), [3, 3, 3, 3, 3])[0]
    assert softer[3][1][0][1] < rows[3][1][0][1] * 0.95                  # 180 km/h: v^1.5 grips more
    assert softer[-1][1][0][1] == pytest.approx(rows[-1][1][0][1])       # 288 km/h: both at max_g
    brakes = carphysics.preview(cfg, dict(sim, brake_max=30.0), [3, 3, 3, 3, 3])[3]
    assert brakes[3] < you[3] * 0.8                                      # shorter braking


def test_steering_not_scaled(cfg):
    from game import carphysics
    p = carphysics.CarPhysics(cfg, dict(carphysics.sim_values(cfg), steering_scaled=False))
    cap = carphysics.steer_cap(p, 80.0, 1.0, 1.0, 1.0, 1.0)
    for a in (0.01, 0.5, 1.0):                                           # env steers a x cap'
        got = a * carphysics.steer_cap(p, 80.0, 1.0, 1.0, 1.0, 1.0, action=a)
        assert got == pytest.approx(min(a * cfg.vehicle.s_max, cap))


def test_car_physics_in_a_race(cfg, policies):
    """More downforce turns tighter at speed; a faster wheel reaches the angle sooner;
    battery, size and the env reset for the next race."""
    from game import carphysics
    sim = carphysics.sim_values(cfg)
    solo = {"middle": 0, "aggressive": 0, "defensive": 0}
    angles = {}
    for name, vals in (("sim", sim), ("plus50", carphysics.presets(cfg)["Downforce +50%"])):
        race = make_race(cfg, policies, track="Monza", ai=solo, laps=1, physics=vals)
        while race.time < 3.0:
            race.step(DriveInput(0.0, 1.0, 0.0), set())
        race.step(DriveInput(1.0, 0.0, 0.0), set())
        angles[name] = float(race.env.std[race.h, 2])
        assert float(race.env.std[race.h, 3]) > 30.0
    assert angles["plus50"] > angles["sim"] * 1.1
    first = {}
    for sv in (1.5, 6.0):
        race = make_race(cfg, policies, track="Monza", ai=solo, laps=1, physics=dict(sim, sv_max=sv))
        race.step(DriveInput(1.0, 0.0, 0.0), set())                  # standing still: full lock wanted
        first[sv] = float(race.env.std[race.h, 2])
    assert first[1.5] == pytest.approx(1.5 * cfg.sim.control_dt, rel=0.05)
    assert first[6.0] > first[1.5] * 3

    from racer.env import RaceEnv
    s = RaceSettings(track="Monza", laps=1, ai={"middle": 1, "aggressive": 0, "defensive": 0}, grid=1,
                     physics=dict(sim, ers_capacity=20.0, ers_start=0.5, width=3.0, length=7.0))
    env = RaceEnv(cfg, s.cars, s.track, ["middle"])
    stock_side = None
    race = Race(cfg, s, env, policies, np.random.default_rng(1))
    h = race.h
    assert env.battery[h] == pytest.approx(10.0) and env.car.ers_capacity[h] == pytest.approx(20.0)
    assert env.side_distances[h].max() > env.side_distances[1 - h].max()       # its body is bigger
    sim_ = env.unwrapped.sim
    stock_side = sim_.scan_cache[1 - h].side_distances.copy()
    s.physics, s.grid = {}, 2
    race = Race(cfg, s, env, policies, np.random.default_rng(1))                # rematch, AI-equal
    assert np.allclose(env.side_distances[0], env.side_distances[1])
    assert np.allclose(sim_.scan_cache[0].side_distances, stock_side)
