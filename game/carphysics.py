"""
Your car's physics, changeable on the Car physics pages: every setting of a
car in the simulator, for your car only (the AI keeps what it was trained
with). The steering cap, the heart of cornering:

    max steer angle = steer_factor x min(s_max, atan(wheelbase x a_lat / max(v, 1)^2))
    a_lat           = 9.81 x grip x min(max_g, base_g + df_share x downforce_g x (v / ref_speed)^exponent) x lateral_share
    lateral_share   = max(combined_floor, sqrt(1 - u^2))          (1 with the traction circle off)
    u               = min(1, |throttle x engine - braking| / (a_max speeding up, brake_max slowing))
    engine          = min(a_max, a_max x v_switch / v)            (x (1 + ERS power boost) deploying)
    braking         = brake x brake_max on the throttle, else min(brake_limit, brake + brake_share) x brake_max
    df_share        = 1 - drs_grip_loss while DRS is open

The wheel moves towards (your steer x that angle) at up to sv_max rad/s.
(The simulator's exponent is 2; it's a setting here so other curves can be
tried.)

Left out: mass, yaw inertia, centre-of-gravity height and mu (the
kinematic model ignores them), v_min (the simulator never lets a car
reverse), the separate s_min / sv_min / lf / lr (mirrors of s_max / sv_max;
only the wheelbase sum matters to the kinematic model) and DRS latch (your
DRS is the game's toggle).

How it's applied, all from outside the simulator's code:
- your car's per-car parameters (design.car_params, rebuilt with your
  values) and its row of the integrator's parameters,
- the simulator's per-step functions (racer.physics max_steer,
  lateral_share, braking, longitudinal_accel, engine_accel, tow_boost,
  ers_update), wrapped to run again for your car with a copy of the
  settings that holds your values (Settings.override),
- a copy of the simulator's step that clips the wheel speed per car,
- your car's lidar body (other cars see its size; its own wall check uses
  it), and its drawn size.
Not per car: the AI's "alongside" sensor and the close-call geometry
assume every car has the stock size.
"""

import dataclasses
import math
import types

import numpy as np

from game import sim  # noqa: F401
from racer import design, physics, sensors

P_LF, P_LR, P_S_MIN, P_S_MAX, P_SV_MIN, P_SV_MAX, P_V_MAX = 3, 4, 8, 9, 10, 11, 15     # the gym's parameter row
LF_SHARE = 1.6 / 3.6                    # the stock front / rear split of the wheelbase


def kmh(x):
    """A speed kept in m/s (as the simulator does), shown in km/h."""
    return f"{x * 3.6:.0f} km/h"


@dataclasses.dataclass(frozen=True)
class Var:
    key: str
    group: str
    label: str
    path: str           # the settings key ("section.key"), or "" for the game's own (grip, exponent, brake_max, wheelbase)
    lo: float
    hi: float
    step: float
    fmt: object         # a format string, a function, or "bool"
    note: str

    def show(self, x):
        if self.fmt == "bool":
            return "on" if x else "off"
        return self.fmt(x) if callable(self.fmt) else self.fmt.format(x)


GROUPS = [
    ("grip", "Grip", "how much the tyres hold in corners, and how it grows with speed"),
    ("traction", "Traction circle and braking", "how braking and throttle share the tyre with cornering"),
    ("steering", "Steering", "how far and how fast the wheel turns"),
    ("engine", "Engine and top speed", "acceleration and straight-line speed"),
    ("tow", "Slipstream", "the pull of the car ahead's wake"),
    ("drs", "DRS", "what the open rear wing gives and costs"),
    ("ers", "ERS battery", "the electric boost and how it recharges"),
    ("size", "Car size", "the body: collisions with walls and cars, and how it's drawn"),
    ("design", "Design levels", "how much each engine / battery / chassis / brakes / wing level changes"),
]

VARS = [
    Var("base_g", "grip", "Base grip", "grip.base_g", 0.5, 5.0, 0.1, "{:.1f} g", "grip with no downforce (slow corners)"),
    Var("downforce_g", "grip", "Downforce grip", "grip.downforce_g", 0.0, 12.0, 0.1, "{:.2f} g",
        "grip downforce adds at the reference speed"),
    Var("ref_speed", "grip", "Downforce reference speed", "grip.ref_speed", 30.0, 120.0, 5.0, kmh,
        "the speed where downforce adds its full downforce grip"),
    Var("exponent", "grip", "Downforce speed exponent", "", 1.0, 3.0, 0.1, "v^{:.1f}",
        "how downforce grows with speed (the simulator: v^2; lower = more grip below the reference speed)"),
    Var("max_g", "grip", "Max grip", "grip.max_g", 1.0, 12.0, 0.25, "{:.2f} g", "the ceiling on cornering grip (fast corners)"),
    Var("grip", "grip", "Grip multiplier", "", 0.5, 2.0, 0.05, "x{:.2f}",
        "all cornering grip, slow and fast (on top of your chassis level)"),
    Var("steering_scaled", "grip", "Steering scaled to grip", "grip.steering_scaled", 0, 1, 1, "bool",
        "on: full steer = the grip limit at your speed · off: full steer = full lock, cut at the grip limit"),
    Var("traction_circle", "traction", "Traction circle", "grip.traction_circle", 0, 1, 1, "bool",
        "braking / accelerating use up cornering grip"),
    Var("combined_floor", "traction", "Grip kept at full brake / throttle", "grip.combined_floor", 0.0, 1.0, 0.05,
        "{:.0%}", "cornering grip left when braking or throttle take all the tyre"),
    Var("a_max", "traction", "Traction limit (a_max)", "car.a_max", 5.0, 60.0, 1.0, "{:.0f} m/s²",
        "the tyre's push: acceleration below v_switch, and engine power (a_max x v_switch)"),
    Var("brake_max", "traction", "Full braking", "", 5.0, 60.0, 1.0, "{:.0f} m/s²", "how hard full brakes slow you"),
    Var("brake_share", "traction", "Engine braking", "engine.brake_share", 0.0, 1.0, 0.05, "{:.0%}",
        "off the throttle: this share of full braking, also on the tyre"),
    Var("brake_limit", "traction", "Brakes + engine braking cap", "engine.brake_limit", 1.0, 2.0, 0.05, "x{:.2f}",
        "brakes plus engine braking reach at most this x full braking"),
    Var("sv_max", "steering", "Wheel speed (sv_max)", "car.sv_max", 0.25, 8.0, 0.25, "{:.2f} rad/s",
        "how fast the wheel turns (matters below ~90 km/h)"),
    Var("s_max", "steering", "Full lock (s_max)", "car.s_max", 0.1, 1.2, 0.05, "{:.2f} rad",
        "the steering angle limit (the tightest low-speed turn)"),
    Var("wheelbase", "steering", "Wheelbase", "", 2.0, 5.0, 0.1, "{:.1f} m", "axle to axle (low-speed turning geometry)"),
    Var("v_max", "engine", "Top speed", "car.v_max", 50.0, 130.0, 1.0, kmh,
        "where engine power equals drag (DRS, slipstream and ERS add to it)"),
    Var("v_switch", "engine", "Power switch speed (v_switch)", "car.v_switch", 10.0, 80.0, 1.0, kmh,
        "above it the engine is power-limited: higher = more power, faster at speed (same top speed)"),
    Var("tow_boost", "tow", "Slipstream top-speed boost", "tow.boost", 0.0, 0.5, 0.01, "+{:.0%}",
        "up to this much more top speed in a car's wake"),
    Var("tow_accel", "tow", "Slipstream pull", "tow.accel", 0.0, 30.0, 0.5, "{:.1f} m/s²", "extra pull on the throttle in a wake"),
    Var("tow_full_dist", "tow", "Full wake within", "tow.full_dist", 0.0, 100.0, 1.0, "{:.0f} m behind", "the full effect this close"),
    Var("tow_max_dist", "tow", "Wake reaches", "tow.max_dist", 10.0, 300.0, 5.0, "{:.0f} m behind", "fading out to nothing here"),
    Var("tow_full_width", "tow", "Full wake width", "tow.full_width", 0.0, 10.0, 0.25, "±{:.2f} m", "the full effect within this sideways"),
    Var("tow_width", "tow", "Wake width", "tow.width", 0.5, 15.0, 0.25, "±{:.2f} m", "fading out to nothing at this sideways"),
    Var("tow_min_speed", "tow", "Slipstream from", "tow.min_speed", 0.0, 60.0, 1.0, kmh, "no slipstream below this speed"),
    Var("drs_boost", "drs", "DRS top-speed boost", "drs.boost", 0.0, 0.3, 0.01, "+{:.0%}", "more top speed while open"),
    Var("drs_accel", "drs", "DRS pull", "drs.accel", 0.0, 10.0, 0.25, "{:.2f} m/s²", "extra pull on the throttle while open"),
    Var("drs_grip_loss", "drs", "DRS downforce loss", "drs.grip_loss", 0.0, 1.0, 0.05, "{:.0%}",
        "share of the downforce grip lost while open"),
    Var("ers_capacity", "ers", "Battery size", "ers.capacity_s", 1.0, 60.0, 1.0, "{:.0f} s", "seconds of full deployment"),
    Var("ers_start", "ers", "Charge at the start", "ers.start_charge", 0.0, 1.0, 0.05, "{:.0%}", "battery level on the grid"),
    Var("ers_power", "ers", "ERS power boost", "ers.power_boost", 0.0, 1.0, 0.05, "+{:.0%}", "engine power while deploying"),
    Var("ers_harvest", "ers", "Recharge rate", "ers.harvest_rate", 0.0, 5.0, 0.1, "x{:.1f}",
        "seconds of boost gained per second of full braking at top speed"),
    Var("ers_close_harvest", "ers", "Chase recharge rate", "ers.close_harvest_rate", 0.0, 5.0, 0.1, "x{:.1f}",
        "the recharge rate for a lap after passing the detection point close behind a car"),
    Var("ers_close_gap", "ers", "Chase recharge within", "ers.close_gap_s", 0.0, 5.0, 0.1, "{:.1f} s",
        "how close behind at the detection point gives the chase recharge"),
    Var("width", "size", "Width", "car.width", 1.0, 4.0, 0.1, "{:.1f} m", "body width"),
    Var("length", "size", "Length", "car.length", 2.0, 10.0, 0.1, "{:.1f} m", "body length"),
    Var("body_x", "size", "Body centre ahead of the rear axle", "car.collision_body_center_x", -2.0, 5.0, 0.1,
        "{:.1f} m", "where the body sits over the wheels"),
    Var("design_step", "design", "Change per level", "design.step", 0.0, 0.2, 0.01, "{:.0%}",
        "what one design level adds / costs"),
    Var("design_max", "design", "Largest change", "design.max_change", 0.0, 0.6, 0.01, "±{:.0%}",
        "no quantity moves more than this from the stock car"),
]
BY_KEY = {v.key: v for v in VARS}


def _get(cfg, path):
    section, key = path.split(".")
    return getattr(getattr(cfg, section), key)


def sim_values(cfg):
    """The simulator's values: what the AI drives with."""
    vals = {}
    for v in VARS:
        if v.path:
            x = _get(cfg, v.path)
            vals[v.key] = bool(x) if v.fmt == "bool" else float(x)
    vals.update(grip=1.0, exponent=2.0, brake_max=float(cfg.car.a_max), wheelbase=float(cfg.wheelbase))
    return vals


def presets(cfg):
    """{name: values}: the simulator's car, and more downforce (x the
    downforce grip, a higher max grip)."""
    base = sim_values(cfg)
    out = {"AI-equal": dict(base)}
    for name, mult, max_g in (("Downforce +30%", 1.3, 6.0), ("Downforce +50%", 1.5, 6.5),
                              ("Downforce +80%", 1.8, 7.5)):
        out[name] = dict(base, downforce_g=round(base["downforce_g"] * mult, 3), max_g=max_g)
    return out


DEFAULT_PRESET = "Downforce +50%"


def clean(cfg, raw, default=None):
    """Saved values, each checked; missing or bad ones from default (the simulator's when None)."""
    vals = dict(sim_values(cfg))
    vals.update(default or {})
    if isinstance(raw, dict):
        for v in VARS:
            x = raw.get(v.key)
            if v.fmt == "bool":
                if isinstance(x, bool):
                    vals[v.key] = x
            elif isinstance(x, (int, float)) and not isinstance(x, bool) and v.lo <= x <= v.hi:
                vals[v.key] = float(x)
    return vals


def differs(cfg, vals, key, sim=None):
    sim = sim or sim_values(cfg)
    a, b = vals[key], sim[key]
    return a != b if BY_KEY[key].fmt == "bool" else not math.isclose(a, b, abs_tol=1e-9)


def changed(cfg, vals, group=None):
    sim = sim_values(cfg)
    return [v.key for v in VARS if (group is None or v.group == group) and differs(cfg, vals, v.key, sim)]


def same(cfg, a, b):
    return not any(differs(cfg, a, k, b) for k in BY_KEY)


def label(cfg, vals):
    """The preset these values are, or 'Custom (n changed)'."""
    for name, p in presets(cfg).items():
        if same(cfg, vals, p):
            return name
    return f"Custom ({len(changed(cfg, vals))} changed)"


class CarPhysics:
    """One set of values, ready to apply: pcfg is the settings copy the
    simulator's functions run with for your car."""

    def __init__(self, cfg, vals):
        self.cfg, self.vals = cfg, dict(vals)
        self.is_sim = not changed(cfg, self.vals)
        v = self.vals
        lf = v["wheelbase"] * LF_SHARE
        over = {var.path: v[var.key] for var in VARS if var.path}
        over.update({"car.s_min": -v["s_max"], "car.sv_min": -v["sv_max"], "car.lf": lf, "car.lr": v["wheelbase"] - lf})
        self.pcfg = cfg.override(over)

    def car(self, design_row):
        """Your car's per-car parameters (design.car_params with your values; brake_max on top)."""
        car = design.car_params(self.pcfg, np.asarray(design_row)[None, :])
        car.brake = car.brake * self.vals["brake_max"] / self.vals["a_max"]
        return car

    def apply(self, env, i, design_row):
        """Set car i up with these values (after env.set_designs, which rebuilds the env's per-car parameters)."""
        mine = self.car(design_row)
        for k, x in vars(mine).items():
            getattr(env.car, k)[i] = x[0]
        rows, v, p = env.param_rows, self.vals, self.pcfg
        steer = float(env.car.steer[i])
        rows[i, P_LF], rows[i, P_LR] = p.car.lf, p.car.lr
        rows[i, P_S_MIN], rows[i, P_S_MAX] = -v["s_max"] * steer, v["s_max"] * steer
        rows[i, P_SV_MIN], rows[i, P_SV_MAX] = -v["sv_max"], v["sv_max"]
        rows[i, P_V_MAX] = max(p.physics_v_max, self.cfg.physics_v_max)
        sim_ = env.unwrapped.sim
        sim_.scan_cache[i] = sim_._build_scan_cache(sim_.scan_sims[i], self.vehicle(sim_.vehicle_params))
        env.side_distances[i] = sim_.scan_cache[i].side_distances

    def vehicle(self, base):
        """The gym's vehicle parameters with your size."""
        v = self.pcfg.vehicle
        return dataclasses.replace(base, width=v.width, length=v.length,
                                   collision_body_center_x=v.collision_body_center_x)

    def at_start(self, env, i):
        """After env.reset: your battery at the start."""
        env.battery[i] = self.vals["ers_start"] * env.car.ers_capacity[i]

    def after_step(self, env, i):
        """Your chase-recharge rule (the simulator sets it at each detection pass)."""
        gap = env.detect_gap[i]
        if np.isfinite(gap) and env.steps * self.cfg.sim.control_dt >= self.cfg.race.detection_from_s:
            env.ers_close[i] = gap < self.vals["ers_close_gap"]


# -------------------------------------------------------------------------
# The steering cap for your car (the simulator's max_steer, with the
# exponent and the steering-scaled switch)
# -------------------------------------------------------------------------

def steer_cap(p, v, grip_factor, df_share, steer_factor, lateral, action=None):
    """Your car's cap at speed v. action: your steer this step - with
    steering scaled off, the env's (action x cap) must come out as
    clip(action x s_max x steer_factor, -cap, cap)."""
    g, vals = p.pcfg.grip, p.vals
    down = g.base_g + df_share * g.downforce_g * (max(v, 0.0) / g.ref_speed) ** vals["exponent"]
    a_lat = 9.81 * grip_factor * vals["grip"] * min(g.max_g, down) * lateral
    cap = steer_factor * min(p.pcfg.vehicle.s_max, math.atan(p.pcfg.wheelbase * a_lat / max(v, 1.0) ** 2))
    if not vals["steering_scaled"] and action is not None and abs(action) > 1e-9:
        return min(abs(action) * p.pcfg.vehicle.s_max * steer_factor, cap) / abs(action)
    return cap


# -------------------------------------------------------------------------
# The simulator's functions, run again for your car
# -------------------------------------------------------------------------

_sim = types.SimpleNamespace(max_steer=physics.max_steer, lateral_share=physics.lateral_share,
                             braking=physics.braking, longitudinal_accel=physics.longitudinal_accel,
                             engine_accel=physics.engine_accel, tow_boost=physics.tow_boost,
                             ers_update=physics.ers_update, car_vertices=sensors.car_vertices)


class Active:
    """Set around each env.step: your car's index, its CarPhysics, the env
    and your steer this step."""
    car = None
    phys = None
    env = None
    steer = None


def _one(car, i):
    return types.SimpleNamespace(**{k: np.asarray(v)[i:i + 1] for k, v in vars(car).items()})


def _at(x, i, n):
    return np.broadcast_to(np.asarray(x, dtype=np.float64), (n,))[i:i + 1]


def _max_steer(cfg, speed, grip_factor=1.0, downforce=1.0, steer_factor=1.0, lateral_share=1.0):
    out = _sim.max_steer(cfg, speed, grip_factor, downforce, steer_factor, lateral_share)
    i, p = Active.car, Active.phys
    if i is None or np.ndim(out) == 0:
        return out
    n = len(out)
    out = np.array(out, dtype=np.float64, copy=True)
    loss = cfg.drs.grip_loss
    drs_open = (1.0 - float(_at(downforce, i, n)[0])) / loss if loss > 0 else 0.0   # the env passes 1 - loss x open
    out[i] = steer_cap(p, float(_at(speed, i, n)[0]), float(_at(grip_factor, i, n)[0]),
                       1.0 - p.vals["drs_grip_loss"] * drs_open, float(_at(steer_factor, i, n)[0]),
                       float(_at(lateral_share, i, n)[0]), Active.steer)
    return out


def _again(name, slots):
    """A wrapper: the simulator's function for every car, then again for
    your car with your settings copy. slots: what each argument is ('car':
    the car namespace, 'x': a per-car number, 'mask': a per-car flag)."""
    original = getattr(_sim, name)

    def wrapper(cfg, *args):
        out = original(cfg, *args)
        i, p = Active.car, Active.phys
        if i is None:
            return out
        out = np.array(out, dtype=np.float64, copy=True)
        n = len(out)
        mine = [(_one(a, i) if kind == "car" else _at(a, i, n) if kind == "x" else
                 np.asarray(a, dtype=bool)[i:i + 1]) for a, kind in zip(args, slots)]
        out[i] = original(p.pcfg, *mine)[0]
        return out

    return wrapper


def _tow_boost(cfg, geo, n):
    out = _sim.tow_boost(cfg, geo, n)
    i, p = Active.car, Active.phys
    if i is None:
        return out
    out = np.array(out, dtype=np.float64, copy=True)
    out[i] = _sim.tow_boost(p.pcfg, geo, n)[i]
    return out


def _ers_update(cfg, car, battery, throttle, brake, ers_request, v, racing, harvest_scale=1.0):
    i, p = Active.car, Active.phys
    if i is None:
        return _sim.ers_update(cfg, car, battery, throttle, brake, ers_request, v, racing, harvest_scale)
    mine = float(battery[i])
    deploy = np.array(_sim.ers_update(cfg, car, battery, throttle, brake, ers_request, v, racing, harvest_scale),
                      dtype=np.float64)
    battery[i] = mine                                  # yours again, with your chase-recharge ratio
    n = len(battery)
    close = float(_at(harvest_scale, i, n)[0]) != 1.0
    e = p.pcfg.ers
    scale = np.array([e.close_harvest_rate / e.harvest_rate if close and e.harvest_rate > 0 else 1.0])
    deploy[i] = _sim.ers_update(p.pcfg, _one(car, i), battery[i:i + 1], _at(throttle, i, n), _at(brake, i, n),
                                np.asarray(ers_request, dtype=bool)[i:i + 1], _at(v, i, n),
                                np.asarray(racing, dtype=bool)[i:i + 1], scale)[0]
    return deploy


def _car_vertices(cfg, xy, yaw):
    """Every racing car's lidar body; yours with your size (the scan update
    passes the racing cars in index order)."""
    verts, centres = _sim.car_vertices(cfg, xy, yaw)
    i, p, env = Active.car, Active.phys, Active.env
    if i is None or env is None or env.done[i]:
        return verts, centres
    racing = np.flatnonzero(~env.done)
    if len(racing) != len(xy):
        return verts, centres
    k = int(np.searchsorted(racing, i))
    v2, c2 = _sim.car_vertices(p.pcfg, xy[k:k + 1], yaw[k:k + 1])
    verts, centres = verts.copy(), centres.copy()
    verts[k], centres[k] = v2[0], c2[0]
    return verts, centres


physics.max_steer = _max_steer           # RaceEnv.step calls these through the module
physics.lateral_share = _again("lateral_share", ("car", "x"))
physics.braking = _again("braking", ("car", "x", "x"))
physics.longitudinal_accel = _again("longitudinal_accel", ("car", "x", "x", "x", "x", "x", "x", "x", "mask"))
physics.engine_accel = _again("engine_accel", ("car", "x", "x", "x"))
physics.tow_boost = _tow_boost
physics.ers_update = _ers_update
sensors.car_vertices = _car_vertices     # the racing scan update calls it through the module


def reset(env):
    """Every car back to the stock body for its lidar and wall check (a new
    race on a reused env; env.set_designs rebuilds the rest)."""
    sim_ = env.unwrapped.sim
    sim_.scan_cache[:] = env._game_stock_cache
    env.side_distances = env._game_stock_side.copy()


def install(env):
    """Per env, once: the simulator's racing step with the wheel speed
    clipped per car (each car's sv_min / sv_max in its parameter row: the
    same for every car the AI drives), and room in the scan's car culling
    for a car bigger than stock."""
    sim_ = env.unwrapped.sim
    env._game_stock_cache = list(sim_.scan_cache)
    env._game_stock_side = env.side_distances.copy()

    def step(sim_self, control_inputs):
        steer = control_inputs[:, 0].astype(np.float32)
        accel = control_inputs[:, 1].astype(np.float32)
        st = sim_self.state
        st.control_input[:, 1] = accel
        if st.delay_buffer is not None:
            steer = st.push_delay(steer)
        st.control_input[:, 0] = steer
        cars = np.flatnonzero(~env.done)
        if len(cars):
            rows = env.param_rows
            rate = np.clip((steer[cars] - st.state[cars, 2]) / sim_self.time_step, rows[cars, P_SV_MIN],
                           rows[cars, P_SV_MAX])
            physics._integrate(st.state, cars, rate.astype(np.float64), accel[cars].astype(np.float64),
                               sim_self.substeps, sim_self.integrator_dt, env.param_rows)
            s = st.state[cars]
            st.standard_state[cars, :5] = s
            st.standard_state[cars, 5:] = 0.0
            st.poses[cars] = s[:, [0, 1, 4]]
        base = sim_self.vehicle_params
        p = Active.phys if Active.car is not None else None
        if p is not None and (p.vals["width"] > base.width or p.vals["length"] > base.length):
            sim_self.vehicle_params = dataclasses.replace(base, width=max(base.width, p.vals["width"]),
                                                          length=max(base.length, p.vals["length"]))
        try:
            sim_self._update_scans()                   # its car culling radius comes from vehicle_params
        finally:
            sim_self.vehicle_params = base
        sim_self._update_agent_collisions()
        st.sim_time += sim_self.time_step

    sim_.step = types.MethodType(step, sim_)


# -------------------------------------------------------------------------
# The pages' preview
# -------------------------------------------------------------------------

PREVIEW_KMH = (54, 90, 126, 180, 234, 288)
CASES = ((None, None), (0.0, 0.0), (0.0, 0.5), (1.0, 0.0))     # steady, lifted, half brake, full throttle


def preview(cfg, vals, design_row):
    """Turning: per speed (km/h, [(lateral g, radius m) steady, lifted,
    half brake, full throttle], the same for the AI's car); the lock radius
    and the time to full lock; and the straight line for you and the AI:
    (top speed, top speed deploying ERS, 0-200 km/h s, 300-100 km/h braking m)."""
    p = CarPhysics(cfg, vals)
    you = p.car(design_row)
    ai_p = CarPhysics(cfg, sim_values(cfg))
    ai = ai_p.car(design.stock(cfg))
    rows = []
    for kmh_ in PREVIEW_KMH:
        v = kmh_ / 3.6
        rows.append((kmh_, [_turn(p, you, v, th, br) for th, br in CASES],
                     [_turn(ai_p, ai, v, th, br) for th, br in CASES]))
    lock = vals["wheelbase"] / math.tan(vals["s_max"] * float(you.steer[0]))
    return rows, lock, vals["s_max"] / vals["sv_max"], _straight(p, you), _straight(ai_p, ai)


def _turn(p, car, v, throttle, brake):
    """throttle None: steady (the tyre's whole grip for cornering)."""
    if throttle is None:
        lat = 1.0
    else:
        th, br, vv = np.array([throttle]), np.array([brake]), np.array([v])
        engine = _sim.engine_accel(p.pcfg, car, vv, np.ones(1), np.zeros(1))
        decel = _sim.braking(p.pcfg, car, th, br)
        lat = float(_sim.lateral_share(p.pcfg, car, th * engine - decel)[0])
    angle = steer_cap(p, v, float(car.grip[0]), 1.0, float(car.steer[0]), lat)
    radius = p.pcfg.wheelbase / math.tan(max(angle, 1e-6))
    return v * v / radius / 9.81, radius


def _straight(p, car):
    cfg = p.pcfg
    top = float(car.v_top[0])
    top_ers = float(((1 + cfg.ers.power_boost) * car.power[0] / car.drag[0]) ** (1 / 3))
    dt, v, t = 0.01, 0.0, 0.0

    def accel(v, throttle, brake):
        vv, th, br = np.array([v]), np.array([throttle]), np.array([brake])
        engine = _sim.engine_accel(cfg, car, vv, np.ones(1), np.zeros(1))
        decel = _sim.braking(cfg, car, th, br)
        return float(_sim.longitudinal_accel(cfg, car, vv, th, decel, engine, np.ones(1), np.zeros(1), np.zeros(1),
                                             np.zeros(1, dtype=bool))[0])

    while v < 200 / 3.6 and t < 60:
        v, t = v + accel(v, 1.0, 0.0) * dt, t + dt
    zero_200 = t if v >= 200 / 3.6 else None
    v, d = min(300 / 3.6, top), 0.0
    while v > 100 / 3.6 and d < 5000:
        v_new = v + accel(v, 0.0, 1.0) * dt
        d += (v + v_new) / 2 * dt
        v = v_new
    return top * 3.6, top_ers * 3.6, zero_200, d
