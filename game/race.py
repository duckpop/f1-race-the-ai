"""
One race: you against the AI drivers in the simulator's RaceEnv, stepped at
its 20 Hz (the AI was trained at that rate). No Qt in here: the window
(game/app.py) reads the state and draws it.

What the game adds to the simulator's race, all from outside its code:

- Your car takes your inputs. DRS: one press opens it when it's available
  right now (same rules as the AI); braking, the zone ending or a second
  press closes it. ERS: one press on, one off; off by itself when the
  battery is empty.
- The AI keeps its 2 s forced full-throttle launch; you launch yourself.
- Laps races: you finish at your Nth crossing of the line; once the leader
  has finished, every car finishes at its next crossing and leaves the
  track. Timed races end as the simulator's do.
- Respawn mode: on contact (or 6 s stuck) your car is put back on the centre
  line where it was, at a standstill, and is a ghost for 3 s: the other
  cars' sensors don't see it and nothing can touch it (walls still count).
- Car physics (the menu's Car physics page, game/carphysics.py): your car
  can have its own grip, downforce, traction circle, braking and steering
  values; the AI keeps the simulator's.
"""

import math
import types

import numpy as np

from game import sim  # noqa: F401  (the simulator on the import path)
from f1tenth_gym.envs.lidar import check_ttc_jit
from racer import design, sensors
from racer.config import A_BRAKE, A_DRS, A_ERS, A_STEER, A_THROTTLE
from racer.scoring import f1_points
from racer.track import SV
from watch import race_order

from game import carphysics
from game.settings import DIFFICULTY

YOU = "you"
GHOST_S = 3.0
OUT = ("collision", "contact", "no_progress", "sim_error")
MARK_M = 5.0                           # gap timing resolution: a time stamp every 5 m of race distance


# -------------------------------------------------------------------------
# Ghosting: hidden from the other cars' geometry (mirrors, alongside, tow)
# and from their lidar / contact checks
# -------------------------------------------------------------------------

class _GhostGeometry(sensors.Geometry):
    """The simulator's pairwise geometry without the hidden cars."""
    hidden = ()

    def __init__(self, cfg, poses, speed, racing):
        if _GhostGeometry.hidden:
            racing = np.array(racing, dtype=bool, copy=True)
            racing[list(_GhostGeometry.hidden)] = False
        super().__init__(cfg, poses, speed, racing)


sensors.Geometry = _GhostGeometry      # RaceEnv builds sensors.Geometry(...) every step


def _install_ghost_scans(env, race):
    """Wrap the simulator's scan update: while you're a ghost, the other
    cars are scanned as if your car were off the track, and yours sees only
    walls (no car contact either way: car contact is a scan check)."""
    sim_ = env.unwrapped.sim
    original = sim_._update_scans

    def update_scans(sim_self):
        g = race.ghost
        if g is None or env.done[g]:
            return original()
        env.done[g] = True
        try:
            original()
        finally:
            env.done[g] = False
        _wall_scan(sim_self, env, g)

    sim_._update_scans = types.MethodType(update_scans, sim_)


def _wall_scan(sim_, env, i):
    """The simulator's racing scan for one car, walls only."""
    st = sim_.state
    tx, ty, tyaw = sim_.config.lidar_config.base_link_to_lidar_tf
    x, y, yaw = (float(v) for v in st.poses[i])
    c, s = math.cos(yaw), math.sin(yaw)
    origin = np.array([x + tx * c - ty * s, y + tx * s + ty * c, yaw + tyaw])
    scanner, cache = sim_.scan_sims[i], sim_.scan_cache[i]
    clean = scanner.scan(origin, rng=None)
    hit = check_ttc_jit(clean, st.standard_state[i, SV], cache.angles, cache.cosines, cache.side_distances,
                        sim_.ttc_threshold)
    env.wall_hit[i] = hit
    if hit:
        st.state[i, 3:] = 0.0
        st.collisions[i] = 1.0
    else:
        st.collisions[i] = 0.0
    env.wall_scans[i] = clean
    sim_._adjusted_scans[i] = clean.copy()
    noisy = clean + sim_.scan_rngs[i].normal(0.0, scanner.std_dev, size=scanner.num_beams)
    st.scans[i] = np.clip(noisy, scanner.min_range, scanner.max_range)


# -------------------------------------------------------------------------
# Pure helpers
# -------------------------------------------------------------------------

def line_crossings(prev, now, length, t_end, dt):
    """[(time, car, line multiple)] of every forward crossing of the line this
    step, in time order. prev / now: each car's distance past the start line
    (laps x length at its kth crossing); t_end: the clock after the step."""
    out = []
    for i in np.flatnonzero(np.floor(now / length) > np.floor(prev / length)):
        k = int(np.floor(now[i] / length))
        span = now[i] - prev[i]
        frac = (k * length - prev[i]) / span if span > 0 else 1.0
        out.append((t_end - dt + min(max(frac, 0.0), 1.0) * dt, int(i), k))
    return sorted(out)


def signed_s(s, length):
    """Arc length as a distance from the line: (-length/2, length/2]."""
    return np.where(s > length / 2, s - length, s)


def lap_text(n):
    return f"+{n} lap" + ("s" if n > 1 else "")


# -------------------------------------------------------------------------
# The race
# -------------------------------------------------------------------------

class Race:
    def __init__(self, cfg, settings, env, policies, rng, records=None):
        """env: a RaceEnv with settings.cars cars on the track; policies:
        {profile: FrozenPolicy}; records: PlayerRecords (None: don't save)."""
        self.cfg, self.s, self.env, self.policies, self.rng = cfg, settings, env, policies, rng
        self.records = records
        n = env.num_agents
        assert n == settings.cars
        self.h = int(rng.integers(n)) if settings.grid == "random" else min(int(settings.grid), n) - 1
        ai = [p for p, k in settings.ai.items() for _ in range(k)]
        rng.shuffle(ai)
        self.profiles = ai[:self.h] + [YOU] + ai[self.h:]
        self.ai = np.array([i for i in range(n) if i != self.h], dtype=np.int64)
        self.groups = {p: np.array([i for i, q in enumerate(self.profiles) if q == p], dtype=np.int64)
                       for p in dict.fromkeys(ai)}
        env.set_profiles(["middle" if p == YOU else p for p in self.profiles])      # rewards: unused
        env.speed_share[self.ai] = DIFFICULTY[settings.difficulty]
        designs = design.stock(cfg, n)
        designs[self.h] = settings.design
        env.set_designs(designs)
        self.phys = carphysics.CarPhysics(cfg, carphysics.clean(cfg, settings.physics))
        self.ghost = None                                  # your car while it's a ghost, else None
        env._game_race = self                              # the scan wrapper reads this race's ghost
        if not getattr(env, "_game_ghost_scans", False):
            env._game_ghost_scans = True
            _install_ghost_scans(env, _RaceRef(env))
            carphysics.install(env)                        # the wheel speed limit per car
        carphysics.reset(env)
        if not self.phys.is_sim:
            self.phys.apply(env, self.h, settings.design)  # your car's own physics values

        self.obs = env.reset(rng, spawn="grid")
        if not self.phys.is_sim:
            self.phys.at_start(env, self.h)
        env.launch_steps = 0                               # the AI's launch is applied below, not yours
        dt = cfg.sim.control_dt
        self.dt = dt
        self.laps = settings.laps if settings.format == "laps" else None
        env.max_steps = int(round(settings.seconds / dt)) if self.laps is None else 10 ** 9
        self.length = env.tracker.length
        self.line0 = signed_s(env.s.copy(), self.length)  # each car's start, from the line
        self.flag = False                                  # the leader has finished (laps races)
        self.finish = {}                                   # car -> (time, line distance at its finish)
        self.out = {}                                      # car -> (time, reason)
        self.ghost_until = -np.inf
        self.last_respawn = -np.inf
        self.respawns = 0
        self.drs_request = False
        self.ers_on = False
        self.events = []
        self.new_record = False
        self.laps_seen = 0
        top = self.length * (self.laps + 1.5) if self.laps else settings.seconds * cfg.physics_v_max
        self.mark_base = float(self.line0.min()) - 2 * MARK_M
        self.marks = np.full((n, int((top - self.mark_base) / MARK_M) + 2), np.nan)
        self.start_pos = {int(c): k + 1 for k, c in enumerate(race_order(env))}
        self._mark(None, self.line_distance(), 0.0, np.arange(n))

    # ---- state -----------------------------------------------------------------
    @property
    def time(self):
        return self.env.steps * self.dt

    def line_distance(self):
        """Each car's distance past the start line (the kth crossing at k laps)."""
        return self.line0 + self.env.progress

    def you_racing(self):
        return not self.env.done[self.h]

    def you_done(self):
        """Finished or out: the rest of the race can be watched or skipped."""
        return self.h in self.finish or self.h in self.out

    def over(self):
        return bool(self.env.done.all())

    def ghost_active(self):
        return self.you_racing() and self.time < self.ghost_until

    def time_left(self):
        return max(self.env.max_steps * self.dt - self.time, 0.0) if self.laps is None else None

    def your_steer_cap(self, v):
        """Your car's steering cap at speed v, coasting, DRS shut."""
        env, h = self.env, self.h
        return carphysics.steer_cap(self.phys, v, float(env.car.grip[h]), 1.0, float(env.car.steer[h]), 1.0)

    def lap_of(self, i):
        """The lap car i is on (1-based), capped at the race's laps."""
        k = int(np.floor(self.line_distance()[i] / self.length)) + 1
        k = max(k, 1)
        return min(k, self.laps) if self.laps else k

    # ---- one physics step -------------------------------------------------------
    def step(self, inp, presses):
        """inp: DriveInput; presses: actions pressed since the last step."""
        cfg, env, h = self.cfg, self.env, self.h
        self.events = []
        n = env.num_agents
        a = np.zeros((n, cfg.action_dim), dtype=np.float32)
        for prof, idx in self.groups.items():
            live = idx[~env.done[idx]]
            if len(live):
                a[live] = self.policies[prof].act(self.obs[live])
        if env.steps < cfg.launch_steps:                   # lights out: the AI launches flat out
            a[self.ai, A_THROTTLE] = 1.0
            a[self.ai, A_BRAKE] = np.minimum(a[self.ai, A_BRAKE], 0.0)
        if not env.done[h]:
            a[h, A_STEER], a[h, A_THROTTLE], a[h, A_BRAKE] = inp.steer, inp.throttle, inp.brake
            if "drs" in presses:
                if env.drs_open[h] > 0:
                    env.drs_open[h] = 0.0                  # a second press closes it
                    self.events.append(("drs", "closed"))
                elif env.drs_available[h] > 0:
                    self.drs_request = True
                else:
                    self.events.append(("drs", "unavailable"))
            a[h, A_DRS] = 1.0 if self.drs_request or env.drs_open[h] > 0 else -1.0
            if "ers" in presses:
                if self.ers_on:
                    self.ers_on = False
                    self.events.append(("ers", "off"))
                elif env.battery[h] > 0:
                    self.ers_on = True
                    self.events.append(("ers", "on"))
                else:
                    self.events.append(("ers", "empty"))
            a[h, A_ERS] = 1.0 if self.ers_on else -1.0
        ghost = self.ghost_active()
        self.ghost = h if ghost else None
        _GhostGeometry.hidden = (h,) if ghost else ()
        carphysics.Active.car = None if self.phys.is_sim else h
        carphysics.Active.phys, carphysics.Active.env, carphysics.Active.steer = self.phys, env, float(a[h, A_STEER])
        prev = self.line_distance()
        active = np.flatnonzero(~env.done)
        was_open = env.drs_open[h] > 0
        self.obs, _, term, _, _ = env.step(a)
        _GhostGeometry.hidden = ()
        carphysics.Active.car = carphysics.Active.env = None
        if not self.phys.is_sim and not env.done[h]:
            self.phys.after_step(env, h)
        self.actions = a
        t = self.time
        if self.drs_request and env.drs_open[h] > 0:
            self.events.append(("drs", "open"))
        elif was_open and env.drs_open[h] == 0 and not env.done[h]:
            self.events.append(("drs", "closed"))
        self.drs_request = False
        if self.ers_on and env.battery[h] <= 1e-9:
            self.ers_on = False
            self.events.append(("ers", "empty"))

        for i in np.flatnonzero(term):
            reason = env.termination_reason[i]
            if i == h:
                self.events.append(("crash", reason))
                if self.s.crashes == "respawn" and reason in OUT[:3]:
                    self._respawn()
                    continue
            self.out[int(i)] = (t, reason)
        if self.laps:
            self._finishes(prev, t)
        self._mark(prev, self.line_distance(), t, active)
        self._lap_record()
        if self.laps is None and self.over():
            for i in range(n):
                if env.finish_position[i] is not None:
                    self.finish[i] = (t, float(self.line_distance()[i]))

    def _respawn(self):
        """Back on the centre line where you were, at a standstill, as a ghost."""
        env, h = self.env, self.h
        st = env.unwrapped.sim.state
        x, y, yaw = env.tracker.pose_at(env.s[h])
        st.state[h] = (x, y, 0.0, 0.0, yaw)
        st.standard_state[h] = 0.0
        st.standard_state[h, 0], st.standard_state[h, 1], st.standard_state[h, 4] = x, y, yaw
        st.poses[h] = (x, y, yaw)
        st.collisions[h] = 0.0
        env.done[h] = False
        env.termination_reason[h] = None
        env.wall_hit[h] = False
        env.steps_since_best[h] = 0
        env.prev_action[h] = 0.0
        env.drs_open[h] = 0.0
        env.ers_on[h] = 0.0
        env.prev_body_gap[h, :] = np.inf
        env.prev_body_gap[:, h] = np.inf
        env.s[h] = env.tracker.reset_agent(h, x, y)
        self.drs_request = self.ers_on = False
        self.ghost_until = self.time + GHOST_S
        self.last_respawn = self.time
        self.respawns += 1
        self.events.append(("respawn", None))

    def _finishes(self, prev, t):
        """Laps races: the leader finishes at the Nth crossing, then every car
        at its next crossing. Finished cars leave the track."""
        env = self.env
        now = self.line_distance()
        racing = ~env.done
        done_now = []
        for when, i, k in line_crossings(np.where(racing, prev, now), now, self.length, t, self.dt):
            if not racing[i] or (not self.flag and k < self.laps):
                continue
            self.flag = True
            self.finish[i] = (when, k * self.length)
            env.done[i] = True
            env.termination_reason[i] = "finished"
            done_now.append(i)
            if i == self.h:
                self.events.append(("finished", None))
        if done_now and not env.done.all():
            env._clear_cars(np.array(done_now))             # finished cars leave the track

    def _mark(self, prev, now, t, cars):
        """Time stamps every MARK_M of race distance for cars (for the gaps),
        each interpolated inside the step; prev None: the start (time 0 up
        to where each car stands)."""
        top = self.marks.shape[1] - 1
        for i in cars:
            hi = min(int(np.floor((now[i] - self.mark_base) / MARK_M)), top)
            if prev is None:
                self.marks[i, :hi + 1] = 0.0
                continue
            lo = int(np.floor((prev[i] - self.mark_base) / MARK_M)) + 1
            if hi < lo:
                continue
            ks = np.arange(lo, hi + 1)
            span = now[i] - prev[i]
            frac = (self.mark_base + ks * MARK_M - prev[i]) / span if span > 0 else np.ones(len(ks))
            fresh = np.isnan(self.marks[i, ks])
            self.marks[i, ks[fresh]] = (t - self.dt + np.clip(frac, 0.0, 1.0) * self.dt)[fresh]

    def passed_at(self, i, dist):
        """When car i passed race distance dist (interpolated), or None."""
        x = (dist - self.mark_base) / MARK_M
        k = int(np.floor(x))
        if k < 0 or k + 1 >= self.marks.shape[1]:
            return None
        a, b = self.marks[i, k], self.marks[i, k + 1]
        if np.isnan(a):
            return None
        if np.isnan(b):
            return float(a)
        return float(a + (b - a) * (x - k))

    def _lap_record(self):
        laps = self.env.lap_times[self.h]
        if len(laps) <= self.laps_seen:
            return
        self.laps_seen = len(laps)
        lap_s = laps[-1]
        started = self.env.lap_start[self.h] - lap_s
        clean = self.last_respawn < started
        record = False
        if clean and self.records is not None:
            record = self.records.offer(self.env.map_name, lap_s, design.text(self.s.design),
                                        carphysics.label(self.cfg, self.phys.vals))
        self.new_record = self.new_record or record
        self.events.append(("lap", (lap_s, clean, record)))

    # ---- standings ---------------------------------------------------------------
    def classification(self):
        """Every car, best first: [(car, status)], status 'finished',
        'racing' or 'out'. Laps races: by distance past the line (finished
        cars at their finish line distance, ties by finish time); racing cars
        among themselves in the simulator's race order (no flicker side by
        side). Timed races: the simulator's race order, its final order at
        the end. Out cars last, furthest first."""
        env = self.env
        line = self.line_distance()
        racing = [int(i) for i in race_order(env)]
        if self.laps is None and self.finish:
            done = sorted(self.finish, key=lambda i: env.finish_position[i])
            rows = [(i, "finished") for i in done]
        else:
            done = sorted(self.finish, key=lambda i: (-self.finish[i][1], self.finish[i][0]))
            rows, k = [], 0
            for i in racing:                              # merge: finished cars by their finish distance
                while k < len(done) and self.finish[done[k]][1] >= line[i]:
                    rows.append((done[k], "finished"))
                    k += 1
                rows.append((i, "racing"))
            rows += [(i, "finished") for i in done[k:]]
        seen = {i for i, _ in rows}
        out = sorted((i for i in range(env.num_agents) if i not in seen), key=lambda i: -line[i])
        return rows + [(i, "out") for i in out]

    def interval(self, c, p):
        """How far car c is behind car p, as text: '+1.234' or '+1 lap'.
        Both measured where c is now (or where it finished)."""
        line = self.line_distance()
        x = self.finish[c][1] if c in self.finish else line[c]
        t_c = self.finish[c][0] if c in self.finish else self.time
        d_p = self.finish[p][1] if p in self.finish else line[p]
        laps_down = int((d_p - x) // self.length)
        if laps_down >= 1:
            return lap_text(laps_down)
        t_p = self.passed_at(p, x)
        if t_p is None:
            return ""
        return f"+{max(t_c - t_p, 0.0):.3f}"

    def results(self):
        """The final table: [dict] in order (unfinished racing cars by their
        running order, after the finishers)."""
        rows = []
        cls = self.classification()
        leader = cls[0][0] if cls else None
        best = self.env.best_laps()
        pos = 0
        for i, status in cls:
            classified = status != "out"
            if classified:
                pos += 1
            if not classified:
                gap = {"collision": "DNF (wall)", "contact": "DNF (car)", "no_progress": "DNF (stuck)",
                       "sim_error": "DNF (error)"}.get(self.out.get(i, (0, ""))[1], "DNF")
            elif i == leader:
                gap = "winner" if status == "finished" else "leading"
            else:
                gap = self.interval(i, leader)
            rows.append(dict(car=i, pos=pos if classified else None, label=self.label(i), profile=self.profiles[i],
                             design=design.text(self.env.designs[i]), best=best[i], gap=gap,
                             points=f1_points(pos) if classified else 0, you=i == self.h,
                             running=status == "racing"))
        return rows

    def label(self, i):
        return "You" if i == self.h else f"#{i + 1} {self.profiles[i]}"

    def fastest_lap(self):
        """(lap_s, car) of the race's fastest lap, or None."""
        best = [(t, i) for i, t in enumerate(self.env.best_laps()) if t]
        return min(best) if best else None


class _RaceRef:
    """The scan wrapper reads `ghost` from whichever Race uses the env now
    (an env is reused for rematches)."""

    def __init__(self, env):
        self.env = env

    @property
    def ghost(self):
        race = getattr(self.env, "_game_race", None)
        return race.ghost if race is not None else None
