"""
The game window and its loop. One window holds three pages: the setup
menu, the race (the simulator's track view with the HUD over it) and the
mapping screen. A ~60 fps timer reads the controls, steps the race at the
simulator's 20 Hz (catching up at most 3 steps a frame) and draws every
frame, blending the cars between steps.

States: menu -> lights -> racing -> (spectate) -> results, plus paused and
mapping, which return to where they came from.
"""

import gc
import math
import time

import numpy as np

from game import sim  # noqa: F401
from PyQt6 import QtCore, QtGui, QtWidgets

from game import carphysics, inputs, menu, records, settings as settings_mod, store, ui
from game.hud import Hud
from game.mapping import MappingPage
from game.physics_page import PhysicsPage
from game.race import Race
from game.settings import ZOOM_RANGE
from game.view import RaceView, rgb

YOUR_COLOR = "#984ea3"
FRAME_MS = 16
MAX_STEPS_PER_FRAME = 3
LIGHT_GAP_S = 1.0
LIGHTS_OUT_DELAY_S = (0.2, 3.0)
BANNER_S = 2.2


def load_policies(cfg):
    """The three AI drivers' best checkpoints (numpy forward pass)."""
    from racer import config
    from racer.agent import read_actor_weights
    from racer.policy import FrozenPolicy
    out = {}
    for prof in settings_mod.PROFILES:
        path = cfg.checkpoint_path(prof, "best")
        if not path.exists():
            raise SystemExit(f"no AI checkpoint at {config.shown(path)} - is the simulator submodule complete?")
        params, data = read_actor_weights(path, cfg)
        out[prof] = FrozenPolicy(params)
        print(f"[play] AI {prof:<10} {config.shown(path)} ({data['model'].get('total_it', '?')} updates)")
    return out


class Window(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.game = app
        self.setWindowTitle("F1: race the AI")
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)
        self.setStyleSheet("QMainWindow { background: #101114; }")

    def closeEvent(self, event):
        self.game.shutdown()
        event.accept()


class App(QtCore.QObject):
    def __init__(self, qapp):
        super().__init__()
        from racer import config
        from racer.track import available_maps
        self.qapp = qapp
        self.cfg = config.load()
        self.maps = available_maps(self.cfg)
        self.settings = settings_mod.load(self.cfg, self.maps)
        self.records = records.PlayerRecords()
        self.controls = inputs.Controls(store)
        self.policies = load_policies(self.cfg)
        self.rng = np.random.default_rng()
        self.dt = self.cfg.sim.control_dt

        self.win = Window(self)
        self.menu = menu.setup_menu(self)
        self.menu_page = ui.MenuPage(self.menu, extra_paint=self._menu_status)
        self.race_page = QtWidgets.QWidget()
        self.race_layout = QtWidgets.QVBoxLayout(self.race_page)
        self.race_layout.setContentsMargins(0, 0, 0, 0)
        self.mapping_page = MappingPage(self)
        self.physics_page = PhysicsPage(self)
        for page in (self.menu_page, self.race_page, self.mapping_page, self.physics_page):
            self.win.stack.addWidget(page)

        self.state = "menu"
        self.back_state = None                  # where paused / mapping return to
        self.env = self.view = self.race = self.hud = None
        self.env_key = None
        self.loading = ""
        self.rotate = True
        self.acc = 0.0
        self.pending = set()
        self.lights = None
        self.banners = []
        self.inp = inputs.DriveInput()
        self.ai_record = None
        self.results_rows = []
        self.crash_rumble_until = 0.0
        self.pause_menu = menu.pause_menu(self)
        self.results_menu_obj = menu.results_menu(self)

        qapp.installEventFilter(self)
        self.win.showMaximized()
        self.last = time.monotonic()
        self.timer = QtCore.QTimer()
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(FRAME_MS)                    # pads already plugged in arrive as plug events

    # ---- keyboard ----------------------------------------------------------------------
    def eventFilter(self, obj, event):
        t = event.type()
        if t in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease) and self.win.isActiveWindow():
            try:
                name = QtCore.Qt.Key(event.key()).name
            except ValueError:
                return False
            if t == QtCore.QEvent.Type.KeyPress:
                if name == "Key_F11" and not event.isAutoRepeat():
                    self.win.showMaximized() if self.win.isFullScreen() else self.win.showFullScreen()
                    return True
                self.controls.key_down(name, event.isAutoRepeat())
            else:
                self.controls.key_up(name, event.isAutoRepeat())
            return True
        if t == QtCore.QEvent.Type.ApplicationDeactivate or (obj is self.win and t ==
                                                               QtCore.QEvent.Type.WindowDeactivate):
            self.controls.release_all()
        return False

    # ---- the loop ----------------------------------------------------------------------
    def tick(self):
        now = time.monotonic()
        dt = min(now - self.last, 0.25)
        self.last = now
        need = self.controls.poll_pads()
        if need and self.state != "mapping":
            self._new_pad(need[0])
        inp, presses, menu_ev = self.controls.frame(dt)
        keys = self.controls.frame_keys
        self.inp = inp
        st = self.state
        if st == "menu":
            self.menu.handle(menu_ev)
            self.menu_page.update()
        elif st == "mapping":
            self.mapping_page.tick(menu_ev, keys)
        elif st == "physics":
            self.physics_page.handle(menu_ev)
        elif st == "lights":
            if "pause" in presses:
                self.pause()
            elif "restart" in presses:
                self.restart()
            else:
                if "camera" in presses:
                    self.rotate = not self.rotate
                self._lights_tick(now)
        elif st in ("racing", "spectate"):
            if "pause" in presses:
                self.pause()
            elif st == "racing" and "restart" in presses:
                self.restart()
            elif st == "spectate" and (keys or presses or menu_ev):
                self._show_results()
            else:
                if "camera" in presses:
                    self.rotate = not self.rotate
                self._physics(dt, inp, presses)
        elif st == "paused":
            if "pause" in presses or "back" in menu_ev:
                self.resume()
            else:
                self.pause_menu.handle(menu_ev)
        elif st == "results":
            self.results_menu_obj.handle(menu_ev)
        if self.race is not None and self.win.stack.currentWidget() is self.race_page:
            self._draw(dt)

    def _physics(self, dt, inp, presses):
        race = self.race
        self.pending |= presses & {"drs", "ers"}
        self.acc += dt
        steps = 0
        while self.acc >= self.dt and steps < MAX_STEPS_PER_FRAME:
            race.step(inp, self.pending)
            self.pending = set()
            self.acc -= self.dt
            steps += 1
            self.view.stepped()
            self.view.set_ghost(race.h if race.ghost_active() else None)
            self._events(race.events)
            self._grip_rumble()
            if race.over():
                self._show_results()
                return
            if self.state == "racing" and race.you_done():
                self.state = "spectate"
        self.acc = min(self.acc, self.dt)                 # a long stall doesn't fast-forward

    def _draw(self, dt):
        race, view = self.race, self.view
        if self.state in ("racing", "spectate"):
            alpha = self.acc / self.dt
        else:
            alpha = 1.0
        follow = race.h
        if race.you_done():
            order = [i for i, s in race.classification() if s == "racing"]
            follow = order[0] if order else None
        speed = float(race.env.std[follow, 3]) if follow is not None else 0.0
        view.draw(alpha, follow, self.rotate, self.settings.zoom, dt, speed)
        if self.hud.size() != view.view.size():
            self.hud.setGeometry(view.view.rect())
        self.hud.update()

    # ---- race start / lights -----------------------------------------------------------
    def start_race(self):
        s = self.settings
        s.clamp_grid()
        settings_mod.save(s)
        map_name = str(self.rng.choice(sorted(self.maps))) if s.track == "Random" else s.track
        self._prepare(map_name)
        self._new_race()

    def _prepare(self, map_name):
        """The env (and its view) for this track and car count; reused when
        they match the last race's."""
        key = (map_name, self.settings.cars)
        if key == self.env_key:
            return
        self.loading = f"Loading {map_name}..."
        self.menu_page.repaint()
        self.qapp.processEvents()
        if self.env is not None:
            self.view.close()
            self.hud.deleteLater()
            self.env.close()
            self.env = self.view = self.hud = None
            gc.collect()
        from racer.env import RaceEnv
        show = QtWidgets.QMainWindow.show
        QtWidgets.QMainWindow.show = lambda w: None      # the gym opens its own window: keep it hidden
        try:
            self.env = RaceEnv(self.cfg, self.settings.cars, map_name, ["middle"], render=True)
        finally:
            QtWidgets.QMainWindow.show = show
        self.view = RaceView(self.cfg, self.env, self.maps[map_name], self._zoom)
        self.race_layout.addWidget(self.view.view)
        self.hud = Hud(self.view.view, self)
        self.hud.show()
        self.env_key = key
        self.ai_record = records.ai_record(self.cfg, map_name)
        self.loading = ""

    def _new_race(self):
        """A race on the current env with the current settings: the lights next."""
        s = self.settings
        self.race = Race(self.cfg, s, self.env, self.policies, self.rng, self.records)
        stock, mine = self.cfg.vehicle, self.race.phys.pcfg.vehicle
        self.view.set_sizes([(mine.length, mine.width) if i == self.race.h else (stock.length, stock.width)
                             for i in range(self.env.num_agents)])
        self.view.paint([rgb(self.car_hex(i)) for i in range(self.env.num_agents)])
        self.view.snap()
        self.hud.cache_step = -1
        self.acc, self.pending, self.banners = 0.0, set(), []
        self.lights = {"start": time.monotonic(), "out_after": 5 * LIGHT_GAP_S + float(self.rng.uniform(
            *LIGHTS_OUT_DELAY_S)), "out_at": None}
        self.state = "lights"
        self.hud.set_clickable(False)
        self.win.stack.setCurrentWidget(self.race_page)
        r = self.race
        length = ui.laps_text(r.laps) if r.laps else ui.clock(s.seconds)
        print(f"[play] {self.env.map_name} · {length} · {self.env.num_agents} cars · "
              f"{menu.DIFFICULTY_NAMES[s.difficulty]} · crashes: {s.crashes} · car physics: "
              f"{carphysics.label(self.cfg, r.phys.vals)} · you start P{r.h + 1}")

    def _lights_tick(self, now):
        L = self.lights
        if now - L["start"] >= L["out_after"]:
            L["out_at"] = now
            self.state = "racing"
            self.last = now
            self.acc = 0.0

    def lights_state(self):
        """(lights lit 0-5, the time they went out or None) for the HUD; (None, None) without lights."""
        L = self.lights
        if L is None:
            return None, None
        if L["out_at"] is not None:
            return 5, L["out_at"]
        start = L.get("paused_at", time.monotonic())
        return min(int((start - L["start"]) / LIGHT_GAP_S), 5), None

    # ---- pause / restart / menu / quit ---------------------------------------------------
    def pause(self):
        self.back_state = self.state
        self.state = "paused"
        if self.lights and self.lights["out_at"] is None:
            self.lights["paused_at"] = time.monotonic()
        self.pause_menu.sel = 0
        self.hud.set_clickable(True)
        self.controls.rumble(0, 0, 1)

    def resume(self):
        if self.state != "paused":
            return
        L = self.lights
        if L and "paused_at" in L:
            L["start"] += time.monotonic() - L.pop("paused_at")
        self.state = self.back_state or "racing"
        self.hud.set_clickable(False)
        self.last = time.monotonic()

    def restart(self):
        """The same race again (same track; a new slot if yours is random)."""
        self._new_race()

    def to_menu(self):
        self.state = "menu"
        self.race = None
        self.win.stack.setCurrentWidget(self.menu_page)
        self.controls.rumble(0, 0, 1)

    def quit(self):
        self.win.close()

    def shutdown(self):
        settings_mod.save(self.settings)
        self.controls.save()
        self.timer.stop()
        self.controls.rumble(0, 0, 1)
        if self.env is not None:
            self.env.close()

    # ---- the mapping screen --------------------------------------------------------------
    def open_mapping(self, pad=None, message=""):
        if self.state != "mapping":
            self.back_state = self.state
        self.state = "mapping"
        self.mapping_page.open(pad, self.close_mapping, message)
        self.win.stack.setCurrentWidget(self.mapping_page)

    def close_mapping(self):
        self.state = self.back_state or "menu"
        page = self.race_page if self.state in ("paused", "results") else self.menu_page
        self.win.stack.setCurrentWidget(page)

    # ---- the car physics page ------------------------------------------------------------
    def open_physics(self):
        self.state = "physics"
        self.physics_page.open(self.close_physics)
        self.win.stack.setCurrentWidget(self.physics_page)

    def close_physics(self):
        settings_mod.save(self.settings)
        self.state = "menu"
        self.win.stack.setCurrentWidget(self.menu_page)

    def _new_pad(self, pad):
        """A controller with an unknown layout: map it now (pausing a race)."""
        if self.state in ("lights", "racing", "spectate"):
            self.pause()
        self.open_mapping(pad, f"New controller: {pad.name} - map its controls, then Done")

    # ---- results ------------------------------------------------------------------------
    def _show_results(self):
        self.results_rows = self.race.results()
        self.state = "results"
        self.results_menu_obj.sel = 0
        self.hud.set_clickable(True)
        self.controls.rumble(0, 0, 1)
        r = self.race
        print(f"\n[play] results · {r.env.map_name} · " + (ui.laps_text(r.laps) if r.laps else ui.clock(r.s.seconds)))
        print(f"  {'pos':>4}  {'driver':<16} {'design':<10} {'best lap':>9}  {'gap':<12} {'pts':>3}")
        for row in self.results_rows:
            pos = f"P{row['pos']}" if row["pos"] else "-"
            name = row["label"] + (" (running)" if row["running"] else "")
            print(f"  {pos:>4}  {name:<16} {row['design']:<10} {ui.lap_time(row['best']) if row['best'] else '-':>9}"
                  f"  {row['gap']:<12} {row['points']:>3}")
        if r.new_record:
            print("  new lap record for you here!")

    def overlay_menu(self):
        return self.pause_menu if self.state == "paused" else self.results_menu_obj if self.state == "results" \
            else None

    # ---- feedback: messages and rumble ---------------------------------------------------
    def _banner(self, msg, color="#eeeeee", seconds=BANNER_S):
        now = time.monotonic()
        self.banners = [b for b in self.banners if b[2] > now][-2:] + [(msg, color, now + seconds)]

    def _events(self, events):
        for kind, data in events:
            if kind == "drs":
                self._banner({"open": "DRS open", "closed": "DRS closed",
                              "unavailable": "DRS not available here"}[data],
                             "#4caf50" if data == "open" else "#9e9e9e", 1.2)
            elif kind == "ers":
                self._banner({"on": "ERS on", "off": "ERS off", "empty": "Battery empty - ERS off"}[data],
                             "#42a5f5" if data == "on" else "#9e9e9e", 1.2)
            elif kind == "crash":
                self.controls.rumble(1.0, 1.0, 400)
                self.crash_rumble_until = time.monotonic() + 0.4
                if self.settings.crashes == "out" or data not in ("collision", "contact", "no_progress"):
                    self._banner("Out: " + {"collision": "hit the wall", "contact": "hit a car",
                                            "no_progress": "stuck"}.get(data, data), "#ff5252", 3.0)
            elif kind == "respawn":
                self._banner("Respawned · ghost for 3 s", "#ce93d8")
            elif kind == "finished":
                self._banner("Chequered flag!", "#eeeeee", 3.0)
            elif kind == "lap":
                lap_s, clean, record = data
                msg = f"Lap {ui.lap_time(lap_s)}" + (" · new record!" if record else "") + \
                      ("" if clean else " · not a record lap (respawn)")
                self._banner(msg, "#4caf50" if record else "#eeeeee")

    def _grip_rumble(self):
        """A light buzz while your tyres are at the limit: steering at the
        grip limit (in the speed range where grip, not lock, caps it) or
        heavy braking at speed."""
        race = self.race
        if not race.you_racing() or time.monotonic() < self.crash_rumble_until:
            return
        env, h = race.env, race.h
        v = float(env.std[h, 3])
        a = race.actions[h]
        grip_capped = race.your_steer_cap(v) < self.cfg.vehicle.s_max * env.car.steer[h] * 0.999
        at_limit = (grip_capped and abs(a[0]) >= 0.95) or (a[2] >= 0.9 and v > 25.0)
        if at_limit:
            self.controls.rumble(0.25, 0.05, 120)

    # ---- for the HUD and the menu ----------------------------------------------------------
    def car_hex(self, i):
        if self.race is None:
            return "#888888"
        return YOUR_COLOR if i == self.race.h else self.cfg.color(self.race.profiles[i])

    def shown_input(self):
        """(steer +right, throttle, brake): what your car used last step,
        or what you're pressing before lights out."""
        race = self.race
        if self.state == "lights" or not race.you_racing() or not hasattr(race, "actions"):
            return self.inp.shown_steer, self.inp.throttle, self.inp.brake
        a = race.actions[race.h]
        return -float(a[0]), max(float(a[1]), 0.0), max(float(a[2]), 0.0)

    def _zoom(self, factor):
        lo, hi = ZOOM_RANGE
        self.settings.zoom = float(min(max(self.settings.zoom * factor, lo), hi))

    def _menu_status(self, p, rect, scale):
        pads = self.controls.connected()
        s = ("Controller: " + ", ".join(p_.name for p_ in pads)) if pads else "No controller connected · keyboard"
        ui.text(p, rect.x() + 28 * scale, rect.bottom() - 40 * scale, s, ui.DIM, ui.font(13 * scale))
        if self.loading:
            ui.text(p, rect.right() - 28 * scale, rect.y() + 52 * scale, self.loading, ui.AMBER,
                    ui.font(17 * scale, bold=True), align="right")
