"""
The Car physics pages (your car only, game/carphysics.py):

- the hub: the preset, every group of settings with how many you changed,
  "Changed values" (only what differs from the AI's car), reset all,
- one page per group: its settings, changed with left / right,
- beside every page: the formula and a live preview - cornering grip and
  the tightest turn per speed (steady, lifted, half brake, full throttle)
  and the straight line (top speed, 0-200 km/h, braking), against the
  AI's car doing the same.
"""

from game import carphysics, ui
from game.ui import QtCore, QtGui, QtWidgets

FORMULA = [
    "max steer = steer_factor × min( s_max , atan( wheelbase × a_lat / v² ) )",
    "a_lat     = 9.81 × grip × min( max_g , base_g + downforce_g × (v / ref_speed)^exponent ) × lateral_share",
    "lateral_share = max( floor , √(1 − u²) )   u = how much of the tyre braking / throttle use",
    "turning radius at the limit = v² / a_lat   ·   your steer × max steer, at up to sv_max rad/s",
]
GROUP_NAMES = {key: name for key, name, _ in carphysics.GROUPS}


class PhysicsPage(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.cfg = app.cfg
        self.sim = carphysics.sim_values(self.cfg)
        self.on_done = None
        self.cache = (None, None)
        self.hub = self._hub()
        self.groups = {key: self._group(key) for key, _, _ in carphysics.GROUPS}
        self.menu = self.hub

    @property
    def vals(self):
        return self.app.settings.physics

    def open(self, on_done):
        self.on_done = on_done
        self.menu = self.hub
        self.hub.sel = 0

    def handle(self, events):
        self.menu.handle(events)
        self.update()

    # ---- rows ----------------------------------------------------------------------------
    def _value(self, var):
        x = self.vals[var.key]
        text = var.show(x)
        if carphysics.differs(self.cfg, self.vals, var.key, self.sim):
            text += f"   AI {var.show(self.sim[var.key])}"
        return text

    def _step(self, var, d):
        if var.fmt == "bool":
            self.vals[var.key] = not self.vals[var.key]
            return
        x = round(self.vals[var.key] + d * var.step, 6)
        self.vals[var.key] = float(min(max(x, var.lo), var.hi))

    def _item(self, var, menu_ref, prefix=""):
        def note():
            m = menu_ref()
            return var.note if m is not None and m.items[m.sel].label == prefix + var.label else None

        return ui.Item(prefix + var.label, lambda: self._value(var), lambda: self._step(var, -1),
                       lambda: self._step(var, 1), note=note)

    def _go(self, menu):
        self.menu = menu
        menu.sel = 0
        menu._skip(1)

    def _back(self):
        """To the hub, on the row you came from."""
        self.menu = self.hub

    def _hub(self):
        cfg = self.cfg
        names = list(carphysics.presets(cfg))
        hub = None

        def set_preset(d):
            current = carphysics.label(cfg, self.vals)
            k = names.index(current) if current in names else (-1 if d > 0 else 0)
            self.vals.update(carphysics.presets(cfg)[names[(k + d) % len(names)]])

        def note_for(label_text, text):
            return lambda: text if hub is not None and hub.items[hub.sel].label == label_text else None

        items = [ui.Item("Preset", lambda: carphysics.label(cfg, self.vals), lambda: set_preset(-1),
                         lambda: set_preset(1), note=note_for("Preset", "AI-equal is exactly the AI's car"))]
        for key, name, about in carphysics.GROUPS:
            def count(k=key):
                n = len(carphysics.changed(cfg, self.vals, k))
                return f"{n} changed" if n else "the AI's values"
            items.append(ui.Item(name, count, activate=(lambda k=key: self._go(self.groups[k])),
                                 right=(lambda k=key: self._go(self.groups[k])), note=note_for(name, about),
                                 gap_before=key == carphysics.GROUPS[0][0]))
        items += [
            ui.Item("Changed values", lambda: str(len(carphysics.changed(cfg, self.vals))) + " changed",
                    activate=self._open_changed, right=self._open_changed, gap_before=True,
                    enabled=lambda: bool(carphysics.changed(cfg, self.vals))),
            ui.Item("Reset everything to the AI's car", activate=lambda: self.vals.update(self.sim)),
            ui.Item("Done", activate=self._done),
        ]
        hub = ui.Menu("Car physics · your car only", items, on_back=self._done,
                      hint="up / down: choose · Enter / A or right: open · left / right: preset · Esc / B: done")
        return hub

    def _group(self, key):
        menu = None
        vars_ = [v for v in carphysics.VARS if v.group == key]
        items = [self._item(v, lambda: menu) for v in vars_]
        items += [ui.Item("Reset this group to the AI's", gap_before=True,
                          activate=lambda: self.vals.update({v.key: self.sim[v.key] for v in vars_})),
                  ui.Item("Back", activate=self._back)]
        menu = ui.Menu(f"Car physics · {GROUP_NAMES[key]}", items, on_back=self._back,
                       hint="up / down: choose · left / right: change · Esc / B: back")
        return menu

    def _open_changed(self):
        """Only the values that differ from the AI's car, each with its group."""
        menu = None
        keys = carphysics.changed(self.cfg, self.vals)
        items = []
        for k in keys:
            var = carphysics.BY_KEY[k]
            item = self._item(var, lambda: menu, prefix=f"{GROUP_NAMES[var.group]} · ")

            def reset_one(var=var):
                self.vals[var.key] = self.sim[var.key]
                self._open_changed()
            item.activate = reset_one
            items.append(item)
        items.append(ui.Item("Back", activate=self._back, gap_before=bool(keys)))
        menu = ui.Menu("Car physics · changed values", items, on_back=self._back,
                       hint="left / right: change · Enter / A: back to the AI's value · Esc / B: back")
        self._go(menu)

    def _done(self):
        if self.on_done:
            self.on_done()

    # ---- paint ---------------------------------------------------------------------------
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), ui.PAGE_BG)
        sc = max(min(self.height() / 900, 1.5), 0.6)
        sc = min(sc, (self.height() - 48) / self.menu.content_height(), self.width() / 1560)
        left = QtCore.QRectF(20, 24 * sc, 700 * sc, self.height() - 48 * sc)
        ui.panel(p, left, QtGui.QColor(28, 29, 33), 12)
        self.menu.paint(p, left, sc, label_w=330 * sc)
        right = QtCore.QRectF(left.right() + 16, left.y(), self.width() - left.right() - 36, left.height())
        ui.panel(p, right, QtGui.QColor(28, 29, 33), 12)
        self._preview(p, right, sc)
        p.end()

    def mousePressEvent(self, event):
        if self.menu.click(event.position(), event.button()):
            self.update()

    def _preview(self, p, r, sc):
        x, y = r.x() + 24 * sc, r.y() + 44 * sc
        ui.text(p, x, y, "The formula", ui.TEXT, ui.font(18 * sc, bold=True))
        y += 26 * sc
        mono = ui.font(12 * sc, mono=True)
        for line in FORMULA:
            ui.text(p, x, y, line, ui.DIM, mono)
            y += 19 * sc
        key = (tuple(sorted(self.vals.items())), tuple(self.app.settings.design))
        if self.cache[0] != key:
            self.cache = (key, carphysics.preview(self.cfg, self.vals, self.app.settings.design))
        rows, lock, lock_s, you, ai = self.cache[1]

        y += 24 * sc
        ui.text(p, x, y, "Cornering", ui.TEXT, ui.font(18 * sc, bold=True))
        y += 22 * sc
        ui.text(p, x, y, "grip · tightest turning radius at full steering · steady: no braking or throttle "
                         "on the tyre · lifted: off the throttle (engine braking)", ui.DIM, ui.font(12.5 * sc))
        y += 28 * sc
        cols = [x, x + 100 * sc, x + 260 * sc, x + 420 * sc, x + 580 * sc, x + 760 * sc]
        for name, cx in zip(("speed", "steady", "lifted", "half brake", "full throttle", "AI car, steady"), cols):
            ui.text(p, cx, y, name, ui.DIM, ui.font(13 * sc, bold=True))
        y += 6 * sc
        cell = ui.font(14 * sc, mono=True)
        for kmh, cells, ai_cells in rows:
            y += 26 * sc
            ui.text(p, cols[0], y, f"{kmh} km/h", ui.TEXT, ui.font(14 * sc))
            for (g, radius), (_, ai_radius), cx in zip(cells, ai_cells, cols[1:5]):
                color = ui.GOOD if radius < ai_radius * 0.98 else ui.BAD if radius > ai_radius * 1.02 else ui.TEXT
                ui.text(p, cx, y, f"{g:4.1f} g {radius:5.0f} m", color, cell)
            ui.text(p, cols[5], y, f"{ai_cells[0][0]:4.1f} g {ai_cells[0][1]:5.0f} m", ui.DIM, cell)
        y += 32 * sc
        ui.text(p, x, y, f"Low speed: tightest turn {lock:.1f} m (wheelbase / tan(s_max)) · "
                         f"the wheel reaches full lock in {lock_s:.2f} s", ui.TEXT, ui.font(14 * sc))

        y += 40 * sc
        ui.text(p, x, y, "Straight line", ui.TEXT, ui.font(18 * sc, bold=True))
        y += 30 * sc
        scols = [x, x + 200 * sc, x + 380 * sc, x + 560 * sc]
        for name, cx in zip(("", "top speed", "with ERS", "0-200 km/h · 300-100 km/h braking"), scols):
            ui.text(p, cx, y, name, ui.DIM, ui.font(13 * sc, bold=True))
        for who, (top, top_ers, t200, brake_m), other in (("you", you, ai), ("AI car", ai, None)):
            y += 26 * sc
            ui.text(p, scols[0], y, who, ui.TEXT, ui.font(14 * sc))

            def col(mine, theirs, lower_better=False):
                if other is None or theirs is None or mine is None:
                    return ui.DIM if other is None else ui.TEXT
                if abs(mine - theirs) <= 0.01 * abs(theirs):
                    return ui.TEXT
                return ui.GOOD if (mine < theirs) == lower_better else ui.BAD

            ui.text(p, scols[1], y, f"{top:5.0f} km/h", col(top, other[0] if other else None), cell)
            ui.text(p, scols[2], y, f"{top_ers:5.0f} km/h", col(top_ers, other[1] if other else None), cell)
            ui.text(p, scols[3], y, f"{t200:4.1f} s" if t200 else "  -  ",
                    col(t200, other[2] if other else None, lower_better=True), cell)
            ui.text(p, scols[3] + 120 * sc, y, f"{brake_m:4.0f} m", col(brake_m, other[3] if other else None,
                                                                          lower_better=True), cell)
        y += 34 * sc
        ui.text(p, x, y, "green / red: better / worse than the AI's car doing the same · the AI always drives "
                         "the simulator's values", ui.DIM, ui.font(12.5 * sc))
        y += 20 * sc
        ui.text(p, x, y, "car size: the AI's short-range 'alongside' sensor still assumes the stock size",
                ui.DIM, ui.font(12.5 * sc))
