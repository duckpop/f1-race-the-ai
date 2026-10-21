"""
The race HUD, one transparent widget over the track view:

- left: the standings (interval to the car ahead, places gained / lost),
- top right: lap N of M or the time left, your current / last / best lap,
  the race's fastest lap, your record and the AI's record here,
- under it: the gaps to the cars ahead and behind you,
- bottom right: your car (place, speed, steer / throttle / brake, DRS and
  ERS lights, battery, the detection point),
- the start lights, short messages, and the pause / results boxes.
"""

import math
import time

import numpy as np

from game import sim  # noqa: F401
from racer.track import SV

from game import ui
from game.ui import QtCore, QtGui, QtWidgets, Qt

OUT_TEXT = {"collision": "hit the wall", "contact": "hit a car", "no_progress": "stuck", "sim_error": "sim error"}


class Hud(QtWidgets.QWidget):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.cache_step, self.rows = -1, []

    def set_clickable(self, on):
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not on)

    def mousePressEvent(self, event):
        menu = self.app.overlay_menu()
        if menu is not None and menu.click(event.position(), event.button()):
            self.update()

    # ---- standings data (once per physics step) ------------------------------------
    def standings(self):
        race = self.app.race
        if race.env.steps != self.cache_step or not self.rows:
            self.cache_step = race.env.steps
            rows, above = [], None
            for k, (i, status) in enumerate(race.classification()):
                if status == "out":
                    gap = "DNF"
                elif above is None:
                    gap = "Finish" if status == "finished" else "Leader"
                else:
                    gap = race.interval(i, above)
                rows.append((i, status, gap))
                if status != "out":
                    above = i
            self.rows = rows
        return self.rows

    # ---- paint -----------------------------------------------------------------------
    def paintEvent(self, event):
        app = self.app
        if app.race is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        sc = max(min(self.height() / 900, 1.6), 0.7)
        self.sc = sc
        self._standings(p, sc)
        right = self.width() - 12 * sc
        y = self._laps(p, sc, right)
        self._gaps(p, sc, right, y)
        self._car(p, sc, right)
        self._lights(p, sc)
        self._banners(p, sc)
        if app.state == "spectate":
            self._spectate(p, sc)
        text_hint = ("C: north-up camera" if app.rotate else "C: rotating camera") + " · wheel: zoom · Esc / Start: pause"
        ui.text(p, 14 * sc, self.height() - 12 * sc, text_hint, ui.FAINT, ui.font(12 * sc))
        if app.state == "paused":
            self._menu_box(p, sc, app.pause_menu, 420)
        elif app.state == "results":
            self._results(p, sc)
        p.end()

    def _standings(self, p, sc):
        race = self.app.race
        rows = self.standings()
        row_h = 19 * sc
        w = 300 * sc
        h = (len(rows) * row_h) + 44 * sc
        ui.panel(p, QtCore.QRectF(10 * sc, 10 * sc, w, h))
        if race.laps:
            lap = race.lap_of(race.h) if not race.you_done() else race.laps
            leader_lap = max((race.lap_of(i) for i, st, _ in rows if st == "racing"), default=race.laps)
            title = f"Lap {min(leader_lap, race.laps)} / {race.laps}" + ("  · chequered flag" if race.flag else "")
        else:
            title = f"{ui.clock(race.time_left())} left"
        ui.text(p, 22 * sc, 34 * sc, title, ui.TEXT, ui.font(15 * sc, bold=True))
        ui.text(p, 10 * sc + w - 12 * sc, 34 * sc, ui.clock(race.time), ui.DIM, ui.font(13 * sc), align="right")
        y = 46 * sc
        pos = 0
        mono = ui.font(13 * sc, mono=True)
        for i, status, gap in rows:
            you = i == race.h
            if you:
                ui.panel(p, QtCore.QRectF(14 * sc, y - 1, w - 8 * sc, row_h), ui.ACCENT_SOFT, 4)
            base = y + row_h * 0.74
            if status != "out":
                pos += 1
                ui.text(p, 40 * sc, base, f"P{pos}", ui.TEXT, ui.font(13 * sc, bold=True), align="right")
                moved = race.start_pos.get(i, pos) - pos
                arrow, col = (f"▲{moved}", ui.GOOD) if moved > 0 else (f"▼{-moved}", ui.BAD) if moved < 0 \
                    else ("–", ui.FAINT)
                ui.text(p, 76 * sc, base, arrow, col, ui.font(12 * sc), align="right")
            else:
                ui.text(p, 40 * sc, base, "–", ui.FAINT, ui.font(13 * sc), align="right")
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(self.app.car_hex(i)))
            p.drawRect(QtCore.QRectF(86 * sc, y + 4 * sc, 10 * sc, 10 * sc))
            name = race.label(i) + (" ✓" if status == "finished" else "")
            ui.text(p, 102 * sc, base, name, ui.TEXT if status != "out" else ui.FAINT,
                    ui.font(13 * sc, bold=you))
            gap_col = ui.BAD if gap == "DNF" else ui.DIM
            ui.text(p, 10 * sc + w - 12 * sc, base, gap, gap_col, mono, align="right")
            y += row_h

    def _laps(self, p, sc, right):
        app, race = self.app, self.app.race
        env, h = race.env, race.h
        w, x = 300 * sc, right - 300 * sc
        lines = []
        laps = env.lap_times[h]
        current = race.time - env.lap_start[h] if np.isfinite(env.lap_start[h]) and race.you_racing() else None
        lines.append(("Current", ui.lap_time(current) if current is not None else "-", ui.TEXT))
        lines.append(("Last", ui.lap_time(laps[-1]) if laps else "-", ui.TEXT))
        lines.append(("Best", ui.lap_time(min(laps)) if laps else "-", ui.GOOD if laps else ui.TEXT))
        fast = race.fastest_lap()
        lines.append(("Race fastest", f"{ui.lap_time(fast[0])}  {race.label(fast[1])}" if fast else "-",
                      ui.QtGui.QColor("#ce93d8") if fast and fast[1] == h else ui.DIM))
        mine = app.records.get(env.map_name)
        lines.append(("Your record", ui.lap_time(mine["lap_s"]) if mine else "-", ui.DIM))
        ai = app.ai_record
        lines.append(("AI record", f"{ui.lap_time(ai[0])}  {ai[1]}" if ai else "-", ui.DIM))
        h_px = (54 + 20 * len(lines)) * sc
        ui.panel(p, QtCore.QRectF(x, 10 * sc, w, h_px))
        if race.laps:
            big = f"LAP {race.lap_of(h) if not race.you_done() else race.laps} / {race.laps}"
        else:
            big = f"TIME LEFT {ui.clock(race.time_left())}"
        ui.text(p, x + 14 * sc, 42 * sc, big, ui.TEXT, ui.font(22 * sc, bold=True))
        y = 66 * sc
        for label, value, col in lines:
            ui.text(p, x + 14 * sc, y, label, ui.DIM, ui.font(13 * sc))
            ui.text(p, x + 120 * sc, y, value, col, ui.font(13 * sc, mono=True))
            y += 20 * sc
        return 10 * sc + h_px + 8 * sc

    def _gaps(self, p, sc, right, y):
        race = self.app.race
        rows = [r for r in self.standings() if r[1] != "out"]
        idx = next((k for k, r in enumerate(rows) if r[0] == race.h), None)
        w, x = 300 * sc, right - 300 * sc
        ui.panel(p, QtCore.QRectF(x, y, w, 58 * sc))
        ahead = rows[idx - 1][0] if idx is not None and idx > 0 else None
        behind = rows[idx + 1][0] if idx is not None and idx + 1 < len(rows) else None
        for k, (label, other, gap) in enumerate((
                ("Ahead", ahead, race.interval(race.h, ahead) if ahead is not None else ""),
                ("Behind", behind, race.interval(behind, race.h).replace("+", "-", 1) if behind is not None else ""))):
            yy = y + (24 + 22 * k) * sc
            ui.text(p, x + 14 * sc, yy, label, ui.DIM, ui.font(13 * sc))
            if other is None:
                ui.text(p, x + 80 * sc, yy, "–", ui.FAINT, ui.font(13 * sc))
                continue
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(self.app.car_hex(other)))
            p.drawRect(QtCore.QRectF(x + 80 * sc, yy - 10 * sc, 10 * sc, 10 * sc))
            ui.text(p, x + 96 * sc, yy, race.label(other), ui.TEXT, ui.font(13 * sc))
            ui.text(p, x + w - 14 * sc, yy, gap, ui.TEXT, ui.font(14 * sc, bold=True, mono=True), align="right")

    def _car(self, p, sc, right):
        app, race = self.app, self.app.race
        env, h = race.env, race.h
        w, hh = 330 * sc, 214 * sc
        x, y = right - w, self.height() - hh - 10 * sc
        ui.panel(p, QtCore.QRectF(x, y, w, hh))
        rows = [r for r in self.standings() if r[1] != "out"]
        place = next((k + 1 for k, r in enumerate(rows) if r[0] == h), None)
        if race.you_racing():
            head = f"P{place} / {env.num_agents}" if place else ""
        elif h in race.finish:
            head = f"Finished P{place}" if place else "Finished"
        else:
            head = "Out: " + OUT_TEXT.get(race.out.get(h, (0, ""))[1], "out")
        ui.text(p, x + 14 * sc, y + 30 * sc, head, ui.TEXT, ui.font(20 * sc, bold=True))
        kmh = float(env.std[h, SV]) * 3.6 if race.you_racing() else 0.0
        ui.text(p, x + w - 14 * sc, y + 30 * sc, f"{kmh:.0f} km/h", ui.TEXT, ui.font(20 * sc, bold=True),
                align="right")
        if race.ghost_active():
            ui.text(p, x + w / 2, y + 30 * sc, f"GHOST {race.ghost_until - race.time:.1f}s", ui.DIM,
                    ui.font(12 * sc, bold=True), align="center")
        a = app.shown_input()
        bx, bw = x + 86 * sc, w - 150 * sc
        for k, (name, v, fill, centred) in enumerate((("steer", a[0], "#e0e0e0", True),
                                                       ("throttle", a[1], "#4caf50", False),
                                                       ("brake", a[2], "#ef5350", False))):
            yy = y + (46 + 22 * k) * sc
            ui.text(p, x + 14 * sc, yy + 11 * sc, name, ui.DIM, ui.font(12 * sc))
            ui.bar(p, bx, yy, bw, 13 * sc, v, fill, centred)
            ui.text(p, bx + bw + 8 * sc, yy + 11 * sc, f"{v:+.2f}" if centred else f"{v:.0%}", ui.TEXT,
                    ui.font(12 * sc, mono=True))

        def light(cx, cy, name, ring, fill, state):
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2.5 * sc) if ring else QtGui.QPen(QtGui.QColor(90, 90, 90), 1))
            p.setBrush(QtGui.QColor(fill))
            p.drawEllipse(QtCore.QPointF(cx, cy), 14 * sc, 14 * sc)
            ui.text(p, cx, cy + 30 * sc, name, ui.DIM, ui.font(12 * sc), align="center")
            ui.text(p, cx, cy + 45 * sc, state, ui.TEXT, ui.font(11 * sc), align="center")

        ly = y + 132 * sc
        drs_open, avail, armed = env.drs_open[h] > 0, env.drs_available[h] > 0, bool(env.drs_armed[h])
        drs_fill = "#4caf50" if drs_open else "#ffb300" if avail else "#6d5200" if armed else "#424242"
        light(x + 36 * sc, ly, "DRS", drs_open, drs_fill,
              "open" if drs_open else "available" if avail else "armed" if armed else "off")
        deploying = env.ers_on[h] > 0
        light(x + 100 * sc, ly, "ERS", race.ers_on, "#42a5f5" if deploying else "#1d4f75" if race.ers_on else "#424242",
              "deploying" if deploying else "on" if race.ers_on else "off")
        battery = float(env.battery[h] / env.car.ers_capacity[h])
        ui.text(p, x + 146 * sc, ly - 8 * sc, "battery", ui.DIM, ui.font(12 * sc))
        ui.bar(p, x + 146 * sc, ly - 2 * sc, w - 210 * sc, 12 * sc, battery, "#42a5f5")
        ui.text(p, x + w - 14 * sc, ly + 9 * sc, f"{battery:.0%}", ui.TEXT, ui.font(12 * sc, mono=True), align="right")
        gap = float(env.detect_gap[h])
        notes = []
        if race.time < app.cfg.drs.from_s:
            notes.append((f"DRS from {app.cfg.drs.from_s:.0f} s", ui.DIM))
        elif np.isfinite(gap):
            notes.append((f"detection: {gap:.2f} s behind", ui.AMBER if armed else ui.DIM))
            if armed:
                notes.append(("DRS armed", ui.AMBER))
        else:
            notes.append(("detection: not passed yet", ui.DIM))
        if env.ers_close[h]:
            notes.append(("chase recharge ×1.5", ui.GOOD))
        for k, (note, col) in enumerate(notes):
            ui.text(p, x + 146 * sc, ly + (30 + 16 * k) * sc, note, col, ui.font(12 * sc))

    def _lights(self, p, sc):
        lit, out_at = self.app.lights_state()
        if lit is None:
            return
        now = time.monotonic()
        if out_at is not None and now - out_at > 1.2:
            return
        r = 22 * sc
        w = 5 * 2 * r + 4 * 16 * sc + 40 * sc
        x0 = self.width() / 2 - w / 2
        ui.panel(p, QtCore.QRectF(x0, 14 * sc, w, 2 * r + 30 * sc), ui.QtGui.QColor(10, 10, 10, 230), 10)
        for k in range(5):
            cx = x0 + 20 * sc + r + k * (2 * r + 16 * sc)
            on = out_at is None and k < lit
            p.setPen(QtGui.QPen(QtGui.QColor(70, 70, 70), 2))
            p.setBrush(QtGui.QColor("#e53935") if on else QtGui.QColor(35, 35, 35))
            p.drawEllipse(QtCore.QPointF(cx, 14 * sc + 15 * sc + r), r, r)
        if out_at is not None:
            ui.text(p, self.width() / 2, 14 * sc + 2 * r + 70 * sc, "GO", ui.GOOD, ui.font(40 * sc, bold=True),
                    align="center")

    def _banners(self, p, sc):
        now = time.monotonic()
        y = self.height() * 0.22
        for msg, color, until in self.app.banners:
            if until < now:
                continue
            f = ui.font(22 * sc, bold=True)
            p.setFont(f)
            w = p.fontMetrics().horizontalAdvance(msg) + 36 * sc
            ui.panel(p, QtCore.QRectF(self.width() / 2 - w / 2, y - 30 * sc, w, 42 * sc), ui.BG, 8)
            ui.text(p, self.width() / 2, y, msg, QtGui.QColor(color), f, align="center")
            y += 50 * sc

    def _spectate(self, p, sc):
        race = self.app.race
        h = race.h
        place = next((k + 1 for k, r in enumerate(r for r in self.standings() if r[1] != "out") if r[0] == h), None)
        what = f"You finished P{place}" if h in race.finish else \
            f"You're out ({OUT_TEXT.get(race.out.get(h, (0, ''))[1], 'out')})"
        msg = f"{what} · watching the race · any key or button: results"
        f = ui.font(16 * sc, bold=True)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(msg) + 40 * sc
        ui.panel(p, QtCore.QRectF(self.width() / 2 - w / 2, self.height() - 70 * sc, w, 40 * sc))
        ui.text(p, self.width() / 2, self.height() - 44 * sc, msg, ui.TEXT, f, align="center")

    def _menu_box(self, p, sc, menu, width, top=None, height=None):
        n = len(menu.items)
        w = width * sc
        hh = height if height is not None else (110 + 34 * n) * sc
        x = self.width() / 2 - w / 2
        y = top if top is not None else self.height() / 2 - hh / 2
        r = QtCore.QRectF(x, y, w, hh)
        ui.panel(p, r, ui.QtGui.QColor(24, 24, 28, 240), 12)
        menu.paint(p, r, sc, label_w=180 * sc)
        return r

    def _results(self, p, sc):
        app, race = self.app, self.app.race
        rows = app.results_rows
        row_h = 22 * sc
        w = min(820 * sc, self.width() - 40)
        table_h = (90 + len(rows) * 22) * sc
        menu_h = (10 + 34 * len(app.results_menu_obj.items) + 40) * sc
        total = table_h + menu_h
        x = self.width() / 2 - w / 2
        y = max(self.height() / 2 - total / 2, 10)
        ui.panel(p, QtCore.QRectF(x, y, w, total), ui.QtGui.QColor(24, 24, 28, 245), 12)
        title = f"Results · {race.env.map_name} · " + (ui.laps_text(race.laps) if race.laps else
                                                     ui.clock(race.s.seconds))
        ui.text(p, x + 24 * sc, y + 40 * sc, title, ui.TEXT, ui.font(24 * sc, bold=True))
        if race.new_record:
            ui.text(p, x + w - 24 * sc, y + 40 * sc, "new lap record!", ui.GOOD, ui.font(15 * sc, bold=True),
                    align="right")
        cols = [("Pos", 24, "left"), ("Driver", 74, "left"), ("Design", 290, "left"), ("Best lap", 400, "left"),
                ("Gap", 520, "left"), ("Pts", 700, "right")]
        yy = y + 70 * sc
        for name, cx, al in cols:
            ui.text(p, x + cx * sc, yy, name, ui.DIM, ui.font(12 * sc, bold=True), align=al)
        yy += 8 * sc
        mono = ui.font(13 * sc, mono=True)
        for r in rows:
            if r["you"]:
                ui.panel(p, QtCore.QRectF(x + 14 * sc, yy + 2, w - 28 * sc, row_h), ui.ACCENT_SOFT, 4)
            base = yy + row_h * 0.78
            ui.text(p, x + 24 * sc, base, f"P{r['pos']}" if r["pos"] else "–", ui.TEXT, ui.font(13 * sc, bold=True))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(app.car_hex(r["car"])))
            p.drawRect(QtCore.QRectF(x + 74 * sc, base - 10 * sc, 10 * sc, 10 * sc))
            ui.text(p, x + 90 * sc, base, r["label"] + (" (running)" if r["running"] else ""), ui.TEXT,
                    ui.font(13 * sc, bold=r["you"]))
            ui.text(p, x + 290 * sc, base, r["design"], ui.DIM, mono)
            ui.text(p, x + 400 * sc, base, ui.lap_time(r["best"]) if r["best"] else "-", ui.TEXT, mono)
            ui.text(p, x + 520 * sc, base, r["gap"], ui.BAD if r["gap"].startswith("DNF") else ui.TEXT, mono)
            ui.text(p, x + 700 * sc, base, str(r["points"]), ui.TEXT, ui.font(13 * sc, bold=True), align="right")
            yy += row_h
        self._menu_box(p, sc, app.results_menu_obj, w / sc, top=y + table_h, height=menu_h)
