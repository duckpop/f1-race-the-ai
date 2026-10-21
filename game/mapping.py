"""
The mapping screen: every action's two keyboard keys and its controller
input, rebound by selecting a cell and pressing / pulling the input; the
stick's deadzone and curve and the triggers' deadzone; a live view of what
each input reads now.

Capturing a controller input watches the raw axes, buttons and d-pad:

- a button or d-pad direction counts once it has been seen released (the
  button that started the capture is still held at first),
- an axis counts once it moves more than half its range; its rest is where
  it settles after you let go (so triggers that rest at -1 or at 0 both
  work) and its full value is the furthest it went,
- throttle, brake and steering prefer an axis when both an axis and a
  button fired (triggers that also click a button at the end of travel),
- steering takes two moves, full left then full right: one axis both ways,
  or two buttons / directions.
"""

import time

import numpy as np

from game import inputs, ui
from game.ui import QtCore, QtGui, QtWidgets, Qt

ANALOG = ("steer", "throttle", "brake")
CAPTURE_TIMEOUT_S = 8.0
SETTLE_S = 0.15
# (label, keyboard action, controller action)
ROWS = [
    ("Steer left", "steer_left", "steer"),
    ("Steer right", "steer_right", "steer"),
    ("Throttle", "throttle", "throttle"),
    ("Brake", "brake", "brake"),
    ("DRS (on / off)", "drs", "drs"),
    ("ERS (on / off)", "ers", "ers"),
    ("Pause menu", "pause", "pause"),
    ("Camera", "camera", "camera"),
    ("Restart race", "restart", "restart"),
    ("Menu: select", None, "confirm"),
    ("Menu: back", None, "back"),
    ("Menu: up", None, "menu_up"),
    ("Menu: down", None, "menu_down"),
    ("Menu: left", None, "menu_left"),
    ("Menu: right", None, "menu_right"),
]
SLIDERS = [("Stick deadzone", "steer_deadzone", 0.0, 0.3, 0.01, lambda v: f"{v:.0%}"),
           ("Stick curve", "steer_curve", 0.0, 1.0, 0.1,
            lambda v: "linear" if v == 0 else "progressive" if v >= 1 else f"{v:.1f}"),
           ("Trigger deadzone", "trigger_deadzone", 0.0, 0.3, 0.01, lambda v: f"{v:.0%}")]
DRIVING = ("steer", "throttle", "brake", "drs", "ers", "pause", "camera", "restart")


def key_label(name):
    if not name:
        return "–"
    try:
        code = getattr(Qt.Key, name)
    except AttributeError:
        return name.removeprefix("Key_")
    s = QtGui.QKeySequence(code.value).toString(QtGui.QKeySequence.SequenceFormat.NativeText)
    return s or name.removeprefix("Key_")


def binding_label(b):
    if not b:
        return "–"
    t = b["type"]
    if t == "button":
        return f"button {b['index']}"
    if t == "hat":
        d = {1: "up", 2: "right", 4: "down", 8: "left"}.get(b["mask"], str(b["mask"]))
        return f"d-pad {d}" + (f" ({b['index']})" if b["index"] else "")
    if t == "axis":
        return f"axis {b['index']} {'+' if b['full'] > b['rest'] else '-'}"
    if t == "axis2":
        return f"axis {b['index']} ←→"
    if t == "pair":
        return f"{binding_label(b['left'])} / {binding_label(b['right'])}"
    return "?"


def same_input(a, b):
    if not a or not b or a["type"] != b["type"] or a.get("index") != b.get("index"):
        return False
    return a["type"] != "hat" or a["mask"] == b["mask"]


class Capture:
    """Watches one controller for the input to bind (see the module doc)."""

    def __init__(self, pad, analog):
        self.pad, self.analog = pad, analog
        st = pad.state()
        self.base = st
        self.t0 = time.monotonic()
        self.seen_up = {i for i, v in enumerate(st.buttons) if not v}
        self.hat_seen_zero = {i for i, v in enumerate(st.hats) if inputs.hat_mask(v) == 0}
        self.dev = np.zeros(len(st.axes))
        self.ext = np.array(st.axes, dtype=float)
        self.button = self.hat = None
        self.started = False
        self.quiet_since = None
        self.last = None

    def tick(self):
        """None while waiting; a result dict once everything has been let go
        and has settled; 'timeout'."""
        now = time.monotonic()
        st = self.pad.state()
        held = False
        for i, v in enumerate(st.buttons):
            if not v:
                self.seen_up.add(i)
            elif i in self.seen_up:
                held = True
                if self.button is None:
                    self.button = i
        for i, v in enumerate(st.hats):
            m = inputs.hat_mask(v)
            if m == 0:
                self.hat_seen_zero.add(i)
            elif i in self.hat_seen_zero:
                held = True
                if self.hat is None:
                    self.hat = (i, m & -m)               # one direction
        axes = np.array(st.axes, dtype=float)
        base = np.array(self.base.axes, dtype=float)
        changing = False
        if len(axes) == len(base) and len(axes):
            d = np.abs(axes - base)
            grow = d > self.dev
            self.ext[grow] = axes[grow]
            self.dev = np.maximum(self.dev, d)
            moving = self.dev > 0.5
            if moving.any():
                k = int(np.argmax(np.where(moving, self.dev, -1.0)))
                # still held while it's nearer its extreme than 60% of the way back
                if abs(axes[k] - self.ext[k]) < 0.6 * abs(self.ext[k] - base[k]):
                    held = True
            changing = self.last is not None and bool((np.abs(axes - self.last) > 0.02).any())
            self.last = axes
        if self.button is not None or self.hat is not None or (self.dev > 0.5).any():
            self.started = True
        if not self.started:
            return "timeout" if now - self.t0 > CAPTURE_TIMEOUT_S else None
        if held or changing:
            self.quiet_since = None
            return "timeout" if now - self.t0 > CAPTURE_TIMEOUT_S * 2 else None
        if self.quiet_since is None:
            self.quiet_since = now
            return None
        if now - self.quiet_since < SETTLE_S:
            return None
        axis = int(np.argmax(self.dev)) if (self.dev > 0.5).any() else None
        return {"axis": axis, "rest": float(axes[axis]) if axis is not None else None,
                "full": float(self.ext[axis]) if axis is not None else None,
                "button": self.button, "hat": self.hat}


def to_binding(res, analog):
    """A capture's result -> a unipolar binding (axes first for analog
    actions, buttons first otherwise)."""
    axis = None
    if res["axis"] is not None:
        axis = {"type": "axis", "index": res["axis"], "rest": res["rest"], "full": res["full"]}
        if abs(axis["full"] - axis["rest"]) < 0.4:           # it never really moved from where it rests
            axis = None
    button = {"type": "button", "index": res["button"]} if res["button"] is not None else None
    hat = {"type": "hat", "index": res["hat"][0], "mask": res["hat"][1]} if res["hat"] is not None else None
    order = (axis, button, hat) if analog else (button, hat, axis)
    return next((b for b in order if b), None)


class MappingPage(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.controls = app.controls
        self.pad_index = 0
        self.row, self.col = 0, 0                # col: 0 key 1, 1 key 2, 2 controller
        self.capture = None                      # dict: kind, action, slot / phase, Capture
        self.message = ""
        self.rects = {}
        self.on_done = None

    # ---- rows ----------------------------------------------------------------------
    def items(self):
        """('pad',) selector, ('bind', k), ('slider', k), ('button', name)."""
        out = [("pad",)] + [("bind", k) for k in range(len(ROWS))] + [("slider", k) for k in range(len(SLIDERS))]
        return out + [("button", "reset_pad"), ("button", "reset_keys"), ("button", "done")]

    def pad(self):
        pads = self.controls.connected()
        if not pads:
            return None
        self.pad_index %= len(pads)
        return pads[self.pad_index]

    def open(self, pad=None, on_done=None, message=""):
        self.on_done = on_done
        self.capture = None
        self.controls.capturing = False
        self.message = message
        pads = self.controls.connected()
        if pad is not None and pad in pads:
            self.pad_index = pads.index(pad)
            self.row, self.col = 1, 2
        else:
            self.row, self.col = 1, 0

    # ---- input --------------------------------------------------------------------
    def tick(self, menu_events, keys):
        """menu_events: the controls' menu directions; keys: Qt key names
        pressed this frame."""
        if self.capture is not None:
            self._capture_tick(keys)
        else:
            if "Key_Delete" in keys or "Key_Backspace" in keys:
                self._clear()
                menu_events = [e for e in menu_events if e != "back"]
            for e in menu_events:
                self._nav(e)
        self.update()

    def _nav(self, e):
        items = self.items()
        kind = items[self.row]
        if e == "up":
            self.row = (self.row - 1) % len(items)
        elif e == "down":
            self.row = (self.row + 1) % len(items)
        elif e in ("left", "right"):
            d = -1 if e == "left" else 1
            if kind[0] == "bind":
                self.col = (self.col + d) % 3
            elif kind[0] == "pad":
                self.pad_index += d
            elif kind[0] == "slider":
                self._slide(kind[1], d)
        elif e == "confirm":
            self._activate(kind)
        elif e == "back":
            self._done()

    def _activate(self, kind):
        if kind[0] == "bind":
            self._start(kind[1], self.col)
        elif kind[0] == "pad":
            self.pad_index += 1
        elif kind[0] == "slider":
            self._slide(kind[1], 1)
        elif kind[1] == "reset_pad" and self.pad() is not None:
            self.controls.reset_pad(self.pad())
            self.controls.save()
            self.message = "controller back to its defaults"
        elif kind[1] == "reset_keys":
            self.controls.keys = inputs.clean_keys(None)
            self.controls.save()
            self.message = "keyboard back to its defaults"
        elif kind[1] == "done":
            self._done()

    def _done(self):
        self.controls.save()
        if self.on_done:
            self.on_done()

    def _slide(self, k, d):
        pad = self.pad()
        if pad is None:
            return
        _, key, lo, hi, step, _ = SLIDERS[k]
        rec = pad.record
        rec[key] = float(round(min(max(rec[key] + d * step, lo), hi), 3))
        self.controls.save()

    def _clear(self):
        kind = self.items()[self.row]
        if kind[0] != "bind":
            return
        _, key_action, pad_action = ROWS[kind[1]]
        if self.col < 2 and key_action:
            if key_action == "pause" and self.col == 0:
                self.message = "Esc always pauses"
                return
            self.controls.keys[key_action][self.col] = None
        elif self.col == 2 and pad_action and self.pad() is not None:
            self.pad().record["bindings"][pad_action] = None
            self.pad().record["source"] = "user"
        self.controls.save()

    def _start(self, k, col):
        label, key_action, pad_action = ROWS[k]
        if col < 2:
            if not key_action:
                self.message = "menus use the arrows / WASD, Enter and Esc on the keyboard"
                return
            if key_action == "pause" and col == 0:
                self.message = "Esc always pauses; set a second key in the next column"
                return
            self.capture = {"kind": "key", "action": key_action, "slot": col, "label": label}
            return
        pad = self.pad()
        if pad is None:
            self.message = "no controller connected"
            return
        self.controls.capturing = True
        self.capture = {"kind": "pad", "action": pad_action, "label": label, "phase": "left",
                        "watch": Capture(pad, pad_action in ANALOG), "pad": pad}

    def _capture_tick(self, keys):
        c = self.capture
        if "Key_Escape" in keys:
            self._end_capture("cancelled")
            return
        if c["kind"] == "key":
            if keys:
                name = keys[0]
                for action, slots in self.controls.keys.items():      # one action per key
                    for s, k in enumerate(slots):
                        if k == name and not (action == c["action"] and s == c["slot"]) \
                                and not (action == "pause" and s == 0):
                            slots[s] = None
                self.controls.keys[c["action"]][c["slot"]] = name
                self.controls.save()
                self._end_capture(f"{c['label']}: {key_label(name)}")
            return
        res = c["watch"].tick()
        if res is None:
            return
        if res == "timeout":
            self._end_capture("nothing pressed - try again")
            return
        action, pad = c["action"], c["pad"]
        rec = pad.record
        if action != "steer":
            b = to_binding(res, action in ANALOG)
            if b is None:
                self._end_capture("didn't catch that - try again")
                return
            self._bind(rec, action, b)
            self._end_capture(f"{c['label']}: {binding_label(b)}")
            return
        if c["phase"] == "left":
            c["left"] = res
            c["phase"] = "right"
            c["watch"] = Capture(pad, True)
            return
        left, right = c["left"], res
        if left["axis"] is not None and left["axis"] == right["axis"] and \
                abs(left["full"] - right["rest"]) > 0.4 and abs(right["full"] - right["rest"]) > 0.4:
            b = {"type": "axis2", "index": right["axis"], "center": right["rest"], "left": left["full"],
                 "right": right["full"]}
        else:
            lb, rb = to_binding(left, False), to_binding(right, False)
            if lb is None or rb is None:
                self._end_capture("didn't catch that - try again")
                return
            b = {"type": "pair", "left": lb, "right": rb}
        self._bind(rec, "steer", b)
        self._end_capture(f"Steer: {binding_label(b)}")

    def _bind(self, rec, action, b):
        if action in DRIVING:                  # one driving action per input (menus may share)
            for other in DRIVING:
                if other != action and same_input(rec["bindings"].get(other), b):
                    rec["bindings"][other] = None
        rec["bindings"][action] = b
        rec["source"] = "user"
        self.controls.save()

    def _end_capture(self, message):
        self.capture = None
        self.controls.capturing = False
        self.message = message

    # ---- mouse ------------------------------------------------------------------------
    def mousePressEvent(self, event):
        if self.capture is not None:
            return
        pos = event.position()
        for (row, col), r in self.rects.items():
            if r.contains(pos):
                self.row = row
                kind = self.items()[row]
                if kind[0] == "bind" and col is not None:
                    self.col = col
                    self._start(kind[1], col)
                elif kind[0] == "slider":
                    self._slide(kind[1], -1 if event.button() == Qt.MouseButton.RightButton else 1)
                else:
                    self._activate(kind)
                self.update()
                return

    # ---- paint --------------------------------------------------------------------------
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), ui.PAGE_BG)
        sc = max(min(self.height() / 1000, 1.5), 0.7)
        w = min(self.width() - 40, 1040 * sc)
        x0 = (self.width() - w) / 2
        ui.panel(p, QtCore.QRectF(x0, 16 * sc, w, self.height() - 32 * sc), QtGui.QColor(28, 29, 33), 12)
        ui.text(p, x0 + 28 * sc, 60 * sc, "Controls", ui.TEXT, ui.font(26 * sc, bold=True))
        pad = self.pad()
        st = pad.state() if pad is not None else None
        rec = pad.record if pad is not None else None
        cols = [x0 + 28 * sc, x0 + 250 * sc, x0 + 390 * sc, x0 + 530 * sc, x0 + 760 * sc]
        row_h = 30 * sc
        y = 86 * sc
        self.rects = {}
        f, fb, mono = ui.font(15 * sc), ui.font(15 * sc, bold=True), ui.font(14 * sc, mono=True)
        held_keys = self.controls.held
        for k, kind in enumerate(self.items()):
            if kind[0] in ("slider",) and kind[1] == 0 or kind[0] == "button" and kind[1] == "reset_pad":
                y += 10 * sc
            r = QtCore.QRectF(x0 + 12 * sc, y, w - 24 * sc, row_h)
            on = k == self.row
            if on:
                ui.panel(p, r, ui.ACCENT_SOFT, 6)
            base = y + row_h * 0.7
            if kind[0] == "pad":
                if pad is None:
                    s = "Controller: none connected (connect one, it shows up here)"
                else:
                    how = {"sdl": "defaults from SDL's database", "user": "mapped by you"}.get(rec["source"], "")
                    if not any(rec["bindings"].values()):
                        how = "not mapped yet"
                    n = len(self.controls.connected())
                    s = f"Controller: {pad.name}  ·  {how}" + (f"  ·  ‹ › {self.pad_index + 1} of {n}" if n > 1 else "")
                ui.text(p, cols[0], base, s, ui.TEXT, fb if on else f)
                self.rects[(k, None)] = r
                y += row_h + 8 * sc
                for name, cx in zip(("Action", "Key 1", "Key 2", "Controller", "Now"), cols):
                    ui.text(p, cx, y + 14 * sc, name, ui.DIM, ui.font(12 * sc, bold=True))
                y += 22 * sc
                continue
            if kind[0] == "bind":
                label, key_action, pad_action = ROWS[kind[1]]
                ui.text(p, cols[0], base, label, ui.TEXT, fb if on else f)
                for c in range(3):
                    cell = QtCore.QRectF(cols[c + 1] - 6 * sc, y + 2, (cols[c + 2] - cols[c + 1] if c < 2 else
                                                                        cols[4] - cols[3]) - 8 * sc, row_h - 4)
                    self.rects[(k, c)] = cell
                    if on and c == self.col:
                        p.setPen(QtGui.QPen(ui.TEXT, 1.5))
                        p.setBrush(Qt.BrushStyle.NoBrush)
                        p.drawRoundedRect(cell, 4, 4)
                    if c < 2:
                        name = self.controls.keys[key_action][c] if key_action else None
                        s = key_label(name) if key_action else ""
                        if key_action == "pause" and c == 0:
                            s += " (always)"
                        color = ui.GOOD if name and name in held_keys else ui.TEXT
                        ui.text(p, cols[c + 1], base, s, color, f)
                    else:
                        b = rec["bindings"].get(pad_action) if rec else None
                        ui.text(p, cols[3], base, binding_label(b) if rec else "", ui.TEXT, f)
                if st is not None and pad_action:
                    self._live(p, rec, st, pad_action, label, cols[4], y + 8 * sc, 180 * sc, 14 * sc)
            elif kind[0] == "slider":
                label, key, lo, hi, step, fmt = SLIDERS[kind[1]]
                ui.text(p, cols[0], base, label, ui.TEXT, fb if on else f)
                v = fmt(rec[key]) if rec else "–"
                ui.text(p, cols[1], base, f"‹  {v}  ›" if on else v, ui.TEXT, f)
                if rec and kind[1] == 0 and st is not None:
                    a, _ = inputs.pad_steer(rec, st)
                    ui.text(p, cols[3], base, "stick now", ui.DIM, ui.font(12 * sc))
                    ui.bar(p, cols[4], y + 8 * sc, 180 * sc, 14 * sc, a, "#e0e0e0", centred=True)
                self.rects[(k, None)] = r
            else:
                names = {"reset_pad": "Reset this controller to its defaults", "reset_keys": "Reset the keyboard",
                         "done": "Done"}
                ui.text(p, cols[0], base, names[kind[1]], ui.TEXT, fb if on else f)
                self.rects[(k, None)] = r
            y += row_h
        hint_y = self.height() - 30 * sc
        if self.capture is not None:
            c = self.capture
            if c["kind"] == "key":
                msg = f"Press a key for {c['label']}  (Esc: cancel)"
            elif c["action"] == "steer":
                msg = ("Push the stick (or press the button) fully LEFT, then let go" if c["phase"] == "left"
                       else "Now fully RIGHT, then let go") + "  (Esc: cancel)"
            elif c["action"] in ("throttle", "brake"):
                msg = f"Pull / press {c['label']} all the way, then let go  (Esc: cancel)"
            else:
                msg = f"Press the input for {c['label']}, then let go  (Esc: cancel)"
            box = QtCore.QRectF(x0 + 12 * sc, hint_y - 54 * sc, w - 24 * sc, 40 * sc)
            ui.panel(p, box, QtGui.QColor(152, 78, 163, 200), 8)
            ui.text(p, x0 + w / 2, hint_y - 27 * sc, msg, ui.TEXT, ui.font(17 * sc, bold=True), align="center")
        elif self.message:
            ui.text(p, x0 + 28 * sc, hint_y - 28 * sc, self.message, ui.AMBER, ui.font(14 * sc))
        ui.text(p, x0 + 28 * sc, hint_y,
                "up / down: action · left / right: column · Enter / A: rebind · Delete: clear · Esc / B: done",
                ui.DIM, ui.font(13 * sc))
        p.end()

    def _live(self, p, rec, st, action, label, x, y, w, h):
        """What the bound input reads now."""
        b = rec["bindings"]
        if action == "steer":
            a, digital = inputs.pad_steer(rec, st)
            v = a if a else digital
            v = min(-v, 0.0) if label == "Steer left" else max(v, 0.0)
            ui.bar(p, x, y, w, h, abs(v), "#e0e0e0")
        elif action in ("throttle", "brake"):
            v = inputs.shape_trigger(inputs.value(b.get(action), st), rec["trigger_deadzone"])
            ui.bar(p, x, y, w, h, v, "#4caf50" if action == "throttle" else "#ef5350")
        else:
            on = inputs.value(b.get(action), st) >= inputs.PRESS_ON
            p.setPen(QtGui.QPen(ui.FAINT, 1))
            p.setBrush(ui.GOOD if on else QtGui.QColor(50, 50, 50))
            p.drawEllipse(QtCore.QPointF(x + h / 2, y + h / 2), h / 2, h / 2)
