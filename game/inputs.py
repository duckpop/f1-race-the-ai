"""
Your controls: the keyboard (Qt key events, fed in by the window) and any
number of game controllers (read with pygame-ce / SDL2, no pygame window),
both at once. Every action can be rebound (the mapping screen, game/mapping.py)
and the bindings live in player/controls.json: the keyboard's, plus one
record per controller model (SDL GUID) with its own stick / trigger tuning.

A controller binding is one raw input, as JSON:

    {"type": "axis", "index": 4, "rest": -1.0, "full": 1.0}    an axis from rest to full (a trigger)
    {"type": "axis2", "index": 0, "center": 0.0, "left": -1.0, "right": 1.0}   a stick axis, both ways
    {"type": "pair", "left": BINDING, "right": BINDING}         steering on two buttons / d-pad directions
    {"type": "button", "index": 0}
    {"type": "hat", "index": 0, "mask": 1}                      d-pad: 1 up, 2 right, 4 down, 8 left

Controllers SDL's database knows (Xbox, DualSense, DualShock 4, ...) start
from defaults made from SDL's mapping of that model; any other controller
starts unmapped and opens the mapping screen.

Signs: the simulator steers left with +, so `steer` in DriveInput is +1 full
left; everything the player sees (stick values, the mapping screen) is
+1 = right.
"""

import dataclasses
import os

import numpy as np

# Controller actions, in the mapping screen's order. "steer" is one stick
# axis (or a pair of buttons); the menu_* and confirm / back drive the menus.
PAD_ACTIONS = ("steer", "throttle", "brake", "drs", "ers", "pause", "camera", "restart",
               "confirm", "back", "menu_up", "menu_down", "menu_left", "menu_right")
# Keyboard actions (two keys each). The menus always use the arrows / WASD,
# Enter / Space and Esc / Backspace; Esc always pauses too.
KEY_ACTIONS = ("steer_left", "steer_right", "throttle", "brake", "drs", "ers", "pause", "camera", "restart")
LABELS = {"steer": "Steer", "steer_left": "Steer left", "steer_right": "Steer right", "throttle": "Throttle",
          "brake": "Brake", "drs": "DRS (on / off)", "ers": "ERS (on / off)", "pause": "Pause menu",
          "camera": "Camera: rotating / north-up", "restart": "Restart race", "confirm": "Menu: select",
          "back": "Menu: back", "menu_up": "Menu: up", "menu_down": "Menu: down", "menu_left": "Menu: left",
          "menu_right": "Menu: right"}
# Edge-triggered actions (a press does something once).
PRESS_ACTIONS = ("drs", "ers", "pause", "camera", "restart")

DEFAULT_KEYS = {
    "steer_left": ["Key_Left", "Key_A"], "steer_right": ["Key_Right", "Key_D"],
    "throttle": ["Key_Up", "Key_W"], "brake": ["Key_Down", "Key_S"],
    "drs": ["Key_Space", None], "ers": ["Key_Shift", None],
    "pause": ["Key_Escape", None], "camera": ["Key_C", None], "restart": ["Key_R", None],
}
FIXED_PAUSE_KEY = "Key_Escape"            # can't be unbound: there's always a way out
MENU_KEYS = {"up": ("Key_Up", "Key_W"), "down": ("Key_Down", "Key_S"), "left": ("Key_Left", "Key_A"),
             "right": ("Key_Right", "Key_D"), "confirm": ("Key_Return", "Key_Enter", "Key_Space"),
             "back": ("Key_Escape", "Key_Backspace")}

# SDL's standard controller element -> this game's action (A = DRS and select, B = back, ...).
SDL_DEFAULTS = {"leftx": "steer", "righttrigger": "throttle", "lefttrigger": "brake", "a": ("drs", "confirm"),
                "x": "ers", "start": "pause", "back": "camera", "b": "back",
                "dpup": "menu_up", "dpdown": "menu_down", "dpleft": "menu_left", "dpright": "menu_right"}

STEER_DEADZONE = 0.08
STEER_CURVE = 0.0                         # 0 linear ... 1 progressive (x^2.5)
TRIGGER_DEADZONE = 0.05
TRIGGER_TOP = 0.03                        # the last 3% of travel already counts as full
PRESS_ON, PRESS_OFF = 0.6, 0.4            # a press on an analog input: hysteresis
KB_STEER_IN_S = 0.15                      # keyboard / digital steering: centre -> full lock
KB_STEER_OUT_S = 0.08                     # ... and back to the centre
NAV_REPEAT_DELAY_S, NAV_REPEAT_S = 0.4, 0.12


# -------------------------------------------------------------------------
# Raw controller state and binding values (pure functions)
# -------------------------------------------------------------------------

@dataclasses.dataclass
class PadState:
    """One snapshot of a controller's raw inputs."""
    axes: list
    buttons: list
    hats: list                    # pygame (x, y) tuples


def hat_mask(xy):
    x, y = xy
    return (1 if y > 0 else 0) | (2 if x > 0 else 0) | (4 if y < 0 else 0) | (8 if x < 0 else 0)


def unipolar(raw, rest, full):
    """An axis from its rest value (0) to its full value (1), clipped."""
    span = full - rest
    if abs(span) < 1e-6:
        return 0.0
    return float(min(max((raw - rest) / span, 0.0), 1.0))


def bipolar(raw, center, left, right):
    """A stick axis: -1 at its left value, +1 at its right value, 0 at the centre."""
    d = raw - center
    toward_right = (right - center) * d > 0
    span = (right - center) if toward_right else (center - left)
    if abs(span) < 1e-6:
        return 0.0
    return float(min(max(d / span, -1.0), 1.0))


def value(b, st):
    """0..1 for a unipolar binding (axis / button / hat), 0 if unbound or out
    of range (a different controller model with fewer inputs)."""
    if not b:
        return 0.0
    t, i = b.get("type"), b.get("index", -1)
    if t == "button":
        return float(bool(st.buttons[i])) if 0 <= i < len(st.buttons) else 0.0
    if t == "hat":
        return float(bool(hat_mask(st.hats[i]) & b["mask"])) if 0 <= i < len(st.hats) else 0.0
    if t == "axis":
        return unipolar(st.axes[i], b["rest"], b["full"]) if 0 <= i < len(st.axes) else 0.0
    if t == "axis2":                                     # a stick used as a button: either way
        return abs(bipolar(st.axes[i], b["center"], b["left"], b["right"])) if 0 <= i < len(st.axes) else 0.0
    return 0.0


def shape_stick(s, deadzone, curve):
    """Deadzone (rescaled so steering starts from 0 at its edge) and the curve:
    |s|^(1 + 1.5 curve), 0 = linear."""
    x = abs(s)
    if x <= deadzone:
        return 0.0
    x = min((x - deadzone) / max(1.0 - deadzone, 1e-6), 1.0)
    return float(np.copysign(x ** (1.0 + 1.5 * curve), s))


def shape_trigger(v, deadzone):
    if v <= deadzone:
        return 0.0
    return float(min((v - deadzone) / max(1.0 - deadzone - TRIGGER_TOP, 1e-6), 1.0))


def pad_steer(record, st):
    """(analog, digital) steering of a controller, +1 = right: analog from a
    stick axis (deadzone + curve), digital (-1 / 0 / +1) from a button pair."""
    b = record["bindings"].get("steer")
    if not b:
        return 0.0, 0.0
    if b["type"] == "axis2":
        i = b["index"]
        if not 0 <= i < len(st.axes):
            return 0.0, 0.0
        s = bipolar(st.axes[i], b["center"], b["left"], b["right"])
        return shape_stick(s, record["steer_deadzone"], record["steer_curve"]), 0.0
    if b["type"] == "pair":
        return 0.0, value(b.get("right"), st) - value(b.get("left"), st)
    return 0.0, 0.0


def ramp(current, target, dt):
    """Keyboard / digital steering: towards target (-1, 0 or +1) at the
    build-up rate, back towards the centre at the faster return rate."""
    if target != 0 and current * target >= 0:                       # building up (or holding)
        step = dt / KB_STEER_IN_S
        return float(min(current + step, target) if target > 0 else max(current - step, target))
    step = dt / KB_STEER_OUT_S                                       # releasing or reversing: centre first
    if current > 0:
        return float(max(current - step, 0.0))
    if current < 0:
        return float(min(current + step, 0.0))
    return 0.0


# -------------------------------------------------------------------------
# SDL's mapping -> default bindings
# -------------------------------------------------------------------------

def parse_sdl(element, two_sided=False):
    """One SDL mapping element ('b0', 'h0.4', 'a2', '+a2', '-a2', 'a5~') -> a
    binding, or None. two_sided: for a stick axis (leftx)."""
    if not element:
        return None
    try:
        if element[0] == "b":
            return {"type": "button", "index": int(element[1:])}
        if element[0] == "h":
            hat, mask = element[1:].split(".")
            return {"type": "hat", "index": int(hat), "mask": int(mask)}
        half = element[0] if element[0] in "+-" else ""
        body = element[1:] if half else element
        inverted = body.endswith("~")
        body = body.rstrip("~")
        if not body.startswith("a"):
            return None
        index = int(body[1:])
    except ValueError:
        return None
    sign = -1.0 if inverted else 1.0
    if two_sided:
        if half:
            return None                       # a stick split over two half axes: map it yourself
        return {"type": "axis2", "index": index, "center": 0.0, "left": -sign, "right": sign}
    if half == "+":
        return {"type": "axis", "index": index, "rest": 0.0, "full": sign}
    if half == "-":
        return {"type": "axis", "index": index, "rest": 0.0, "full": -sign}
    return {"type": "axis", "index": index, "rest": -sign, "full": sign}     # a trigger over the whole axis


def defaults_from_sdl(mapping):
    """Default bindings from SDL's mapping dict ({'a': 'b0', 'leftx': 'a0', ...})."""
    bindings = {a: None for a in PAD_ACTIONS}
    for element, actions in SDL_DEFAULTS.items():
        b = parse_sdl(mapping.get(element), two_sided=(element == "leftx"))
        for action in (actions if isinstance(actions, tuple) else (actions,)):
            if action in bindings and b is not None:
                bindings[action] = dict(b)
    return bindings


def new_record(guid, name, bindings=None, source="user"):
    return {"guid": guid, "name": name, "source": source,
            "bindings": bindings or {a: None for a in PAD_ACTIONS},
            "steer_deadzone": STEER_DEADZONE, "steer_curve": STEER_CURVE, "trigger_deadzone": TRIGGER_DEADZONE}


def valid_binding(b, two_sided=False):
    """A binding read from the file: the right keys and types, else None."""
    if not isinstance(b, dict):
        return None
    t = b.get("type")
    num = (int, float)
    try:
        if t == "pair" and two_sided:
            left, right = valid_binding(b.get("left")), valid_binding(b.get("right"))
            return {"type": "pair", "left": left, "right": right} if left and right else None
        if t in ("button", "hat", "axis", "axis2") and isinstance(b.get("index"), int) and b["index"] >= 0:
            if t == "hat" and isinstance(b.get("mask"), int):
                return {"type": t, "index": b["index"], "mask": b["mask"]}
            if t == "button":
                return {"type": t, "index": b["index"]}
            if t == "axis" and isinstance(b.get("rest"), num) and isinstance(b.get("full"), num):
                return {"type": t, "index": b["index"], "rest": float(b["rest"]), "full": float(b["full"])}
            if t == "axis2" and all(isinstance(b.get(k), num) for k in ("center", "left", "right")):
                return {"type": t, "index": b["index"], **{k: float(b[k]) for k in ("center", "left", "right")}}
    except (TypeError, ValueError):
        pass
    return None


def clean_record(guid, raw):
    """A controller record from the file, every value checked."""
    if not isinstance(raw, dict):
        return None
    rec = new_record(guid, str(raw.get("name", "controller")), source=str(raw.get("source", "user")))
    bindings = raw.get("bindings") if isinstance(raw.get("bindings"), dict) else {}
    for action in PAD_ACTIONS:
        b = valid_binding(bindings.get(action), two_sided=(action == "steer"))
        if action == "steer" and b and b["type"] not in ("axis2", "pair"):
            b = None
        rec["bindings"][action] = b
    for key, lo, hi in (("steer_deadzone", 0.0, 0.5), ("steer_curve", 0.0, 1.0), ("trigger_deadzone", 0.0, 0.5)):
        v = raw.get(key)
        if isinstance(v, (int, float)) and lo <= v <= hi:
            rec[key] = float(v)
    return rec


def clean_keys(raw):
    """Keyboard bindings from the file: two slots per action, Qt key names."""
    keys = {a: list(v) for a, v in DEFAULT_KEYS.items()}
    if isinstance(raw, dict):
        for action in KEY_ACTIONS:
            v = raw.get(action)
            if isinstance(v, list) and len(v) == 2 and all(k is None or isinstance(k, str) for k in v):
                keys[action] = list(v)
    keys["pause"][0] = FIXED_PAUSE_KEY
    return keys


# -------------------------------------------------------------------------
# Controllers (pygame-ce, SDL2)
# -------------------------------------------------------------------------

class Pad:
    def __init__(self, joystick, known_mapping):
        self.js = joystick
        self.instance_id = joystick.get_instance_id()
        self.guid = joystick.get_guid()
        self.name = joystick.get_name()
        self.sdl_mapping = known_mapping           # SDL's mapping dict, or None (not in its database)
        self.record = None                         # set by Controls
        self.pressed = {}                          # action -> currently held (hysteresis)

    def state(self):
        js = self.js
        return PadState([js.get_axis(i) for i in range(js.get_numaxes())],
                        [js.get_button(i) for i in range(js.get_numbuttons())],
                        [js.get_hat(i) for i in range(js.get_numhats())])

    def rumble(self, low, high, ms):
        try:
            self.js.rumble(low, high, ms)
        except Exception:                          # some backends refuse rumble outright
            pass


class PadManager:
    """Every connected controller, opened as it is plugged in (SDL needs no
    window: SDL_VIDEODRIVER=dummy, and background events allowed since our
    window isn't SDL's)."""

    def __init__(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        import pygame
        from pygame._sdl2 import controller
        self.pygame, self.controller = pygame, controller
        pygame.display.init()
        pygame.joystick.init()
        controller.init()
        self.pads = {}                              # instance_id -> Pad

    def poll(self):
        """Handle plug / unplug events (and drain SDL's queue). Returns the
        newly connected pads."""
        pg = self.pygame
        added = []
        for e in pg.event.get():
            if e.type == pg.JOYDEVICEADDED:
                pad = self._open(e.device_index)
                if pad is not None:
                    added.append(pad)
            elif e.type == pg.JOYDEVICEREMOVED:
                pad = self.pads.pop(e.instance_id, None)
                if pad is not None:
                    print(f"[play] controller disconnected: {pad.name}")
        return added

    def _open(self, device_index):
        try:
            js = self.pygame.joystick.Joystick(device_index)
        except self.pygame.error:
            return None
        if js.get_instance_id() in self.pads:
            return None
        mapping = None
        try:
            if self.controller.is_controller(device_index):
                c = self.controller.Controller(device_index)
                mapping = dict(c.get_mapping())
        except Exception:
            mapping = None
        pad = Pad(js, mapping)
        self.pads[pad.instance_id] = pad
        print(f"[play] controller connected: {pad.name} ({'known layout' if mapping else 'unknown layout'})")
        return pad

    def rumble(self, low, high, ms):
        for pad in self.pads.values():
            pad.rumble(low, high, ms)


# -------------------------------------------------------------------------
# Everything together
# -------------------------------------------------------------------------

@dataclasses.dataclass
class DriveInput:
    steer: float = 0.0            # +1 full left (the simulator's sign)
    throttle: float = 0.0
    brake: float = 0.0
    shown_steer: float = 0.0      # +1 full right, for the HUD


class Controls:
    """The keyboard (key_down / key_up from the window's event filter) and
    every controller, combined: steering from whichever is turned further,
    throttle / brake the larger, presses from any of them."""

    def __init__(self, store_mod):
        self.store = store_mod
        raw = store_mod.load("controls", {})
        raw = raw if isinstance(raw, dict) else {}
        self.keys = clean_keys(raw.get("keyboard"))
        self.records = {}
        for guid, rec in (raw.get("pads") or {}).items():
            r = clean_record(guid, rec)
            if r is not None:
                self.records[guid] = r
        self.held = set()                 # Qt key names held now
        self.key_presses = []             # (Qt key name, auto-repeat) pressed since the last read
        self.frame_keys = []              # the names pressed in the last frame (no repeats)
        self.kb_steer = 0.0
        self.nav_hold = {}                # menu direction -> (held since, last repeat)
        self.pads = None
        try:
            self.pads = PadManager()
        except Exception as e:            # no SDL: keyboard only
            print(f"[play] no controller support ({e}); keyboard only")
        self.capturing = False            # the mapping screen reads raw inputs itself

    # ---- persistence -------------------------------------------------------
    def save(self):
        self.store.save("controls", {"keyboard": self.keys, "pads": self.records})

    def record_for(self, pad):
        """The pad's bindings: saved ones, else SDL's defaults, else empty
        (needs mapping). Returns (record, needs_mapping)."""
        rec = self.records.get(pad.guid)
        if rec is None:
            if pad.sdl_mapping:
                rec = new_record(pad.guid, pad.name, defaults_from_sdl(pad.sdl_mapping), source="sdl")
            else:
                rec = new_record(pad.guid, pad.name)
            self.records[pad.guid] = rec
        pad.record = rec
        return rec, not any(rec["bindings"].values())

    def reset_pad(self, pad):
        """Back to SDL's defaults (or unmapped)."""
        self.records.pop(pad.guid, None)
        return self.record_for(pad)[0]

    # ---- keyboard (from the window) ---------------------------------------
    def key_down(self, name, auto_repeat):
        if not auto_repeat:
            self.held.add(name)
        self.key_presses.append((name, auto_repeat))

    def key_up(self, name, auto_repeat):
        if not auto_repeat:
            self.held.discard(name)

    def release_all(self):
        """The window lost focus: nothing stays held."""
        self.held.clear()

    def _key(self, action):
        return any(k in self.held for k in self.keys.get(action, ()) if k)

    # ---- per frame ---------------------------------------------------------
    def poll_pads(self):
        """Plug / unplug handling; returns pads that need mapping (unknown
        layout, nothing saved yet)."""
        if self.pads is None:
            return []
        need = []
        for pad in self.pads.poll():
            _, empty = self.record_for(pad)
            if empty:
                need.append(pad)
        return need

    def connected(self):
        return list(self.pads.pads.values()) if self.pads else []

    def frame(self, dt):
        """Read everything once per frame: (DriveInput, presses, menu
        directions). presses: driving / game actions pressed this frame;
        menu: 'up', 'down', 'left', 'right', 'confirm', 'back' this frame."""
        presses, menu = set(), []
        self.frame_keys = [name for name, repeat in self.key_presses if not repeat]    # for the mapping screen
        for name, repeat in self.key_presses:
            if not repeat:
                for action in PRESS_ACTIONS:
                    if name in self.keys.get(action, ()) or (action == "pause" and name == FIXED_PAUSE_KEY):
                        presses.add(action)
            for direction, names in MENU_KEYS.items():
                if name in names and (not repeat or direction in ("up", "down", "left", "right")):
                    menu.append(direction)
        self.key_presses.clear()

        target = (1.0 if self._key("steer_left") else 0.0) - (1.0 if self._key("steer_right") else 0.0)
        throttle = 1.0 if self._key("throttle") else 0.0
        brake = 1.0 if self._key("brake") else 0.0
        analog = 0.0                        # +1 right
        held_nav = set()
        for pad in self.connected():
            rec = pad.record or self.record_for(pad)[0]
            st = pad.state()
            a, digital = pad_steer(rec, st)
            if abs(a) > abs(analog):
                analog = a
            if digital:
                target = float(np.clip(target - digital, -1.0, 1.0))     # digital +1 = right = env -1
            b = rec["bindings"]
            throttle = max(throttle, shape_trigger(value(b.get("throttle"), st), rec["trigger_deadzone"]))
            brake = max(brake, shape_trigger(value(b.get("brake"), st), rec["trigger_deadzone"]))
            for action in PRESS_ACTIONS + ("confirm", "back", "menu_up", "menu_down", "menu_left", "menu_right"):
                v = value(b.get(action), st)
                was = pad.pressed.get(action, False)
                now = v >= (PRESS_OFF if was else PRESS_ON)
                pad.pressed[action] = now
                if now and not was and not self.capturing:
                    if action in PRESS_ACTIONS:
                        presses.add(action)
                    elif action in ("confirm", "back"):
                        menu.append(action)
                if now and action.startswith("menu_"):
                    held_nav.add(action[5:])
            if abs(a) > 0.6:                                             # the stick steers the menus too
                held_nav.add("right" if a > 0 else "left")
        menu += self._nav_repeat(held_nav)
        self.kb_steer = ramp(self.kb_steer, target, dt)
        steer = self.kb_steer if abs(self.kb_steer) >= abs(analog) else -analog
        if self.capturing:
            return DriveInput(), set(), []
        return DriveInput(steer, throttle, brake, -steer), presses, menu

    def _nav_repeat(self, held):
        """Held d-pad / stick directions: one step, then repeats."""
        import time
        now = time.monotonic()
        out = []
        for d in list(self.nav_hold):
            if d not in held:
                del self.nav_hold[d]
        for d in held:
            if d not in self.nav_hold:
                self.nav_hold[d] = (now, now)
                out.append(d)
            else:
                since, last = self.nav_hold[d]
                if now - since >= NAV_REPEAT_DELAY_S and now - last >= NAV_REPEAT_S:
                    self.nav_hold[d] = (since, now)
                    out.append(d)
        return out

    def rumble(self, low, high, ms):
        if self.pads is not None:
            self.pads.rumble(low, high, ms)
