"""The mapping screen's capture on scripted controller input."""

import pytest

from game import inputs, mapping


class FakePad:
    def __init__(self, axes=6, buttons=12, hats=1):
        self.axes, self.buttons, self.hats = [0.0] * axes, [0] * buttons, [(0, 0)] * hats

    def state(self):
        return inputs.PadState(list(self.axes), list(self.buttons), list(self.hats))


@pytest.fixture
def clock(monkeypatch):
    t = [0.0]
    monkeypatch.setattr(mapping.time, "monotonic", lambda: t[0])
    return t


def run(cap, pad, clock, frames):
    """frames: [(seconds, change)]; returns the capture's result."""
    res = None
    for seconds, change in frames:
        if change:
            change(pad)
        end = clock[0] + seconds
        while clock[0] < end:
            clock[0] += 0.016
            res = cap.tick()
            if res is not None:
                return res
    return res


def test_trigger_from_minus_one(clock):
    pad = FakePad()
    pad.axes[4] = -1.0
    cap = mapping.Capture(pad, analog=True)
    res = run(cap, pad, clock, [(0.1, None), (0.1, lambda p: p.axes.__setitem__(4, 0.2)),
                                (0.2, lambda p: p.axes.__setitem__(4, 1.0)),
                                (1.0, lambda p: p.axes.__setitem__(4, -1.0))])
    assert mapping.to_binding(res, True) == {"type": "axis", "index": 4, "rest": -1.0, "full": 1.0}


def test_trigger_that_reads_zero_until_first_touched(clock):
    pad = FakePad()                                       # axis 5 at 0 before its first press
    cap = mapping.Capture(pad, analog=True)
    res = run(cap, pad, clock, [(0.1, lambda p: p.axes.__setitem__(5, 1.0)),
                                (1.0, lambda p: p.axes.__setitem__(5, -1.0))])
    b = mapping.to_binding(res, True)
    assert (b["index"], b["rest"], b["full"]) == (5, -1.0, 1.0)


def test_button_while_the_starting_button_is_still_held(clock):
    pad = FakePad()
    pad.buttons[0] = 1                                    # A started the capture and is still down
    cap = mapping.Capture(pad, analog=False)
    res = run(cap, pad, clock, [(0.3, None), (0.1, lambda p: p.buttons.__setitem__(0, 0)),
                                (0.1, lambda p: p.buttons.__setitem__(0, 1)),
                                (1.0, lambda p: p.buttons.__setitem__(0, 0))])
    assert mapping.to_binding(res, False) == {"type": "button", "index": 0}


def test_analog_action_prefers_the_axis_over_a_trigger_click(clock):
    pad = FakePad()
    pad.axes[4] = -1.0
    cap = mapping.Capture(pad, analog=True)

    def full(p):
        p.axes[4] = 1.0
        p.buttons[7] = 1                                  # some pads also click a button at full travel

    def release(p):
        p.axes[4] = -1.0
        p.buttons[7] = 0

    res = run(cap, pad, clock, [(0.3, full), (1.0, release)])
    assert mapping.to_binding(res, True)["type"] == "axis"
    assert mapping.to_binding(res, False)["type"] == "button"


def test_dpad_direction(clock):
    pad = FakePad()
    cap = mapping.Capture(pad, analog=False)
    res = run(cap, pad, clock, [(0.2, lambda p: p.hats.__setitem__(0, (-1, 0))),
                                (1.0, lambda p: p.hats.__setitem__(0, (0, 0)))])
    assert mapping.to_binding(res, False) == {"type": "hat", "index": 0, "mask": 8}


def test_nothing_pressed_times_out(clock):
    pad = FakePad()
    cap = mapping.Capture(pad, analog=False)
    assert run(cap, pad, clock, [(mapping.CAPTURE_TIMEOUT_S + 1, None)]) == "timeout"


def test_steering_one_axis_both_ways(clock, monkeypatch):
    """The page's two-step steering capture makes one axis2 binding."""
    pad = FakePad()
    pad.record = inputs.new_record("g", "pad")

    class Controls:
        capturing = False
        keys = inputs.clean_keys(None)

        def connected(self):
            return [pad]

        def save(self):
            pass

    class App:
        controls = Controls()

    page = mapping.MappingPage.__new__(mapping.MappingPage)       # no Qt widget needed for the logic
    page.app, page.controls, page.pad_index, page.message = App, App.controls, 0, ""
    page._start(0, 2)                                             # "Steer left", controller column
    for value in (-1.0, 0.0):
        pad.axes[0] = value
        for _ in range(30):
            clock[0] += 0.016
            page._capture_tick([])
    assert page.capture["phase"] == "right"
    for value in (1.0, 0.0):
        pad.axes[0] = value
        for _ in range(30):
            clock[0] += 0.016
            if page.capture is not None:
                page._capture_tick([])
    assert page.capture is None
    assert pad.record["bindings"]["steer"] == {"type": "axis2", "index": 0, "center": 0.0, "left": -1.0, "right": 1.0}
