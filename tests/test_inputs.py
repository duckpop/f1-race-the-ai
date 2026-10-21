import pytest

from game import inputs


def test_sdl_trigger_over_the_whole_axis_rests_at_minus_one():
    assert inputs.parse_sdl("a5") == {"type": "axis", "index": 5, "rest": -1.0, "full": 1.0}


def test_sdl_half_and_inverted_axes():
    assert inputs.parse_sdl("+a2") == {"type": "axis", "index": 2, "rest": 0.0, "full": 1.0}
    assert inputs.parse_sdl("-a2") == {"type": "axis", "index": 2, "rest": 0.0, "full": -1.0}
    assert inputs.parse_sdl("a5~") == {"type": "axis", "index": 5, "rest": 1.0, "full": -1.0}


def test_sdl_stick_buttons_and_hats():
    assert inputs.parse_sdl("a0", two_sided=True) == {"type": "axis2", "index": 0, "center": 0.0,
                                                      "left": -1.0, "right": 1.0}
    assert inputs.parse_sdl("+a0", two_sided=True) is None
    assert inputs.parse_sdl("b11") == {"type": "button", "index": 11}
    assert inputs.parse_sdl("h0.4") == {"type": "hat", "index": 0, "mask": 4}
    assert inputs.parse_sdl("x3") is None and inputs.parse_sdl("") is None


def test_defaults_from_the_pv58_mapping():
    """SDL's mapping of the SHANWAN pad (the Genesis PV58 here)."""
    mapping = {"a": "b0", "b": "b1", "x": "b3", "y": "b4", "back": "b10", "start": "b11", "leftx": "a0",
               "lefty": "a1", "lefttrigger": "a5", "righttrigger": "a4", "dpup": "h0.1", "dpdown": "h0.4",
               "dpleft": "h0.8", "dpright": "h0.2", "crc": "0ecc", "platform": "Linux"}
    b = inputs.defaults_from_sdl(mapping)
    assert b["throttle"]["index"] == 4 and b["brake"]["index"] == 5
    assert b["drs"] == b["confirm"] == {"type": "button", "index": 0}
    assert b["ers"]["index"] == 3 and b["pause"]["index"] == 11 and b["camera"]["index"] == 10
    assert b["steer"]["type"] == "axis2" and b["restart"] is None


def test_unipolar_and_bipolar_calibration():
    assert inputs.unipolar(-1.0, -1.0, 1.0) == 0.0
    assert inputs.unipolar(0.0, -1.0, 1.0) == 0.5
    assert inputs.unipolar(1.2, -1.0, 1.0) == 1.0
    assert inputs.unipolar(-0.5, 0.0, -1.0) == 0.5            # a half axis pulled the negative way
    assert inputs.bipolar(-0.5, 0.0, -1.0, 1.0) == -0.5
    assert inputs.bipolar(0.5, 0.1, -0.9, 0.9) == pytest.approx(0.5)
    assert inputs.bipolar(0.5, 0.0, 1.0, -1.0) == -0.5         # an inverted stick


def test_stick_deadzone_and_curve():
    assert inputs.shape_stick(0.05, 0.08, 0.0) == 0.0
    assert inputs.shape_stick(1.0, 0.08, 0.0) == 1.0
    assert inputs.shape_stick(-0.54, 0.08, 0.0) == pytest.approx(-0.5)
    assert inputs.shape_stick(0.54, 0.08, 1.0) == pytest.approx(0.5 ** 2.5)


def test_trigger_reaches_full_before_the_end_of_travel():
    assert inputs.shape_trigger(0.04, 0.05) == 0.0
    assert inputs.shape_trigger(0.97, 0.05) == pytest.approx(1.0)


def test_keyboard_ramp_builds_slowly_and_returns_fast():
    v, t = 0.0, 0.0
    while v < 1.0:
        v, t = inputs.ramp(v, 1.0, 0.01), t + 0.01
    assert t == pytest.approx(inputs.KB_STEER_IN_S, abs=0.011)
    v, t = 1.0, 0.0
    while v > 0.0:
        v, t = inputs.ramp(v, 0.0, 0.01), t + 0.01
    assert t == pytest.approx(inputs.KB_STEER_OUT_S, abs=0.011)
    assert inputs.ramp(0.5, -1.0, 0.01) < 0.5                   # reversing goes through the centre first
    assert inputs.ramp(0.5, -1.0, 0.01) >= 0.0


def test_values_of_missing_inputs_are_zero():
    st = inputs.PadState(axes=[0.0], buttons=[1], hats=[(0, 1)])
    assert inputs.value({"type": "button", "index": 5}, st) == 0.0
    assert inputs.value({"type": "hat", "index": 0, "mask": 1}, st) == 1.0
    assert inputs.value({"type": "hat", "index": 0, "mask": 4}, st) == 0.0
    assert inputs.value(None, st) == 0.0


def test_saved_records_are_checked():
    rec = inputs.clean_record("g", {"name": "pad", "bindings": {"throttle": {"type": "axis", "index": 2},
                                                                 "steer": {"type": "button", "index": 0},
                                                                 "drs": {"type": "button", "index": 1}},
                                    "steer_deadzone": 3.0})
    assert rec["bindings"]["throttle"] is None                 # no rest / full
    assert rec["bindings"]["steer"] is None                    # steering needs an axis or a pair
    assert rec["bindings"]["drs"] == {"type": "button", "index": 1}
    assert rec["steer_deadzone"] == inputs.STEER_DEADZONE      # out of range: the default
    keys = inputs.clean_keys({"pause": [None, "Key_P"], "drs": ["Key_X"]})
    assert keys["pause"] == ["Key_Escape", "Key_P"] and keys["drs"] == ["Key_Space", None]
