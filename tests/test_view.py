import pytest

from game.view import LOOK_AHEAD_S, ZOOM_REF, camera_distance

HALF = 0.312          # half the view height per metre of camera distance (60 deg fov, 1869 x 1010 window)


def ahead(dist, below=0.25):
    return dist * HALF * (1 + below)


def test_slow_keeps_the_base_view():
    assert camera_distance(280.0, ZOOM_REF, 20.0, HALF, 0.25) == pytest.approx(280.0)


def test_fast_shows_the_set_time_ahead():
    v = 300 / 3.6
    d = camera_distance(280.0, ZOOM_REF, v, HALF, 0.25)
    assert ahead(d) == pytest.approx(LOOK_AHEAD_S * v)            # 250 m at 300 km/h
    assert camera_distance(280.0, ZOOM_REF, v, HALF, 0.0) * HALF == pytest.approx(LOOK_AHEAD_S * v)  # north-up


def test_the_wheel_scales_it():
    v = 300 / 3.6
    assert camera_distance(280.0, 2 * ZOOM_REF, v, HALF, 0.25) == pytest.approx(
        2 * camera_distance(280.0, ZOOM_REF, v, HALF, 0.25))
