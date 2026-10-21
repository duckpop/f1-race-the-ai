"""
The race view: the simulator's own OpenGL track view (the gym renderer's
GLViewWidget), taken out of the gym's window and put into the game's, with

- every car drawn at ~60 fps, blended between the 20 Hz physics steps,
- a camera on your car that turns with it (your car points up the screen;
  C switches to north-up), looking ahead, zoomed with the mouse wheel,
- DRS zones tinted green, the DRS detection point as an amber line across
  the track and a chequered start / finish line.
"""

import csv
import dataclasses
import math
import time

import numpy as np

from game import sim  # noqa: F401
from racer.sensors import CLEARED_POSE

ZONE_RGBA = (0.26, 0.63, 0.28, 0.50)        # green over the white track
ZONE_EDGE_RGBA = (0.11, 0.37, 0.13, 1.0)    # dark green lines along both edges of a zone
DETECT_RGBA = (1.0, 0.70, 0.0, 1.0)         # amber, like the HUD's "armed"
GHOST_ALPHA = (0.45, 0.80)                  # a ghost pulses between these opacities ...
GHOST_PULSE_HZ = 2.5                        # ... this often, inside a solid outline
GHOST_EDGE_RGBA = (0.30, 0.08, 0.36, 1.0)
# Rotating camera: your car sits 75% of the way from the bottom edge up to
# the middle of the view (this share of the half height below the centre).
CAR_BELOW_CENTRE = 0.25
CAMERA_YAW_TAU = 0.04                       # s, smoothing of the rotating camera
# The camera pulls back with speed so you always see this long ahead of your
# car (never closer than the view at the reference zoom, today's default);
# out quicker than in, so braking for a corner keeps it in view.
LOOK_AHEAD_S = 3.0
ZOOM_REF = 2.0                              # the zoom setting the base view is measured at (the default)
ZOOM_OUT_TAU, ZOOM_IN_TAU = 0.5, 1.2        # s


def camera_distance(base, zoom, speed, half_per_m, car_below_centre):
    """Camera distance (m) showing LOOK_AHEAD_S x speed of track between
    the car and the top edge, at least base; zoom scales it (the mouse
    wheel). half_per_m: half the view's height per metre of distance;
    car_below_centre: the car's place below the centre, as a share of the
    half height (0 = centred)."""
    ahead_per_m = half_per_m * (1.0 + car_below_centre)
    need = LOOK_AHEAD_S * max(speed, 0.0) / ahead_per_m if ahead_per_m > 0 else 0.0
    return zoom / ZOOM_REF * max(base, need)
# Heights above the track image (at -0.01 m), far enough apart for the depth
# buffer even with the whole track in view (camera km away): 3 cm (the first
# version) z-fought with the track image once zoomed out and the zones vanished.
Z_ZONE, Z_FINISH, Z_LINES, Z_CARS, Z_OUTLINE = 0.6, 1.0, 1.4, 2.0, 2.4
JUMP_M = 40.0                               # a car moved further than this in one step: teleported, not driven


def rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def track_widths(cfg, map_path, tracker):
    """(left, right) half widths (m) at each tracker point, from the track's
    centerline file (scaled like the track); the median if the counts differ."""
    try:
        with open(f"{map_path}_centerline.csv") as f:
            rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
        right = np.array([float(r[2]) for r in rows]) * cfg.sim.map_scale
        left = np.array([float(r[3]) for r in rows]) * cfg.sim.map_scale
    except (OSError, ValueError, IndexError):
        return None
    if len(left) in (tracker.n, tracker.n + 1):
        return left[:tracker.n], right[:tracker.n]
    return np.full(tracker.n, np.median(left)), np.full(tracker.n, np.median(right))


class RaceView:
    def __init__(self, cfg, env, map_path, on_zoom):
        """env: a RaceEnv built with render=True; on_zoom(factor): the
        mouse wheel turned (factor < 1: closer)."""
        import pyqtgraph.opengl as gl
        from PyQt6 import QtGui
        self.gl, self.QtGui = gl, QtGui
        self.cfg, self.env = cfg, env
        r = self.renderer = env.unwrapped.renderer
        self.view = r.view
        r.window.takeCentralWidget()               # the GL view moves into the game's window
        r.window.hide()
        r.init = False                             # the gym's render() would show its window again
        r.focused = False                          # the gym's camera never moves: this one does
        r.agent_to_follow = None
        for label in ("lap_label", "fps_label"):
            if getattr(r, label, None) is not None:
                getattr(r, label).hide()
        self._mouse(on_zoom)
        self.tracker = env.tracker
        self.widths = track_widths(cfg, map_path, self.tracker)
        self._zones()
        self._line(env.detect_s, DETECT_RGBA, 4.0)
        self._finish_line()
        n = env.num_agents
        self.prev = np.zeros((n, 3))
        self.cur = np.zeros((n, 3))
        self.cam_yaw = None
        self.cam_dist = None
        self.colors = [(128, 128, 128)] * n
        self.ghost_car = None
        self.ghost_outline = self.gl.GLLinePlotItem(pos=np.zeros((5, 3), dtype=np.float32), color=GHOST_EDGE_RGBA,
                                                    width=3.0, antialias=True, mode="line_strip",
                                                    glOptions="opaque")   # the default (additive) turns white on white
        self.ghost_outline.setDepthValue(20)
        self.ghost_outline.setVisible(False)
        self.view.addItem(self.ghost_outline)

    # ---- mouse: only the wheel (zoom) ------------------------------------------
    def _mouse(self, on_zoom):
        v = self.view

        def wheel(event):
            on_zoom(0.85 if event.angleDelta().y() > 0 else 1 / 0.85)
            event.accept()

        def ignore(event):
            event.accept()

        v.wheelEvent = wheel
        v.mousePressEvent = v.mouseMoveEvent = v.mouseReleaseEvent = ignore

    # ---- track overlays ------------------------------------------------------------
    def _edges(self, s, share=0.96):
        """Left and right track edges (x, y) at arc lengths s (share: of the half widths)."""
        pts = np.array([self.tracker.pose_at(x) for x in s])
        xy, yaw = pts[:, :2], pts[:, 2]
        normal = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
        if self.widths is None:
            wl = wr = np.full(len(s), 13.0)
        else:
            k = np.clip(np.searchsorted(self.tracker.ss, np.asarray(s) % self.tracker.length, side="right") - 1,
                        0, self.tracker.n - 1)
            wl, wr = self.widths[0][k], self.widths[1][k]
        return xy + normal * (share * wl)[:, None], xy - normal * (share * wr)[:, None]

    def _zones(self):
        for start, length in self.tracker.zones():
            s = np.arange(start, start + length + 1e-6, 4.0)
            left, right = self._edges(s)
            m = len(s)
            verts = np.zeros((2 * m, 3), dtype=np.float32)
            verts[0::2, :2], verts[1::2, :2] = left, right
            verts[:, 2] = Z_ZONE
            faces = np.array([[2 * k, 2 * k + 1, 2 * k + 2] for k in range(m - 1)]
                             + [[2 * k + 1, 2 * k + 3, 2 * k + 2] for k in range(m - 1)], dtype=np.int32)
            mesh = self.gl.GLMeshItem(vertexes=verts, faces=faces, color=ZONE_RGBA, smooth=False,
                                      drawEdges=False, glOptions="translucent")
            mesh.setDepthValue(1)
            self.view.addItem(mesh)
            for edge in self._edges(s, share=0.88):      # inside the walls, so they don't merge with them
                pos = np.column_stack([edge, np.full(m, Z_LINES)]).astype(np.float32)
                line = self.gl.GLLinePlotItem(pos=pos, color=ZONE_EDGE_RGBA, width=3.0, antialias=True,
                                              mode="line_strip", glOptions="translucent")
                line.setDepthValue(2)
                self.view.addItem(line)

    def _line(self, s, rgba, width):
        left, right = self._edges([s])
        pos = np.array([[*left[0], Z_LINES], [*right[0], Z_LINES]], dtype=np.float32)
        line = self.gl.GLLinePlotItem(pos=pos, color=rgba, width=width, antialias=True, glOptions="translucent")
        line.setDepthValue(2)
        self.view.addItem(line)

    def _finish_line(self, square=2.2):
        """A chequered strip two squares deep across the track at the line."""
        left, right = self._edges([0.0])
        a, b = left[0], right[0]
        across = b - a
        width = float(np.linalg.norm(across))
        u = across / max(width, 1e-6)
        x, y, yaw = self.tracker.pose_at(0.0)
        f = np.array([math.cos(yaw), math.sin(yaw)])
        cols = max(int(width // square), 1)
        verts, faces, colors = [], [], []
        for row in range(2):
            for col in range(cols):
                p0 = a + u * (col * width / cols) + f * (row - 1) * square
                quad = [p0, p0 + u * width / cols, p0 + u * width / cols + f * square, p0 + f * square]
                k = len(verts)
                verts += [[q[0], q[1], Z_FINISH] for q in quad]
                faces += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
                c = (0.08, 0.08, 0.08, 1.0) if (row + col) % 2 else (0.97, 0.97, 0.97, 1.0)
                colors += [c, c]
        mesh = self.gl.GLMeshItem(vertexes=np.array(verts, dtype=np.float32), faces=np.array(faces, dtype=np.int32),
                                  faceColors=np.array(colors, dtype=np.float32), smooth=False, drawEdges=False)
        mesh.setDepthValue(1)
        self.view.addItem(mesh)

    # ---- cars ----------------------------------------------------------------------
    def _ensure_cars(self):
        if self.renderer.cars is None:             # the gym makes the car meshes on its first update
            self.renderer.update(self.env.unwrapped.render_obs)
        for car in self.renderer.cars:
            car.z_offset = Z_CARS                  # above every track marking (the gym draws them at 0.3 m)

    def set_sizes(self, sizes):
        """Each car's drawn (length, width); meshes rebuilt only where it
        changed (paint() colours them again)."""
        self._ensure_cars()
        base = self.env.unwrapped.sim.vehicle_params
        for car, (length, width) in zip(self.renderer.cars, sizes):
            if (car.car_length, car.car_width) != (length, width):
                car.update_params(dataclasses.replace(base, length=length, width=width))

    def paint(self, colors, ghost_car=None):
        """Every car's colour (RGB 0-255); ghost_car drawn see-through."""
        self._ensure_cars()
        self.colors = list(colors)
        self.ghost_car = ghost_car
        for i, car in enumerate(self.renderer.cars):
            alpha = GHOST_ALPHA[1] if i == ghost_car else 1.0
            car.color = self.colors[i]
            car.opacity = alpha
            car.rgba = np.array([[c / 255 for c in self.colors[i]] + [alpha]] * 2)
            if alpha < 1.0:
                car.mesh.setGLOptions("translucent")
            car.scale = None                       # rebuilt on the next draw

    def set_ghost(self, ghost_car):
        if ghost_car != self.ghost_car:
            self.paint(self.colors, ghost_car)

    def snap(self):
        """Poses after a reset or a teleport: no blending from before."""
        self.cur = np.array(self.env.unwrapped.sim.state.poses, dtype=np.float64)
        self.prev = self.cur.copy()
        self.cam_yaw = None
        self.cam_dist = None

    def stepped(self):
        """After each physics step: blend from the last poses to the new ones."""
        new = np.array(self.env.unwrapped.sim.state.poses, dtype=np.float64)
        self.prev = self.cur
        jumped = np.hypot(*(new[:, :2] - self.prev[:, :2]).T) > JUMP_M
        self.prev[jumped] = new[jumped]
        self.cur = new

    def pose(self, alpha):
        a = min(max(alpha, 0.0), 1.0)
        out = self.prev + (self.cur - self.prev) * a
        dyaw = (self.cur[:, 2] - self.prev[:, 2] + np.pi) % (2 * np.pi) - np.pi
        out[:, 2] = self.prev[:, 2] + dyaw * a
        return out

    # ---- a frame -----------------------------------------------------------------
    def draw(self, alpha, follow, rotate, zoom, dt, speed=0.0):
        """follow: the car the camera is on (None: the whole track); speed:
        its speed (m/s), for the look-ahead zoom."""
        poses = self.pose(alpha)
        g = self.ghost_car
        if g is not None:                          # pulse and outline your ghost
            car = self.renderer.cars[g]
            lo, hi = GHOST_ALPHA
            a = lo + (hi - lo) * (0.5 + 0.5 * math.sin(2 * math.pi * GHOST_PULSE_HZ * time.monotonic()))
            car.rgba = np.array([[c / 255 for c in self.colors[g]] + [a]] * 2)
            car.scale = None
            x, y, yaw = poses[g]
            hl, hw = car.car_length / 2 + 0.4, car.car_width / 2 + 0.4
            c, s = math.cos(yaw), math.sin(yaw)
            corners = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
            pos = np.array([[x + u * c - w * s, y + u * s + w * c, Z_OUTLINE] for u, w in corners], dtype=np.float32)
            self.ghost_outline.setData(pos=pos)
        self.ghost_outline.setVisible(g is not None)
        for i, car in enumerate(self.renderer.cars):
            car.pose = tuple(poses[i])
            car.render(1.0)
        r = self.renderer
        Q = self.QtGui.QVector3D
        if follow is None or poses[follow, 0] >= CLEARED_POSE[0] / 2:
            r._center_camera_on_map()
            self.cam_yaw = self.cam_dist = None
        else:
            x, y, yaw = poses[follow]
            target = camera_distance(r.default_camera_dist * ZOOM_REF, zoom, speed, self._half_height(1.0),
                                     CAR_BELOW_CENTRE if rotate else 0.0)
            if self.cam_dist is None:
                self.cam_dist = target
            tau = ZOOM_OUT_TAU if target > self.cam_dist else ZOOM_IN_TAU
            self.cam_dist += (target - self.cam_dist) * (1.0 - math.exp(-dt / tau))
            dist = self.cam_dist
            if rotate:
                if self.cam_yaw is None:
                    self.cam_yaw = yaw
                k = 1.0 - math.exp(-dt / CAMERA_YAW_TAU)
                self.cam_yaw += ((yaw - self.cam_yaw + math.pi) % (2 * math.pi) - math.pi) * k
                cy = self.cam_yaw
                ahead = CAR_BELOW_CENTRE * self._half_height(dist)
                x, y = x + ahead * math.cos(cy), y + ahead * math.sin(cy)
                azimuth = math.degrees(cy) - 180.0
            else:
                self.cam_yaw = None
                azimuth = 0.0
            self.view.setCameraPosition(pos=Q(x, y, 1), distance=dist, elevation=90, azimuth=azimuth)
        self.view.update()

    def _half_height(self, dist):
        """Half the view's height in metres on the track at camera distance
        dist (pyqtgraph's field of view is the horizontal one)."""
        v = self.view
        return dist * math.tan(math.radians(v.opts["fov"]) / 2) * v.height() / max(v.width(), 1)

    def close(self):
        self.view.setParent(None)
        self.view.deleteLater()
