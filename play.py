"""
Drive a car yourself against the simulator's trained AI drivers.

    uv run play.py

Opens the setup menu (track, laps or time, the AI field, difficulty, crash
rules, grid slot, your car's design, controls). Keyboard and game
controllers both work at any time; see the README for the controls.
"""

import sys

from game import sim  # noqa: F401  (the simulator submodule on the import path)


def main():
    from watch import undo_vscode_snap_env
    undo_vscode_snap_env()
    from PyQt6 import QtGui, QtWidgets
    # The simulator's renderer asks for desktop OpenGL without vsync (Wayland
    # needs it); the format has to be the default before the app exists.
    fmt = QtGui.QSurfaceFormat()
    fmt.setSwapInterval(0)
    fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    qapp.setApplicationName("F1: race the AI")
    from game.app import App
    app = App(qapp)          # noqa: F841  (kept alive by this frame while the loop runs)
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
