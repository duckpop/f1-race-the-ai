"""
Drawing helpers shared by the menus, the mapping screen and the HUD (all
painted with QPainter, in the simulator viewer's dark-panel style), and the
list menu every screen is built from: rows of a label and a value, moved
through with up / down, changed with left / right, chosen with select -
from the keyboard, any controller or the mouse.
"""

from PyQt6 import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt
BG = QtGui.QColor(20, 20, 20, 215)
PAGE_BG = QtGui.QColor(16, 17, 20)
TEXT = QtGui.QColor("#eeeeee")
DIM = QtGui.QColor("#9e9e9e")
FAINT = QtGui.QColor("#6a6a6a")
ACCENT = QtGui.QColor("#984ea3")            # your car's colour
ACCENT_SOFT = QtGui.QColor(152, 78, 163, 90)
GOOD = QtGui.QColor("#4caf50")
BAD = QtGui.QColor("#ff5252")
AMBER = QtGui.QColor("#ffb300")
BLUE = QtGui.QColor("#42a5f5")


def font(px, bold=False, mono=False):
    f = QtGui.QFont("Monospace" if mono else "Sans")
    f.setPixelSize(max(int(px), 8))
    f.setBold(bold)
    if mono:
        f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return f


def panel(p, rect, color=BG, radius=8):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawRoundedRect(QtCore.QRectF(rect), radius, radius)


def text(p, x, y, s, color=TEXT, f=None, align="left"):
    """s with its baseline at y; align left / right / center at x."""
    if f is not None:
        p.setFont(f)
    p.setPen(color)
    w = p.fontMetrics().horizontalAdvance(s)
    if align == "right":
        x -= w
    elif align == "center":
        x -= w / 2
    p.drawText(QtCore.QPointF(x, y), s)
    return w


def bar(p, x, y, w, h, value, fill, centred=False, back=QtGui.QColor(60, 60, 60)):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(back)
    p.drawRect(QtCore.QRectF(x, y, w, h))
    p.setBrush(QtGui.QColor(fill))
    if centred:                                     # -1 left ... +1 right
        mid, half = x + w / 2, w / 2 * min(abs(value), 1.0)
        p.drawRect(QtCore.QRectF(mid if value > 0 else mid - half, y, half, h))
        p.setBrush(DIM)
        p.drawRect(QtCore.QRectF(mid, y - 2, 1, h + 4))
    else:
        p.drawRect(QtCore.QRectF(x, y, w * min(max(value, 0.0), 1.0), h))


def lap_time(seconds):
    if seconds is None:
        return "-"
    m, s = divmod(float(seconds), 60.0)
    return f"{int(m)}:{s:06.3f}"


def laps_text(n):
    return f"{n} lap" + ("" if n == 1 else "s")


def clock(seconds):
    s = max(int(seconds), 0)
    return f"{s // 60}:{s % 60:02d}"


# -------------------------------------------------------------------------
# List menus
# -------------------------------------------------------------------------

class Item:
    """A menu row. value(): the text shown on the right (None: an action
    row); left() / right(): change it; activate(): an action row's choice;
    enabled(): greyed out and skipped when False; note: a dim line under it."""

    def __init__(self, label, value=None, left=None, right=None, activate=None, enabled=None, note=None,
                 gap_before=False):
        self.label, self.value, self.left, self.right = label, value, left, right
        self.activate, self.enabled, self.note, self.gap_before = activate, enabled, note, gap_before

    def is_enabled(self):
        return self.enabled is None or self.enabled()


class Menu:
    """Rows and a selection; handle() takes the controls' menu directions."""

    def __init__(self, title, items, on_back=None, hint=None):
        self.title, self.items, self.on_back = title, items, on_back
        self.hint = hint or "up / down: choose · left / right: change · Enter / A: select · Esc / B: back"
        self.sel = 0
        self.rects = []                              # each row's on-screen rect (last paint)
        self._skip(1)

    def _skip(self, d):
        for _ in range(len(self.items)):
            if self.items[self.sel].is_enabled():
                return
            self.sel = (self.sel + d) % len(self.items)

    def move(self, d):
        self.sel = (self.sel + d) % len(self.items)
        self._skip(d)

    def handle(self, events):
        for e in events:
            item = self.items[self.sel]
            if e == "up":
                self.move(-1)
            elif e == "down":
                self.move(1)
            elif e == "left" and item.left and item.is_enabled():
                item.left()
            elif e == "right" and item.right and item.is_enabled():
                item.right()
            elif e == "confirm" and item.is_enabled():
                if item.activate:
                    item.activate()
                elif item.right:
                    item.right()
            elif e == "back" and self.on_back:
                self.on_back()

    def click(self, pos, button):
        """A mouse click: selects the row; left-click chooses / steps the
        value up, right-click steps it down."""
        for k, r in enumerate(self.rects):
            if r.contains(QtCore.QPointF(pos)) and self.items[k].is_enabled():
                self.sel = k
                item = self.items[k]
                if button == Qt.MouseButton.RightButton and item.left:
                    item.left()
                elif button == Qt.MouseButton.LeftButton:
                    if item.activate:
                        item.activate()
                    elif item.right:
                        item.right()
                return True
        return False

    def paint(self, p, rect, scale=1.0, label_w=None):
        """The menu inside rect (title, rows, hint)."""
        row_h = 34 * scale
        x0, y = rect.x() + 28 * scale, rect.y() + 52 * scale
        if self.title:
            text(p, x0, y, self.title, TEXT, font(26 * scale, bold=True))
            y += 22 * scale
        else:
            y = rect.y() + 10 * scale
        label_w = label_w or 260 * scale
        vx = x0 + label_w
        self.rects = []
        for k, item in enumerate(self.items):
            if item.gap_before:
                y += 12 * scale
            r = QtCore.QRectF(rect.x() + 12 * scale, y, rect.width() - 24 * scale, row_h)
            self.rects.append(r)
            on = k == self.sel
            if on:
                panel(p, r, ACCENT_SOFT, 6)
            color = TEXT if item.is_enabled() else FAINT
            base = y + row_h * 0.68
            text(p, x0, base, item.label, color, font(17 * scale, bold=on))
            v = item.value() if item.value else None
            if v is not None:
                s = f"‹  {v}  ›" if on and (item.left or item.right) else v
                text(p, vx, base, s, color if not on else TEXT, font(17 * scale))
            y += row_h
            if item.note and item.note():
                text(p, x0 + 14 * scale, y + 13 * scale, item.note(), DIM, font(13 * scale))
                y += 20 * scale
        text(p, x0, rect.bottom() - 16 * scale, self.hint, DIM, font(13 * scale))


class MenuPage(QtWidgets.QWidget):
    """A full-window page painting one Menu (the setup screen)."""

    def __init__(self, menu=None, extra_paint=None):
        super().__init__()
        self.menu, self.extra_paint = menu, extra_paint
        self.setMouseTracking(False)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), PAGE_BG)
        scale = max(min(self.height() / 900, 1.6), 0.75)
        w = min(self.width() - 40, 760 * scale)
        r = QtCore.QRectF((self.width() - w) / 2, 24 * scale, w, self.height() - 48 * scale)
        panel(p, r, QtGui.QColor(28, 29, 33), 12)
        if self.menu is not None:
            self.menu.paint(p, r, scale)
        if self.extra_paint:
            self.extra_paint(p, r, scale)
        p.end()

    def mousePressEvent(self, event):
        if self.menu is not None and self.menu.click(event.position(), event.button()):
            self.update()
