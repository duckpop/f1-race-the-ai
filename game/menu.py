"""
The setup screen (track, race length, the AI field, difficulty, crash
rules, grid slot, your car's design), the pause menu and the results menu.
"""

from game import sim  # noqa: F401
from racer import design

from game import ui
from game import carphysics
from game.settings import DIFFICULTY, LAPS_RANGE, MAX_AI, PROFILES, SECONDS_RANGE

CATEGORY_TRADEOFF = {
    "engine": "more acceleration and top speed · less braking",
    "battery": "bigger ERS battery, faster recharge · less steering",
    "chassis": "more cornering grip · less acceleration and top speed",
    "brakes": "stronger braking · more drag, lower top speed",
    "wing": "less drag, higher top speed · less braking, ERS drains faster",
}
DIFFICULTY_NAMES = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "expert": "Expert"}


def _cycle(options, current, d):
    k = options.index(current) if current in options else 0
    return options[(k + d) % len(options)]


def setup_menu(app):
    """The setup screen's Menu; app: start_race(), open_mapping(), quit()."""
    s, cfg = app.settings, app.cfg
    tracks = ["Random"] + sorted(app.maps)
    categories = list(cfg.design.categories)
    menu = None

    def set_track(d):
        s.track = _cycle(tracks, s.track, d)

    def set_count(prof, d):
        new = s.ai[prof] + d
        if 0 <= new and s.ai_total + d <= MAX_AI:
            s.ai[prof] = new
            s.clamp_grid()

    def set_grid(d):
        slots = ["random"] + list(range(1, s.cars + 1))
        s.grid = _cycle(slots, s.grid, d)

    def set_level(k, d):
        new = list(s.design)
        new[k] += d
        if bool(design.legal(cfg, new)):
            s.design = new

    def selected(item_label):
        return menu is not None and menu.items[menu.sel].label == item_label

    items = [
        ui.Item("Track", lambda: s.track, lambda: set_track(-1), lambda: set_track(1)),
        ui.Item("Race", lambda: "Laps" if s.format == "laps" else "Time",
                lambda: setattr(s, "format", "time" if s.format == "laps" else "laps"),
                lambda: setattr(s, "format", "time" if s.format == "laps" else "laps")),
        ui.Item("Laps", lambda: str(s.laps),
                lambda: setattr(s, "laps", max(s.laps - 1, LAPS_RANGE[0])),
                lambda: setattr(s, "laps", min(s.laps + 1, LAPS_RANGE[1])), enabled=lambda: s.format == "laps"),
        ui.Item("Duration", lambda: ui.clock(s.seconds),
                lambda: setattr(s, "seconds", max(s.seconds - 30, SECONDS_RANGE[0])),
                lambda: setattr(s, "seconds", min(s.seconds + 30, SECONDS_RANGE[1])),
                enabled=lambda: s.format == "time"),
    ]
    for k, prof in enumerate(PROFILES):
        items.append(ui.Item(f"{prof.capitalize()} AI", (lambda p=prof: str(s.ai[p])),
                             (lambda p=prof: set_count(p, -1)), (lambda p=prof: set_count(p, 1)),
                             gap_before=k == 0,
                             note=(lambda: f"{s.ai_total} AI + you = {s.cars} cars on the grid")
                             if k == len(PROFILES) - 1 else None))
    items += [
        ui.Item("Difficulty", lambda: f"{DIFFICULTY_NAMES[s.difficulty]} (AI at {DIFFICULTY[s.difficulty]:.0%})",
                lambda: setattr(s, "difficulty", _cycle(list(DIFFICULTY), s.difficulty, -1)),
                lambda: setattr(s, "difficulty", _cycle(list(DIFFICULTY), s.difficulty, 1))),
        ui.Item("Crashes", lambda: "Respawn (ghost for 3 s)" if s.crashes == "respawn" else "Out (the AI's rules)",
                lambda: setattr(s, "crashes", "out" if s.crashes == "respawn" else "respawn"),
                lambda: setattr(s, "crashes", "out" if s.crashes == "respawn" else "respawn")),
        ui.Item("Grid slot", lambda: "Random" if s.grid == "random" else f"P{s.grid}",
                lambda: set_grid(-1), lambda: set_grid(1)),
    ]
    items.append(ui.Item("Car physics", lambda: carphysics.label(cfg, s.physics), activate=app.open_physics,
                         gap_before=True,
                         note=lambda: "your car only: grip, downforce, braking, steering" if selected("Car physics")
                         else None))
    for k, cat in enumerate(categories):
        label = cat.capitalize()
        items.append(ui.Item(label, (lambda k=k: f"{s.design[k]}"), (lambda k=k: set_level(k, -1)),
                             (lambda k=k: set_level(k, 1)),
                             note=(lambda lab=label, c=cat: CATEGORY_TRADEOFF[c] if selected(lab) else None)))
    # The last design row also shows the whole design and the points used.
    items[-1].note = (lambda c=categories[-1]: (CATEGORY_TRADEOFF[c] + "   ·   " if selected(c.capitalize()) else "")
                      + f"your car {design.text(s.design)} · {sum(s.design)} / {cfg.design.max_points} points")
    items += [
        ui.Item("Controls", activate=app.open_mapping, gap_before=True),
        ui.Item("Start race", activate=app.start_race),
        ui.Item("Quit", activate=app.quit),
    ]
    menu = ui.Menu("F1: race the AI", items, on_back=None,
                   hint="up / down: choose · left / right: change · Enter / A: select · F11: fullscreen")
    menu.sel = len(items) - 2                       # Start race
    return menu


def pause_menu(app):
    return ui.Menu("Paused", [
        ui.Item("Resume", activate=app.resume),
        ui.Item("Restart race", activate=app.restart),
        ui.Item("Controls", activate=app.open_mapping),
        ui.Item("Back to menu", activate=app.to_menu),
        ui.Item("Quit", activate=app.quit),
    ], on_back=app.resume, hint="Enter / A: select · Esc / Start / B: resume")


def results_menu(app):
    return ui.Menu("", [
        ui.Item("Rematch", activate=app.restart),
        ui.Item("Menu", activate=app.to_menu),
        ui.Item("Quit", activate=app.quit),
    ], on_back=app.to_menu, hint="Enter / A: select")
