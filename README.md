# F1: race the AI

Drive a car yourself against the trained AI drivers from [f1-simulator](https://github.com/duckpop/f1-simulator): the middle, aggressive and defensive drivers, on the same 23 tracks and with the same physics. Play with the keyboard or a game controller. On a controller, the analog triggers give you partial throttle and brake.

The simulator is a git submodule (`simulator/`), so the game uses the exact cars, tracks and trained drivers from there. The game doesn't change any simulator file.

## Setup

Needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). No GPU needed.

```bash
git clone --recurse-submodules https://github.com/duckpop/f1-race-the-ai.git
cd f1-race-the-ai
uv venv
uv pip install -r requirements.txt
```

If you cloned without `--recurse-submodules`, run `git submodule update --init --recursive`.

To move to a newer simulator version later:

```bash
git submodule update --remote simulator
```

## Play

```bash
uv run play.py
```

The setup menu picks the race:

| Setting | Choices (default first) |
|---|---|
| Track | Random, or any of the 23 tracks |
| Race | Laps (1 to 50, default 5) or Time (30 s to 60 min) |
| Middle / Aggressive / Defensive AI | How many of each, 7 / 7 / 7, at most 21 in all |
| Difficulty | Hard (the AI at 95% of its top speed), Easy 80%, Medium 88%, Expert 100% |
| Crashes | Respawn (ghost for 3 s) or Out (the AI's rules) |
| Grid slot | Random, or P1 to the back |
| Car physics | Your car only: every car setting, grouped (below). Starts at the Downforce +50% preset |
| Engine, Battery, Chassis, Brakes, Wing | Your car's design, levels 1 to 5, 17 points (stock 3-3-3-3-3) |

The menu remembers your last race. F11 toggles fullscreen.

## Controls

| Action | Keyboard | Controller |
|---|---|---|
| Steer | ← → or A D | Left stick |
| Throttle | ↑ or W | Right trigger (RT / R2), analog |
| Brake | ↓ or S | Left trigger (LT / L2), analog |
| DRS on / off | Space | A / Cross |
| ERS on / off | Shift | X / Square |
| Pause menu | Esc | Start / Options |
| Camera: rotating / north-up | C | Back / Select / Share |
| Restart race | R | (not bound) |

Keyboard and controller work at the same time. On the keyboard, steering builds to full lock in 0.15 s and returns to centre in 0.08 s. Full steering means the most the tyres can take at your speed, the same steering model the AI uses, so full lock never spins the car. The mouse wheel zooms.

### Mapping screen

**Controls** in the menu (or the pause menu) rebinds every action: two keys each on the keyboard, one input on the controller. Select a cell, press Enter or A, then press or pull the input you want. Steering takes two moves: fully left, then fully right.

- Triggers are calibrated while you map them: their rest position and full travel are recorded, so triggers that rest at -1 and triggers that rest at 0 both work.
- Sliders set the stick deadzone (default 8%), the stick curve (linear to progressive) and the trigger deadzone (5%).
- The **Now** column shows what each input reads live.
- Delete clears a cell. Esc always pauses and can't be unbound.

Xbox, DualSense, DualShock 4 and other controllers SDL knows get the defaults above. A controller SDL doesn't know opens the mapping screen as soon as it's connected, pausing the race if one is running. Mappings are saved per controller model.

## Car physics

**Car physics** in the menu changes every setting of your car. The AI always drives the simulator's values. The hub page shows a preset, then each group with how many of its values you changed, plus **Changed values**, which lists only what differs from the AI's car (Enter resets one). Every page has the formula and a live preview beside it:

- **Cornering:** grip and the tightest turning radius at 54 to 288 km/h in four cases (steady, lifted off the throttle, half brake, full throttle).
- **Straight line:** top speed, top speed with ERS, 0 to 200 km/h, and braking from 300 to 100 km/h.

Both are shown against the AI's car doing the same, in green where yours is better and red where it's worse.

The steering cap the cornering values feed (the simulator's `racer/physics.py` `max_steer` and its inputs):

```
max steer angle = steer_factor × min( s_max , atan( wheelbase × a_lat / max(v, 1)² ) )
a_lat           = 9.81 × grip × min( max_g , base_g + df_share × downforce_g × (v / ref_speed)^exponent ) × lateral_share
lateral_share   = max( combined_floor , √(1 − u²) )          (1 with the traction circle off)
u               = min( 1 , |throttle × engine − braking| / (a_max speeding up, brake_max slowing) )
braking         = brake × brake_max on the throttle, else min( brake_limit , brake + brake_share ) × brake_max
df_share        = 1 − drs_grip_loss while DRS is open
```

Your steering picks a share of that angle, and the wheel turns toward it at up to `sv_max` rad/s. At speed, the turning radius at the limit is `v² / a_lat`, so only the grip terms matter there. `s_max` and `wheelbase` set the tightest turn at low speed, and `sv_max` only matters below about 90 km/h.

| Group | Settings (the AI's value) |
|---|---|
| Grip | base grip (1.8 g), downforce grip (3.2 g), downforce reference speed (252 km/h), downforce speed exponent (v², the game's own), max grip (5 g), grip multiplier (×1), steering scaled to grip (on) |
| Traction circle and braking | traction circle (on), grip kept at full brake / throttle (25%), traction limit a_max (27 m/s², also engine power), full braking (30 m/s²), engine braking (50%), brakes + engine braking cap (×1.4) |
| Steering | wheel speed sv_max (1.5 rad/s), full lock s_max (0.6 rad), wheelbase (3.6 m) |
| Engine and top speed | top speed (310 km/h), power switch speed v_switch (126 km/h: more power at speed, same top speed) |
| Slipstream | top-speed boost (+10%), pull (10 m/s²), full wake within (15 m), wake reaches (120 m), full wake width (±1.5 m), wake width (±3 m), from (72 km/h) |
| DRS | top-speed boost (+6%), pull (2 m/s²), downforce loss (40%) |
| ERS battery | size (10 s), charge at the start (85%), power boost (+25%), recharge rate (×1), chase recharge rate (×1.5), chase recharge within (1 s) |
| Car size | width (2 m), length (5.6 m), body centre ahead of the rear axle (1.8 m) |
| Design levels | change per level (5%), largest change (±15%) |

The **Preset** row sets everything to the AI's car (AI-equal) or to a downforce level: +30% (max 6 g), +50% (max 6.5 g) or +80% (max 7.5 g). The page starts at +50%.

Not on the pages:

- **Mass, yaw inertia, centre-of-gravity height and mu:** the simulator's kinematic model ignores them.
- **v_min:** the simulator never lets a car reverse.
- **s_min, sv_min, lf, lr:** they mirror s_max and sv_max. Only the wheelbase matters to the kinematic model.
- **DRS latch:** your DRS is the game's toggle.

A car size you change is used for your collisions with walls and cars, for what the other cars' lidar sees and for how your car is drawn. The AI's short-range "alongside" sensor still assumes the stock size.

## Race rules

- **Start:** five red lights, one per second, then a random 0.2 to 3 s before they go out. Nothing moves before lights out. The AI gets its usual 2 s full-throttle launch; you launch yourself.
- **DRS:** the same rules as the AI. You must be in a zone (tinted green on the track), armed (you passed the detection point, the amber line, less than 1 s behind a car), at least 30 s into the race and above 72 km/h. Press once to open it while it's available. Braking, the end of the zone or a second press closes it.
- **ERS:** press once to switch it on and once to switch it off. It deploys while you're on the throttle and switches itself off when the battery runs empty.
- **Laps races:** you finish at your last crossing of the chequered line. Once the leader has finished, every car finishes at its next crossing and leaves the track.
- **Timed races:** cars are placed by distance when time runs out, as in the simulator.
- **Respawn mode:** hitting a wall or a car, or 6 s without progress, puts you back on the centre line where you were, at a standstill, with no time penalty. For 3 s you're a see-through ghost: the AI's sensors don't see you and no car can touch you, but walls still count. An AI car that hits you is out, as always.
- **Out mode:** the AI's rules. Any contact or 6 s stuck ends your race.
- **After you finish or go out:** the camera follows the race leader until every car is done. Any key or button skips to the results. Results show position, best lap, gap and F1 points (25-18-15-12-10-8-6-4-2-1), with Rematch, Menu and Quit.

## The screen

- Your car is purple. The camera turns with it so it always points up, 3/8 of the way up the window; C switches to north-up.
- As you speed up, the camera pulls back so you always see 3 seconds of track ahead (about 250 m at 300 km/h). It never gets closer than the view at low speed, and the mouse wheel scales the zoom.
- **Left:** the standings, with each car's interval to the car ahead and places gained or lost since the start.
- **Top right:** lap N of M (or the time left), your current, last and best lap, the race's fastest lap, your record here and the AI's record here. Under it, your gaps to the cars ahead and behind.
- **Bottom right:** your place and speed, steer / throttle / brake bars, the DRS light (off, armed, available, open), the ERS light (off, on, deploying), the battery and your gap at the detection point.

Your lap records (clean laps only: a lap with a respawn in it doesn't count) are kept in `player/records.json`, beside `player/settings.json` and `player/controls.json`. The `player/` folder is not in git.

## What's where

| Path | Contents |
|---|---|
| `play.py` | Starts the game |
| `game/app.py` | The window, its pages and the 60 fps loop (the race steps at the simulator's 20 Hz) |
| `game/race.py` | One race: your car's inputs, the AI, DRS / ERS toggles, laps, respawn and ghost, gaps, records |
| `game/carphysics.py`, `game/physics_page.py` | Your car's settings (applied to your car only) and the Car physics pages |
| `game/inputs.py` | Keyboard and controllers (pygame-ce / SDL2), bindings, calibration, SDL defaults |
| `game/mapping.py` | The mapping screen |
| `game/view.py` | The simulator's track view in the game window: blended cars, camera, DRS zones, lines |
| `game/hud.py`, `game/menu.py`, `game/ui.py` | HUD, menus and drawing helpers |
| `game/settings.py`, `game/records.py`, `game/store.py` | Saved settings, lap records, the `player/` files |
| `tests/` | `uv run --with pytest pytest` |
| `simulator/` | [f1-simulator](https://github.com/duckpop/f1-simulator) (submodule) |

The game targets Linux and Windows. It has been tested on Linux (Wayland) only.
