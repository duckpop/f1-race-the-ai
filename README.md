# F1: race the AI

Drive a car yourself against the trained AI drivers from [f1-simulator](https://github.com/duckpop/f1-simulator): the middle, aggressive and defensive drivers, on the same 23 tracks and with the same physics.

The game is still being planned. This repo holds the simulator as a git submodule (`simulator/`), so the game uses the exact cars, tracks and trained drivers from there.

## Setup

```bash
git clone --recurse-submodules https://github.com/duckpop/f1-race-the-ai.git
cd f1-race-the-ai
uv venv
uv pip install -r simulator/requirements.txt
```

If you cloned without `--recurse-submodules`, run `git submodule update --init --recursive`.

To move to a newer simulator version later:

```bash
git submodule update --remote simulator
```

## Planned

- Your car among AI cars on the grid, rendered with the simulator's race view
- Controls for steer, throttle, brake, DRS and ERS (to be decided)
- Race results with positions and F1 points
