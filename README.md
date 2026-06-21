# beamng_drift

Software-in-the-loop (SIL) bridge between a Python controller and
BeamNG.drive, for developing/testing a drift controller against the game's
vehicle physics.

## Layout

| Path | What it is |
|------|------------|
| `sil_beamng.py` | Connects to BeamNG, loads an empty world, runs the deterministic SIL loop, reads state, applies control. Hosts `BeamNGSIL` and `HotController`. |
| `controller.py` | The control law (`DriftController`). Hot-reloaded at runtime. |
| `gains.json` | Numeric gains / setpoints. Re-read live when the file changes. |
| `requirements.txt` | Python deps (`beamngpy`, `numpy`, `stable-baselines3`, `torch`). |
| `driftRL/` | Git submodule (`git@github.com:DGames95/driftRL.git`), tracked separately. |
| `models/drift_circle/` | Trained PPO policy (`best_model.zip`, `final_model.zip`) from driftRL. |

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Paths to the BeamNG install and user folder are set at the top of
`sil_beamng.py` (`BNG_HOME`, `BNG_USER`).

## Run

```
.venv\Scripts\python sil_beamng.py
```

Launches BeamNG, loads `smallgrid` (flat empty map), spawns one vehicle, and
runs the loop. Edit `gains.json` or `controller.py` and save to tune while it
runs. `Ctrl-C` to stop.

## State / control interface

`BeamNGSIL.get_state()` returns a dict:

| Key | Meaning | Units |
|-----|---------|-------|
| `X`, `Y` | world position | m |
| `psi` | heading (yaw) | rad |
| `vx`, `vy` | body-frame velocity (long., lat.) | m/s |
| `r` | yaw rate | rad/s |
| `ax`, `ay` | body-frame acceleration | m/s² |
| `speed` | speed magnitude | m/s |

`r`, `ax`, `ay` are finite-differenced over the fixed control step.

`BeamNGSIL.apply_control(throttle, brake, delta)`:
- `throttle`, `brake` in `[0, 1]`
- `delta` is a physical front road-wheel angle in **rad**, mapped internally
  to BeamNG's normalized `[-1, 1]` steering via `MAX_STEER_ANGLE`
  (a calibration constant in `sil_beamng.py`).

The controller is `controller.py:DriftController`, called as
`(state, t, dt) -> (throttle, brake, delta)`.

## RL controller

`controller.py` supports `"mode": "rl"` in `gains.json`, which drives the car
with a trained driftRL PPO policy instead of the hand-written laws. Set
`"model_path"` (default `models/drift_circle/best_model`) to choose the model.
Switch live between `rl`, `path`, and `drift` by editing `mode` and saving.

Notes:
- The policy reads `state["track_obs"]`, an 8-vector scaled by driftRL's
  `OBS_SCALE`. `track.py:obs()` is kept equal to that scale — change both
  together.
- The policy was trained on a 30 m circle, so `TRACK_RADIUS` is set to `30.0`
  in `sil_beamng.py` to match.
- Needs `stable-baselines3` + `torch` (in `requirements.txt`).

## Live tuning

`HotController` (in `sil_beamng.py`) polls file modification times each tick:
- `gains.json` changes → re-read, pushed into the controller.
- `controller.py` changes → `importlib.reload`, instance rebuilt with internal
  state carried over.
- Errors during reload or in the control law are caught so the sim keeps
  running.

## Submodule

```
git clone --recurse-submodules <url>      # fresh clone
git submodule update --init               # after a normal clone
```

`driftRL` is pinned to a specific commit; bump it with
`git add driftRL && git commit` after updating inside it.
