# CLAUDE.md

A SIL bridge to BeamNG.drive for a Python drift controller. Python reads the
car state each tick, computes commands, and applies them; BeamNG runs the
physics.

## Components

- `sil_beamng.py`
  - `BeamNGSIL`: lifecycle (`open`/`close`), `step`, `get_state`,
    `apply_control`, `reset`. Wraps `beamngpy`.
  - `HotController`: file-watch hot-reload of `controller.py` and `gains.json`.
  - `main()`: the run loop.
  - Config constants at the top: `BNG_HOME`, `BNG_USER`, `HOST`/`PORT`, `MAP`,
    `VEHICLE_MODEL`, `PHYSICS_HZ`, `CONTROL_HZ`, `MAX_STEER_ANGLE`.
- `controller.py`: `DriftController(params)` with
  `__call__(state, t, dt) -> (throttle, brake, delta)` and `update_params`.
- `gains.json`: gains/setpoints consumed by `DriftController`.

## Conventions in the current code

- State is body-frame for velocities/accelerations; `psi` is world heading.
- `delta` is a physical road-wheel angle in radians; `apply_control` converts
  it to BeamNG's normalized `[-1, 1]` steering using `MAX_STEER_ANGLE`.
- `r`, `ax`, `ay` are finite-differenced over the fixed control `DT`.
- The sim runs deterministically (`set_deterministic`) and is advanced with
  `bng.control.step`.
- `controller.py` has three modes (`gains.json` `"mode"`): `path`/`drift`
  (analytic) and `rl` (loads an SB3 PPO policy from `"model_path"`, consuming
  `state["track_obs"]`). Models live in `models/`; needs `stable-baselines3`+`torch`.
- `track.py:obs()` scaling must stay equal to driftRL's `OBS_SCALE`
  (`drift_env.py:48`) so trained policies transfer. RL action is
  `[delta (rad), T]`; `T` splits into throttle (`max(0,T)`) / brake (`max(0,-T)`).

## Environment

- Windows. Use the venv interpreter: `.venv\Scripts\python`.
- `beamngpy==1.35.1`, BeamNG.drive v0.36.x.
- `git` Bash tool runs `bash`; use the PowerShell tool for PowerShell.

## Repo

- `beamng_drift` is the working repo; `driftRL/` is a submodule with its own
  history/remote — keep changes to each in their own repo.
- Don't commit `.venv/` or `__pycache__/` (already gitignored).

## Things not yet decided / out of scope right now

- `MAX_STEER_ANGLE` is an uncalibrated guess.
- The controller in `controller.py` is a placeholder drift law, not final.
- No tests, no remote for `beamng_drift` yet.
