"""OLS parameter fitting for the 3-DOF bicycle vehicle model.

All functions take numpy arrays and return a dict with fitted values
and goodness-of-fit stats. No BeamNG dependency.

Model reference (matches drift_env.py conventions):
  alpha_f = arctan((vy + LF*r) / vx) - delta   [front slip angle]
  alpha_r = arctan((vy - LR*r) / vx)            [rear  slip angle]
  Fyf = -CA_F * alpha_f   (linear region)
  Fyr = -CA_R * alpha_r   (linear region)
  M*(vy_dot + vx*r) = Fyf + Fyr
  IZ*r_dot            = LF*Fyf - LR*Fyr
  M*vx_dot            = Fx_r - C_DRAG*vx*|vx|   (straight-line)
  Fx_r                = T * F_DRIVE_MAX
"""

import numpy as np

try:
    from scipy.signal import sosfiltfilt, butter

    def lowpass(x: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
        sos = butter(order, cutoff_hz, fs=fs_hz, output="sos")
        return sosfiltfilt(sos, x)

except ImportError:
    def lowpass(x: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
        k = max(1, int(fs_hz / cutoff_hz / 2))
        return np.convolve(x, np.ones(k) / k, mode="same")


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _lstsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return theta


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def fit_drag(vx: np.ndarray, ax: np.ndarray, M: float) -> dict:
    """Coast-down:  M*ax = -C_DRAG * vx * |vx|

    Args:
        vx: longitudinal speed [m/s]
        ax: longitudinal acceleration [m/s^2], negative during coast-down
        M:  vehicle mass [kg]
    """
    A = (-vx * np.abs(vx)).reshape(-1, 1)
    b = M * ax
    theta = _lstsq(A, b)
    y_hat = A @ theta
    return {
        "C_DRAG": float(theta[0]),
        "R2": float(_r2(b, y_hat)),
        "RMSE": float(np.sqrt(np.mean((b - y_hat) ** 2))),
    }


def fit_drivetrain(T: np.ndarray, vx: np.ndarray, ax: np.ndarray, M: float) -> dict:
    """Straight-line accel:  M*ax = F_DRIVE_MAX*T - C_DRAG*vx*|vx|

    Args:
        T:  normalised throttle [0, 1]
        vx: longitudinal speed [m/s]
        ax: longitudinal acceleration [m/s^2]
        M:  vehicle mass [kg]
    """
    A = np.column_stack([T, -vx * np.abs(vx)])
    b = M * ax
    theta = _lstsq(A, b)
    y_hat = A @ theta
    return {
        "F_DRIVE_MAX": float(theta[0]),
        "C_DRAG": float(theta[1]),
        "R2": float(_r2(b, y_hat)),
        "RMSE": float(np.sqrt(np.mean((b - y_hat) ** 2))),
    }


def fit_understeer_gradient(
    delta_rad: np.ndarray, vx: np.ndarray, r: np.ndarray, g: float = 9.81
) -> dict:
    """Steady-state circle:  delta*R = L + K_us * vx^2/g   where R = vx/r

    K_us > 0 → understeer,  K_us < 0 → oversteer.
    L (estimated wheelbase from intercept) can cross-check the spec value.

    Args:
        delta_rad: commanded steer angle [rad]
        vx:        longitudinal speed at steady state [m/s]
        r:         yaw rate at steady state [rad/s]
        g:         gravitational acceleration [m/s^2]
    """
    R = vx / np.clip(np.abs(r), 1e-3, None) * np.sign(r)
    y = delta_rad * R           # = L + K_us * vx^2/g
    x = vx ** 2 / g
    A = np.column_stack([np.ones_like(x), x])
    theta = _lstsq(A, y)
    y_hat = A @ theta
    return {
        "L": float(theta[0]),       # wheelbase estimate from intercept [m]
        "K_us": float(theta[1]),    # understeer gradient [rad·s²/m]
        "R2": float(_r2(y, y_hat)),
    }


def fit_bicycle_linear(
    delta: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    r: np.ndarray,
    vy_dot: np.ndarray,
    r_dot: np.ndarray,
    M: float,
    LF: float,
    LR: float,
) -> dict:
    """Joint OLS for [CA_F, CA_R, IZ] from a steering chirp dataset.

    Stacks the linearised lateral-force and yaw-moment equations:

      vy row : [phi_f,      phi_r,      0     ] @ [CA_F, CA_R, IZ] = M*(vy_dot + vx*r)
      yaw row: [LF*phi_f, -LR*phi_r, -r_dot  ] @ [CA_F, CA_R, IZ] = 0

    where phi_f = delta - (vy+LF*r)/vx  and  phi_r = -(vy-LR*r)/vx.

    Args:
        delta:  commanded steer angle [rad]
        vx:     longitudinal speed [m/s]
        vy:     lateral speed [m/s]
        r:      yaw rate [rad/s]
        vy_dot: lateral acceleration [m/s^2] (smoothed finite difference)
        r_dot:  yaw acceleration [rad/s^2] (smoothed finite difference)
        M:      vehicle mass [kg]
        LF:     CoG-to-front-axle distance [m]
        LR:     CoG-to-rear-axle distance [m]
    """
    phi_f = delta - (vy + LF * r) / vx
    phi_r = -(vy - LR * r) / vx
    N = len(delta)

    A_vy = np.column_stack([phi_f, phi_r, np.zeros(N)])
    b_vy = M * (vy_dot + vx * r)

    A_r = np.column_stack([LF * phi_f, -LR * phi_r, -r_dot])
    b_r = np.zeros(N)

    A = np.vstack([A_vy, A_r])
    b = np.concatenate([b_vy, b_r])
    theta = _lstsq(A, b)

    CA_F, CA_R, IZ = theta
    return {
        "CA_F": float(CA_F),
        "CA_R": float(CA_R),
        "IZ": float(IZ),
        "R2_vy": float(_r2(b_vy, A_vy @ theta)),
        "R2_r": float(_r2(A_r @ theta, np.zeros(N))),  # how well yaw eq closes
    }


def extract_mu(ay_array: np.ndarray, g: float = 9.81, percentile: float = 95) -> dict:
    """Estimate peak friction coefficient from lateral acceleration samples.

    Uses the given percentile rather than the raw max to reject noise spikes.
    The result is a lower bound on µ — the vehicle may never have been driven
    to the absolute friction limit during sysid.

    Args:
        ay_array:   lateral acceleration [m/s^2], all collected maneuvers
        g:          gravitational acceleration [m/s^2]
        percentile: percentile of |ay| to use for the µ estimate
    """
    MU = float(np.percentile(np.abs(ay_array), percentile) / g)
    return {"MU": MU, "ay_percentile": int(percentile)}


def validate_forward_sim(
    params: dict,
    delta_arr: np.ndarray,
    T_arr: np.ndarray,
    vx0: float,
    vy0: float,
    r0: float,
    vx_meas: np.ndarray,
    vy_meas: np.ndarray,
    r_meas: np.ndarray,
    dt: float,
) -> dict:
    """Simulate the full nonlinear bicycle model and compare to measured signals.

    Uses the same equations as drift_env.py (tanh saturation + friction ellipse)
    so the RMSE reflects how well the identified params reproduce BeamNG behaviour.

    Args:
        params:   dict with keys matching DriftEnv class attributes
        delta_arr, T_arr: commanded inputs at each timestep
        vx0, vy0, r0: initial state
        vx_meas, vy_meas, r_meas: measured state timeseries
        dt: timestep [s]
    """
    M = params["M"]
    IZ = params["IZ"]
    LF = params["LF"]
    LR = params["LR"]
    CA_F = params["CA_F"]
    CA_R = params["CA_R"]
    MU = params["MU"]
    F_DRIVE_MAX = params["F_DRIVE_MAX"]
    C_DRAG = params["C_DRAG"]

    g = 9.81
    FY_MAX_F = MU * M * g * LR / (LF + LR)
    FY_MAX_R = MU * M * g * LF / (LF + LR)

    N = len(delta_arr)
    vx_sim = np.empty(N)
    vy_sim = np.empty(N)
    r_sim = np.empty(N)

    vx, vy, r = float(vx0), float(vy0), float(r0)
    for i in range(N):
        d = float(delta_arr[i])
        T = float(T_arr[i])
        vx_s = max(vx, 0.5)

        alpha_f = np.arctan2(vy + LF * r, vx_s) - d
        alpha_r = np.arctan2(vy - LR * r, vx_s)

        Fx_r = float(np.clip(T * F_DRIVE_MAX, -FY_MAX_R, FY_MAX_R))
        fy_max_r_eff = FY_MAX_R * np.sqrt(max(1.0 - (Fx_r / FY_MAX_R) ** 2, 1e-3))

        Fyf = -FY_MAX_F * np.tanh(CA_F * alpha_f / FY_MAX_F)
        Fyr = -fy_max_r_eff * np.tanh(CA_R * alpha_r / fy_max_r_eff)

        Fx = Fx_r - C_DRAG * vx * abs(vx)
        vx_dot = (Fx - Fyf * np.sin(d)) / M + r * vy
        vy_dot = (Fyf * np.cos(d) + Fyr) / M - r * vx
        r_dot = (LF * Fyf * np.cos(d) - LR * Fyr) / IZ

        vx_sim[i] = vx
        vy_sim[i] = vy
        r_sim[i] = r

        vx += dt * vx_dot
        vy += dt * vy_dot
        r += dt * r_dot
        vx = max(vx, 0.0)

    return {
        "RMSE_vx": float(np.sqrt(np.mean((vx_sim - vx_meas) ** 2))),
        "RMSE_vy": float(np.sqrt(np.mean((vy_sim - vy_meas) ** 2))),
        "RMSE_r": float(np.sqrt(np.mean((r_sim - r_meas) ** 2))),
    }
