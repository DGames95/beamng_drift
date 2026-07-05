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


def _ols(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS solve returning (theta, standard_errors, covariance).

    Standard errors come from the usual sigma^2 (A^T A)^-1 covariance, where
    sigma^2 = RSS / (n - p). They quantify how well-constrained each parameter
    is *given the model* — wide SEs flag a param the data barely pins down, which
    is exactly the "general uncertainty" signal we want to carry into the
    randomization ranges. (They do NOT capture model mismatch, i.e. BeamNG not
    actually being a linear bicycle; the fit R2 and regime coverage do that.)
    """
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ theta
    n, p = A.shape
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
        se = np.full(p, np.nan)
    return theta, se, cov


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
    theta, se, cov = _ols(A, b)
    y_hat = A @ theta
    return {
        "C_DRAG": float(theta[0]),
        "C_DRAG_se": float(se[0]),
        "cov": cov.tolist(),              # 1x1 covariance for [C_DRAG]
        "params": ["C_DRAG"],
        "R2": float(_r2(b, y_hat)),
        "RMSE": float(np.sqrt(np.mean((b - y_hat) ** 2))),
        "n": int(len(b)),
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
    theta, se, cov = _ols(A, b)
    y_hat = A @ theta
    # correlation between the two estimates — for a short accel these are
    # strongly (negatively) correlated, which is exactly why a JOINT
    # F_DRIVE_MAX / C_DRAG fit is ill-conditioned on this data.
    denom = se[0] * se[1]
    corr = float(cov[0, 1] / denom) if denom > 0 else float("nan")
    return {
        "F_DRIVE_MAX": float(theta[0]),
        "F_DRIVE_MAX_se": float(se[0]),
        "C_DRAG": float(theta[1]),
        "C_DRAG_se": float(se[1]),
        "cov": cov.tolist(),              # 2x2 covariance for [F_DRIVE_MAX, C_DRAG]
        "params": ["F_DRIVE_MAX", "C_DRAG"],
        "corr_Fdrive_Cdrag": corr,
        "R2": float(_r2(b, y_hat)),
        "RMSE": float(np.sqrt(np.mean((b - y_hat) ** 2))),
        "n": int(len(b)),
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


def fit_drive_fixed_drag(T: np.ndarray, vx: np.ndarray, ax: np.ndarray,
                         M: float, C_DRAG: float) -> dict:
    """F_DRIVE_MAX with C_DRAG FIXED (from the coast-down), decoupling the two.

    The joint accel fit is ill-conditioned (F_DRIVE_MAX and C_DRAG are strongly
    correlated over a short pull) and on a drift car wheelspin throws away the
    high-throttle samples, collapsing it to a single throttle level -> a
    degenerate, sometimes negative, estimate. With C_DRAG known, each
    non-wheelspin sample gives a direct estimate

        F_DRIVE_MAX_i = (M*ax_i + C_DRAG*vx_i*|vx_i|) / T_i ,

    valid only where the rear isn't spinning (otherwise Fx_r is traction-capped,
    not T*F_DRIVE_MAX). We report the median and the robust spread; if the engine
    map is nonlinear in throttle the per-throttle medians will fan out, which the
    returned breakdown exposes.
    """
    T = np.asarray(T, float); vx = np.asarray(vx, float); ax = np.asarray(ax, float)
    keep = T > 0.02
    est = (M * ax[keep] + C_DRAG * vx[keep] * np.abs(vx[keep])) / T[keep]
    est = est[np.isfinite(est)]
    if len(est) < 6:
        return {"F_DRIVE_MAX": None, "n": int(len(est)), "method": "fixed-C_DRAG"}
    med = float(np.median(est))
    # robust std from the IQR (1.349 sigma between the quartiles)
    q25, q75 = np.percentile(est, [25, 75])
    robust_std = float((q75 - q25) / 1.349)
    by_T = {}
    for t in np.unique(np.round(T[keep], 2)):
        m = np.round(T[keep], 2) == t
        if m.sum() >= 3:
            by_T[float(t)] = round(float(np.median(est[m])), 1)
    return {
        "F_DRIVE_MAX": med,
        "F_DRIVE_MAX_robust_std": robust_std,
        "per_throttle_median": by_T,
        "n": int(len(est)),
        "method": "fixed-C_DRAG per-sample median",
    }


def fit_axle_tire(alpha: np.ndarray, Fy: np.ndarray, alpha_lin_deg: float = 2.0) -> dict:
    """Fit a single axle's tyre curve from steady-state (slip, lateral-force) data.

    This is the robust replacement for the old joint-chirp OLS. The skidpad
    maneuver hands us per-axle lateral force directly from steady-state force
    balance (no noisy differentiation), so we just need to characterise the
    curve Fy(alpha):

      * cornering stiffness CA  — slope at small slip, where Fy ≈ -CA*alpha.
        Fit through the origin on the near-linear core (|alpha| < alpha_lin).
      * saturation Fy_sat       — the peak |Fy| the axle reached; this sets MU
        once divided by the axle's vertical load. Flagged as a *lower bound*
        unless the data actually rolled into saturation (outer-band slope well
        below CA), since a gentle run may never have hit the limit.

    Args:
        alpha:        slip angle [rad] (sign: positive slip -> negative Fy)
        Fy:           lateral force on the axle [N]
        alpha_lin_deg: half-width of the near-linear core for the stiffness fit
    """
    a = np.asarray(alpha, dtype=float)
    f = np.asarray(Fy, dtype=float)

    alin = float(np.radians(alpha_lin_deg))
    core = np.abs(a) < alin
    if core.sum() < 6:                       # sparse core -> widen to inner 50%
        alin = float(np.percentile(np.abs(a), 50)) if len(a) else alin
        core = np.abs(a) <= alin

    A = (-a[core]).reshape(-1, 1)            # Fy = -CA * alpha
    theta, se, cov = _ols(A, f[core])
    CA = float(theta[0])
    CA_se = float(se[0])
    CA_var = float(cov[0, 0])

    absf = np.abs(f)
    Fy_sat = float(np.percentile(absf, 95)) if len(f) else float("nan")
    amax = float(np.max(np.abs(a))) if len(a) else 0.0

    # outer-band incremental slope: a big drop vs CA means the tyre rolled into
    # saturation in this dataset (so Fy_sat is a real peak, not a lower bound)
    outer = np.abs(a) > alin
    sat_ratio = float("nan")
    if outer.sum() >= 6:
        Ao = (-a[outer]).reshape(-1, 1)
        to, _, _ = _ols(Ao, f[outer])
        sat_ratio = float(to[0] / CA) if CA != 0 else float("nan")
    reached_sat = (not np.isnan(sat_ratio)) and sat_ratio < 0.6 and amax > 1.5 * alin

    return {
        "CA": CA,
        "CA_se": CA_se,
        "CA_var": CA_var,
        "Fy_sat": Fy_sat,
        "alpha_lin_deg": float(np.degrees(alin)),
        "alpha_max_deg": float(np.degrees(amax)),
        "outer_slope_ratio": sat_ratio,
        "reached_saturation": bool(reached_sat),
        "n_core": int(core.sum()),
        "n_total": int(len(a)),
        "R2_core": float(_r2(f[core], A @ theta)) if core.sum() else float("nan"),
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
