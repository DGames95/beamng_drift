"""Track geometry for BeamNG path-following and RL.

Centerline sampled every DS meters with heading psi, curvature kappa, and
arc-length s at each sample.  Two generators:

    Track.circle(radius)       closed circular track (constant curvature)
    Track.random_track(rng)    open track, piecewise-linear curvature profile,
                               |kappa| bounded in [KAPPA_LO, KAPPA_HI]

Track-frame convention (matches driftRL for future RL policy transfer):
    e_y   > 0  means the car is LEFT of the centerline (w.r.t. travel direction)
    e_psi      heading error (car psi - track psi), wrapped to (-pi, pi]
    positive curvature = left turn

Scale: DS = 1.0 m, radii 30-80 m, half-width 5 m — appropriate for BeamNG
world-space (1 unit = 1 m) at control rates of 50 Hz.

RL observation interface:
    track.obs(X, Y, psi, vx, vy, r, hint, lookahead_dists)
    -> np.ndarray [e_y, e_psi, kappa_0, kappa_1, ..., kappa_N]
    Same shape/meaning as driftRL obs slice [3:8], so a trained policy can
    read from it directly once the full obs vector is assembled in the same
    order as DriftEnv.
"""

import numpy as np

DS          = 1.0            # centerline sample spacing [m]
HALF_WIDTH  = 20.0            # track half-width [m]

# random track curvature bounds: radii 30-80 m
KAPPA_LO    = 1.0 / 80.0
KAPPA_HI    = 1.0 / 30.0
SEG_LEN     = (40.0, 120.0)  # arc-length range between curvature knots [m]

# default lookahead distances [m] — matches driftRL (0, 10, 25 m ahead)
DEFAULT_LOOKAHEAD = (0.0, 10.0, 25.0)

# exit threshold: car is considered off-track when |e_y| exceeds this
OFF_TRACK_MARGIN = HALF_WIDTH


class Track:
    def __init__(self, xy, psi, kappa, closed):
        self.xy     = xy       # (N, 2) centerline world positions [m]
        self.psi    = psi      # (N,)   tangent heading [rad]
        self.kappa  = kappa    # (N,)   signed curvature [1/m]
        self.closed = closed
        self.n      = len(xy)
        self.s      = np.arange(self.n) * DS
        self.length = self.n * DS
        self.half_width = HALF_WIDTH

        normal = np.stack([-np.sin(psi), np.cos(psi)], axis=1)  # left-pointing normal
        self.left  = xy + HALF_WIDTH * normal   # boundary polylines (for rendering)
        self.right = xy - HALF_WIDTH * normal
        self._normal = normal

    # ---------------------------------------------------------------------- generators

    @classmethod
    def circle(cls, radius: float = 40.0, origin=(0.0, 0.0)):
        """Closed circular track centred at `origin`."""
        n  = max(4, int(round(2 * np.pi * radius / DS)))
        th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        xy = np.stack([
            origin[0] + radius * np.cos(th),
            origin[1] + radius * np.sin(th),
        ], axis=1)
        psi   = th + np.pi / 2.0           # CCW travel
        kappa = np.full(n, 1.0 / radius)
        return cls(xy, psi, kappa, closed=True)

    @classmethod
    def random_track(cls, rng, length: float = 600.0, origin=(0.0, 0.0)):
        """Open track starting at `origin` facing +X.

        Curvature is a piecewise-linear profile: random knots every SEG_LEN
        metres with |kappa| in [KAPPA_LO, KAPPA_HI], alternating sign so the
        track keeps turning and doesn't self-intersect too quickly.
        """
        knot_s, knot_k = [0.0], [0.0]
        sign = 1.0
        while knot_s[-1] < length:
            knot_s.append(knot_s[-1] + rng.uniform(*SEG_LEN))
            sign = -sign  # alternate turns for variety; can remove for purely random
            knot_k.append(sign * rng.uniform(KAPPA_LO, KAPPA_HI))

        s     = np.arange(int(length / DS)) * DS
        kappa = np.interp(s, knot_s, knot_k)
        psi   = np.concatenate([[0.0], np.cumsum(kappa[:-1]) * DS])
        x     = origin[0] + np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1])) * DS])
        y     = origin[1] + np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1])) * DS])
        return cls(np.stack([x, y], axis=1), psi, kappa, closed=False)

    # ---------------------------------------------------------------------- queries

    def nearest(self, x: float, y: float, hint: int) -> int:
        """Index of the nearest centerline sample, searched in a window around `hint`.

        Window covers +-50 m (50 samples at DS=1 m) — more than a car can travel
        in one control tick, so the nearest sample is always inside it.
        """
        w = 50
        if self.closed:
            idx = np.arange(hint - w, hint + w) % self.n
        else:
            idx = np.arange(max(hint - w, 0), min(hint + w, self.n))
        d2 = np.sum((self.xy[idx] - [x, y]) ** 2, axis=1)
        return int(idx[np.argmin(d2)])

    def frame(
        self,
        x: float,
        y: float,
        psi: float,
        hint: int,
        lookahead_dists=DEFAULT_LOOKAHEAD,
    ):
        """Track-frame errors and curvature preview.

        Returns
        -------
        e_y     : float   lateral error [m], positive = left of centreline
        e_psi   : float   heading error [rad], wrapped to (-pi, pi]
        kappa_p : ndarray curvature at each lookahead distance
        idx     : int     nearest centreline index (use as next hint)
        """
        i     = self.nearest(x, y, hint)
        e_y   = float(np.dot([x, y] - self.xy[i], self._normal[i]))
        e_psi = float((psi - self.psi[i] + np.pi) % (2 * np.pi) - np.pi)

        kappa_p = []
        for d in lookahead_dists:
            j = i + int(d / DS)
            j = j % self.n if self.closed else min(j, self.n - 1)
            kappa_p.append(self.kappa[j])

        return e_y, e_psi, np.array(kappa_p), i

    def off_track(self, e_y: float) -> bool:
        """True when the car has left the driveable surface."""
        return abs(e_y) > OFF_TRACK_MARGIN

    def at_end(self, idx: int) -> bool:
        """True when an open track is nearly finished (within 4 m of end)."""
        return (not self.closed) and idx >= self.n - int(4.0 / DS)

    # ---------------------------------------------------------------------- RL interface

    def obs(
        self,
        X: float,
        Y: float,
        psi: float,
        vx: float,
        vy: float,
        r: float,
        hint: int,
        lookahead_dists=DEFAULT_LOOKAHEAD,
    ):
        """Build the full RL-style observation vector.

        Layout matches DriftEnv (driftRL/drift_env.py):
            [vx, vy, r, e_y, e_psi, kappa_0, kappa_1, ..., kappa_N]

        Scaled by the same OBS_SCALE constants as driftRL (drift_env.py
        OBS_SCALE) so a policy trained there transfers here directly:
            vx /4, vy /1, r /0.5, e_y /1.5, e_psi /0.3, kappa /0.05
        These MUST equal driftRL's OBS_SCALE — a mismatch silently feeds the
        policy out-of-distribution inputs and it drives badly for no obvious
        reason. (Earlier this used [20,10,2,4,pi,0.05...], which did not match.)
        """
        e_y, e_psi, kappa_p, idx = self.frame(X, Y, psi, hint, lookahead_dists)
        raw = np.array([vx, vy, r, e_y, e_psi, *kappa_p], dtype=np.float32)
        scale = np.array(
            [4.0, 1.0, 0.5, 1.5, 0.3] + [0.05] * len(lookahead_dists),
            dtype=np.float32,
        )
        return raw / scale, e_y, e_psi, kappa_p, idx
