"""Pose landmark validation, smoothing, and torso geometry for Phase 5.2."""
import math
import time
import numpy as np

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
REQUIRED = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
SMOOTH_IDS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.015, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.t = self.x = self.dx = None

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, value, timestamp=None):
        x = np.asarray(value, dtype=np.float64)
        t = time.perf_counter() if timestamp is None else timestamp
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(1 / 240, min(t - self.t, 0.2))
        derivative = (x - self.x) / dt
        ad = self.alpha(self.d_cutoff, dt)
        dx_hat = ad * derivative + (1 - ad) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = self.alpha(cutoff, dt)
        filtered = a * x + (1 - a) * self.x
        self.x, self.dx, self.t = filtered, dx_hat, t
        return filtered.copy()

    def reset(self):
        self.t = self.x = self.dx = None


class LandmarkTracker:
    """Returns visible MediaPipe pose points (0..32); torso anchors remain mandatory."""
    def __init__(self, min_visibility=0.45):
        self.filters = {i: OneEuro() for i in SMOOTH_IDS}
        self.min_visibility = min_visibility
        self.last, self.last_seen = None, None

    def update(self, landmarks, width, height, timestamp=None):
        now = time.perf_counter() if timestamp is None else timestamp
        if landmarks is None:
            valid_anchors = False
        else:
            valid_anchors = all(
                len(landmarks) > i
                and float(landmarks[i].visibility) >= self.min_visibility
                and math.isfinite(landmarks[i].x)
                and math.isfinite(landmarks[i].y)
                for i in REQUIRED
            )
        if valid_anchors:
            pts = {}
            for i, lm in enumerate(landmarks[:33]):
                if (float(lm.visibility) < 0.30
                        or not math.isfinite(lm.x) or not math.isfinite(lm.y)):
                    continue
                value = (lm.x * width, lm.y * height)
                pts[i] = self.filters[i].update(value, now) if i in self.filters else np.asarray(value, dtype=np.float64)
            self.last, self.last_seen = pts, now
            return pts, "TRACKING"
        if self.last is not None and now - self.last_seen < 0.30:
            return self.last, "HOLDING TRACK"
        if self.last_seen is not None and now - self.last_seen >= 0.30:
            self.reset()
        return None, "NO PERSON DETECTED"

    def reset(self):
        for f in self.filters.values():
            f.reset()
        self.last = self.last_seen = None


def torso_geometry(pts):
    if pts is None or any(i not in pts for i in REQUIRED):
        return None
    ls, rs, lh, rh = (pts[i] for i in REQUIRED)
    shoulder = (ls + rs) / 2
    hip = (lh + rh) / 2
    width = float(np.linalg.norm(rs - ls))
    length = float(np.linalg.norm(hip - shoulder))
    if width < 25 or length < 35:
        return None
    horizontal = (rs - ls) / width
    vertical = (hip - shoulder) / length
    if abs(float(horizontal[0] * vertical[1] - horizontal[1] * vertical[0])) < 0.30:
        return None
    return shoulder, horizontal, vertical, width, length
