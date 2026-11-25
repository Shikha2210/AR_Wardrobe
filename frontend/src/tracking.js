// Port of backend/tracking.py: One Euro smoothing, landmark validation, torso geometry.
const REQUIRED = [11, 12, 23, 24];
const SMOOTH_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
export const mul = (a, k) => [a[0] * k, a[1] * k];
export const norm = (a) => Math.hypot(a[0], a[1]);

class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.015, dCutoff = 1.0) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.reset();
  }
  reset() { this.t = null; this.x = null; this.dx = null; }
  static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  update(v, t) {
    if (this.x === null) { this.x = [v[0], v[1]]; this.dx = [0, 0]; this.t = t; return [v[0], v[1]]; }
    const dt = Math.max(1 / 240, Math.min(t - this.t, 0.2));
    const d = [(v[0] - this.x[0]) / dt, (v[1] - this.x[1]) / dt];
    const ad = OneEuro.alpha(this.dCutoff, dt);
    const dxHat = [ad * d[0] + (1 - ad) * this.dx[0], ad * d[1] + (1 - ad) * this.dx[1]];
    const cutoff = this.minCutoff + this.beta * norm(dxHat);
    const a = OneEuro.alpha(cutoff, dt);
    const f = [a * v[0] + (1 - a) * this.x[0], a * v[1] + (1 - a) * this.x[1]];
    this.x = f; this.dx = dxHat; this.t = t;
    return [f[0], f[1]];
  }
}

export class LandmarkTracker {
  constructor(minVisibility = 0.45) {
    this.filters = {};
    SMOOTH_IDS.forEach((i) => { this.filters[i] = new OneEuro(); });
    this.minVisibility = minVisibility;
    this.last = null; this.lastSeen = null;
  }
  // landmarks: array of {x,y,visibility} normalised 0..1; t in seconds.
  update(landmarks, width, height, t) {
    const ok = landmarks && REQUIRED.every((i) => {
      const lm = landmarks[i];
      return lm && (lm.visibility ?? 1) >= this.minVisibility && Number.isFinite(lm.x) && Number.isFinite(lm.y);
    });
    if (ok) {
      const pts = {};
      landmarks.slice(0, 33).forEach((lm, i) => {
        if ((lm.visibility ?? 1) < 0.30 || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) return;
        const v = [lm.x * width, lm.y * height];
        pts[i] = this.filters[i] ? this.filters[i].update(v, t) : v;
      });
      this.last = pts; this.lastSeen = t;
      return { pts, status: 'TRACKING' };
    }
    if (this.last && t - this.lastSeen < 0.30) return { pts: this.last, status: 'HOLDING TRACK' };
    if (this.lastSeen !== null && t - this.lastSeen >= 0.30) this.reset();
    return { pts: null, status: 'NO PERSON DETECTED' };
  }
  reset() {
    Object.values(this.filters).forEach((f) => f.reset());
    this.last = null; this.lastSeen = null;
  }
}

export function torsoGeometry(pts) {
  if (!pts || REQUIRED.some((i) => !pts[i])) return null;
  const [ls, rs, lh, rh] = REQUIRED.map((i) => pts[i]);
  const shoulder = mul(add(ls, rs), 0.5);
  const hip = mul(add(lh, rh), 0.5);
  const width = norm(sub(rs, ls));
  const length = norm(sub(hip, shoulder));
  if (width < 25 || length < 35) return null;
  const horizontal = mul(sub(rs, ls), 1 / width);
  const vertical = mul(sub(hip, shoulder), 1 / length);
  if (Math.abs(horizontal[0] * vertical[1] - horizontal[1] * vertical[0]) < 0.30) return null;
  return { shoulder, horizontal, vertical, width, length };
}
