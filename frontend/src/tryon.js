// Port of backend/tryon.py: piecewise-affine T-shirt mesh warp drawn with Canvas 2D.
import { torsoGeometry, sub, add, mul, norm } from './tracking.js';

// RGB tints (the Python source stored them as BGR).
export const COLORS = {
  white: [235, 235, 235], black: [36, 32, 32], blue: [48, 112, 205],
  red: [205, 65, 55], green: [75, 145, 70], purple: [155, 85, 155],
};
export const DEMO_ID = 'levis_black_tee';

const makeCanvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
const ctxOf = (c) => c.getContext('2d', { willReadFrequently: false });

function tint(base, rgb) {
  const c = makeCanvas(base.width, base.height);
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(base, 0, 0);
  const img = g.getImageData(0, 0, c.width, c.height), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = (0.114 * d[i + 2] + 0.587 * d[i + 1] + 0.299 * d[i]) / 255; // same luma weights as cv2 BGR2GRAY
    const k = 0.48 + 0.52 * gray;
    d[i] = Math.min(255, rgb[0] * k); d[i + 1] = Math.min(255, rgb[1] * k); d[i + 2] = Math.min(255, rgb[2] * k);
  }
  g.putImageData(img, 0, 0);
  return c;
}

// Row-wise horizontal expansion of the narrow bundled demo shirt (Python _widen_torso_source).
function widenTorso(src) {
  const w = src.width, h = src.height, out = makeCanvas(w, h), g = out.getContext('2d');
  g.imageSmoothingQuality = 'high';
  const center = (w - 1) * 0.5;
  for (let y = 0; y < h; y++) {
    const v = y / Math.max(h - 1, 1);
    let t = Math.min(1, Math.max(0, (v - 0.27) / 0.20));
    t = t * t * (3 - 2 * t);
    const s = 1 + 0.78 * t;
    g.drawImage(src, 0, y, w, 1, center - center * s, y, w * s, 1);
  }
  return out;
}

function fitNeckline(dst, shoulder, horizontal, vertical, bodyW, torsoH) {
  const centerCol = 3;
  const neckCenter = sub(shoulder, mul(vertical, 0.105 * torsoH));
  const half = 0.19 * bodyW;
  const orig = dst[2].map((p) => [p[0], p[1]]);
  const targets = {
    [centerCol - 1]: sub(sub(neckCenter, mul(horizontal, half)), mul(vertical, 0.025 * torsoH)),
    [centerCol]: add(neckCenter, mul(vertical, 0.035 * torsoH)),
    [centerCol + 1]: sub(add(neckCenter, mul(horizontal, half)), mul(vertical, 0.025 * torsoH)),
  };
  const deltas = {};
  for (const col of Object.keys(targets).map(Number)) {
    deltas[col] = sub(targets[col], orig[col]);
    dst[2][col] = targets[col];
  }
  for (const col of Object.keys(deltas).map(Number)) dst[3][col] = add(dst[3][col], mul(deltas[col], 0.30));
  for (const col of [centerCol - 2, centerCol + 2]) {
    const nearest = col < centerCol ? centerCol - 1 : centerCol + 1;
    dst[2][col] = add(dst[2][col], mul(deltas[nearest], 0.22));
    dst[3][col] = add(dst[3][col], mul(deltas[nearest], 0.10));
  }
}

function fitShoulders(dst, pts, bodyW) {
  if (![11, 12, 23, 24].every((i) => pts[i])) return;
  const left = pts[11], right = pts[12];
  const axis = sub(right, left);
  const sw = norm(axis);
  if (sw < 1) return;
  const h = mul(axis, 1 / sw);
  const ext = 0.055 * (bodyW ?? sw);
  const leftSeam = sub(left, mul(h, ext)), rightSeam = add(right, mul(h, ext));
  dst[3][1] = leftSeam; dst[3][5] = rightSeam;
  for (const [row, f] of [[1, 0.10], [2, 0.055]]) {
    dst[row][1] = sub(dst[row][1], mul(h, f * sw));
    dst[row][5] = add(dst[row][5], mul(h, f * sw));
  }
  const hipMid = mul(add(pts[23], pts[24]), 0.5), shMid = mul(add(left, right), 0.5);
  const down = sub(hipMid, shMid), dl = norm(down);
  const vert = mul(down, 1 / Math.max(dl, 1));
  const lift = mul(vert, 0.055 * bodyW);
  dst[1][1] = sub(sub(leftSeam, mul(h, 0.19 * sw)), lift);
  dst[1][5] = sub(add(rightSeam, mul(h, 0.19 * sw)), lift);
  dst[2][1] = sub(leftSeam, mul(h, 0.105 * sw));
  dst[2][5] = add(rightSeam, mul(h, 0.105 * sw));
}

function drawTriangle(g, src, s0, s1, s2, d0, d1, d2) {
  const det = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s2[0] - s0[0]) * (s1[1] - s0[1]);
  if (Math.abs(det) < 1e-6) return;
  const a = ((d1[0] - d0[0]) * (s2[1] - s0[1]) - (d2[0] - d0[0]) * (s1[1] - s0[1])) / det;
  const c = ((d2[0] - d0[0]) * (s1[0] - s0[0]) - (d1[0] - d0[0]) * (s2[0] - s0[0])) / det;
  const b = ((d1[1] - d0[1]) * (s2[1] - s0[1]) - (d2[1] - d0[1]) * (s1[1] - s0[1])) / det;
  const d = ((d2[1] - d0[1]) * (s1[0] - s0[0]) - (d1[1] - d0[1]) * (s2[0] - s0[0])) / det;
  const e = d0[0] - a * s0[0] - c * s0[1];
  const f = d0[1] - b * s0[0] - d * s0[1];
  // Clip to the destination triangle, grown ~1.6px so neighbouring triangles leave no seams.
  const cx = (d0[0] + d1[0] + d2[0]) / 3, cy = (d0[1] + d1[1] + d2[1]) / 3;
  const grow = (p) => { const dx = p[0] - cx, dy = p[1] - cy, l = Math.hypot(dx, dy) || 1; return [p[0] + dx / l * 1.6, p[1] + dy / l * 1.6]; };
  const p0 = grow(d0), p1 = grow(d1), p2 = grow(d2);
  g.save();
  g.beginPath(); g.moveTo(p0[0], p0[1]); g.lineTo(p1[0], p1[1]); g.lineTo(p2[0], p2[1]); g.closePath(); g.clip();
  g.setTransform(a, b, c, d, e, f);
  const x0 = Math.max(0, Math.floor(Math.min(s0[0], s1[0], s2[0]) - 1)), y0 = Math.max(0, Math.floor(Math.min(s0[1], s1[1], s2[1]) - 1));
  const x1 = Math.min(src.width, Math.ceil(Math.max(s0[0], s1[0], s2[0]) + 1)), y1 = Math.min(src.height, Math.ceil(Math.max(s0[1], s1[1], s2[1]) + 1));
  if (x1 > x0 && y1 > y0) g.drawImage(src, x0, y0, x1 - x0, y1 - y0, x0, y0, x1 - x0, y1 - y0);
  g.restore();
}

export class TryOnEngine {
  constructor() {
    this.garments = {};      // id -> canvas (already widened where needed)
    this.layer = null; this.armMask = null; this.armPix = null;
  }

  loadBase(image) {
    const base = makeCanvas(image.naturalWidth || image.width, image.naturalHeight || image.height);
    base.getContext('2d').drawImage(image, 0, 0);
    for (const [name, rgb] of Object.entries(COLORS)) this.garments[name] = widenTorso(tint(base, rgb));
    this.garments[DEMO_ID] = widenTorso(base);
  }

  // canvas: RGBA, tightly cropped; real uploads keep their own proportions (no widening).
  setCustom(canvas) { this.garments.custom_upload = canvas; }

  _warp(g, garment, pts, geo, w, h) {
    const { shoulder, horizontal, vertical, width: bodyW, length: torsoH } = geo;
    const hip = mul(add(pts[23], pts[24]), 0.5);
    const hipW = norm(sub(pts[24], pts[23]));
    const hipRatio = Math.min(1.12, Math.max(0.62, hipW / Math.max(bodyW, 1)));
    const gw = garment.width, gh = garment.height;
    const rows = 7, cols = 7;
    const sx = [0.0, 0.22, 0.38, 0.50, 0.62, 0.78, 1.0].map((v) => v * (gw - 1));
    const sy = [0.0, 72.0, 108.0, 145.0, 205.0, 430.0, 759.0].map((v) => v * (gh - 1) / 759.0);
    const profile = [2.28, 2.44, 2.30, 1.86, 1.58, 1.62, 1.62];
    const top = sub(shoulder, mul(vertical, 0.055 * torsoH));
    const bottom = add(hip, mul(vertical, 0.035 * torsoH));
    const src = sy.map((y) => sx.map((x) => [x, y]));
    const dst = [];
    for (let r = 0; r < rows; r++) {
      const v = sy[r] / Math.max(gh - 1, 1);
      const center = add(mul(top, 1 - v), mul(bottom, v));
      const hipBlend = Math.max(0, Math.min(1, (v - 0.48) / 0.52));
      let width = bodyW * profile[r] * (1 - hipBlend * (1 - hipRatio) * 0.30);
      if (v >= 0.38 && v <= 0.78) width *= 1 - 0.012 * Math.sin(Math.PI * (v - 0.38) / 0.40);
      dst.push(sx.map((x) => {
        const u = (x / Math.max(gw - 1, 1)) * 2 - 1;
        return add(center, mul(horizontal, u * width * 0.5));
      }));
    }
    fitNeckline(dst, shoulder, horizontal, vertical, bodyW, torsoH);
    fitShoulders(dst, pts, bodyW);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'medium';
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        drawTriangle(g, garment, src[r][c], src[r][c + 1], src[r + 1][c], dst[r][c], dst[r][c + 1], dst[r + 1][c]);
        drawTriangle(g, garment, src[r][c + 1], src[r + 1][c + 1], src[r + 1][c], dst[r][c + 1], dst[r + 1][c + 1], dst[r + 1][c]);
      }
    }
  }

  _armMask(w, h, pts, bodyW) {
    if (!this.armMask || this.armMask.width !== w || this.armMask.height !== h) this.armMask = makeCanvas(w, h);
    const g = this.armMask.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, w, h);
    g.strokeStyle = '#fff'; g.lineCap = 'round';
    for (const [sid, eid, wid] of [[11, 13, 15], [12, 14, 16]]) {
      if (!pts[sid] || !pts[eid]) continue;
      const s = pts[sid], e = pts[eid];
      const a = add(s, mul(sub(e, s), 0.48));
      g.lineWidth = Math.max(5, bodyW * 0.055);
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(e[0], e[1]); g.stroke();
      if (pts[wid]) {
        g.lineWidth = Math.max(6, bodyW * 0.070);
        g.beginPath(); g.moveTo(e[0], e[1]); g.lineTo(pts[wid][0], pts[wid][1]); g.stroke();
      }
    }
    return this.armMask;
  }

  /** Draws the garment onto ctx (which already holds the mirrored camera frame). */
  render(ctx, frame, pts, outfit, personMask) {
    const geo = torsoGeometry(pts);
    const garment = this.garments[outfit];
    if (!geo || !garment) return false;
    const w = frame.width, h = frame.height;
    if (!this.layer || this.layer.width !== w || this.layer.height !== h) { this.layer = makeCanvas(w, h); this.armPix = makeCanvas(w, h); }
    const g = this.layer.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'source-over'; g.clearRect(0, 0, w, h);
    this._warp(g, garment, pts, geo, w, h);
    if (personMask) { g.globalCompositeOperation = 'destination-in'; g.drawImage(personMask, 0, 0); g.globalCompositeOperation = 'source-over'; }
    ctx.drawImage(this.layer, 0, 0);

    // Restore the real forearms/upper arms in front of the shirt.
    const mask = this._armMask(w, h, pts, geo.width);
    const a = this.armPix.getContext('2d');
    a.setTransform(1, 0, 0, 1, 0, 0); a.globalCompositeOperation = 'source-over'; a.clearRect(0, 0, w, h);
    const sigma = Math.max(0.8, geo.width * 0.008);
    if ('filter' in a) a.filter = `blur(${sigma}px)`;
    a.drawImage(mask, 0, 0);
    if ('filter' in a) a.filter = 'none';
    a.globalCompositeOperation = 'source-in';
    a.drawImage(frame, 0, 0);
    a.globalCompositeOperation = 'source-over';
    ctx.drawImage(this.armPix, 0, 0);
    return true;
  }
}
