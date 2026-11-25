// Browser replacement for backend/garment_ai.py (no GrabCut here: colour-distance + flood fill,
// or pose polygon + person mask for photos of a model).
import { loadImagePose } from './pose.js';

const MAX = 800;

async function toCanvas(file) {
  let bmp;
  try { bmp = await createImageBitmap(file); } catch { throw new Error('Could not open the uploaded image.'); }
  if (Math.min(bmp.width, bmp.height) < 80) throw new Error('Image must be at least 80 x 80 pixels.');
  const q = Math.min(1, MAX / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * q); c.height = Math.round(bmp.height * q);
  c.getContext('2d', { willReadFrequently: true }).drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close();
  return c;
}

function largestComponent(mask, w, h) {
  const label = new Int32Array(w * h); const stack = new Int32Array(w * h);
  let best = 0, bestId = 0, id = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || label[i]) continue;
    id++; let sp = 0, n = 0; stack[sp++] = i; label[i] = id;
    while (sp) {
      const p = stack[--sp]; n++;
      const x = p % w, y = (p / w) | 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q = ny * w + nx;
        if (mask[q] && !label[q]) { label[q] = id; stack[sp++] = q; }
      }
    }
    if (n > best) { best = n; bestId = id; }
  }
  const out = new Uint8Array(w * h);
  if (bestId) for (let i = 0; i < out.length; i++) out[i] = label[i] === bestId ? 1 : 0;
  return out;
}

// Plain-background removal: flood fill from the border over pixels close to the border colour.
function backgroundMask(data, w, h) {
  const px = (x, y) => (y * w + x) * 4;
  const border = [];
  for (let x = 0; x < w; x++) border.push(px(x, 0), px(x, 1), px(x, h - 1), px(x, h - 2));
  for (let y = 0; y < h; y++) border.push(px(0, y), px(1, y), px(w - 1, y), px(w - 2, y));
  const med = (ch) => { const v = border.map((i) => data[i + ch]).sort((a, b) => a - b); return v[v.length >> 1]; };
  const bg = [med(0), med(1), med(2)];
  const dist = (i) => Math.hypot(data[i] - bg[0], data[i + 1] - bg[1], data[i + 2] - bg[2]);
  const bd = border.map(dist).sort((a, b) => a - b);
  const thr = Math.min(90, Math.max(32, bd[Math.floor(bd.length * 0.95)] * 1.5 + 12));
  const isBg = new Uint8Array(w * h);
  const stack = new Int32Array(w * h); let sp = 0;
  const push = (x, y) => {
    const p = y * w + x;
    if (isBg[p] || dist(p * 4) > thr) return;
    isBg[p] = 1; stack[sp++] = p;
  };
  for (let x = 0; x < w; x++) { push(x, 0); push(x, h - 1); }
  for (let y = 0; y < h; y++) { push(0, y); push(w - 1, y); }
  while (sp) {
    const p = stack[--sp]; const x = p % w, y = (p / w) | 0;
    if (x > 0) push(x - 1, y);
    if (x < w - 1) push(x + 1, y);
    if (y > 0) push(x, y - 1);
    if (y < h - 1) push(x, y + 1);
  }
  const fg = new Uint8Array(w * h);
  for (let i = 0; i < fg.length; i++) fg[i] = isBg[i] ? 0 : 1;
  return fg;
}

function finish(canvas, fg, w, h) {
  const comp = largestComponent(fg, w, h);
  let area = 0; for (let i = 0; i < comp.length; i++) area += comp[i];
  if (area < 0.02 * w * h || area > 0.97 * w * h) throw new Error('No clear garment found. Use a centred garment on an uncluttered background.');
  // 1px erosion (removes background halo) then 3x3 box blur for a soft edge.
  const er = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const p = y * w + x;
    er[p] = comp[p] && comp[p - 1] && comp[p + 1] && comp[p - w] && comp[p + w] ? 255 : 0;
  }
  const alpha = new Uint8ClampedArray(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let s = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += er[(y + dy) * w + x + dx];
    alpha[y * w + x] = s / 9;
  }
  const g = canvas.getContext('2d');
  const img = g.getImageData(0, 0, w, h);
  let x0 = w, y0 = h, x1 = 0, y1 = 0;
  for (let i = 0; i < alpha.length; i++) {
    img.data[i * 4 + 3] = alpha[i];
    if (alpha[i] > 12) { const x = i % w, y = (i / w) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  if (x1 - x0 < 20 || y1 - y0 < 20) throw new Error('The clothing region is too small or unclear. Try a closer, well-lit photo.');
  g.putImageData(img, 0, 0);
  const out = document.createElement('canvas');
  out.width = x1 - x0 + 1; out.height = y1 - y0 + 1;
  out.getContext('2d').drawImage(canvas, x0, y0, out.width, out.height, 0, 0, out.width, out.height);
  return out;
}

async function modelMask(canvas, w, h) {
  const pose = await loadImagePose();
  const res = pose.detect(canvas);
  const lm = res.landmarks?.[0];
  if (!lm) throw new Error('No person detected. Use a front-facing upper-body photo.');
  if ([11, 12, 23, 24].some((i) => (lm[i].visibility ?? 1) < 0.45)) throw new Error('Shoulders and hips must be visible.');
  const P = (i) => [lm[i].x * w, lm[i].y * h];
  const [ls, rs, le, re, lh, rh] = [11, 12, 13, 14, 23, 24].map(P);
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const sw = d(ls, rs);
  if (sw < 25 || d([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2], [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]) < 35) throw new Error('Person is too small; upload a closer photo.');
  const poly = [
    [ls[0] - 0.1 * sw, ls[1] - 0.02 * sw], [rs[0] + 0.1 * sw, rs[1] - 0.02 * sw],
    [re[0] + (rs[0] - re[0]) * 0.1, re[1] + (rs[1] - re[1]) * 0.1],
    [rh[0] + 0.06 * sw, rh[1] + 0.03 * d(rh, ls)], [lh[0] - 0.06 * sw, lh[1] + 0.03 * d(lh, rs)],
    [le[0] + (ls[0] - le[0]) * 0.1, le[1] + (ls[1] - le[1]) * 0.1],
  ];
  const pc = document.createElement('canvas'); pc.width = w; pc.height = h;
  const pg = pc.getContext('2d', { willReadFrequently: true }); pg.fillStyle = '#fff';
  pg.beginPath(); poly.forEach((p, i) => (i ? pg.lineTo(p[0], p[1]) : pg.moveTo(p[0], p[1]))); pg.closePath(); pg.fill();
  const pd = pg.getImageData(0, 0, w, h).data;
  const seg = res.segmentationMasks?.[0];
  const sm = seg ? seg.getAsFloat32Array() : null;
  const fg = new Uint8Array(w * h);
  for (let i = 0; i < fg.length; i++) fg[i] = pd[i * 4 + 3] > 127 && (!sm || sm[i] > 0.5) ? 1 : 0;
  res.segmentationMasks?.forEach((m) => m.close());
  return fg;
}

/** Returns a tightly-cropped transparent RGBA canvas of the garment. */
export async function extractGarment(file, mode) {
  const canvas = await toCanvas(file);
  const { width: w, height: h } = canvas;
  const g = canvas.getContext('2d', { willReadFrequently: true });
  let fg;
  if (mode === 'garment') fg = backgroundMask(g.getImageData(0, 0, w, h).data, w, h);
  else if (mode === 'model') fg = await modelMask(canvas, w, h);
  else throw new Error('Unknown extraction mode.');
  return finish(canvas, fg, w, h);
}
