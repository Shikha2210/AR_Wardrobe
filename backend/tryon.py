"""2D garment renderer for AR Wardrobe.

Uses a real transparent product garment when loaded, a body-proportioned
piecewise-affine mesh, and pose-based arm foreground restoration. This is not
3D cloth simulation; it is a lightweight webcam try-on approximation.
"""
from pathlib import Path
import cv2
import numpy as np
from tracking import torso_geometry

ASSET_DIR = Path(__file__).resolve().parent / "assets"
COLORS = {
    "white": (235, 235, 235), "black": (32, 32, 36), "blue": (205, 112, 48),
    "red": (55, 65, 205), "green": (70, 145, 75), "purple": (155, 85, 155)
}


class TryOnEngine:
    def __init__(self, grid_cols=7, grid_rows=7):
        self.garments = {}
        self.garment_types = {}
        self.grid_cols = max(7, int(grid_cols))
        self.grid_rows = max(7, int(grid_rows))
        # These are fallback color swatches only. The real demo garment is
        # loaded separately and is the default selection in the application.
        for name in COLORS:
            path = ASSET_DIR / "tshirt.png"
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            if image.ndim != 3 or image.shape[2] != 4:
                continue
            image = image.copy()
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            tint = np.asarray(COLORS[name], np.float32).reshape(1, 1, 3)
            image[:, :, :3] = np.uint8(np.clip(tint * (0.48 + 0.52 * gray[:, :, None]), 0, 255))
            self.garments[name] = image

    def load_garment_image(self, name, image_path, remove_light_background=True, garment_type="top"):
        """Load a front-view catalog image, remove a plain light background, retain fabric pixels."""
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read garment photo: {path}")
        has_alpha = image.ndim == 3 and image.shape[2] == 4
        bgr = image[:, :, :3] if has_alpha else image
        if remove_light_background:
            # Smooth alpha from near-white neutral background; preserve colored/dark fabric.
            mn = bgr.min(axis=2).astype(np.float32)
            mx = bgr.max(axis=2).astype(np.float32)
            chroma = mx - mn
            neutral = chroma < 38
            alpha = np.full(mn.shape, 255.0, np.float32)
            fade = neutral & (mn > 218) & (mn < 242)
            alpha[fade] = np.clip((242.0 - mn[fade]) * (255.0 / 24.0), 0, 255)
            alpha[neutral & (mn >= 242)] = 0
            # Only retain the largest connected foreground component to discard dust/background specks.
            count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 20).astype(np.uint8), 8)
            if count > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                alpha[labels != largest] = 0
            if has_alpha:
                alpha *= image[:, :, 3].astype(np.float32) / 255.0
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = np.uint8(alpha)
        else:
            rgba = image.copy() if has_alpha else cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        ys, xs = np.where(rgba[:, :, 3] > 12)
        if len(xs) < 20 or len(ys) < 20:
            raise ValueError("Could not isolate the T-shirt. Use a front-facing product image on a plain background.")
        # Tight crop; retain a small transparent safety margin.
        px = max(2, int((xs.max() - xs.min() + 1) * .015))
        py = max(2, int((ys.max() - ys.min() + 1) * .015))
        x0, x1 = max(0, xs.min()-px), min(rgba.shape[1], xs.max()+px+1)
        y0, y1 = max(0, ys.min()-py), min(rgba.shape[0], ys.max()+py+1)
        rgba = rgba[y0:y1, x0:x1]
        scale = min(1.0, 720.0 / max(rgba.shape[:2]))
        if scale < 1:
            rgba = cv2.resize(rgba, (int(rgba.shape[1]*scale), int(rgba.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        self.garments[name] = rgba
        self.garment_types[name] = garment_type
        return rgba

    @staticmethod
    def _warp_mesh(source, src_grid, dst_grid, frame_shape):
        """Build one inverse remap for the complete mesh (instead of warping each triangle separately)."""
        h, w = frame_shape[:2]
        gh, gw = source.shape[:2]
        map_x = np.full((h, w), -1, np.float32)
        map_y = np.full((h, w), -1, np.float32)

        rows, cols = src_grid.shape[:2]
        for r in range(rows - 1):
            for c in range(cols - 1):
                triangles = (
                    (src_grid[r, c], src_grid[r, c + 1], src_grid[r + 1, c],
                     dst_grid[r, c], dst_grid[r, c + 1], dst_grid[r + 1, c]),
                    (src_grid[r, c + 1], src_grid[r + 1, c + 1], src_grid[r + 1, c],
                     dst_grid[r, c + 1], dst_grid[r + 1, c + 1], dst_grid[r + 1, c]),
                )
                for s0, s1, s2, d0, d1, d2 in triangles:
                    dst = np.asarray([d0, d1, d2], np.float32)
                    x, y, bw, bh = cv2.boundingRect(dst)
                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(w, x + bw), min(h, y + bh)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    local = dst - np.array([x0, y0], np.float32)
                    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    cv2.fillConvexPoly(mask, np.round(local).astype(np.int32), 255)
                    if not np.any(mask):
                        continue
                    matrix = cv2.getAffineTransform(dst, np.asarray([s0, s1, s2], np.float32))
                    yy, xx = np.indices(mask.shape, dtype=np.float32)
                    xx += x0
                    yy += y0
                    sx = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
                    sy = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
                    region_x = map_x[y0:y1, x0:x1]
                    region_y = map_y[y0:y1, x0:x1]
                    valid = mask > 0
                    region_x[valid] = sx[valid]
                    region_y[valid] = sy[valid]

        return cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    @staticmethod
    def _widen_torso_source(garment):
        """Expand the narrow shirt-body portion while preserving the shoulder/sleeve silhouette.

        The supplied vector asset has a very narrow torso (about 35% of its
        canvas width). A uniform mesh cannot fix that without making sleeves
        excessively wide, so widen rows below the sleeve/underarm transition.
        """
        h, w = garment.shape[:2]
        center = (w - 1) * 0.5
        yy, xx = np.indices((h, w), dtype=np.float32)
        v = yy / max(h - 1, 1)
        # Smoothly ramp from no change at the shoulder/sleeve area to a
        # 1.78x torso expansion below the underarm. The previous 1.48x
        # still left the user's original shirt visible along both waist sides.
        t = np.clip((v - 0.27) / 0.20, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        scale = 1.0 + 0.78 * t
        map_x = center + (xx - center) / scale
        return cv2.remap(
            garment, map_x, yy, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

    @staticmethod
    def _fit_neckline(dst_grid, shoulder, horizontal, vertical, body_w, torso_h):
        """Place the collar around the neck base using explicit neck anchors.

        MediaPipe does not provide a reliable neck landmark. Estimate the
        neck base from the shoulder midpoint and torso axis, then place the
        left/centre/right collar controls around that point. The centre sits
        slightly lower than the sides to create a crew-neck U curve.
        """
        rows, cols = dst_grid.shape[:2]
        if rows < 7 or cols < 7:
            return dst_grid

        center_col = cols // 2
        # Neck base lies above the shoulder midpoint. Approximate its width
        # from shoulder width, not from the shirt image's full sleeve span.
        neck_center = shoulder - vertical * (0.105 * torso_h)
        neck_half_width = 0.19 * body_w

        # Source row 2 passes through the collar. Set three controls explicitly
        # so the collar spans the neck rather than following a straight chest line.
        original_row = dst_grid[2].copy()
        collar_targets = {
            center_col - 1: neck_center - horizontal * neck_half_width - vertical * (0.025 * torso_h),
            center_col:     neck_center + vertical * (0.035 * torso_h),
            center_col + 1: neck_center + horizontal * neck_half_width - vertical * (0.025 * torso_h),
        }
        deltas = {}
        for col, target in collar_targets.items():
            deltas[col] = target - original_row[col]
            dst_grid[2, col] = target

        # Blend the anchor corrections into the next row to avoid a visible kink.
        for col, delta in deltas.items():
            dst_grid[3, col] += delta * 0.30

        # Smooth the two neighbouring controls without moving the sleeve/shoulder edges.
        for col in (center_col - 2, center_col + 2):
            nearest = center_col - 1 if col < center_col else center_col + 1
            dst_grid[2, col] += (collar_targets[nearest] - original_row[nearest]) * 0.22
            dst_grid[3, col] += (collar_targets[nearest] - original_row[nearest]) * 0.10
        return dst_grid

    @staticmethod
    def _fit_shoulders(dst_grid, pts, body_w=None):
        """Align shoulder seams to shoulder joints and preserve sleeve coverage.

        The collar row is row 2; the actual shoulder seam is row 3. Pinning
        row 2 (the collar row) to the shoulder joints pulls the neck/sleeve
        transition inward. Row 3 is aligned instead, with a small outward
        allowance so the garment seam sits just beyond the shoulder joint.
        """
        if dst_grid.shape[0] < 5 or dst_grid.shape[1] < 7:
            return dst_grid
        if not all(i in pts for i in (11, 12, 23, 24)):
            return dst_grid

        left = np.asarray(pts[11], dtype=np.float32)
        right = np.asarray(pts[12], dtype=np.float32)
        shoulder_axis = right - left
        shoulder_width = float(np.linalg.norm(shoulder_axis))
        if shoulder_width < 1:
            return dst_grid
        horizontal = shoulder_axis / shoulder_width

        # Extend the shoulder seam slightly past the anatomical joint.
        extension = (0.055 * shoulder_width) if body_w is None else (0.055 * body_w)
        left_seam = left - horizontal * extension
        right_seam = right + horizontal * extension

        # Row 3 maps to the source shoulder seam, just below the collar.
        dst_grid[3, 1] = left_seam
        dst_grid[3, 5] = right_seam

        # Carry the shoulder cap outward on the two rows above the seam.
        # These controls shape the fabric over the top of the shoulder; they
        # do not alter the collar centre or the torso/hem width.
        for row, factor in ((1, 0.10), (2, 0.055)):
            dst_grid[row, 1] = dst_grid[row, 1] - horizontal * (factor * shoulder_width)
            dst_grid[row, 5] = dst_grid[row, 5] + horizontal * (factor * shoulder_width)

        # Lift and extend the upper shoulder-cap controls toward the anatomical
        # shoulder points. This fills the dark wedges between the collar slope
        # and shoulder joints without widening the chest panel.
        hip_mid = (np.asarray(pts[23], np.float32) + np.asarray(pts[24], np.float32)) * 0.5
        shoulder_mid = (left + right) * 0.5
        down_axis = hip_mid - shoulder_mid
        down_len = float(np.linalg.norm(down_axis))
        vertical = down_axis / max(down_len, 1.0)
        lift = vertical * (0.055 * body_w)
        cap_extension = horizontal * (0.075 * shoulder_width)
        dst_grid[1, 1] = left_seam - horizontal * (0.19 * shoulder_width) - lift
        dst_grid[1, 5] = right_seam + horizontal * (0.19 * shoulder_width) - lift
        dst_grid[2, 1] = left_seam - horizontal * (0.105 * shoulder_width)
        dst_grid[2, 5] = right_seam + horizontal * (0.105 * shoulder_width)

        # Leave row 4 on the continuous mesh profile. Hard-pinning this row
        # toward the torso created abrupt underarm kinks and pointed blue tabs
        # when the arms rotated. The neighboring row controls already define
        # a smooth transition from sleeve cap to torso.
        return dst_grid

    def _warp_garment(self, garment, pts, geometry, frame_shape, preserve_proportions=False):
        """Warp a catalog tee using shoulder, hip and arm anchors.

        Bundled illustration assets can use a source-specific torso widening
        correction. Uploaded real garments must retain their original
        proportions, sleeve length and width, so that correction is disabled
        for custom uploads.

        The outer columns follow the upper-arm direction while the lower torso
        tapers toward the measured hip width. This remains a 2D approximation.
        """
        shoulder, horizontal, vertical, body_w, torso_h = geometry
        hip = (pts[23] + pts[24]) * 0.5
        hip_w = float(np.linalg.norm(pts[24] - pts[23]))
        hip_ratio = float(np.clip(hip_w / max(body_w, 1.0), 0.62, 1.12))
        # The bundled vector test shirt has a deliberately narrow body and
        # needs its historical widening correction. Real uploaded garments
        # already contain their own cut and proportions; never reshape them
        # with that demo-asset-specific correction.
        if not preserve_proportions:
            garment = self._widen_torso_source(garment)
        gh, gw = garment.shape[:2]
        rows, cols = self.grid_rows, self.grid_cols
        if cols == 7:
            # Extra controls at the sleeve edges and collar boundaries.
            sx = np.array([0.0, 0.22, 0.38, 0.50, 0.62, 0.78, 1.0], np.float32) * (gw - 1)
        else:
            sx = np.linspace(0, gw - 1, cols, dtype=np.float32)
        if rows == 7:
            # Non-uniform rows put control points around the actual collar.
            sy = np.array([0.0, 72.0, 108.0, 145.0, 205.0, 430.0, 759.0], np.float32)
            sy *= (gh - 1) / 759.0
        else:
            sy = np.linspace(0, gh - 1, rows, dtype=np.float32)
        src_grid = np.stack(np.meshgrid(sx, sy), axis=-1)

        # Width values describe sleeve-to-hem silhouette. Scale the waist/hem
        # by measured hip width rather than carrying shoulder width downward.
        # Full garment width must extend beyond shoulder landmarks to cover sleeves.
        # The top rows represent sleeve span; lower rows represent torso width.
        # Broaden only the shoulder/sleeve zone. The previous profile left
        # black fabric exposed over both shoulder caps; keep the waist/hem
        # comparatively unchanged so the torso does not become boxy.
        # Wider sleeve/shoulder coverage and a fuller waist, tapering gently.
        # The hem is shortened separately below so the tee does not hang too low.
        # Broad shoulder caps, a cleaner/narrower chest edge, then enough
        # torso width at the waist to cover the original shirt without a long hem.
        profile = np.array([2.28, 2.44, 2.30, 1.86, 1.58, 1.62, 1.62], np.float32)
        vlist = sy / max(gh - 1, 1)
        if rows != len(profile):
            profile = np.interp(vlist, np.linspace(0, 1, len(profile)), profile)
        top = shoulder - vertical * (0.055 * torso_h)
        # Stop the hem just below the hip anchor rather than extending far
        # down the abdomen; the previous 0.14 torso-height extension looked long.
        bottom = hip + vertical * (0.035 * torso_h)
        dst_grid = np.zeros((rows, cols, 2), np.float32)

        # Keep sleeve geometry tied to the garment silhouette. The previous
        # elbow-driven outer-vertex pull created sharp triangular sleeve tips
        # when elbows moved inward or landmark confidence fluctuated.
        # The pose-driven torso frame still rotates/scales the entire shirt.
        # Keep each destination control aligned with its corresponding
        # non-uniform source column. A uniform u-grid here mismatches the
        # collar/shoulder source controls and visibly pinches the shoulders.
        us = (sx / max(gw - 1, 1)) * 2.0 - 1.0
        for r, v in enumerate(vlist):
            center = top * (1.0 - v) + bottom * v
            hip_blend = max(0.0, min(1.0, (v - 0.48) / 0.52))
            width_scale = float(profile[r]) * (1.0 - hip_blend * (1.0 - hip_ratio) * 0.30)
            width = body_w * width_scale
            if 0.38 <= v <= 0.78:
                width *= 1.0 - 0.012 * np.sin(np.pi * (v - 0.38) / 0.40)
            for c, u in enumerate(us):
                dst_grid[r, c] = center + horizontal * (u * width * 0.5)

        dst_grid = self._fit_neckline(dst_grid, shoulder, horizontal, vertical, body_w, torso_h)
        # Apply shoulder pins last so collar smoothing cannot pull the seams inward.
        dst_grid = self._fit_shoulders(dst_grid, pts, body_w)

        return self._warp_mesh(garment, src_grid, dst_grid, frame_shape)

    @staticmethod
    def _arm_foreground_mask(frame_shape, pts, body_w):
        """Build a narrow arm-only foreground mask, excluding the shoulder/sleeve cap.

        Starting the restoration stroke at the shoulder was repainting the
        original dark T-shirt sleeve over the virtual garment. Start farther
        down the upper arm and use a narrower stroke so the shirt overlay can
        cover the shoulder cap while skin/arms remain in front.
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for sid, eid, wid in ((11, 13, 15), (12, 14, 16)):
            if sid not in pts or eid not in pts:
                continue
            shoulder = np.asarray(pts[sid], dtype=np.float32)
            elbow = np.asarray(pts[eid], dtype=np.float32)
            # Leave the top 42% of the shoulder-to-elbow segment to the tee.
            # This prevents original dark sleeve pixels from being restored
            # over the virtual garment in the upper-arm area.
            arm_start = shoulder + 0.48 * (elbow - shoulder)
            a = tuple(np.round(arm_start).astype(int))
            e = tuple(np.round(elbow).astype(int))
            upper_width = max(5, int(body_w * 0.055))
            cv2.line(mask, a, e, 255, upper_width, cv2.LINE_AA)
            if wid in pts:
                wrist = tuple(np.round(np.asarray(pts[wid]).astype(np.float32)).astype(int))
                cv2.line(mask, e, wrist, 255, max(6, int(body_w * 0.070)), cv2.LINE_AA)
        return cv2.GaussianBlur(mask, (0, 0), max(0.8, body_w * 0.008))

    @staticmethod
    def _restore_arms(original, composed, pts, body_w):
        """Restore only the arm foreground; let the virtual tee cover shoulder sleeves."""
        mask = TryOnEngine._arm_foreground_mask(original.shape, pts, body_w)
        a = (mask.astype(np.float32) / 255.0)[:, :, None]
        return np.uint8(np.clip(original.astype(np.float32)*a + composed.astype(np.float32)*(1-a),0,255))

    def render(self, frame, pts, outfit="levis_black_tee", person_mask=None):
        geometry = torso_geometry(pts)
        if geometry is None or outfit not in self.garments:
            return frame, False
        shoulder, horizontal, vertical, body_w, torso_h = geometry
        original = frame.copy()
        warped = self._warp_garment(
            self.garments[outfit], pts, geometry, frame.shape,
            preserve_proportions=(outfit == "custom_upload")
        )
        ys, xs = np.where(warped[:, :, 3] > 2)
        if len(xs) == 0:
            return frame, False
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        region = frame[y0:y1, x0:x1]
        garment = warped[y0:y1, x0:x1]
        alpha_u8 = garment[:, :, 3]
        # Optional MediaPipe person segmentation clips the garment overlay
        # to the detected person silhouette, preventing fabric pixels from
        # spilling into the background around shoulders/arms.
        if person_mask is not None:
            mask = np.asarray(person_mask, dtype=np.float32)
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = np.clip(mask, 0.0, 1.0)
            # MediaPipe's segmentation edge can sit a couple of pixels inside
            # dark sleeves/shoulders. A very small dilation closes that seam;
            # the garment alpha still limits where any extra pixels are drawn.
            grow = max(1, min(3, int(round(body_w * 0.012))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)
            )
            mask_u8 = np.uint8(mask * 255.0)
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
            mask = mask_u8.astype(np.float32) / 255.0
            person_roi = mask[y0:y1, x0:x1]
            # Keep the soft mask boundary; don't hard-threshold and create
            # jagged cutouts at the silhouette.
            alpha_u8 = np.uint8(alpha_u8.astype(np.float32) * person_roi)
        # Tiny edge feather only: retain the shirt silhouette but remove
        # harsh pixel steps from the piecewise-affine mesh.
        alpha_u8 = cv2.GaussianBlur(alpha_u8, (3, 3), 0.9)
        alpha = alpha_u8[:, :, None].astype(np.float32) / 255.0
        composed = frame.copy()
        composed[y0:y1, x0:x1] = np.uint8(np.clip(
            region.astype(np.float32) * (1.0 - alpha) +
            garment[:, :, :3].astype(np.float32) * alpha, 0, 255
        ))
        composed = self._restore_arms(original, composed, pts, body_w)
        return composed, True
