import unittest, numpy as np, cv2
from pathlib import Path
from tracking import OneEuro, LandmarkTracker, torso_geometry
from tryon import TryOnEngine
class P:
    def __init__(self,x,y,v=1): self.x=x; self.y=y; self.visibility=v
class Tests(unittest.TestCase):
    def setUp(self):
        self.lm=[P(0,0,0) for _ in range(33)]
        for i,x,y in [(11,.35,.25),(12,.65,.25),(23,.4,.72),(24,.6,.72)]: self.lm[i]=P(x,y)
    def test_filter(self):
        f=OneEuro(); np.testing.assert_allclose(f.update([10,10],1),[10,10]); self.assertGreater(f.update([100,10],1.03)[0],10)
    def test_tracker(self):
        t=LandmarkTracker(); pts,status=t.update(self.lm,640,480,1); self.assertEqual(status,'TRACKING'); self.assertIsNotNone(torso_geometry(pts)); self.assertEqual(t.update(None,640,480,1.1)[1],'HOLDING TRACK')
    def test_render_with_asymmetric_hip_width(self):
        t=LandmarkTracker(); pts,_=t.update(self.lm,640,480,1)
        pts[23]=np.array([270.0,345.0]); pts[24]=np.array([390.0,350.0])
        engine=TryOnEngine(grid_cols=7,grid_rows=7)
        frame=np.zeros((480,640,3),np.uint8)
        out,ok=engine.render(frame,pts,'blue')
        self.assertTrue(ok)
        self.assertEqual(out.shape,frame.shape)
        self.assertGreater(np.count_nonzero(out),0)


    def test_tshirt_mesh_uses_lightweight_grid(self):
        engine = TryOnEngine()
        self.assertEqual(engine.grid_cols, 7)
        self.assertEqual(engine.grid_rows, 7)

    def test_neckline_fit_raises_collar_toward_neck(self):
        grid = np.zeros((7, 7, 2), dtype=np.float32)
        vertical = np.array([0.0, 1.0], dtype=np.float32)
        shoulder = np.array([100.0, 100.0], dtype=np.float32)
        horizontal = np.array([1.0, 0.0], dtype=np.float32)
        fitted = TryOnEngine._fit_neckline(
            grid.copy(), shoulder, horizontal, vertical, 120.0, 200.0
        )
        # Collar center is slightly below the estimated neck base, while
        # the left/right collar anchors sit higher and span the neck.
        self.assertAlmostEqual(float(fitted[2, 3, 1]), 100.0 - 21.0 + 7.0, places=3)
        self.assertAlmostEqual(float(fitted[2, 2, 0]), 100.0 - 22.8, places=3)
        self.assertAlmostEqual(float(fitted[2, 4, 0]), 100.0 + 22.8, places=3)
        self.assertEqual(float(fitted[2, 0, 1]), 0.0)

    def test_tshirt_asset_has_transparent_neck_opening(self):
        asset_path = Path(__file__).resolve().parents[1] / "assets" / "tshirt.png"
        garment = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(garment)
        self.assertEqual(garment.shape[2], 4)
        self.assertEqual(int(garment[108, 300, 3]), 0)
        self.assertGreater(int(garment[155, 300, 3]), 200)

    def test_shoulder_mesh_controls_match_pose_landmarks(self):
        grid = np.zeros((7, 7, 2), dtype=np.float32)
        pts = {
            11: np.array([140.0, 120.0], dtype=np.float32),
            12: np.array([360.0, 124.0], dtype=np.float32),
            23: np.array([170.0, 330.0], dtype=np.float32),
            24: np.array([330.0, 334.0], dtype=np.float32),
        }
        fitted = TryOnEngine._fit_shoulders(grid.copy(), pts, body_w=220.0)
        # Shoulder seam is on row 3, slightly outside the anatomical joints.
        self.assertLess(float(fitted[3, 1, 0]), float(pts[11][0]))
        self.assertGreater(float(fitted[3, 5, 0]), float(pts[12][0]))
        # Row 4 is intentionally left to the continuous mesh profile.
        # Avoiding a hard positional pin prevents abrupt underarm kinks.
        np.testing.assert_array_equal(fitted[4], grid[4])
        # Collar centre remains untouched; shoulder-cap controls extend outward.
        np.testing.assert_allclose(fitted[2, 3], grid[2, 3], atol=1e-5)
        self.assertLess(float(fitted[1, 1, 0]), float(pts[11][0]))
        self.assertGreater(float(fitted[1, 5, 0]), float(pts[12][0]))
        self.assertLess(float(fitted[1, 1, 1]), float(fitted[3, 1, 1]))
        self.assertLess(float(fitted[1, 5, 1]), float(fitted[3, 5, 1]))

    def test_tshirt_underarm_notches_are_filled(self):
        asset_path = Path(__file__).resolve().parents[1] / "assets" / "tshirt.png"
        garment = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(garment)
        # Pixels inside the former transparent underarm wedges are now fabric.
        self.assertGreater(int(garment[290, 170, 3]), 200)
        self.assertGreater(int(garment[290, 430, 3]), 200)

    def test_torso_source_expansion_keeps_dimensions(self):
        source = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets" / "tshirt.png"), cv2.IMREAD_UNCHANGED)
        expanded = TryOnEngine._widen_torso_source(source)
        self.assertEqual(expanded.shape, source.shape)
        self.assertGreater(int(np.count_nonzero(expanded[:, :, 3])), 0)

    def test_arm_restore_mask_starts_below_shoulder(self):
        pts = {
            11: np.array([160.0, 120.0], np.float32),
            13: np.array([80.0, 240.0], np.float32),
            15: np.array([20.0, 300.0], np.float32),
            12: np.array([360.0, 120.0], np.float32),
            14: np.array([440.0, 240.0], np.float32),
            16: np.array([500.0, 300.0], np.float32),
        }
        mask = TryOnEngine._arm_foreground_mask((400, 560, 3), pts, 200.0)
        # Shoulder cap pixels should no longer be restored from the original
        # black shirt, while the mid-upper-arm remains protected in front.
        self.assertEqual(int(mask[120, 160]), 0)
        self.assertEqual(int(mask[145, 143]), 0)
        self.assertGreater(int(mask[205, 103]), 0)

    def test_person_mask_clips_garment_overlay(self):
        tracker = LandmarkTracker()
        pts, _ = tracker.update(self.lm, 640, 480, 1)
        engine = TryOnEngine()
        frame = np.zeros((480, 640, 3), np.uint8)
        full = np.ones((480, 640), np.float32)
        clipped = np.zeros((480, 640), np.float32)
        clipped[:, :320] = 1.0
        out_full, ok_full = engine.render(frame, pts, "blue", person_mask=full)
        out_clip, ok_clip = engine.render(frame, pts, "blue", person_mask=clipped)
        self.assertTrue(ok_full and ok_clip)
        self.assertGreater(np.count_nonzero(out_full), np.count_nonzero(out_clip))
        self.assertEqual(out_clip.shape, frame.shape)

    def test_render(self):
        t=LandmarkTracker(); pts,_=t.update(self.lm,640,480,1); engine=TryOnEngine(); frame=np.zeros((480,640,3),np.uint8); out,ok=engine.render(frame,pts,'blue'); self.assertTrue(ok); self.assertEqual(out.shape,frame.shape); self.assertGreater(np.count_nonzero(out),0)
    def test_phase24_dark_pixel_patch_not_used(self):
        self.assertFalse(hasattr(TryOnEngine, "_cover_dark_upper_sleeve_gaps"))


    def test_real_upload_path_preserves_garment_proportions(self):
        # Real uploads are rendered with preserve_proportions=True, so the
        # bundled illustration's source widening must not be applied.
        sample = np.zeros((100, 60, 4), dtype=np.uint8)
        sample[10:95, 20:40, :3] = (40, 120, 205)
        sample[10:95, 20:40, 3] = 255
        unchanged = TryOnEngine._widen_torso_source(sample)
        self.assertEqual(unchanged.shape, sample.shape)
        self.assertFalse(np.array_equal(unchanged, sample))



if __name__=='__main__': unittest.main()

