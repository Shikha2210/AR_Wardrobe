"""FastAPI service: camera frames in, 2D garment try-on frames out."""
import json, os, time
from pathlib import Path
import cv2, mediapipe as mp, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from tracking import LandmarkTracker
from tryon import TryOnEngine, COLORS

cv2.setNumThreads(1)
app = FastAPI(title="AR Wardrobe API", version="1.26.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = TryOnEngine(grid_cols=7, grid_rows=7)
ASSET_DIR = Path(__file__).resolve().parent / "assets"
WARDROBE_DIR = ASSET_DIR / "wardrobe"
WARDROBE_DIR.mkdir(parents=True, exist_ok=True)
DEMO_ID = "levis_black_tee"
SAFE_DEMO_FILE = ASSET_DIR / "tshirt.png"
CUSTOM_FILE = WARDROBE_DIR / "custom_extracted.png"


def _load_cached_demo():
    """Load only a known isolated local fallback; never auto-load model photos."""
    if DEMO_ID not in engine.garments and SAFE_DEMO_FILE.exists():
        engine.load_garment_image(DEMO_ID, SAFE_DEMO_FILE, remove_light_background=False)


_load_cached_demo()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "AR Wardrobe API",
        "mediapipe_solutions": hasattr(mp, "solutions"),
        "pipeline": "MediaPipe Pose + person segmentation mask + smoothed landmarks + optimized T-shirt mesh + local CV extraction"
    }


@app.get("/api/outfits")
def outfits():
    return [{"id":k,"name":k.title(),"color":f"#{v[2]:02x}{v[1]:02x}{v[0]:02x}"} for k,v in COLORS.items()]


@app.get("/api/wardrobe")
def wardrobe():
    return {
        "demo": {
            "id": DEMO_ID,
            "name": "Clean test T-shirt illustration",
            "loaded": DEMO_ID in engine.garments,
            "preview_url": "/api/wardrobe/demo-preview" if DEMO_ID in engine.garments else None
        },
        "custom": {"loaded": "custom_upload" in engine.garments}
    }


@app.post("/api/wardrobe/demo")
def load_demo_garment():
    """Load the bundled isolated test shirt; no fragile retailer download."""
    try:
        _load_cached_demo()
        if DEMO_ID not in engine.garments:
            raise ValueError("Bundled assets/tshirt.png is missing or unreadable")
        return {"status": "loaded", "id": DEMO_ID, "preview_url": "/api/wardrobe/demo-preview",
                "note": "Using the bundled clean illustration as a safe test garment."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/wardrobe/extract-auto")
async def extract_garment_auto(image: UploadFile = File(...), mode: str = Form("garment")):
    """Extract a T-shirt automatically from a garment-only or model photo using local CV."""
    if mode not in {"garment", "model"}:
        raise HTTPException(status_code=400, detail="Mode must be 'garment' or 'model'.")
    if not image.content_type or image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG or WebP image.")
    raw = await image.read()
    if not raw or len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be non-empty and no larger than 12 MB.")
    try:
        from garment_ai import extract_garment_rgba
        rgba, method_note = extract_garment_rgba(raw, mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI extraction failed: {exc}") from exc
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise HTTPException(status_code=422, detail="AI could not find a usable clothing region. Try a clearer, front-facing photo.")
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 12)
    if len(xs) < 100 or len(ys) < 100:
        raise HTTPException(status_code=422, detail="The clothing region is too small or unclear. Try a closer, well-lit photo.")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    rgba = rgba[y0:y1, x0:x1]
    scale = min(1.0, 900.0 / max(rgba.shape[:2]))
    if scale < 1:
        rgba = cv2.resize(rgba, (int(rgba.shape[1]*scale), int(rgba.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(CUSTOM_FILE), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
        raise HTTPException(status_code=500, detail="Could not save extracted garment.")
    engine.garments["custom_upload"] = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    engine.garment_types["custom_upload"] = "top"
    return {"status": "extracted", "id": "custom_upload", "preview_url": "/api/wardrobe/custom-preview",
            "width": int(rgba.shape[1]), "height": int(rgba.shape[0]), "method": mode, "garment_type": "top",
            "note": method_note + " Inspect the preview; AI masks can need correction on difficult photos."}


@app.get("/api/wardrobe/demo-preview")
def demo_preview():
    return FileResponse(SAFE_DEMO_FILE, media_type="image/png")


@app.get("/api/wardrobe/custom-preview")
def custom_preview():
    if not CUSTOM_FILE.exists():
        raise HTTPException(status_code=404, detail="No extracted garment yet")
    return FileResponse(CUSTOM_FILE, media_type="image/png", filename="custom_extracted.png")


@app.websocket("/ws/tryon")
async def tryon_socket(ws: WebSocket):
    await ws.accept()
    tracker=LandmarkTracker(); outfit=DEMO_ID if DEMO_ID in engine.garments else "blue"
    show_landmarks=False; fps=0.0; last=time.perf_counter()
    connections=mp.solutions.pose.POSE_CONNECTIONS
    try:
        with mp.solutions.pose.Pose(static_image_mode=False,model_complexity=0,smooth_landmarks=False,enable_segmentation=True,min_detection_confidence=.55,min_tracking_confidence=.55) as pose:
            while True:
                message=await ws.receive()
                if message.get("text") is not None:
                    try:
                        cmd=json.loads(message["text"])
                        if cmd.get("type")=="settings":
                            candidate=cmd.get("outfit",outfit)
                            if candidate in engine.garments: outfit=candidate
                            show_landmarks=bool(cmd.get("showLandmarks",show_landmarks))
                    except (ValueError,TypeError): pass
                    continue
                data=message.get("bytes")
                if not data: continue
                frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if frame is None: continue
                frame=cv2.flip(frame,1); h,w=frame.shape[:2]
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
                result=pose.process(rgb)
                landmarks=result.pose_landmarks.landmark if result.pose_landmarks else None
                pts,status=tracker.update(landmarks,w,h)
                if pts is not None:
                    person_mask = result.segmentation_mask if result.segmentation_mask is not None else None
                    frame,_=engine.render(frame,pts,outfit,person_mask=person_mask)
                    if show_landmarks:
                        for a,b in connections:
                            if a in pts and b in pts:
                                cv2.line(frame,tuple(np.int32(pts[a])),tuple(np.int32(pts[b])),(70,220,110),2,cv2.LINE_AA)
                        for idx,point in pts.items():
                            cv2.circle(frame,tuple(np.int32(point)),4 if idx in (11,12,13,14,15,16,23,24) else 2,(40,240,100),-1,cv2.LINE_AA)
                now=time.perf_counter(); inst=1.0/max(now-last,1e-5); fps=inst if not fps else .2*inst+.8*fps; last=now
                cv2.putText(frame,f"{status} | {fps:.0f} FPS",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(80,255,130),2,cv2.LINE_AA)
                ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,68])
                if ok: await ws.send_bytes(encoded.tobytes())
    except WebSocketDisconnect: pass
    except Exception as exc:
        try: await ws.send_text(json.dumps({"type":"error","message":str(exc)}))
        except Exception: pass


# Serve the built frontend (Docker/Hugging Face deployment). Must stay last.
_DIST = Path(__file__).resolve().parent / "static"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
