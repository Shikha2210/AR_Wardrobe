# AR Wardrobe — T-shirt-focused build

A local webcam virtual try-on prototype using React/Vite, FastAPI, MediaPipe Pose and OpenCV.

## Current scope
- T-shirts only. Dress-specific loading, fitting and UI have been removed from this build.
- Live webcam pose tracking with MediaPipe and One Euro landmark smoothing.
- T-shirt overlay using shoulder, torso, hip and elbow-aware 2D mesh geometry.
- Local automatic T-shirt extraction from either a clothing-only product photo or a catalogue/model photo.
- No manual outline points or manual masking UI.
- Frontend/backend communication through HTTP and WebSocket.

## Important limitations
This is a 2D pose-driven overlay, not 3D cloth simulation or photorealistic clothing removal. The renderer approximates how a T-shirt sits on the torso. It cannot reconstruct body pixels hidden by the original shirt or infer true fabric physics. The model-photo extraction path is approximate and can include arms or miss fabric hidden behind arms. Best extraction results come from a centred, front-facing T-shirt against a simple contrasting background.

## Project layout
- `backend/app.py` — FastAPI API and WebSocket frame loop
- `backend/tracking.py` — MediaPipe landmark validation and smoothing
- `backend/tryon.py` — T-shirt fitting and rendering
- `backend/garment_ai.py` — local OpenCV garment extraction
- `backend/assets/tshirt.png` — bundled clean T-shirt illustration
- `backend/assets/wardrobe/` — uploaded extracted garment cache
- `frontend/src/main.jsx` — React interface and webcam client
- `frontend/src/style.css` — interface styling

## Run on Windows

Backend terminal:
```powershell
cd .\backend
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Frontend terminal:
```powershell
cd .\frontend
npm install
npm run dev
```

Open the Vite URL printed in the frontend terminal (usually `http://localhost:5173` or `http://localhost:5174`).

## Tests
From `backend`:
```powershell
python -m unittest discover -s tests -v
```

## Performance notes
The T-shirt renderer uses a 5×5 mesh and a single OpenCV remap per frame rather than separate affine image warps for every triangle. Webcam upload is capped at 400 pixels wide and JPEG quality 0.60 to reduce transfer and inference load. Actual FPS depends on the computer, camera and browser.
