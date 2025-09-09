
# -------- Root endpoint --------

# Place this after app initialization

# ...existing code...


# server.py
import os
import io
import json
import base64
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage import filters, transform as sk_transform

BASE_DIR = os.path.dirname(__file__)

# -------- Config --------
EMBEDDING_DIM = 128
IMAGE_SIZE = 224
MODEL_WEIGHTS = os.path.join(BASE_DIR, "signature_embedding.pth")  # saved embedding model weights (embedding network)
EMBED_DB = os.path.join(BASE_DIR, "embeddings_db.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "registered_images")
VERIFY_THRESHOLD = 0.75  # cosine similarity threshold (0..1). Tune with validation (EER)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(SAMPLES_DIR, exist_ok=True)

# -------- Lifespan/startup --------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.MODEL = load_model(MODEL_WEIGHTS, device=DEVICE)
    app.state.EMB_DB_DATA = load_embeddings_db()
    print("Server ready. Device:", DEVICE)
    yield
    # optional cleanup here

app = FastAPI(title="Signature Verification Backend", lifespan=lifespan)

if os.path.isdir(FRONTEND_DIR):
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    except Exception as e:
        print("Could not mount frontend:", e)


# -------- Utilities --------
def save_embeddings_db(db: Dict[str, Any]):
    with open(EMBED_DB, "w") as f:
        json.dump(db, f, indent=2)


def load_embeddings_db() -> Dict[str, Any]:
    if not os.path.exists(EMBED_DB):
        return {}
    with open(EMBED_DB, "r") as f:
        return json.load(f)


def ensure_user_db_entry(db: Dict[str, Any], user_id: str):
    if user_id not in db:
        db[user_id] = {"embeddings": [], "mean_embedding": None, "samples": []}


# ---------- Model (ResNet50 -> EMBEDDING_DIM-d embedding) ----------
class EmbeddingNet(nn.Module):
    def __init__(self, out_dim=EMBEDDING_DIM):
        super().__init__()
        from torchvision.models import ResNet50_Weights
        base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, out_dim)
        self.base = base

    def forward(self, x):
        return self.base(x)


def load_model(weights_path: Optional[str] = None, device=DEVICE):
    model = EmbeddingNet().to(device)
    model.eval()
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        state = torch.load(weights_path, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception:
            # try partial load
            model_state = model.state_dict()
            for k, v in state.items():
                if k in model_state and v.shape == model_state[k].shape:
                    model_state[k] = v
            model.load_state_dict(model_state)
        print("Weights loaded.")
    else:
        print("No weights found; model running with randomly initialized head (for testing only).")
    return model


# -------- Preprocessing pipeline ----------
def render_strokes_to_image(strokes: List[List[Dict[str, float]]], size=(IMAGE_SIZE, IMAGE_SIZE), bg=255):
    if not strokes:
        return None
    pts = []
    for stroke in strokes:
        for p in stroke:
            try:
                pts.append((p["x"], p["y"]))
            except KeyError as e:
                print(f"Malformed stroke point: {p}, missing key: {e}")
                continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(xs) == 0:
        return None
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = maxx - minx if maxx > minx else 1.0
    h = maxy - miny if maxy > miny else 1.0
    scale = 0.9 * min(size[0] / w, size[1] / h)
    img = Image.new("L", size, color=bg)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        norm_pts = []
        for p in stroke:
            try:
                norm_pts.append(((p["x"] - minx) * scale + size[0] * 0.05, (p["y"] - miny) * scale + size[1] * 0.05))
            except KeyError as e:
                print(f"Malformed stroke point in drawing: {p}, missing key: {e}")
                continue
        if len(norm_pts) >= 2:
            draw.line(norm_pts, fill=0, width=3)
        elif len(norm_pts) == 1:
            x, y = norm_pts[0]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=0)
    return img


def decode_base64_image(data_url: str) -> Image.Image:
    if "," in data_url:
        header, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("L")
    return img


def preprocess_image(img: Image.Image, size=IMAGE_SIZE) -> Image.Image:
    """Normalize, crop around dark strokes, keep white background.

    Previous version inverted foreground/background and then cropped using the
    background mask, producing mostly black images. This keeps original polarity.
    """
    arr = np.array(img.convert("L"))
    if arr.dtype != np.uint8:
        arr = (255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)).astype(np.uint8)

    # Otsu threshold – foreground are darker (< thresh)
    try:
        thresh = filters.threshold_otsu(arr)
    except Exception:
        thresh = 128
    fg_mask = arr < thresh  # True where strokes (dark)
    coords = np.argwhere(fg_mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = arr[y0:y1 + 1, x0:x1 + 1]
    else:
        cropped = arr

    h, w = cropped.shape
    max_side = max(h, w)
    pad_h = (max_side - h) // 2
    pad_w = (max_side - w) // 2
    # Create white square canvas then paste cropped (top-left offset)
    canvas = np.full((max_side, max_side), 255, dtype=np.uint8)
    canvas[pad_h:pad_h + h, pad_w:pad_w + w] = cropped

    resized = sk_transform.resize(canvas, (size, size), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return Image.fromarray(resized, mode="L")


to_tensor_and_normalize = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def image_to_embedding(model: nn.Module, pil_img: Image.Image, device=DEVICE) -> np.ndarray:
    tensor = to_tensor_and_normalize(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = emb.cpu().numpy().reshape(-1)
    norm = np.linalg.norm(emb) + 1e-10
    emb = emb / norm
    return emb.astype(float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# -------- FastAPI / Request models --------
class SampleModel(BaseModel):
    type: str  # 'upload' or 'pad'
    image: Optional[str] = None  # base64 dataurl
    strokes: Optional[List[List[Dict[str, float]]]] = None
    created: Optional[str] = None


class RegisterPayload(BaseModel):
    user_id: str
    samples: List[SampleModel]


# Allow your local frontend to call (adjust origins if serving frontend differently)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5500", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Endpoints --------
@app.post("/process_signature_registration", summary="Process Signature Registration", response_description="Registration result", response_model=Dict[str, Any])
async def process_signature_registration(payload: RegisterPayload):
    user_id = payload.user_id
    samples = payload.samples
    if not samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    db = app.state.EMB_DB_DATA
    ensure_user_db_entry(db, user_id)
    MODEL = app.state.MODEL

    saved = []
    for i, s in enumerate(samples):
        try:
            if s.strokes:
                pil = render_strokes_to_image(s.strokes, size=(IMAGE_SIZE, IMAGE_SIZE))
            elif s.image:
                pil = decode_base64_image(s.image)
            else:
                continue
            pil = preprocess_image(pil, size=IMAGE_SIZE)
            emb = image_to_embedding(MODEL, pil, device=DEVICE).tolist()

            timestamp = s.created or time.strftime("%Y-%m-%d %H:%M:%S")
            sample_name = f"{user_id}_{int(time.time())}_{i}.png"
            sample_path = os.path.join(SAMPLES_DIR, sample_name)
            pil.convert("L").save(sample_path)

            db[user_id]["embeddings"].append(emb)
            db[user_id]["samples"].append({"path": sample_path, "time": timestamp, "type": s.type})
            saved.append(sample_path)
        except Exception as e:
            print("Failed to process sample:", e)
            continue

    if db[user_id]["embeddings"]:
        arr = np.array(db[user_id]["embeddings"], dtype=float)
        mean_emb = np.mean(arr, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
        db[user_id]["mean_embedding"] = mean_emb.tolist()
    save_embeddings_db(db)
    app.state.EMB_DB_DATA = db
    return {"success": True, "saved_samples": saved, "count": len(db[user_id]["embeddings"])}

@app.post("/verify_signature", summary="Verify Signature", response_description="Verification result", response_model=Dict[str, Any])
async def verify_signature(req: Request):
    """
    Verify a signature for a user. Accepts either base64 image or strokes.
    Example payloads:
    {
        "user_id": "demo_user",
        "image": "data:image/png;base64,..."
    }
    or
    {
        "user_id": "demo_user",
        "strokes": [
            [
                {"x": 10, "y": 20, "t": 1234567890},
                {"x": 15, "y": 25, "t": 1234567891}
            ]
        ]
    }
    """
    body = await req.json()
    user_id = body.get("user_id")
    image_b64 = body.get("image")
    strokes = body.get("strokes")

    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    db = app.state.EMB_DB_DATA
    if user_id not in db or (not db[user_id].get("mean_embedding") and not db[user_id].get("embeddings")):
        return {"success": False, "error": "No registration found for user"}

    if strokes:
        pil = render_strokes_to_image(strokes, size=(IMAGE_SIZE, IMAGE_SIZE))
    elif image_b64:
        pil = decode_base64_image(image_b64)
    else:
        raise HTTPException(status_code=400, detail="Missing image or strokes")

    pil = preprocess_image(pil, size=IMAGE_SIZE)
    emb = image_to_embedding(app.state.MODEL, pil, device=DEVICE)

    user_mean = np.array(db[user_id].get("mean_embedding")) if db[user_id].get("mean_embedding") else None
    user_embs = np.array(db[user_id].get("embeddings")) if db[user_id].get("embeddings") else None

    best_score = -1.0
    scores = []
    if user_mean is not None:
        score = cosine_similarity(emb, user_mean)
        scores.append(score)
        best_score = max(best_score, score)
    if user_embs is not None and user_embs.shape[0] > 0:
        for ue in user_embs:
            s = cosine_similarity(emb, np.array(ue))
            scores.append(s)
            if s > best_score:
                best_score = s

    match = bool(best_score >= VERIFY_THRESHOLD)
    return {"success": True, "match": match, "score": float(best_score), "all_scores": scores}


@app.get("/user/{user_id}/signatures")
def list_user_signatures(user_id: str):
    db = app.state.EMB_DB_DATA
    if user_id not in db:
        raise HTTPException(status_code=404, detail="user not found")
    return {"user_id": user_id, "count": len(db[user_id].get("embeddings", [])), "samples": db[user_id].get("samples", []), "mean_embedding": db[user_id].get("mean_embedding")}


@app.post("/retrain")
async def retrain_endpoint():
    return {"accepted": True, "message": "retrain endpoint is a stub — wire training here when ready."}


@app.get("/")
def root():
    # Redirect to frontend UI if it exists, else show JSON message
    if os.path.isdir(FRONTEND_DIR):
        return RedirectResponse(url="/frontend")
    return {"message": "Signature Verification Backend running. Frontend not found.", "docs": "/docs"}


# -------- Static image serving (registered samples) --------
@app.get("/registered_image/{filename}", summary="Get a registered signature image")
def get_registered_image(filename: str):
    """Serve a stored registered signature image by filename."""
    path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png")


# -------- Maintenance / Admin Endpoints --------
@app.delete("/user/{user_id}/signatures", summary="Delete all signatures for a user")
def delete_user_signatures(user_id: str):
    db = app.state.EMB_DB_DATA
    if user_id not in db:
        raise HTTPException(status_code=404, detail="user not found")
    # Delete image files for that user
    removed_files = []
    for sample in db[user_id].get("samples", []):
        p = sample.get("path")
        if p and os.path.isfile(p):
            try:
                os.remove(p)
                removed_files.append(p)
            except Exception as e:
                print("Failed removing", p, e)
    # Reset user entry
    db[user_id] = {"embeddings": [], "mean_embedding": None, "samples": []}
    save_embeddings_db(db)
    app.state.EMB_DB_DATA = db
    return {"success": True, "user_id": user_id, "removed_files": removed_files, "message": "User signatures cleared."}


@app.delete("/admin/clear_all_signatures", summary="Delete ALL users' signatures and embeddings")
def clear_all_signatures(confirm: str = "no"):
    if confirm.lower() != "yes":
        raise HTTPException(status_code=400, detail="Set confirm=yes to actually perform the purge.")
    db = app.state.EMB_DB_DATA
    # Remove all stored sample images
    removed = []
    for user_id, info in db.items():
        for sample in info.get("samples", []):
            p = sample.get("path")
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                    removed.append(p)
                except Exception as e:
                    print("Failed removing", p, e)
        # reset user
        db[user_id] = {"embeddings": [], "mean_embedding": None, "samples": []}
    # Optionally also clear any orphaned files left in directory
    try:
        for fname in os.listdir(SAMPLES_DIR):
            fp = os.path.join(SAMPLES_DIR, fname)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    save_embeddings_db(db)
    app.state.EMB_DB_DATA = db
    return {"success": True, "removed_files": removed, "message": "All signatures purged."}