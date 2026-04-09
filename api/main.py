"""
REST API for dog image matching (found vs lost dogs).

Run from project root:
  CHECKPOINT_PATH=checkpoints/best.pth GALLERY_PATH=gallery_embeddings.pt uvicorn api.main:app --host 0.0.0.0 --port 8000

Optional: set GALLERY_API_URL so POST /admin/rebuild-gallery can refresh the gallery from your Express API
  (e.g. http://localhost:4000/api/lost-dogs/for-gallery) when a new lost dog is created.
  Requires: pip install requests (for rebuild-gallery).

Windows PowerShell:
  $env:CHECKPOINT_PATH="checkpoints/best.pth"; $env:GALLERY_PATH="gallery_embeddings.pt"; uvicorn api.main:app --host 0.0.0.0 --port 8000

Re-ranking (optional env): MATCH_RETRIEVE_K (default 10), MATCH_DEFAULT_TOP_K (default 5),
ENSEMBLE_W_FUSED / ENSEMBLE_W_FRONT / ENSEMBLE_W_LAT (default 0.5/0.25/0.25),
RERANK_MIN_COSINE in [-1,1] (default -1 = off). Rebuild gallery after upgrade to store
frontal_embeddings + lateral_embeddings for dual-view ensemble re-ranking.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Project root (parent of api/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import io
import torch
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import DualViewFusionModel
from src.preprocessing import get_test_transforms
from src.utils.evaluation import cosine_similarity_search, rerank_with_dual_view_ensemble


# --- Config from environment ---
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "best.pth")
GALLERY_PATH = os.environ.get("GALLERY_PATH", "gallery_embeddings.pt")
GALLERY_API_URL = os.environ.get("GALLERY_API_URL")  # Express endpoint for lost dogs (for rebuild-gallery)
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Re-ranking: retrieve a wide candidate set, then re-score with fused + per-view cosine ensemble
MATCH_RETRIEVE_K = int(os.environ.get("MATCH_RETRIEVE_K", "10"))
MATCH_DEFAULT_TOP_K = int(os.environ.get("MATCH_DEFAULT_TOP_K", "5"))
# Ensemble weights (normalized to sum to 1): fused vs frontal-only vs lateral-only gallery vectors
_wf = float(os.environ.get("ENSEMBLE_W_FUSED", "0.5"))
_wfr = float(os.environ.get("ENSEMBLE_W_FRONT", "0.25"))
_wlat = float(os.environ.get("ENSEMBLE_W_LAT", "0.25"))
_s = _wf + _wfr + _wlat
ENSEMBLE_W_FUSED = _wf / _s
ENSEMBLE_W_FRONT = _wfr / _s
ENSEMBLE_W_LAT = _wlat / _s
# Min cosine in [-1, 1] for re-ranked score; -1 disables (see /match min_rerank_cosine)
RERANK_MIN_COSINE = float(os.environ.get("RERANK_MIN_COSINE", "-1"))

# CORS: allow Flutter web and common dev origins (required when Flutter web calls this API from the browser)
_CORS_ORIGINS_ENV = os.environ.get("CORS_ORIGINS", "").strip()
CORS_ORIGINS = [
    "http://localhost:50585",
    "http://localhost:3000",
    "http://127.0.0.1:50585",
    "http://127.0.0.1:3000",
]
if _CORS_ORIGINS_ENV:
    CORS_ORIGINS.extend(o.strip() for o in _CORS_ORIGINS_ENV.split(",") if o.strip())


def load_model_and_gallery():
    """Load model and gallery embeddings once at startup."""
    device = torch.device(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    embedding_dim = checkpoint.get("embedding_dim") or (checkpoint.get("args") or {}).get("embedding_dim", 512)
    
    model = DualViewFusionModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    gallery_embeddings = None
    gallery_frontal = None
    gallery_lateral = None
    gallery_ids = []
    
    if os.path.exists(GALLERY_PATH):
        data = torch.load(GALLERY_PATH, map_location=device)
        gallery_embeddings = data["embeddings"]
        gallery_ids = data["ids"]
        gallery_frontal = data.get("frontal_embeddings")
        gallery_lateral = data.get("lateral_embeddings")
        if gallery_frontal is not None:
            gallery_frontal = gallery_frontal.to(device)
        if gallery_lateral is not None:
            gallery_lateral = gallery_lateral.to(device)
    
    return model, gallery_embeddings, gallery_frontal, gallery_lateral, gallery_ids, device


def _fetch_dogs_from_api(api_url: str) -> List[Dict[str, Any]]:
    """GET api_url; expect JSON list of dogs or { dogs: [...] } / { data: [...] }."""
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "dogs" in data:
        return data["dogs"]
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError("API response must be a list or object with 'dogs' or 'data' array")


def _get_front_and_side_urls(dog: Dict[str, Any]) -> tuple:
    """Get (front_url, side_url) from one dog. Supports frontImageUrl/sideImageUrl or images[].viewType."""
    if "frontImageUrl" in dog and "sideImageUrl" in dog:
        return dog["frontImageUrl"], dog["sideImageUrl"]
    if "frontalImageUrl" in dog and "lateralImageUrl" in dog:
        return dog["frontalImageUrl"], dog["lateralImageUrl"]
    if "images" in dog and isinstance(dog["images"], list):
        front = side = None
        for img in dog["images"]:
            vt = (img.get("viewType") or img.get("view_type") or "").lower()
            url = img.get("url") or img.get("imageUrl")
            if not url:
                continue
            if vt in ("frontal", "front"):
                front = url
            elif vt in ("lateral", "side"):
                side = url
        if front and side:
            return front, side
    raise ValueError(f"Dog record missing front+side URLs: {list(dog.keys())}")


def _download_image(url: str, timeout: int = 15) -> bytes:
    """Download image bytes from URL."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _rebuild_gallery_from_api() -> int:
    """
    Fetch lost dogs from GALLERY_API_URL, compute embeddings with the loaded model,
    save to GALLERY_PATH, and return the new gallery size.
    """
    if not GALLERY_API_URL:
        raise ValueError("GALLERY_API_URL is not set. Set it to your Express for-gallery endpoint.")
    if _model is None:
        raise RuntimeError("Model not loaded.")

    dogs = _fetch_dogs_from_api(GALLERY_API_URL)
    embeddings_list: List[torch.Tensor] = []
    frontal_list: List[torch.Tensor] = []
    lateral_list: List[torch.Tensor] = []
    ids_list: List[str] = []

    for i, dog in enumerate(dogs):
        dog_id = dog.get("dogId") or dog.get("dog_id") or dog.get("id") or str(i)
        try:
            front_url, side_url = _get_front_and_side_urls(dog)
        except ValueError:
            continue
        try:
            front_bytes = _download_image(front_url)
            side_bytes = _download_image(side_url)
        except Exception:
            continue
        try:
            front_t = preprocess_image(front_bytes).to(_device)
            side_t = preprocess_image(side_bytes).to(_device)
        except Exception:
            continue
        with torch.no_grad():
            fused = _model(front_t, side_t).squeeze(0).cpu()
            fe = _model.encode_frontal(front_t).squeeze(0).cpu()
            le = _model.encode_lateral(side_t).squeeze(0).cpu()
        embeddings_list.append(fused)
        frontal_list.append(fe)
        lateral_list.append(le)
        ids_list.append(dog_id)

    if not embeddings_list:
        raise RuntimeError("No valid dogs to embed. Check API response and image URLs.")

    gallery_tensor = torch.stack(embeddings_list)
    frontal_tensor = torch.stack(frontal_list)
    lateral_tensor = torch.stack(lateral_list)
    torch.save(
        {
            "embeddings": gallery_tensor,
            "frontal_embeddings": frontal_tensor,
            "lateral_embeddings": lateral_tensor,
            "ids": ids_list,
            "checkpoint_path": CHECKPOINT_PATH,
        },
        GALLERY_PATH,
    )
    return len(ids_list)


def _reload_gallery_from_disk() -> int:
    """Re-read gallery from GALLERY_PATH and update in-memory state. Returns new gallery size."""
    global _gallery_embeddings, _gallery_frontal, _gallery_lateral, _gallery_ids
    if not os.path.exists(GALLERY_PATH):
        _gallery_embeddings = None
        _gallery_frontal = None
        _gallery_lateral = None
        _gallery_ids = []
        return 0
    data = torch.load(GALLERY_PATH, map_location=_device)
    _gallery_embeddings = data["embeddings"]
    _gallery_ids = data["ids"]
    _gallery_frontal = data.get("frontal_embeddings")
    _gallery_lateral = data.get("lateral_embeddings")
    if _gallery_frontal is not None:
        _gallery_frontal = _gallery_frontal.to(_device)
    if _gallery_lateral is not None:
        _gallery_lateral = _gallery_lateral.to(_device)
    return len(_gallery_ids)


# Global state (loaded at startup)
_model = None
_gallery_embeddings = None
_gallery_frontal = None
_gallery_lateral = None
_gallery_ids = None
_device = None


app = FastAPI(
    title="Dog Image Matching API",
    description="Match found dog images (frontal + lateral) against a gallery of lost dogs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
def startup():
    global _model, _gallery_embeddings, _gallery_frontal, _gallery_lateral, _gallery_ids, _device
    try:
        _model, _gallery_embeddings, _gallery_frontal, _gallery_lateral, _gallery_ids, _device = load_model_and_gallery()
    except Exception as e:
        raise RuntimeError(f"Failed to load model or gallery: {e}") from e

    # If no gallery file but GALLERY_API_URL is set (e.g. on Railway), build gallery from backend at startup
    if (_gallery_embeddings is None or len(_gallery_ids) == 0) and GALLERY_API_URL:
        try:
            _rebuild_gallery_from_api()
            _reload_gallery_from_disk()
        except Exception as e:
            import logging
            logging.getLogger("uvicorn.error").warning(
                "Gallery not loaded; build from API failed (set GALLERY_API_URL to your for-gallery URL): %s", e
            )


class MatchItem(BaseModel):
    dog_id: str
    similarity: float
    percentage: float
    confidence: float


class MatchResponse(BaseModel):
    success: bool
    matches: List[MatchItem]
    message: Optional[str] = None


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Load image from bytes and apply test transforms; return tensor [1, C, H, W]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_test_transforms()
    tensor = transform(img).unsqueeze(0)
    return tensor


@app.get("/health")
def health():
    """Readiness check."""
    return {
        "status": "ok",
        "checkpoint": CHECKPOINT_PATH,
        "gallery_size": len(_gallery_ids) if _gallery_ids else 0,
        "device": str(_device),
    }


@app.post("/match", response_model=MatchResponse)
async def match(
    frontal: UploadFile = File(..., description="Frontal view image"),
    lateral: UploadFile = File(..., description="Lateral view image"),
    top_k: int = Form(MATCH_DEFAULT_TOP_K, description="Final number of matches after re-ranking"),
    retrieve_k: int = Form(
        MATCH_RETRIEVE_K,
        description="Initial retrieval size (e.g. 10); top candidates are re-ranked",
    ),
    min_rerank_cosine: Optional[float] = Form(
        None,
        description="Min re-rank cosine in [-1,1]; omit to use RERANK_MIN_COSINE env (-1 = off)",
    ),
):
    """
    Match a found dog (frontal + lateral images) against the lost-dogs gallery.

    Flow: retrieve top ``retrieve_k`` by fused embedding cosine → re-rank with stricter
    ensemble scores (fused + frontal + lateral cosines when gallery stores per-view
    vectors) → return top ``top_k`` with confidence in [0, 1].
    """
    if _gallery_embeddings is None or len(_gallery_ids) == 0:
        raise HTTPException(
            status_code=503,
            detail="Gallery not loaded. Set GALLERY_PATH to a valid gallery_embeddings.pt file.",
        )
    
    try:
        frontal_bytes = await frontal.read()
        lateral_bytes = await lateral.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploads: {e}") from e
    
    try:
        frontal_tensor = preprocess_image(frontal_bytes).to(_device)
        lateral_tensor = preprocess_image(lateral_bytes).to(_device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e
    
    rk = max(1, min(retrieve_k, len(_gallery_ids)))
    min_cos = RERANK_MIN_COSINE if min_rerank_cosine is None else float(min_rerank_cosine)

    with torch.no_grad():
        query_fused = _model(frontal_tensor, lateral_tensor).squeeze(0)
        query_frontal = _model.encode_frontal(frontal_tensor).squeeze(0)
        query_lateral = _model.encode_lateral(lateral_tensor).squeeze(0)

    similarities, indices = cosine_similarity_search(
        query_fused, _gallery_embeddings, top_k=rk
    )

    reranked = rerank_with_dual_view_ensemble(
        query_fused,
        query_frontal,
        query_lateral,
        _gallery_embeddings,
        indices,
        _gallery_frontal,
        _gallery_lateral,
        ENSEMBLE_W_FUSED,
        ENSEMBLE_W_FRONT,
        ENSEMBLE_W_LAT,
        min_cos,
    )

    out_n = max(1, min(top_k, len(reranked)))
    matches = []
    for gallery_row_idx, score, conf in reranked[:out_n]:
        dog_id = _gallery_ids[gallery_row_idx]
        percentage = ((score + 1) / 2) * 100
        matches.append(
            MatchItem(
                dog_id=str(dog_id),
                similarity=round(float(score), 4),
                percentage=round(float(percentage), 2),
                confidence=round(float(conf), 4),
            )
        )

    msg = None
    if _gallery_frontal is None or _gallery_lateral is None:
        msg = (
            "Gallery has no per-view embeddings; rebuild with POST /admin/rebuild-gallery "
            "to enable frontal+lateral re-ranking ensemble."
        )

    return MatchResponse(success=True, matches=matches, message=msg)


@app.post("/admin/rebuild-gallery")
def admin_rebuild_gallery():
    """
    Rebuild the gallery from the Express API (GALLERY_API_URL).
    Call this after adding a new lost dog so matches include the new dog.
    Requires GALLERY_API_URL to be set (e.g. http://localhost:3000/api/lost-dogs/for-gallery).
    """
    try:
        new_size = _rebuild_gallery_from_api()
        _reload_gallery_from_disk()
        return {
            "success": True,
            "message": "Gallery rebuilt from API and reloaded.",
            "gallery_size": new_size,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch from gallery API: {e}")


@app.post("/admin/reload-gallery")
def admin_reload_gallery():
    """
    Reload the gallery from disk (GALLERY_PATH) without rebuilding.
    Use after running scripts/build_gallery_from_urls.py externally.
    """
    try:
        new_size = _reload_gallery_from_disk()
        return {
            "success": True,
            "message": "Gallery reloaded from disk.",
            "gallery_size": new_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Dog Matching API",
        "docs": "/docs",
        "health": "/health",
        "match": "POST /match (frontal, lateral; top_k, retrieve_k, min_rerank_cosine optional)",
        "admin": "POST /admin/rebuild-gallery (from GALLERY_API_URL), POST /admin/reload-gallery",
    }
