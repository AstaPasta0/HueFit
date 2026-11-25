import io
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.color_math import describe_oklab_color, hex_to_oklab
from src.config import (
    ARTIFACTS_DIR,
    FASHIONSIGLIP_MODEL_ID,
    ROLES,
    STATIC_DIR,
    get_amp_dtype,
    get_device,
)
from src.recommender import HueFitRecommender
from src.vision import GarmentVisionEngine, prepare_image, sample_patch_color

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_MIME_PREFIXES = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
)
MAX_SESSION_CACHE = 16
MAX_SAVED_LOOKS = 50
SAVED_LOOKS_PATH = ARTIFACTS_DIR / "saved_looks.json"

ROLE_DISPLAY_LABELS = {
    "top": "Upper Garment (Top)",
    "bottom": "Lower Garment (Bottom)",
    "outerwear": "Outerwear (Jacket / Coat)",
    "footwear": "Footwear (Shoes)",
}

ROLE_ZERO_SHOT_PROMPTS: Dict[str, List[str]] = {
    "top": [
        "a photo of a t-shirt, shirt, blouse, knit sweater, sweatshirt, or tank top",
        "an upper body indoor shirt or pullover top",
    ],
    "outerwear": [
        "a photo of a jacket, coat, blazer, trench coat, parka, leather jacket, or cardigan outerwear",
        "an outer layer coat, tailored blazer, denim jacket, or winter jacket",
    ],
    "bottom": [
        "a photo of pants, trousers, jeans, chinos, leggings, shorts, or a skirt",
        "lower body trousers, denim jeans, or skirt",
    ],
    "footwear": [
        "a photo of shoes, sneakers, boots, loafers, heels, pumps, or sandals",
        "a pair of footwear shoes or boots",
    ],
}


class LockedSlotSpec(BaseModel):
    hex: str
    secondary_hex: Optional[str] = None
    primary_share: float = Field(default=1.0, ge=0.5, le=1.0)
    region_index: Optional[int] = None


class RecommendRequest(BaseModel):
    locks: Dict[str, LockedSlotSpec]
    style: str = "All"
    outerwear_mode: str = "any"
    upload_id: Optional[str] = None
    shuffle_seed: int = 0
    top_k: int = Field(default=8, ge=1, le=12)


class SamplePixelRequest(BaseModel):
    upload_id: str
    x_pct: float = Field(..., ge=0.0, le=100.0)
    y_pct: float = Field(..., ge=0.0, le=100.0)
    target_role: Optional[str] = None


class SaveLookRequest(BaseModel):
    set_id: str
    style: str = "All"
    has_outerwear: bool = False
    colors: Dict[str, Optional[str]]
    secondary_colors: Dict[str, Optional[str]] = Field(default_factory=dict)
    secondary_color_names: Dict[str, Optional[str]] = Field(default_factory=dict)
    primary_shares: Dict[str, Optional[float]] = Field(default_factory=dict)
    color_names: Dict[str, Optional[str]] = Field(default_factory=dict)
    titles: Dict[str, Optional[str]] = Field(default_factory=dict)
    locked_roles: List[str] = Field(default_factory=list)
    compat_score: float = 0.0
    delta_e2000: float = 0.0
    harmony_label: str = "Balanced Palette"


def _look_palette_key(item: Dict[str, object]) -> Tuple[object, ...]:
    c = item.get("colors") or {}
    return (
        str(item.get("set_id") or ""),
        c.get("top"),
        c.get("outerwear"),
        c.get("bottom"),
        c.get("footwear"),
    )


class SavedLooksStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.items: List[Dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.items = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self.items = data[:MAX_SAVED_LOOKS]
        except Exception:
            self.items = []

    def _write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.items, fh, indent=2)
        os.replace(tmp, self.path)

    def list_all(self) -> List[Dict[str, object]]:
        with self.lock:
            return list(self.items)

    def add(self, payload: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
        with self.lock:
            key = _look_palette_key(payload)
            for existing in self.items:
                if _look_palette_key(existing) == key:
                    return existing, False
            entry = dict(payload)
            entry["saved_id"] = uuid.uuid4().hex[:12]
            entry["saved_at"] = time.strftime("%Y-%m-%d %H:%M")
            self.items.insert(0, entry)
            if len(self.items) > MAX_SAVED_LOOKS:
                self.items = self.items[:MAX_SAVED_LOOKS]
            self._write_atomic()
            return entry, True

    def delete(self, saved_id: str) -> bool:
        with self.lock:
            before = len(self.items)
            self.items = [
                it for it in self.items if str(it.get("saved_id")) != saved_id
            ]
            if len(self.items) < before:
                self._write_atomic()
                return True
            return False


class EngineState:
    def __init__(self) -> None:
        self.device = get_device()
        self.amp_dtype = get_amp_dtype(self.device)
        self.gpu_lock = threading.Lock()
        self.vision: Optional[GarmentVisionEngine] = None
        self.clip_model = None
        self.clip_preprocess = None
        self.role_text_embs: Optional[np.ndarray] = None
        self.recommender: Optional[HueFitRecommender] = None
        self.saved_store = SavedLooksStore(SAVED_LOOKS_PATH)
        self.sessions: "OrderedDict[str, Dict[str, object]]" = OrderedDict()

    @torch.inference_mode()
    def _build_role_text_embeddings(self) -> Optional[np.ndarray]:
        try:
            tokenizer = open_clip.get_tokenizer(FASHIONSIGLIP_MODEL_ID)
            role_vecs: List[np.ndarray] = []
            for role in ROLES:
                prompts = ROLE_ZERO_SHOT_PROMPTS[role]
                tokens = tokenizer(prompts).to(self.device)
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        t_feat = self.clip_model.encode_text(tokens)
                else:
                    t_feat = self.clip_model.encode_text(tokens)
                t_feat = F.normalize(t_feat.float(), dim=-1)
                mean_vec = F.normalize(t_feat.mean(dim=0, keepdim=True), dim=-1)
                role_vecs.append(mean_vec[0].cpu().numpy().astype(np.float32))
            return np.stack(role_vecs, axis=0)
        except Exception as exc:
            print(f"[WARN] Zero-shot role classifier unavailable ({exc}).")
            return None

    def classify_crop_role(
        self,
        crop_emb: np.ndarray,
        allowed_roles: Tuple[str, ...] = ROLES,
    ) -> Tuple[str, float]:
        if self.role_text_embs is None:
            return allowed_roles[0], 0.0
        idxs = [ROLES.index(r) for r in allowed_roles]
        sims = (self.role_text_embs[idxs] @ crop_emb.astype(np.float32)) * 12.0
        sims = sims - float(np.max(sims))
        probs = np.exp(sims)
        probs = probs / float(np.sum(probs))
        best_i = int(np.argmax(probs))
        return allowed_roles[best_i], round(float(probs[best_i]), 3)

    def load_all(self) -> None:
        use_biref = (self.device.type == "cuda") or (
            os.environ.get("HUEFIT_CPU_BIREFNET") == "1"
        )
        self.vision = GarmentVisionEngine(
            device=self.device, load_segformer=True, load_birefnet=use_biref
        )
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            FASHIONSIGLIP_MODEL_ID
        )
        self.clip_model = clip_model.to(self.device).eval()
        self.clip_preprocess = clip_preprocess
        self.role_text_embs = self._build_role_text_embeddings()
        self.recommender = HueFitRecommender(device=self.device)

        dummy_img = Image.new("RGB", (256, 256), (245, 245, 245))
        ImageDraw.Draw(dummy_img).rectangle([64, 36, 192, 220], fill=(29, 53, 87))
        _ = self.vision.analyze_photo(dummy_img, use_birefnet=True, return_crops=False)
        dummy_emb = self.encode_crops([dummy_img])[0]
        _ = self.recommender.recommend(
            locks={"top": {"hex": "#1D3557"}},
            style="All",
            outerwear_mode="any",
            top_k=4,
        )
        _ = self.recommender.recommend(
            locks={"top": {"hex": "#1D3557", "visual_embedding": dummy_emb}},
            style="All",
            outerwear_mode="any",
            top_k=4,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.inference_mode()
    def encode_crops(self, crops: List[Image.Image]) -> np.ndarray:
        tensors = torch.stack(
            [self.clip_preprocess(c.convert("RGB")) for c in crops], dim=0
        ).to(self.device)
        if self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                feats = self.clip_model.encode_image(tensors)
        else:
            feats = self.clip_model.encode_image(tensors)
        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def put_session(self, upload_id: str, payload: Dict[str, object]) -> None:
        self.sessions[upload_id] = payload
        self.sessions.move_to_end(upload_id)
        while len(self.sessions) > MAX_SESSION_CACHE:
            self.sessions.popitem(last=False)

    def get_session(self, upload_id: str) -> Optional[Dict[str, object]]:
        sess = self.sessions.get(upload_id)
        if sess is not None:
            self.sessions.move_to_end(upload_id)
        return sess


STATE = EngineState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE.load_all()
    yield
    STATE.sessions.clear()


app = FastAPI(title="HueFit — Outfit Color Recommender", lifespan=lifespan)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


if STATIC_DIR.exists():
    app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")


def _validate_hex(hx: str) -> str:
    clean = hx.strip().lstrip("#").upper()
    if len(clean) != 6 or not all(c in "0123456789ABCDEF" for c in clean):
        raise HTTPException(status_code=400, detail=f"Invalid hex color: '{hx}'")
    return f"#{clean}"


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            media_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse(
        "<h2>HueFit API is live</h2><p>Check <a href='/api/health'>/api/health</a>.</p>"
    )


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    ico_path = STATIC_DIR / "huefit.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path), media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="Favicon not generated yet.")


@app.get("/api/health")
def health():
    if STATE.recommender is None or STATE.vision is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    vram_mb = (
        round(torch.cuda.memory_allocated() / (1024**2), 1)
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "status": "ok",
        "device": str(STATE.device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "vram_allocated_mb": vram_mb,
        "catalog_outfits": len(STATE.recommender.catalog),
        "saved_looks_count": len(STATE.saved_store.list_all()),
        "birefnet_loaded": STATE.vision.birefnet is not None,
        "role_classifier_loaded": STATE.role_text_embs is not None,
        "metrics": STATE.recommender.eval_summary,
    }


@app.get("/api/saved-looks")
def list_saved_looks():
    return {"looks": STATE.saved_store.list_all()}


@app.post("/api/saved-looks")
def save_look(req: SaveLookRequest):
    entry, created = STATE.saved_store.add(req.model_dump())
    return {
        "saved": entry,
        "created": created,
        "total": len(STATE.saved_store.list_all()),
    }


@app.delete("/api/saved-looks/{saved_id}")
def delete_saved_look(saved_id: str):
    deleted = STATE.saved_store.delete(saved_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Saved look not found.")
    return {"deleted": True, "total": len(STATE.saved_store.list_all())}


@app.post("/api/analyze-image")
def analyze_image(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    if STATE.vision is None or STATE.recommender is None:
        raise HTTPException(status_code=503, detail="Vision engine not initialized.")

    content_type = (file.content_type or "").lower()
    if content_type and not any(
        content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Upload JPG, PNG, or WEBP.",
        )

    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB size limit.")

    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        ) from exc

    rgb_img, _ = prepare_image(pil_img, max_side=1024)

    with STATE.gpu_lock:
        res = STATE.vision.analyze_photo(
            pil_img, use_birefnet=True, return_crops=True, return_debug=False
        )
        regions = res["regions"]
        crops = [r.pop("crop_image", rgb_img) for r in regions]
        crop_embs = STATE.encode_crops(crops)

        for i, reg in enumerate(regions):
            reg["region_index"] = i
            if res["mode"] == "flatlay":
                pred_role, conf = STATE.classify_crop_role(
                    crop_embs[i], allowed_roles=ROLES
                )
                if conf >= 0.42:
                    reg["role"] = pred_role
                    reg["label"] = ROLE_DISPLAY_LABELS[pred_role]
                reg["role_confidence"] = conf
            elif (
                res["mode"] == "outfit"
                and reg["role"] == "top"
                and "Dress" not in str(reg.get("label", ""))
            ):
                pred_role, conf = STATE.classify_crop_role(
                    crop_embs[i], allowed_roles=("top", "outerwear")
                )
                if pred_role == "outerwear" and conf >= 0.62:
                    reg["role"] = "outerwear"
                    reg["label"] = ROLE_DISPLAY_LABELS["outerwear"]
                reg["role_confidence"] = conf
            else:
                reg["role_confidence"] = 0.90

        audit = None
        if len(regions) >= 2:
            audit = STATE.recommender.audit_outfit(regions, crop_embs)

    upload_id = uuid.uuid4().hex[:16]
    with STATE.gpu_lock:
        STATE.put_session(
            upload_id,
            {
                "rgb_image": rgb_img,
                "regions": regions,
                "crop_embs": crop_embs,
            },
        )

    res["upload_id"] = upload_id
    res["outfit_audit"] = audit
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return res


@app.post("/api/sample-pixel")
def sample_pixel(req: SamplePixelRequest):
    with STATE.gpu_lock:
        sess = STATE.get_session(req.upload_id)
    if sess is None:
        raise HTTPException(
            status_code=404,
            detail="Upload session expired. Please re-upload the image.",
        )

    rgb_img: Image.Image = sess["rgb_image"]
    regions: List[Dict[str, object]] = sess["regions"]

    patch = sample_patch_color(rgb_img, req.x_pct, req.y_pct, radius=2)
    xn, yn = req.x_pct / 100.0, req.y_pct / 100.0

    matched_idx = 0
    best_dist = 1e9
    for idx, reg in enumerate(regions):
        x1, y1, x2, y2 = reg["bbox"]
        if x1 <= xn <= x2 and y1 <= yn <= y2:
            matched_idx = idx
            break
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        d = (xn - cx) ** 2 + (yn - cy) ** 2
        if d < best_dist:
            best_dist = d
            matched_idx = idx

    matched_reg = regions[matched_idx]
    assigned_role = (
        req.target_role if (req.target_role in ROLES) else matched_reg["role"]
    )

    patch["region_index"] = matched_idx
    patch["detected_role"] = matched_reg["role"]
    patch["role"] = assigned_role
    patch["label"] = ROLE_DISPLAY_LABELS.get(assigned_role, matched_reg["label"])
    patch["has_visual_embedding"] = True
    return patch


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    if STATE.recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized.")
    if not req.locks:
        raise HTTPException(status_code=400, detail="Lock at least one garment slot.")
    if req.outerwear_mode not in ("any", "with", "without"):
        raise HTTPException(
            status_code=400,
            detail="outerwear_mode must be 'any', 'with', or 'without'.",
        )

    sess = None
    if req.upload_id:
        with STATE.gpu_lock:
            sess = STATE.get_session(req.upload_id)

    locks_payload: Dict[str, Dict[str, object]] = {}
    for role, spec in req.locks.items():
        if role not in ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{role}'. Must be one of {ROLES}.",
            )
        hx = _validate_hex(spec.hex)
        sec_hx = _validate_hex(spec.secondary_hex) if spec.secondary_hex else hx

        vis_emb = None
        if sess is not None and spec.region_index is not None:
            crop_embs: np.ndarray = sess["crop_embs"]
            if 0 <= spec.region_index < len(crop_embs):
                vis_emb = crop_embs[spec.region_index]

        locks_payload[role] = {
            "hex": hx,
            "secondary_hex": sec_hx,
            "primary_share": spec.primary_share,
            "visual_embedding": vis_emb,
        }

    with STATE.gpu_lock:
        result = STATE.recommender.recommend(
            locks=locks_payload,
            style=req.style,
            outerwear_mode=req.outerwear_mode,
            top_k=req.top_k,
            shuffle_seed=req.shuffle_seed,
        )

    result["locked_summary"] = {
        role: {
            "hex": payload["hex"],
            "color_name": describe_oklab_color(hex_to_oklab(payload["hex"])),
            "multimodal": payload["visual_embedding"] is not None,
        }
        for role, payload in locks_payload.items()
    }
    return result
