import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy import ndimage

from src.color_math import (
    describe_oklab_color,
    oklab_to_hex,
    rgb_to_hex,
    srgb_to_oklab,
)
from src.config import BIREFNET_MODEL_ID, SEGFORMER_MODEL_ID, get_amp_dtype, get_device

ATR_HUMAN_BODY_IDS = (2, 11, 12, 13, 14, 15)
ATR_ALL_GARMENT_IDS = (1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17)

ATR_LIVE_ROLES: List[Tuple[str, str, Tuple[int, ...]]] = [
    ("top", "Upper Garment (Top)", (4,)),
    ("top", "Dress / One-Piece", (7,)),
    ("bottom", "Lower Garment (Pants)", (6,)),
    ("bottom", "Lower Garment (Skirt)", (5,)),
    ("footwear", "Footwear", (9, 10)),
]

POLYVORE_CAT_TO_ATR_IDS: Dict[str, Tuple[int, ...]] = {
    "tops": (4, 7),
    "outerwear": (4, 7),
    "bottoms": (6, 5),
    "shoes": (9, 10),
    "all-body": (7, 4, 5, 6),
    "bags": (16,),
    "hats": (1,),
    "sunglasses": (3,),
    "scarves": (17,),
    "accessories": (8, 17, 1, 3, 16),
    "jewellery": (8, 17, 4, 7),
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_MORPH_STRUCT_3X3 = np.ones((3, 3), dtype=bool)


def prepare_image(
    img: Image.Image, max_side: int = 1024
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    img = ImageOps.exif_transpose(img)
    alpha_mask = None
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        if max(rgba.size) > max_side:
            rgba.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        arr = np.asarray(rgba)
        alpha = arr[:, :, 3]
        if (alpha < 128).any() and (alpha >= 128).any():
            alpha_mask = alpha >= 128
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg, alpha_mask

    rgb = img.convert("RGB")
    if max(rgb.size) > max_side:
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return rgb, None


def sample_patch_color(
    img: Image.Image, x_pct: float, y_pct: float, radius: int = 2
) -> Dict[str, object]:
    rgb_img, _ = prepare_image(img, max_side=1024)
    w, h = rgb_img.size
    cx = int(np.clip(round((x_pct / 100.0) * (w - 1)), 0, w - 1))
    cy = int(np.clip(round((y_pct / 100.0) * (h - 1)), 0, h - 1))
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)

    patch = np.asarray(rgb_img.crop((x0, y0, x1, y1)), dtype=np.float32).reshape(-1, 3)
    med_rgb = np.median(patch, axis=0)
    ok = srgb_to_oklab(med_rgb)
    hx = rgb_to_hex(med_rgb)
    return {
        "hex": hx,
        "oklab": [round(float(v), 5) for v in ok],
        "color_name": describe_oklab_color(ok),
        "x_pct": round(float(np.clip(x_pct, 0.0, 100.0)), 1),
        "y_pct": round(float(np.clip(y_pct, 0.0, 100.0)), 1),
    }


def _is_uniform_swatch(small_rgb: np.ndarray) -> bool:
    flat = small_rgb.reshape(-1, 3).astype(np.float32)
    if float(flat.std(axis=0).mean()) >= 2.5:
        return False
    p02 = np.percentile(flat, 2, axis=0)
    p98 = np.percentile(flat, 98, axis=0)
    return float((p98 - p02).max()) < 10.0


def _estimate_corner_backdrop(oklab_img: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w, _ = oklab_img.shape
    p = max(3, min(6, min(h, w) // 16))
    patches = [
        oklab_img[:p, :p].reshape(-1, 3),
        oklab_img[:p, -p:].reshape(-1, 3),
        oklab_img[-p:, :p].reshape(-1, 3),
        oklab_img[-p:, -p:].reshape(-1, 3),
        oklab_img[:p, w // 2 - p : w // 2 + p].reshape(-1, 3),
        oklab_img[-p:, w // 2 - p : w // 2 + p].reshape(-1, 3),
        oklab_img[h // 2 - p : h // 2 + p, :p].reshape(-1, 3),
        oklab_img[h // 2 - p : h // 2 + p, -p:].reshape(-1, 3),
    ]
    meds = np.stack([np.median(pt, axis=0) for pt in patches], axis=0)
    corner_meds = meds[:4]
    dists = np.linalg.norm(corner_meds[:, None, :] - corner_meds[None, :, :], axis=-1)
    best_corner = int(np.argmin(dists.sum(axis=1)))
    anchor = corner_meds[best_corner]
    consistent = meds[np.linalg.norm(meds - anchor[None, :], axis=1) < 0.05]
    bg_ok = np.median(consistent, axis=0) if len(consistent) > 0 else anchor
    chroma = float(np.hypot(bg_ok[1], bg_ok[2]))
    tol = 0.022 if (bg_ok[0] > 0.94 and chroma < 0.020) else 0.042
    return bg_ok.astype(np.float32), tol


def corner_floodfill_mask(rgb_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = rgb_arr.shape
    ok = srgb_to_oklab(rgb_arr)
    bg_ok, tol = _estimate_corner_backdrop(ok)
    bg_cand = np.linalg.norm(ok - bg_ok[None, None, :], axis=2) <= (tol * 1.08)

    labeled_bg, _ = ndimage.label(bg_cand)
    border_labels = np.unique(
        np.concatenate(
            [labeled_bg[0, :], labeled_bg[-1, :], labeled_bg[:, 0], labeled_bg[:, -1]]
        )
    )
    border_labels = border_labels[border_labels > 0]
    raw_fg = ~np.isin(labeled_bg, border_labels)

    cleaned_fg = ndimage.binary_opening(raw_fg, structure=_MORPH_STRUCT_3X3)
    labeled, n_comp = ndimage.label(cleaned_fg)
    if n_comp > 1:
        comp_sizes = np.bincount(labeled.ravel())[1:]
        max_size = int(comp_sizes.max())
        min_keep = max(24, int(max_size * 0.08))
        keep_ids = np.where(comp_sizes >= min_keep)[0] + 1
        if len(keep_ids) > 0:
            cleaned_fg = np.isin(labeled, keep_ids)
    if cleaned_fg.sum() < (h * w * 0.03):
        cleaned_fg = raw_fg
    if cleaned_fg.sum() < (h * w * 0.025):
        yy, xx = np.mgrid[0:h, 0:w]
        cleaned_fg = (
            ((xx / max(1, w - 1) - 0.5) / 0.28) ** 2
            + ((yy / max(1, h - 1) - 0.5) / 0.34) ** 2
        ) <= 1.0
    return cleaned_fg, bg_ok


def _fast_kmeans_oklab(
    points: np.ndarray,
    k: int,
    max_iter: int = 12,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(points)
    if n == 0:
        return np.zeros((1, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    k = min(k, n)
    if k == 1:
        return np.mean(points, axis=0, keepdims=True).astype(np.float32), np.zeros(
            (n,), dtype=np.int32
        )

    rng = np.random.default_rng(seed)
    centers = np.empty((k, 3), dtype=np.float32)
    centers[0] = points[
        int(np.argmin(np.linalg.norm(points - np.median(points, axis=0), axis=1)))
    ]
    for c_i in range(1, k):
        d2 = np.min(
            np.sum((points[:, None, :] - centers[None, :c_i, :]) ** 2, axis=2), axis=1
        )
        total = float(d2.sum())
        if total <= 1e-12:
            centers[c_i] = points[int(rng.integers(0, n))]
        else:
            centers[c_i] = points[int(np.argmax(d2))]

    labels = np.zeros((n,), dtype=np.int32)
    for _ in range(max_iter):
        d2 = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_i in range(k):
            mask = labels == c_i
            if mask.any():
                centers[c_i] = points[mask].mean(axis=0)
    return centers, labels


def extract_mask_colors_oklab(
    rgb_arr: np.ndarray,
    fg_mask: np.ndarray,
    max_clusters: int = 3,
    secondary_min_share: float = 0.12,
) -> Dict[str, object]:
    h, w, _ = rgb_arr.shape
    if fg_mask.sum() < 4:
        yy, xx = np.mgrid[0:h, 0:w]
        fg_mask = (np.abs(xx / max(1, w - 1) - 0.5) <= 0.25) & (
            np.abs(yy / max(1, h - 1) - 0.5) <= 0.25
        )

    eroded = ndimage.binary_erosion(fg_mask, structure=_MORPH_STRUCT_3X3)
    sample_mask = (
        eroded if eroded.sum() >= max(24, int(fg_mask.sum() * 0.20)) else fg_mask
    )

    ys, xs = np.where(sample_mask)
    pix_rgb = rgb_arr[ys, xs].astype(np.float32)
    pix_ok = srgb_to_oklab(pix_rgb)

    if len(pix_ok) > 2048:
        idx = np.linspace(0, len(pix_ok) - 1, 2048, dtype=np.int32)
        sub_ok, sub_xs, sub_ys = pix_ok[idx], xs[idx], ys[idx]
    else:
        sub_ok, sub_xs, sub_ys = pix_ok, xs, ys

    if float(pix_ok.std(axis=0).mean()) < 0.015:
        mean_ok = pix_ok.mean(axis=0).astype(np.float32)
        med_x = float(np.median(sub_xs)) / max(1, w - 1) * 100.0
        med_y = float(np.median(sub_ys)) / max(1, h - 1) * 100.0
        hx = oklab_to_hex(mean_ok)
        cname = describe_oklab_color(mean_ok)
        return {
            "primary_oklab": [round(float(v), 5) for v in mean_ok],
            "secondary_oklab": [round(float(v), 5) for v in mean_ok],
            "primary_share": 1.0,
            "primary_hex": hx,
            "secondary_hex": hx,
            "primary_name": cname,
            "secondary_name": cname,
            "x_pct": round(med_x, 1),
            "y_pct": round(med_y, 1),
        }

    centers, labels = _fast_kmeans_oklab(sub_ok, k=max_clusters, max_iter=12, seed=42)
    counts = np.bincount(labels, minlength=len(centers))
    order = np.argsort(counts)[::-1]

    merged: List[Dict[str, object]] = []
    total_pts = float(len(sub_ok))
    for c_i in order:
        cnt = int(counts[c_i])
        if cnt == 0:
            continue
        share = cnt / total_pts
        c_ok = centers[c_i].astype(np.float32)
        c_mask = labels == c_i

        was_merged = False
        for existing in merged:
            ex_ok = existing["ok"]
            d_l = abs(float(c_ok[0] - ex_ok[0]))
            d_ab = float(np.hypot(c_ok[1] - ex_ok[1], c_ok[2] - ex_ok[2]))
            d_e = float(np.linalg.norm(c_ok - ex_ok))
            if d_e < 0.055 or (d_ab < 0.038 and d_l < 0.22):
                old_w = float(existing["weight"])
                new_w = old_w + share
                existing["ok"] = ((ex_ok * old_w + c_ok * share) / new_w).astype(
                    np.float32
                )
                existing["weight"] = new_w
                existing["mask"] = existing["mask"] | c_mask
                was_merged = True
                break
        if not was_merged:
            merged.append({"ok": c_ok, "weight": share, "mask": c_mask})

    merged.sort(key=lambda x: float(x["weight"]), reverse=True)
    primary = merged[0]
    p_ok = primary["ok"]

    if len(merged) >= 2 and float(merged[1]["weight"]) >= secondary_min_share:
        sec = merged[1]
        s_ok = sec["ok"]
        p_share = float(primary["weight"] / (primary["weight"] + sec["weight"]))
    else:
        s_ok = p_ok
        p_share = 1.0

    p_mask = primary["mask"]
    p_xs = sub_xs[p_mask] / max(1, w - 1)
    p_ys = sub_ys[p_mask] / max(1, h - 1)
    p_ok_pts = sub_ok[p_mask]
    med_x, med_y = float(np.median(p_xs)), float(np.median(p_ys))
    best_idx = int(
        np.argmin(
            (p_xs - med_x) ** 2
            + (p_ys - med_y) ** 2
            + np.linalg.norm(p_ok_pts - p_ok[None, :], axis=1) * 0.25
        )
    )

    return {
        "primary_oklab": [round(float(v), 5) for v in p_ok],
        "secondary_oklab": [round(float(v), 5) for v in s_ok],
        "primary_share": round(float(np.clip(p_share, 0.5, 1.0)), 4),
        "primary_hex": oklab_to_hex(p_ok),
        "secondary_hex": oklab_to_hex(s_ok),
        "primary_name": describe_oklab_color(p_ok),
        "secondary_name": describe_oklab_color(s_ok),
        "x_pct": round(float(p_xs[best_idx] * 100.0), 1),
        "y_pct": round(float(p_ys[best_idx] * 100.0), 1),
    }


def mask_to_bbox(mask: np.ndarray, pad: float = 0.03) -> List[float]:
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 1.0, 1.0]
    x1 = max(0.0, float(xs.min()) / max(1, w - 1) - pad)
    y1 = max(0.0, float(ys.min()) / max(1, h - 1) - pad)
    x2 = min(1.0, float(xs.max()) / max(1, w - 1) + pad)
    y2 = min(1.0, float(ys.max()) / max(1, h - 1) + pad)
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def crop_image_by_bbox(img: Image.Image, bbox: Sequence[float]) -> Image.Image:
    w, h = img.size
    x1 = max(0, int(round(bbox[0] * w)))
    y1 = max(0, int(round(bbox[1] * h)))
    x2 = min(w, max(x1 + 8, int(round(bbox[2] * w))))
    y2 = min(h, max(y1 + 8, int(round(bbox[3] * h))))
    return img.crop((x1, y1, x2, y2))


class GarmentVisionEngine:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        load_segformer: bool = True,
        load_birefnet: bool = True,
    ) -> None:
        self.device = device or get_device()
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = get_amp_dtype(self.device)
        self.seg_processor = None
        self.seg_model = None
        self.birefnet = None

        if load_segformer:
            self._load_segformer()
        if load_birefnet:
            self._load_birefnet()

    def _load_segformer(self) -> None:
        from transformers import (
            AutoModelForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self.seg_processor = SegformerImageProcessor.from_pretrained(
            SEGFORMER_MODEL_ID,
            size={"height": 256, "width": 256},
        )
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(
            SEGFORMER_MODEL_ID
        )
        self.seg_model.to(self.device).eval()

    def _load_birefnet(self) -> None:
        try:
            from transformers import AutoModelForImageSegmentation

            self.birefnet = AutoModelForImageSegmentation.from_pretrained(
                BIREFNET_MODEL_ID,
                trust_remote_code=True,
            )
            self.birefnet.to(self.device).eval()
        except Exception as exc:
            print(
                f"[WARN] BiRefNet_lite unavailable ({exc}); using Oklab corner flood-fill fallback."
            )
            self.birefnet = None

    @torch.inference_mode()
    def segment_segformer_batch(
        self,
        images: Sequence[Image.Image],
        out_size: int = 128,
    ) -> np.ndarray:
        if self.seg_model is None or self.seg_processor is None:
            raise RuntimeError("SegFormer-B2-Clothes is not loaded.")
        inputs = self.seg_processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                logits = self.seg_model(pixel_values=pixel_values).logits
        else:
            logits = self.seg_model(pixel_values=pixel_values).logits
        up = F.interpolate(
            logits.float(),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )
        return up.argmax(dim=1).cpu().numpy().astype(np.uint8)

    @torch.inference_mode()
    def segment_birefnet(
        self,
        img: Image.Image,
        infer_size: int = 512,
        out_size: int = 128,
    ) -> Optional[np.ndarray]:
        if self.birefnet is None:
            return None
        resized = img.resize((infer_size, infer_size), Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        norm = (chw - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(norm).unsqueeze(0).to(self.device)
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                out = self.birefnet(tensor)
        else:
            out = self.birefnet(tensor)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        prob = torch.sigmoid(pred.float())
        prob_small = F.interpolate(
            prob, size=(out_size, out_size), mode="bilinear", align_corners=False
        )
        mask = prob_small[0, 0].cpu().numpy() >= 0.50
        mask = ndimage.binary_opening(mask, structure=_MORPH_STRUCT_3X3)
        ratio = float(mask.mean())
        if ratio < 0.025 or ratio > 0.985:
            return None
        return mask

    def extract_polyvore_batch(
        self,
        images: Sequence[Image.Image],
        semantic_categories: Sequence[str],
        work_size: int = 128,
    ) -> List[Dict[str, object]]:
        rgb_imgs = [img.convert("RGB") for img in images]
        seg_maps = self.segment_segformer_batch(rgb_imgs, out_size=work_size)

        results: List[Dict[str, object]] = []
        for img, sem_cat, seg in zip(rgb_imgs, semantic_categories, seg_maps):
            small_rgb = np.asarray(
                img.resize((work_size, work_size), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
            human_ratio = float(np.isin(seg, ATR_HUMAN_BODY_IDS).mean())
            is_model_shot = human_ratio >= 0.012
            cat_key = str(sem_cat or "").strip().lower()

            if is_model_shot:
                target_ids = POLYVORE_CAT_TO_ATR_IDS.get(cat_key, ATR_ALL_GARMENT_IDS)
                fg_mask = np.isin(seg, target_ids)
                if fg_mask.mean() < 0.015:
                    fg_mask = np.isin(seg, ATR_ALL_GARMENT_IDS)
                if fg_mask.mean() < 0.015:
                    fg_mask, _ = corner_floodfill_mask(small_rgb)
                else:
                    cleaned = ndimage.binary_opening(
                        fg_mask, structure=_MORPH_STRUCT_3X3
                    )
                    if cleaned.sum() >= 24:
                        fg_mask = cleaned
                method = "segformer_category_guided"
            else:
                fg_mask, _ = corner_floodfill_mask(small_rgb)
                method = "oklab_floodfill"

            col = extract_mask_colors_oklab(small_rgb, fg_mask, max_clusters=3)
            col["is_model_shot"] = is_model_shot
            col["method"] = method
            col["mask_coverage"] = round(float(fg_mask.mean()), 4)
            results.append(col)
        return results

    def analyze_photo(
        self,
        img: Image.Image,
        use_birefnet: bool = True,
        return_crops: bool = True,
        return_debug: bool = False,
        work_size: int = 128,
    ) -> Dict[str, object]:
        t0 = time.perf_counter()
        rgb_img, alpha_mask = prepare_image(img, max_side=1024)
        small_img = rgb_img.resize((work_size, work_size), Image.Resampling.LANCZOS)
        small_rgb = np.asarray(small_img, dtype=np.uint8)

        if _is_uniform_swatch(small_rgb):
            ok = srgb_to_oklab(small_rgb.reshape(-1, 3).mean(axis=0))
            hx = oklab_to_hex(ok)
            cname = describe_oklab_color(ok)
            reg = {
                "role": "top",
                "label": "Solid Color Swatch",
                "hex": hx,
                "oklab": [round(float(v), 5) for v in ok],
                "color_name": cname,
                "secondary_hex": hx,
                "secondary_oklab": [round(float(v), 5) for v in ok],
                "secondary_name": cname,
                "primary_share": 1.0,
                "share": 100.0,
                "x_pct": 50.0,
                "y_pct": 50.0,
                "bbox": [0.1, 0.1, 0.9, 0.9],
            }
            if return_crops:
                reg["crop_image"] = rgb_img
            res = {
                "mode": "swatch",
                "segmenter": "swatch_shortcut",
                "primary_hex": hx,
                "bg_hex": None,
                "roi_box": [0.1, 0.1, 0.9, 0.9],
                "regions": [reg],
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
            if return_debug:
                res["_debug"] = {
                    "fg_mask": np.ones((work_size, work_size), dtype=bool),
                    "seg_map": None,
                }
            return res

        seg_map = (
            self.segment_segformer_batch([rgb_img], out_size=work_size)[0]
            if self.seg_model is not None
            else None
        )
        human_ratio = (
            float(np.isin(seg_map, ATR_HUMAN_BODY_IDS).mean())
            if seg_map is not None
            else 0.0
        )

        detected_garments: List[Tuple[str, str, np.ndarray]] = []
        if seg_map is not None:
            for role, label, class_ids in ATR_LIVE_ROLES:
                m = np.isin(seg_map, class_ids)
                cleaned = ndimage.binary_opening(m, structure=_MORPH_STRUCT_3X3)
                if cleaned.sum() >= 20:
                    m = cleaned
                min_area = 0.008 if role == "footwear" else 0.022
                if float(m.mean()) >= min_area:
                    detected_garments.append((role, label, m))

        use_segformer_regions = (seg_map is not None) and (
            (human_ratio >= 0.012 and len(detected_garments) >= 1)
            or len({r for r, _, _ in detected_garments}) >= 2
        )

        flood_mask, bg_ok = corner_floodfill_mask(small_rgb)
        bg_hex = oklab_to_hex(bg_ok)

        if use_segformer_regions:
            union_mask = np.zeros((work_size, work_size), dtype=bool)
            for _, _, m in detected_garments:
                union_mask |= m
            total_fg = max(1, int(union_mask.sum()))

            regions = []
            seen_roles = set()
            detected_garments.sort(key=lambda item: int(item[2].sum()), reverse=True)
            for role, label, m in detected_garments:
                if role in seen_roles:
                    continue
                seen_roles.add(role)
                col = extract_mask_colors_oklab(small_rgb, m, max_clusters=3)
                bbox = mask_to_bbox(m, pad=0.03)
                reg = {
                    "role": role,
                    "label": label,
                    "hex": col["primary_hex"],
                    "oklab": col["primary_oklab"],
                    "color_name": col["primary_name"],
                    "secondary_hex": col["secondary_hex"],
                    "secondary_oklab": col["secondary_oklab"],
                    "secondary_name": col["secondary_name"],
                    "primary_share": col["primary_share"],
                    "share": round(float(m.sum()) / total_fg * 100.0, 1),
                    "x_pct": col["x_pct"],
                    "y_pct": col["y_pct"],
                    "bbox": bbox,
                }
                if return_crops:
                    reg["crop_image"] = crop_image_by_bbox(rgb_img, bbox)
                regions.append(reg)

            res = {
                "mode": "outfit",
                "segmenter": "segformer_b2_clothes",
                "primary_hex": regions[0]["hex"],
                "bg_hex": bg_hex,
                "roi_box": mask_to_bbox(union_mask, pad=0.01),
                "regions": regions,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
            if return_debug:
                res["_debug"] = {"fg_mask": union_mask, "seg_map": seg_map}
            return res

        if alpha_mask is not None:
            alpha_small = (
                np.asarray(
                    Image.fromarray(alpha_mask.astype(np.uint8) * 255).resize(
                        (work_size, work_size), Image.Resampling.NEAREST
                    )
                )
                > 128
            )
            fg_mask = ndimage.binary_opening(alpha_small, structure=_MORPH_STRUCT_3X3)
            if fg_mask.sum() < 24:
                fg_mask = alpha_small
            segmenter_name = "alpha_channel"
        else:
            biref_mask = (
                self.segment_birefnet(rgb_img, out_size=work_size)
                if use_birefnet
                else None
            )
            if biref_mask is not None:
                fg_mask = biref_mask
                segmenter_name = "birefnet_lite"
            else:
                fg_mask = flood_mask
                segmenter_name = "oklab_floodfill"

        inferred_role = "top"
        inferred_label = "Primary Garment"
        if len(detected_garments) == 1:
            inferred_role, inferred_label, _ = detected_garments[0]

        col = extract_mask_colors_oklab(small_rgb, fg_mask, max_clusters=3)
        bbox = mask_to_bbox(fg_mask, pad=0.02)
        reg = {
            "role": inferred_role,
            "label": inferred_label,
            "hex": col["primary_hex"],
            "oklab": col["primary_oklab"],
            "color_name": col["primary_name"],
            "secondary_hex": col["secondary_hex"],
            "secondary_oklab": col["secondary_oklab"],
            "secondary_name": col["secondary_name"],
            "primary_share": col["primary_share"],
            "share": 100.0,
            "x_pct": col["x_pct"],
            "y_pct": col["y_pct"],
            "bbox": bbox,
        }
        if return_crops:
            reg["crop_image"] = crop_image_by_bbox(rgb_img, bbox)

        res = {
            "mode": "flatlay",
            "segmenter": segmenter_name,
            "primary_hex": reg["hex"],
            "bg_hex": bg_hex,
            "roi_box": bbox,
            "regions": [reg],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        if return_debug:
            res["_debug"] = {"fg_mask": fg_mask, "seg_map": seg_map}
        return res
