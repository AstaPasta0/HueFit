import io
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote

import numpy as np
import open_clip
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.color_math import (
    describe_oklab_color,
    encode_garment_color,
    oklab_to_hex,
)
from src.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    FASHIONSIGLIP_MODEL_ID,
    MAX_OUTFIT_LEN,
    ROLES,
    VISUAL_DIM,
    ensure_dirs,
    get_device,
)
from src.vision import (
    ATR_ALL_GARMENT_IDS,
    ATR_HUMAN_BODY_IDS,
    POLYVORE_CAT_TO_ATR_IDS,
    GarmentVisionEngine,
    corner_floodfill_mask,
    extract_mask_colors_oklab,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_MORPH_3X3 = np.ones((3, 3), dtype=bool)

CATEGORY_TO_ROLE = {
    "tops": "top",
    "bottoms": "bottom",
    "outerwear": "outerwear",
    "shoes": "footwear",
}

SEMANTIC_CATS = (
    "other",
    "tops",
    "bottoms",
    "outerwear",
    "shoes",
    "all-body",
    "bags",
    "accessories",
    "jewellery",
    "hats",
    "sunglasses",
    "scarves",
)
SEM_CAT_TO_IDX = {c: i + 1 for i, c in enumerate(SEMANTIC_CATS)}

FORMAL_RE = re.compile(
    r"\b(blazer|suit|tailored|tuxedo|silk|satin|oxford|loafer|pump|stiletto|heel|"
    r"pencil|trouser|blouse|trench|cashmere|velvet|chiffon|evening|gown|cocktail|dressy)\b",
    re.IGNORECASE,
)
CASUAL_RE = re.compile(
    r"\b(t-shirt|tee|hoodie|sweatshirt|sneaker|trainer|jeans|denim|jogger|cargo|"
    r"flannel|bomber|chino|legging|canvas|flat|bootie|athletic|sport|running|gym|"
    r"fleece|parka|shorts|tank|espadrille|sandal)\b",
    re.IGNORECASE,
)

RETAILER_JUNK_RE = re.compile(
    r"(?:shein\s*\(\s*sheinside\s*\)|\bsheinside\b|\byoins\b|\bpre-?owned\b|https?://\S+)",
    re.IGNORECASE,
)
PRICE_RE = re.compile(r"(?:\$|£|€)\s*\d+(?:[.,]\d{1,2})?")
EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_title(raw_title: str, fallback: str = "Classic Garment") -> str:
    if not raw_title:
        return fallback
    s = unquote(str(raw_title))
    s = unicodedata.normalize("NFKC", s)
    s = PRICE_RE.sub("", s)
    s = RETAILER_JUNK_RE.sub("", s)
    s = EMPTY_PARENS_RE.sub("", s)
    s = s.replace("??", "").replace("//", " ").replace("|", " ")
    s = re.sub(r"\s*[-–—:,]+\s*$", "", s)
    s = re.sub(r"^\s*[-–—:,]+\s*", "", s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()
    if len(s) < 3:
        return fallback
    if s.islower() or s.isupper():
        s = s.title()
    return s[:68].strip()


def compute_formality(meta: dict) -> Tuple[float, int, int]:
    text = f"{meta.get('title') or ''} {meta.get('url_name') or ''} {meta.get('description') or ''}"
    f_hits = len(FORMAL_RE.findall(text))
    c_hits = len(CASUAL_RE.findall(text))
    score = (f_hits - c_hits) / (f_hits + c_hits + 1.0)
    return float(score), f_hits, c_hits


def _extract_single_color(
    args: Tuple[Image.Image, str, np.ndarray, int],
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    img, sem_cat, seg, work_size = args
    small_rgb = np.asarray(
        img.resize((work_size, work_size), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    human_ratio = float(np.isin(seg, ATR_HUMAN_BODY_IDS).mean())
    is_model = human_ratio >= 0.012
    cat_key = str(sem_cat or "").strip().lower()

    if is_model:
        target_ids = POLYVORE_CAT_TO_ATR_IDS.get(cat_key, ATR_ALL_GARMENT_IDS)
        fg_mask = np.isin(seg, target_ids)
        if fg_mask.mean() < 0.015:
            fg_mask = np.isin(seg, ATR_ALL_GARMENT_IDS)
        if fg_mask.mean() < 0.015:
            fg_mask, _ = corner_floodfill_mask(small_rgb)
        else:
            cleaned = ndimage.binary_opening(fg_mask, structure=_MORPH_3X3)
            if cleaned.sum() >= 24:
                fg_mask = cleaned
    else:
        fg_mask, _ = corner_floodfill_mask(small_rgb)

    col = extract_mask_colors_oklab(small_rgb, fg_mask, max_clusters=3)
    p_ok = np.asarray(col["primary_oklab"], dtype=np.float32)
    s_ok = np.asarray(col["secondary_oklab"], dtype=np.float32)
    p_share = float(col["primary_share"])
    return p_ok, s_ok, p_share, is_model


def _decode_pil(cell_val) -> Optional[Image.Image]:
    if cell_val is None:
        return None
    if isinstance(cell_val, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(cell_val)).convert("RGB")
        except Exception:
            return None
    if isinstance(cell_val, dict):
        raw = cell_val.get("bytes")
        if raw is not None:
            try:
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                return None
    return None


@torch.inference_mode()
def encode_fashionsiglip_batch(
    clip_model: torch.nn.Module,
    clip_preprocess,
    images: Sequence[Image.Image],
    device: torch.device,
) -> np.ndarray:
    tensors = torch.stack([clip_preprocess(img) for img in images], dim=0).to(
        device, non_blocking=True
    )
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feats = clip_model.encode_image(tensors)
    else:
        feats = clip_model.encode_image(tensors)
    feats = F.normalize(feats.float(), dim=-1)
    return feats.cpu().numpy().astype(np.float16)


def parse_all_splits(
    data_dir: Path,
    iid_to_row: Dict[str, int],
) -> Tuple[Dict[str, object], Dict[str, List[dict]]]:
    splits_raw: Dict[str, List[dict]] = {}
    set_index_to_item: Dict[str, str] = {}

    for split in ("train", "valid", "test"):
        with open(data_dir / "disjoint" / f"{split}.json", "r", encoding="utf-8") as fh:
            outfits = json.load(fh)
        splits_raw[split] = outfits
        for o in outfits:
            sid = str(o["set_id"])
            for it in o.get("items", []):
                iid = str(it["item_id"])
                set_index_to_item[f"{sid}_{it['index']}"] = iid

    out: Dict[str, object] = {}

    for split in ("train", "valid", "test"):
        idx_rows: List[List[int]] = []
        y_rows: List[int] = []
        cpath = data_dir / "disjoint" / f"compatibility_{split}.txt"
        with open(cpath, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                label = int(parts[0])
                seq: List[int] = []
                for tok in parts[1:]:
                    iid = set_index_to_item.get(tok, tok)
                    r = iid_to_row.get(iid)
                    if r is not None:
                        seq.append(r)
                        if len(seq) == MAX_OUTFIT_LEN:
                            break
                if len(seq) >= 2:
                    idx_rows.append(seq + [0] * (MAX_OUTFIT_LEN - len(seq)))
                    y_rows.append(label)
        out[f"idx_{split}"] = np.asarray(idx_rows, dtype=np.int32)
        out[f"y_{split}"] = np.asarray(y_rows, dtype=np.float32)

    for split in ("valid", "test"):
        fpath = data_dir / "disjoint" / f"fill_in_blank_{split}.json"
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            questions = json.load(fh)
        q_mat: List[List[int]] = []
        a_mat: List[List[int]] = []
        gt_vec: List[int] = []

        for q in questions:
            raw_q = [str(tok) for tok in q.get("question", []) if tok and tok != "___"]
            raw_a = [str(tok) for tok in q.get("answers", []) if tok]
            if not raw_q or len(raw_a) != 4:
                continue
            q_set_id = raw_q[0].split("_")[0] if "_" in raw_q[0] else ""
            q_rows = [
                iid_to_row.get(set_index_to_item.get(tok, tok), 0) for tok in raw_q
            ]
            q_rows = [r for r in q_rows if r > 0][: MAX_OUTFIT_LEN - 1]
            a_rows = [
                iid_to_row.get(set_index_to_item.get(tok, tok), 0) for tok in raw_a
            ]
            if not q_rows or any(r == 0 for r in a_rows):
                continue

            gt_idx = 0
            for ai, tok in enumerate(raw_a):
                if "_" in tok and tok.split("_")[0] == q_set_id:
                    gt_idx = ai
                    break

            q_mat.append(q_rows + [0] * ((MAX_OUTFIT_LEN - 1) - len(q_rows)))
            a_mat.append(a_rows)
            gt_vec.append(gt_idx)

        out[f"fitb_q_{split}"] = np.asarray(q_mat, dtype=np.int32)
        out[f"fitb_a_{split}"] = np.asarray(a_mat, dtype=np.int32)
        out[f"fitb_gt_{split}"] = np.asarray(gt_vec, dtype=np.int64)

    return out, splits_raw


def build_clean_catalog(
    train_outfits: List[dict],
    item_meta: Dict[str, dict],
    iid_to_row: Dict[str, int],
    item_primary_oklab: np.ndarray,
    item_secondary_oklab: np.ndarray,
    item_primary_share: np.ndarray,
    item_color_feats: np.ndarray,
    item_formality: np.ndarray,
    item_cat_id: np.ndarray,
    item_sem_cat: np.ndarray,
    item_has_image: np.ndarray,
    item_embeddings: np.ndarray,
    strict_train_only_ids: Set[str],
) -> Tuple[List[dict], Dict[str, np.ndarray], np.ndarray]:
    catalog: List[dict] = []
    seen_palettes: Set[Tuple[str, str, Optional[str], str]] = set()
    slot_global_rows: List[List[int]] = []

    for outfit in train_outfits:
        items = outfit.get("items", [])
        raw_iids = [str(it["item_id"]) for it in items]
        if any(iid not in strict_train_only_ids for iid in raw_iids):
            continue

        slots: Dict[str, Tuple[str, int, dict]] = {}
        f_sum = 0
        c_sum = 0

        for iid in raw_iids:
            row = iid_to_row.get(iid)
            if row is None or not bool(item_has_image[row]):
                continue
            meta = item_meta.get(iid, {})
            sem = str(meta.get("semantic_category") or "").strip().lower()
            role = CATEGORY_TO_ROLE.get(sem)
            if role is None or role in slots:
                continue
            slots[role] = (iid, row, meta)
            _, fh, ch = compute_formality(meta)
            f_sum += fh
            c_sum += ch

        if not ("top" in slots and "bottom" in slots and "footwear" in slots):
            continue

        has_outer = "outerwear" in slots
        if f_sum > c_sum:
            style = "Formal"
        elif c_sum > f_sum:
            style = "Casual"
        else:
            style = "Versatile"

        hex_map: Dict[str, Optional[str]] = {}
        sec_hex_map: Dict[str, Optional[str]] = {}
        oklab_map: Dict[str, Optional[List[float]]] = {}
        cname_map: Dict[str, Optional[str]] = {}
        title_map: Dict[str, Optional[str]] = {}
        share_map: Dict[str, Optional[float]] = {}
        g_rows = [0, 0, 0, 0]

        for r_i, role in enumerate(ROLES):
            if role not in slots:
                hex_map[role] = None
                sec_hex_map[role] = None
                oklab_map[role] = None
                cname_map[role] = None
                title_map[role] = None
                share_map[role] = None
                g_rows[r_i] = 0
                continue
            iid, row, meta = slots[role]
            g_rows[r_i] = row
            p_ok = item_primary_oklab[row]
            s_ok = item_secondary_oklab[row]
            hx = oklab_to_hex(p_ok)
            s_hx = oklab_to_hex(s_ok)
            cname = describe_oklab_color(p_ok)
            fallback = f"{cname} {role.title()}"
            raw_t = str(meta.get("title") or meta.get("url_name") or "")

            hex_map[role] = hx
            sec_hex_map[role] = s_hx
            oklab_map[role] = [round(float(v), 5) for v in p_ok]
            cname_map[role] = cname
            title_map[role] = clean_title(raw_t, fallback=fallback)
            share_map[role] = round(float(item_primary_share[row]), 3)

        pal_key = (
            hex_map["top"],
            hex_map["bottom"],
            hex_map["outerwear"],
            hex_map["footwear"],
        )
        if pal_key in seen_palettes:
            continue
        seen_palettes.add(pal_key)
        slot_global_rows.append(g_rows)

        catalog.append(
            {
                "set_id": str(outfit["set_id"]),
                "style": style,
                "has_outerwear": has_outer,
                "colors": hex_map,
                "secondary_colors": sec_hex_map,
                "primary_shares": share_map,
                "oklabs": oklab_map,
                "color_names": cname_map,
                "titles": title_map,
            }
        )

    n_cat = len(catalog)
    unique_cat_global_rows = sorted(
        {r for rows in slot_global_rows for r in rows if r > 0}
    )
    global_to_local = {g: idx + 1 for idx, g in enumerate(unique_cat_global_rows)}

    cat_item_emb = np.zeros(
        (len(unique_cat_global_rows) + 1, item_embeddings.shape[1]), dtype=np.float16
    )
    for g_row, l_row in global_to_local.items():
        cat_item_emb[l_row] = item_embeddings[g_row]

    slot_item_idx = np.zeros((n_cat, 4), dtype=np.int32)
    slot_oklab = np.zeros((n_cat, 4, 3), dtype=np.float32)
    slot_color_feats = np.zeros((n_cat, 4, 13), dtype=np.float32)
    slot_cat_ids = np.zeros((n_cat, 4), dtype=np.int64)
    slot_sem_cats = np.zeros((n_cat, 4), dtype=np.int64)
    slot_formality = np.zeros((n_cat, 4), dtype=np.float32)
    slot_mask = np.zeros((n_cat, 4), dtype=bool)
    style_codes = np.zeros((n_cat,), dtype=np.int8)
    has_outer_arr = np.zeros((n_cat,), dtype=bool)
    neutral_ratio = np.zeros((n_cat,), dtype=np.float32)

    style_to_code = {"Versatile": 0, "Casual": 1, "Formal": 2}
    for i, (entry, g_rows) in enumerate(zip(catalog, slot_global_rows)):
        style_codes[i] = style_to_code[entry["style"]]
        has_outer_arr[i] = bool(entry["has_outerwear"])
        chromas: List[float] = []
        for r_i, g_row in enumerate(g_rows):
            if g_row == 0:
                continue
            slot_item_idx[i, r_i] = global_to_local[g_row]
            slot_oklab[i, r_i] = item_primary_oklab[g_row]
            slot_color_feats[i, r_i] = item_color_feats[g_row]
            slot_cat_ids[i, r_i] = item_cat_id[g_row]
            slot_sem_cats[i, r_i] = item_sem_cat[g_row]
            slot_formality[i, r_i] = item_formality[g_row]
            slot_mask[i, r_i] = True
            chromas.append(
                float(
                    np.hypot(item_primary_oklab[g_row, 1], item_primary_oklab[g_row, 2])
                )
            )
        neutral_ratio[i] = float(
            sum(1 for c in chromas if c < 0.045) / max(1, len(chromas))
        )

    cat_tensors = {
        "slot_item_idx": slot_item_idx,
        "slot_oklab": slot_oklab,
        "slot_color_feats": slot_color_feats,
        "slot_cat_ids": slot_cat_ids,
        "slot_sem_cats": slot_sem_cats,
        "slot_formality": slot_formality,
        "slot_mask": slot_mask,
        "style_codes": style_codes,
        "has_outerwear": has_outer_arr,
        "neutral_ratio": neutral_ratio,
    }
    return catalog, cat_tensors, cat_item_emb


def main() -> None:
    t_start = time.perf_counter()
    ensure_dirs()
    device = get_device()
    print(
        f"=== HueFit v2 — Step 3: Offline Feature Extraction & Clean Catalog ({device}) ==="
    )

    manifest_path = ARTIFACTS_DIR / "disjoint_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    used_item_ids: List[str] = manifest["used_item_ids"]
    train_ids_set = set(manifest["train_item_ids"])
    valid_ids_set = set(manifest["valid_item_ids"])
    test_ids_set = set(manifest["test_item_ids"])
    strict_train_only_ids = train_ids_set - valid_ids_set - test_ids_set

    n_items = len(used_item_ids)
    iid_to_row = {iid: idx + 1 for idx, iid in enumerate(used_item_ids)}
    print(
        f"Unique disjoint items: {n_items:,} | "
        f"Strict train-only pool (zero valid/test overlap): {len(strict_train_only_ids):,}"
    )

    with open(DATA_DIR / "polyvore_item_metadata.json", "r", encoding="utf-8") as fh:
        item_meta = json.load(fh)

    train_cat_keys = sorted(
        {
            str(
                item_meta.get(iid, {}).get("category_id")
                or item_meta.get(iid, {}).get("semantic_category")
                or "other"
            )
            for iid in strict_train_only_ids
        }
    )
    cat_to_idx = {c: i + 2 for i, c in enumerate(train_cat_keys)}
    num_cats = len(cat_to_idx) + 2

    item_cat_id = np.zeros((n_items + 1,), dtype=np.int64)
    item_sem_cat = np.zeros((n_items + 1,), dtype=np.int64)
    item_formality = np.zeros((n_items + 1,), dtype=np.float32)
    item_sem_str = [""] * (n_items + 1)

    for iid, row in iid_to_row.items():
        meta = item_meta.get(iid, {})
        sem = str(meta.get("semantic_category") or "other").strip().lower()
        cat_key = str(meta.get("category_id") or sem or "other")
        item_cat_id[row] = cat_to_idx.get(cat_key, 1)
        item_sem_cat[row] = SEM_CAT_TO_IDX.get(sem, SEM_CAT_TO_IDX["other"])
        item_sem_str[row] = sem
        f_score, _, _ = compute_formality(meta)
        item_formality[row] = f_score

    item_embeddings = np.zeros((n_items + 1, VISUAL_DIM), dtype=np.float16)
    item_color_feats = np.zeros((n_items + 1, 13), dtype=np.float32)
    item_primary_oklab = np.zeros((n_items + 1, 3), dtype=np.float32)
    item_secondary_oklab = np.zeros((n_items + 1, 3), dtype=np.float32)
    item_primary_share = np.ones((n_items + 1,), dtype=np.float32)
    item_has_image = np.zeros((n_items + 1,), dtype=bool)
    item_is_model_shot = np.zeros((n_items + 1,), dtype=bool)

    ckpt_path = ARTIFACTS_DIR / "_preprocess_ckpt.npz"
    if ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path.name}")
        ck = np.load(ckpt_path)
        item_embeddings[:] = ck["item_embeddings"]
        item_color_feats[:] = ck["item_color_feats"]
        item_primary_oklab[:] = ck["item_primary_oklab"]
        item_secondary_oklab[:] = ck["item_secondary_oklab"]
        item_primary_share[:] = ck["item_primary_share"]
        item_has_image[:] = ck["item_has_image"]
        item_is_model_shot[:] = ck["item_is_model_shot"]
        print(f"  Already processed: {int(item_has_image.sum()):,}/{n_items:,} items")

    print(
        "Loading SegFormer-B2-Clothes (256x256) + Marqo-FashionSigLIP (FP16) on GPU ..."
    )
    vision_engine = GarmentVisionEngine(
        device=device, load_segformer=True, load_birefnet=False
    )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        FASHIONSIGLIP_MODEL_ID
    )
    clip_model = clip_model.to(device).eval()

    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated() / (1024**2)
        print(f"Models loaded | Allocated VRAM: {vram_mb:.1f} MB")

    parquet_sources = [
        ROOT_DIR / src["relative_path"] for src in manifest["image_sources"]
    ]
    gpu_batch_size = 64
    work_size = 96
    processed_since_ckpt = 0
    ckpt_interval = 25000

    pbar = tqdm(
        total=n_items,
        initial=int(item_has_image.sum()),
        desc="Extracting Oklab + FashionSigLIP",
        unit="img",
    )

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        buf_rows: List[int] = []
        buf_imgs: List[Image.Image] = []
        buf_cats: List[str] = []

        def flush_batch() -> None:
            nonlocal processed_since_ckpt
            if not buf_imgs:
                return
            seg_maps = vision_engine.segment_segformer_batch(
                buf_imgs, out_size=work_size
            )
            vis_embs = encode_fashionsiglip_batch(
                clip_model, clip_preprocess, buf_imgs, device
            )

            tasks = [
                (img, cat, seg, work_size)
                for img, cat, seg in zip(buf_imgs, buf_cats, seg_maps)
            ]
            color_results = list(pool.map(_extract_single_color, tasks))

            for i, row in enumerate(buf_rows):
                p_ok, s_ok, p_share, is_model = color_results[i]
                item_embeddings[row] = vis_embs[i]
                item_primary_oklab[row] = p_ok
                item_secondary_oklab[row] = s_ok
                item_primary_share[row] = p_share
                item_color_feats[row] = encode_garment_color(p_ok, s_ok, p_share)
                item_has_image[row] = True
                item_is_model_shot[row] = is_model

            n_done = len(buf_rows)
            pbar.update(n_done)
            processed_since_ckpt += n_done
            buf_rows.clear()
            buf_imgs.clear()
            buf_cats.clear()

            if processed_since_ckpt >= ckpt_interval:
                np.savez(
                    ckpt_path,
                    item_embeddings=item_embeddings,
                    item_color_feats=item_color_feats,
                    item_primary_oklab=item_primary_oklab,
                    item_secondary_oklab=item_secondary_oklab,
                    item_primary_share=item_primary_share,
                    item_has_image=item_has_image,
                    item_is_model_shot=item_is_model_shot,
                )
                processed_since_ckpt = 0

        for pq_path in parquet_sources:
            pf = pq.ParquetFile(pq_path)
            for batch in pf.iter_batches(batch_size=256, columns=["item_id", "image"]):
                iids = batch.column("item_id").to_pylist()
                imgs_col = batch.column("image").to_pylist()
                for raw_id, img_cell in zip(iids, imgs_col):
                    iid = str(raw_id)
                    row = iid_to_row.get(iid)
                    if row is None or item_has_image[row]:
                        continue
                    pil_img = _decode_pil(img_cell)
                    if pil_img is None:
                        continue
                    buf_rows.append(row)
                    buf_imgs.append(pil_img)
                    buf_cats.append(item_sem_str[row])
                    if len(buf_imgs) >= gpu_batch_size:
                        flush_batch()
        flush_batch()

    pbar.close()

    extracted_cnt = int(item_has_image.sum())
    model_shot_cnt = int(item_is_model_shot.sum())
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    )
    print(
        f"\nExtracted {extracted_cnt:,}/{n_items:,} images ({extracted_cnt / n_items * 100:.2f}%) | "
        f"Model shots: {model_shot_cnt:,} ({model_shot_cnt / max(1, extracted_cnt) * 100:.1f}%) | "
        f"Peak VRAM: {peak_vram:.1f} MB"
    )

    print(
        "Building indexed Disjoint splits (compatibility + FITB) & train-only negative pools ..."
    )
    split_arrays, splits_raw = parse_all_splits(DATA_DIR, iid_to_row)

    train_pool_by_cat: Dict[str, List[int]] = {}
    for iid in sorted(strict_train_only_ids):
        row = iid_to_row[iid]
        if not bool(item_has_image[row]):
            continue
        c_key = str(int(item_cat_id[row]))
        train_pool_by_cat.setdefault(c_key, []).append(row)

    np.save(ARTIFACTS_DIR / "item_embeddings.npy", item_embeddings)
    np.savez_compressed(
        ARTIFACTS_DIR / "training_dataset.npz",
        item_color_feats=item_color_feats,
        item_primary_oklab=item_primary_oklab,
        item_secondary_oklab=item_secondary_oklab,
        item_primary_share=item_primary_share,
        item_cat_id=item_cat_id,
        item_sem_cat=item_sem_cat,
        item_formality=item_formality,
        item_has_image=item_has_image,
        num_cats=np.int64(num_cats),
        num_sem_cats=np.int64(len(SEMANTIC_CATS) + 1),
        **split_arrays,
    )

    with open(ARTIFACTS_DIR / "item_mappings.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "cat_to_idx": cat_to_idx,
                "sem_cat_to_idx": SEM_CAT_TO_IDX,
                "num_cats": num_cats,
                "num_sem_cats": len(SEMANTIC_CATS) + 1,
                "train_pool_by_cat": train_pool_by_cat,
            },
            fh,
        )

    print(
        "Assembling clean retrieval catalog (strictly from train-only outfits, zero fake outerwear) ..."
    )
    catalog, cat_tensors, cat_item_emb = build_clean_catalog(
        train_outfits=splits_raw["train"],
        item_meta=item_meta,
        iid_to_row=iid_to_row,
        item_primary_oklab=item_primary_oklab,
        item_secondary_oklab=item_secondary_oklab,
        item_primary_share=item_primary_share,
        item_color_feats=item_color_feats,
        item_formality=item_formality,
        item_cat_id=item_cat_id,
        item_sem_cat=item_sem_cat,
        item_has_image=item_has_image,
        item_embeddings=item_embeddings,
        strict_train_only_ids=strict_train_only_ids,
    )

    with open(ARTIFACTS_DIR / "catalog.json", "w", encoding="utf-8") as fh:
        json.dump(catalog, fh)
    np.savez_compressed(ARTIFACTS_DIR / "catalog_tensors.npz", **cat_tensors)
    np.save(ARTIFACTS_DIR / "catalog_item_embeddings.npy", cat_item_emb)

    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except OSError:
            pass

    n_4pc = int(cat_tensors["has_outerwear"].sum())
    n_3pc = len(catalog) - n_4pc
    n_formal = int((cat_tensors["style_codes"] == 2).sum())
    n_casual = int((cat_tensors["style_codes"] == 1).sum())
    n_versatile = int((cat_tensors["style_codes"] == 0).sum())
    total_min = (time.perf_counter() - t_start) / 60.0

    print("\n=== Step 3 Preprocessing Summary ===")
    print(f"Elapsed time                  : {total_min:.2f} min")
    print(
        f"item_embeddings.npy           : {item_embeddings.shape} ({(ARTIFACTS_DIR / 'item_embeddings.npy').stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"catalog_item_embeddings.npy   : {cat_item_emb.shape} ({(ARTIFACTS_DIR / 'catalog_item_embeddings.npy').stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"Clean curated catalog         : {len(catalog):,} outfits ({n_3pc:,} 3-piece + {n_4pc:,} 4-piece, ZERO synthetic outerwear)"
    )
    print(
        f"Catalog styles                : Casual={n_casual:,} | Formal={n_formal:,} | Versatile={n_versatile:,}"
    )
    print(
        f"FITB test questions indexed   : {len(split_arrays.get('fitb_gt_test', [])):,}"
    )
    print(
        f"Unique train categories       : {num_cats} (train-only negative pool buckets: {len(train_pool_by_cat)})"
    )


if __name__ == "__main__":
    main()
