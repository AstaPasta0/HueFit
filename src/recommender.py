import json
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.color_math import (
    ciede2000_hex,
    ciede2000_oklab,
    classify_palette_harmony,
    describe_hex_color,
    describe_oklab_color,
    encode_garment_color,
    hex_to_oklab,
)
from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_SEM_CAT_TO_IDX,
    ROLES,
    VISUAL_DIM,
    get_device,
)
from src.model import HueFitNetV2

ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}
STYLE_TO_CODE = {"Versatile": 0, "Casual": 1, "Formal": 2}

DEFAULT_ROLE_SEM_CAT = {
    "top": "tops",
    "bottom": "bottoms",
    "outerwear": "outerwear",
    "footwear": "shoes",
}

DEFAULT_ROLE_FORMALITY = {
    "top": 0.0,
    "bottom": 0.0,
    "outerwear": 0.1,
    "footwear": 0.0,
}

SECONDARY_MIN_DELTA_E = 12.0
SECONDARY_MAX_PRIMARY_SHARE = 0.88


def _sigmoid_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.clip(a * logits + b, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _calibrate_audit_item_scores(
    raw_item_logits: np.ndarray, overall_prob_01: float
) -> np.ndarray:
    overall_p = float(np.clip(overall_prob_01, 0.02, 0.98))
    overall_logit = float(np.log(overall_p / (1.0 - overall_p)))
    centered = raw_item_logits - float(np.mean(raw_item_logits))
    cal_logits = np.clip(overall_logit + 1.35 * centered, -10.0, 10.0)
    return (1.0 / (1.0 + np.exp(-cal_logits))) * 100.0


def _resolve_distinct_secondary(
    primary_hex: Optional[str],
    secondary_hex: Optional[str],
    primary_share: Optional[float],
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    if not primary_hex or not secondary_hex or primary_share is None:
        return None, None, primary_share
    p_hex = primary_hex.upper()
    s_hex = secondary_hex.upper()
    share = float(primary_share)
    if s_hex == p_hex or share > SECONDARY_MAX_PRIMARY_SHARE:
        return None, None, 1.0
    if ciede2000_hex(p_hex, s_hex) < SECONDARY_MIN_DELTA_E:
        return None, None, 1.0
    return s_hex, describe_hex_color(s_hex), round(share, 3)


class HueFitRecommender:
    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or get_device()

        with open(ARTIFACTS_DIR / "catalog.json", "r", encoding="utf-8") as fh:
            self.catalog: List[dict] = json.load(fh)

        map_path = ARTIFACTS_DIR / "item_mappings.json"
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as fh:
                mappings = json.load(fh)
            self.sem_cat_to_idx: Dict[str, int] = mappings.get(
                "sem_cat_to_idx", DEFAULT_SEM_CAT_TO_IDX
            )
        else:
            self.sem_cat_to_idx = dict(DEFAULT_SEM_CAT_TO_IDX)

        ct = np.load(ARTIFACTS_DIR / "catalog_tensors.npz")
        self.slot_item_idx: np.ndarray = ct["slot_item_idx"]
        self.slot_oklab: np.ndarray = ct["slot_oklab"]
        self.slot_color_feats: np.ndarray = ct["slot_color_feats"]
        self.slot_cat_ids: np.ndarray = ct["slot_cat_ids"]
        self.slot_sem_cats: np.ndarray = ct["slot_sem_cats"]
        self.slot_formality: np.ndarray = ct["slot_formality"]
        self.slot_mask: np.ndarray = ct["slot_mask"]
        self.style_codes: np.ndarray = ct["style_codes"]
        self.has_outerwear: np.ndarray = ct["has_outerwear"]
        self.neutral_ratio: np.ndarray = ct["neutral_ratio"]

        cat_emb_np = np.load(ARTIFACTS_DIR / "catalog_item_embeddings.npy")
        self.cat_item_emb = torch.tensor(
            cat_emb_np, dtype=torch.float16, device=self.device
        )

        self.default_role_cat_id: Dict[str, int] = {}
        for r_i, role in enumerate(ROLES):
            valid_cats = self.slot_cat_ids[self.slot_mask[:, r_i], r_i]
            if len(valid_cats) > 0:
                vals, cnts = np.unique(valid_cats, return_counts=True)
                self.default_role_cat_id[role] = int(vals[int(np.argmax(cnts))])
            else:
                self.default_role_cat_id[role] = 1

        ckpt = torch.load(
            ARTIFACTS_DIR / "huefit_v2.pt", map_location=self.device, weights_only=True
        )
        self.model = HueFitNetV2(**ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.platt_a_multi: float = float(
            ckpt.get(
                "platt_a_multimodal", 1.0 / float(ckpt.get("temp_multimodal", 1.0))
            )
        )
        self.platt_b_multi: float = float(ckpt.get("platt_b_multimodal", 0.0))
        self.platt_a_color: float = float(
            ckpt.get("platt_a_color", 1.0 / float(ckpt.get("temp_color", 1.0)))
        )
        self.platt_b_color: float = float(ckpt.get("platt_b_color", 0.0))
        self.thresh_multi: float = float(ckpt.get("thresh_multimodal", 0.50))
        self.thresh_color: float = float(ckpt.get("thresh_color", 0.50))

        self.eval_summary: Dict[str, object] = {}
        rep_path = ARTIFACTS_DIR / "evaluation_report.json"
        if rep_path.exists():
            with open(rep_path, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            self.eval_summary = {
                "multimodal_auc": rep["test_compatibility"]["huefit_v2_multimodal"][
                    "roc_auc"
                ],
                "color_only_auc": rep["test_compatibility"]["huefit_v2_color_only"][
                    "roc_auc"
                ],
                "fitb_multimodal": rep["test_fitb"]["multimodal_top1"],
                "fitb_color_only": rep["test_fitb"]["color_only_top1"],
            }

    def _select_pool(
        self,
        locked_roles: List[str],
        style: str,
        outerwear_mode: str,
    ) -> Tuple[np.ndarray, Optional[str]]:
        n = len(self.catalog)
        base_mask = np.ones((n,), dtype=bool)

        if "outerwear" in locked_roles:
            outerwear_mode = "with"

        if outerwear_mode == "with":
            base_mask &= self.has_outerwear
        elif outerwear_mode == "without":
            base_mask &= ~self.has_outerwear

        notice: Optional[str] = None
        style_norm = style.strip().capitalize() if style else "All"

        if style_norm in ("Casual", "Formal"):
            code = STYLE_TO_CODE[style_norm]
            strict_mask = base_mask & (self.style_codes == code)
            if int(strict_mask.sum()) >= 16:
                return np.where(strict_mask)[0], None

            relaxed_mask = base_mask & (
                (self.style_codes == code)
                | (self.style_codes == STYLE_TO_CODE["Versatile"])
            )
            if int(relaxed_mask.sum()) >= 8:
                notice = (
                    f"Only {int(strict_mask.sum())} strict {style_norm} outfits matched your filters; "
                    f"expanded pool with Versatile ensembles."
                )
                return np.where(relaxed_mask)[0], notice

            if int(base_mask.sum()) >= 8:
                notice = (
                    f"Fewer than 8 {style_norm} outfits matched your exact layer/pin constraints; "
                    f"showing best matches across All Styles."
                )
                return np.where(base_mask)[0], notice

        if int(base_mask.sum()) >= 1:
            return np.where(base_mask)[0], notice

        notice = "No outfits matched the strict layer filter; expanded to full catalog."
        return np.arange(n, dtype=np.int64), notice

    @torch.inference_mode()
    def recommend(
        self,
        locks: Dict[str, Dict[str, object]],
        style: str = "All",
        outerwear_mode: str = "any",
        top_k: int = 8,
        shuffle_seed: int = 0,
    ) -> Dict[str, object]:
        t0 = time.perf_counter()
        if not locks:
            raise ValueError("At least one garment slot must be locked.")

        valid_locks: Dict[str, Dict[str, object]] = {}
        for role, spec in locks.items():
            if role not in ROLE_TO_IDX:
                continue
            hx = str(spec["hex"]).strip().upper()
            if not hx.startswith("#"):
                hx = f"#{hx}"
            p_ok = hex_to_oklab(hx)
            sec_hx = str(spec.get("secondary_hex") or hx).strip().upper()
            if not sec_hx.startswith("#"):
                sec_hx = f"#{sec_hx}"
            s_ok = hex_to_oklab(sec_hx)
            p_share = float(spec.get("primary_share", 1.0))
            vis_emb = spec.get("visual_embedding", None)
            valid_locks[role] = {
                "hex": hx,
                "secondary_hex": sec_hx,
                "primary_oklab": p_ok,
                "secondary_oklab": s_ok,
                "primary_share": p_share,
                "color_feat": encode_garment_color(p_ok, s_ok, p_share),
                "color_name": describe_oklab_color(p_ok),
                "visual_embedding": vis_emb,
            }

        if not valid_locks:
            raise ValueError("No valid garment roles supplied in locks.")

        locked_roles = list(valid_locks.keys())
        vis_flags = [v["visual_embedding"] is not None for v in valid_locks.values()]
        use_multimodal = all(vis_flags)
        is_mixed_locks = any(vis_flags) and not use_multimodal

        pool_indices, filter_notice = self._select_pool(
            locked_roles, style, outerwear_mode
        )

        dist_sum = np.zeros((len(pool_indices),), dtype=np.float32)
        for role, spec in valid_locks.items():
            r_i = ROLE_TO_IDX[role]
            cand_ok = self.slot_oklab[pool_indices, r_i, :]
            d = np.linalg.norm(cand_ok - spec["primary_oklab"][None, :], axis=1)
            dist_sum += d
        mean_dists = dist_sum / float(len(valid_locks))

        k_tier1 = min(80, len(pool_indices))
        if k_tier1 < len(pool_indices):
            part1 = np.argpartition(mean_dists, k_tier1 - 1)[:k_tier1]
            part1 = part1[np.argsort(mean_dists[part1])]
        else:
            part1 = np.argsort(mean_dists)

        tier1_set = set(part1.tolist())
        rem_local = np.array(
            [i for i in range(len(pool_indices)) if i not in tier1_set], dtype=np.int64
        )

        tier2_local = np.empty((0,), dtype=np.int64)
        if len(rem_local) > 0:
            rem_neut = self.neutral_ratio[pool_indices[rem_local]]
            rem_dist = mean_dists[rem_local]
            tier2_score = rem_neut * 0.75 + np.exp(-rem_dist / 0.25) * 0.25
            k_tier2 = min(40, len(rem_local))
            if k_tier2 < len(rem_local):
                t2_idx = np.argpartition(-tier2_score, k_tier2 - 1)[:k_tier2]
            else:
                t2_idx = np.arange(len(rem_local), dtype=np.int64)
            tier2_local = rem_local[t2_idx]

        cand_local = np.concatenate([part1, tier2_local], axis=0)
        cand_global = pool_indices[cand_local]
        cand_dists = mean_dists[cand_local]
        n_cands = len(cand_global)

        b_col = self.slot_color_feats[cand_global].copy()
        b_oklab = self.slot_oklab[cand_global].copy()
        b_cats = self.slot_cat_ids[cand_global].copy()
        b_sems = self.slot_sem_cats[cand_global].copy()
        b_frms = self.slot_formality[cand_global].copy()
        b_mask = self.slot_mask[cand_global].copy()
        b_item_idx = self.slot_item_idx[cand_global]

        if use_multimodal:
            vis_tensor = self.cat_item_emb[
                torch.tensor(b_item_idx, dtype=torch.long, device=self.device)
            ].float()
        else:
            vis_tensor = torch.zeros(
                (n_cands, 4, VISUAL_DIM), dtype=torch.float32, device=self.device
            )

        for role, spec in valid_locks.items():
            r_i = ROLE_TO_IDX[role]
            b_col[:, r_i, :] = spec["color_feat"][None, :]
            b_oklab[:, r_i, :] = spec["primary_oklab"][None, :]
            b_mask[:, r_i] = True
            if use_multimodal:
                v_live = torch.tensor(
                    spec["visual_embedding"], dtype=torch.float32, device=self.device
                )
                vis_tensor[:, r_i, :] = v_live.unsqueeze(0)

        col_t = torch.tensor(b_col, dtype=torch.float32, device=self.device)
        cats_t = torch.tensor(b_cats, dtype=torch.long, device=self.device)
        sems_t = torch.tensor(b_sems, dtype=torch.long, device=self.device)
        frms_t = torch.tensor(b_frms, dtype=torch.float32, device=self.device)
        pad_t = torch.tensor(~b_mask, dtype=torch.bool, device=self.device)
        hv_t = (~pad_t) if use_multimodal else torch.zeros_like(pad_t)

        o_logits, _ = self.model(vis_tensor, col_t, cats_t, sems_t, frms_t, pad_t, hv_t)
        logits_np = o_logits.cpu().numpy()

        if use_multimodal:
            probs = _sigmoid_platt(logits_np, self.platt_a_multi, self.platt_b_multi)
        else:
            probs = _sigmoid_platt(logits_np, self.platt_a_color, self.platt_b_color)

        is_shuffled = shuffle_seed > 0
        rng = np.random.default_rng(shuffle_seed) if is_shuffled else None
        jitter_amp = 0.095 if is_shuffled else 0.0

        candidates: List[Dict[str, object]] = []
        seen_keys = set()
        unlocked_roles = [r for r in ROLES if r not in valid_locks]

        for i in range(n_cands):
            g_idx = int(cand_global[i])
            entry = self.catalog[g_idx]
            colors = dict(entry["colors"])
            sec_colors = dict(entry["secondary_colors"])
            color_names = dict(entry["color_names"])
            titles = dict(entry["titles"])
            primary_shares = dict(entry["primary_shares"])

            for role, spec in valid_locks.items():
                colors[role] = spec["hex"]
                sec_colors[role] = spec["secondary_hex"]
                color_names[role] = spec["color_name"]
                primary_shares[role] = round(float(spec["primary_share"]), 3)
                titles[role] = f"Pinned {role.title()}"

            combo_key = (
                colors["top"],
                colors["bottom"],
                colors["outerwear"],
                colors["footwear"],
            )
            if combo_key in seen_keys:
                continue
            seen_keys.add(combo_key)

            prob = float(probs[i])
            ok_dist = float(cand_dists[i])
            prox = math.exp(-ok_dist / 0.16)
            jitter = (
                float(rng.uniform(-jitter_amp, jitter_amp)) if rng is not None else 0.0
            )
            rank_score = 0.78 * prob + 0.22 * prox + jitter

            candidates.append(
                {
                    "catalog_idx": g_idx,
                    "set_id": entry["set_id"],
                    "style": entry["style"],
                    "has_outerwear": bool(entry["has_outerwear"]),
                    "colors": colors,
                    "secondary_colors": sec_colors,
                    "primary_shares": primary_shares,
                    "color_names": color_names,
                    "titles": titles,
                    "locked_roles": locked_roles,
                    "compat_score": round(prob * 100.0, 1),
                    "oklab_dist": round(ok_dist, 4),
                    "_rank": rank_score,
                    "_swapped_oklab": b_oklab[i],
                    "_slot_mask": b_mask[i],
                    "_orig_oklab": self.slot_oklab[g_idx],
                }
            )

        candidates.sort(key=lambda x: float(x["_rank"]), reverse=True)

        div_weight = 0.26 if is_shuffled else 0.20
        selected: List[Dict[str, object]] = []
        if candidates:
            selected.append(candidates[0])
            remaining = candidates[1:]
            while len(selected) < top_k and remaining:
                best_i = 0
                best_val = -1e9
                for idx_r, cand in enumerate(remaining):
                    min_div = 1e9
                    for sel_item in selected:
                        slot_ds: List[float] = []
                        for role in unlocked_roles:
                            ri = ROLE_TO_IDX[role]
                            c_active = bool(cand["_slot_mask"][ri])
                            s_active = bool(sel_item["_slot_mask"][ri])
                            if c_active and s_active:
                                slot_ds.append(
                                    float(
                                        np.linalg.norm(
                                            cand["_swapped_oklab"][ri]
                                            - sel_item["_swapped_oklab"][ri]
                                        )
                                    )
                                )
                            elif c_active != s_active:
                                slot_ds.append(0.22)
                        d_mean = float(np.mean(slot_ds)) if slot_ds else 0.0
                        if d_mean < min_div:
                            min_div = d_mean

                    div_bonus = div_weight * min(min_div / 0.25, 1.0)
                    dup_pen = 0.0
                    if "bottom" in unlocked_roles and "footwear" in unlocked_roles:
                        if any(
                            (cand["colors"]["bottom"], cand["colors"]["footwear"])
                            == (s["colors"]["bottom"], s["colors"]["footwear"])
                            for s in selected
                        ):
                            dup_pen = 0.07
                    val = float(cand["_rank"]) + div_bonus - dup_pen
                    if val > best_val:
                        best_val = val
                        best_i = idx_r
                selected.append(remaining.pop(best_i))

        clean_options: List[Dict[str, object]] = []
        for item in selected:
            de00_vals = [
                ciede2000_oklab(
                    valid_locks[role]["primary_oklab"],
                    item["_orig_oklab"][ROLE_TO_IDX[role]],
                )
                for role in locked_roles
            ]
            active_oklabs = [
                item["_swapped_oklab"][ri]
                for ri in range(4)
                if bool(item["_slot_mask"][ri])
            ]
            harmony_label = classify_palette_harmony(active_oklabs)

            gated_sec_hex: Dict[str, Optional[str]] = {}
            gated_sec_name: Dict[str, Optional[str]] = {}
            gated_shares: Dict[str, Optional[float]] = {}
            for role in ROLES:
                s_hx, s_name, s_share = _resolve_distinct_secondary(
                    item["colors"].get(role),
                    item["secondary_colors"].get(role),
                    item["primary_shares"].get(role),
                )
                gated_sec_hex[role] = s_hx
                gated_sec_name[role] = s_name
                gated_shares[role] = s_share

            clean_options.append(
                {
                    "set_id": item["set_id"],
                    "style": item["style"],
                    "has_outerwear": item["has_outerwear"],
                    "colors": item["colors"],
                    "secondary_colors": gated_sec_hex,
                    "secondary_color_names": gated_sec_name,
                    "primary_shares": gated_shares,
                    "color_names": item["color_names"],
                    "titles": item["titles"],
                    "locked_roles": item["locked_roles"],
                    "compat_score": item["compat_score"],
                    "delta_e2000": round(float(np.mean(de00_vals)), 2),
                    "harmony_label": harmony_label,
                }
            )

        if len(clean_options) < top_k and filter_notice is None:
            filter_notice = f"Showing all {len(clean_options)} unique palettes matching your active filters."

        mode_notice = (
            "Mixing a photo pin with a manual color pin — scoring by perceptual color harmony so manual shades stay balanced."
            if is_mixed_locks
            else None
        )

        return {
            "options": clean_options,
            "total_matches": len(clean_options),
            "scoring_mode": "multimodal" if use_multimodal else "color_only",
            "mode_notice": mode_notice,
            "filter_notice": filter_notice,
            "pool_size": int(len(pool_indices)),
            "device": str(self.device),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    @torch.inference_mode()
    def audit_outfit(
        self,
        regions: Sequence[Dict[str, object]],
        visual_embeddings: Sequence[np.ndarray],
    ) -> Optional[Dict[str, object]]:
        if len(regions) < 2:
            return None

        n_reg = min(len(regions), 4)
        vis_arr = np.stack(
            [np.asarray(visual_embeddings[i], dtype=np.float32) for i in range(n_reg)],
            axis=0,
        )
        col_arr = np.zeros((n_reg, 13), dtype=np.float32)
        cat_arr = np.zeros((n_reg,), dtype=np.int64)
        sem_arr = np.zeros((n_reg,), dtype=np.int64)
        frm_arr = np.zeros((n_reg,), dtype=np.float32)
        oklabs_list: List[Sequence[float]] = []

        for i in range(n_reg):
            reg = regions[i]
            role = str(reg.get("role") or "top")
            p_ok = reg["oklab"]
            s_ok = reg.get("secondary_oklab") or p_ok
            p_share = float(reg.get("primary_share", 1.0))
            col_arr[i] = encode_garment_color(p_ok, s_ok, p_share)
            cat_arr[i] = self.default_role_cat_id.get(role, 1)
            sem_key = DEFAULT_ROLE_SEM_CAT.get(role, "tops")
            sem_arr[i] = self.sem_cat_to_idx.get(sem_key, 1)
            frm_arr[i] = DEFAULT_ROLE_FORMALITY.get(role, 0.0)
            oklabs_list.append(p_ok)

        vis_t = torch.tensor(
            vis_arr, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        col_t = torch.tensor(
            col_arr, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        cat_t = torch.tensor(cat_arr, dtype=torch.long, device=self.device).unsqueeze(0)
        sem_t = torch.tensor(sem_arr, dtype=torch.long, device=self.device).unsqueeze(0)
        frm_t = torch.tensor(
            frm_arr, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        pm_t = torch.zeros((1, n_reg), dtype=torch.bool, device=self.device)
        hv_t = torch.ones((1, n_reg), dtype=torch.bool, device=self.device)

        o_logit, i_logits = self.model(vis_t, col_t, cat_t, sem_t, frm_t, pm_t, hv_t)
        overall_prob_01 = float(
            _sigmoid_platt(
                o_logit.cpu().numpy(), self.platt_a_multi, self.platt_b_multi
            )[0]
        )
        overall_prob = overall_prob_01 * 100.0
        item_probs = _calibrate_audit_item_scores(
            i_logits[0].cpu().numpy(), overall_prob_01
        ).tolist()

        item_scores: List[Dict[str, object]] = []
        for i in range(n_reg):
            item_scores.append(
                {
                    "role": regions[i]["role"],
                    "label": regions[i]["label"],
                    "hex": regions[i]["hex"],
                    "color_name": regions[i]["color_name"],
                    "fit_score": round(float(item_probs[i]), 1),
                }
            )

        weakest = min(item_scores, key=lambda x: float(x["fit_score"]))
        harmony = classify_palette_harmony(oklabs_list)

        if overall_prob >= 75.0:
            verdict = "Cohesive Ensemble"
            advice = (
                f"Strong {harmony.lower()} pairing ({overall_prob:.1f}% compatibility). "
                "Pin any piece below to explore alternate combinations."
            )
        elif overall_prob >= 55.0:
            verdict = "Balanced — Room to Refine"
            advice = (
                f"Moderate cohesion ({overall_prob:.1f}%). Consider pinning your strongest pieces and "
                f"refreshing the {weakest['label']} ({weakest['color_name']}, fit {weakest['fit_score']}%)."
            )
        else:
            verdict = "Style or Color Clash Detected"
            advice = (
                f"Low outfit compatibility ({overall_prob:.1f}%). The {weakest['label']} "
                f"({weakest['color_name']}, fit {weakest['fit_score']}%) is the weakest link — "
                "pin another garment to generate harmonized replacements."
            )

        return {
            "overall_compat_pct": round(overall_prob, 1),
            "verdict": verdict,
            "harmony_label": harmony,
            "advice": advice,
            "weakest_role": weakest["role"],
            "weakest_label": weakest["label"],
            "item_scores": item_scores,
        }
