import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.color_math import encode_garment_color, hex_to_oklab
from src.config import ARTIFACTS_DIR, SEED, get_device
from src.model import HueFitNetV2


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def fit_platt_scaling(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    lt = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(labels, dtype=torch.float64)
    log_a = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.LBFGS(
        [log_a, b], lr=0.5, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        a = torch.exp(log_a)
        loss = F.binary_cross_entropy_with_logits(a * lt + b, yt)
        loss.backward()
        return loss

    opt.step(closure)
    a_val = float(torch.exp(log_a).item())
    b_val = float(b.item())
    return round(a_val, 5), round(b_val, 5)


def apply_platt_probs(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.clip(a * logits + b, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def tune_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    best_th = 0.50
    best_score = -1.0
    for th in np.linspace(0.35, 0.65, 61):
        preds = (probs >= th).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        score = 0.5 * f1 + 0.5 * acc
        if score > best_score:
            best_score = score
            best_th = float(th)
    return round(best_th, 4)


def compute_metrics(
    name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    platt_a: float,
    platt_b: float,
    thresh: float,
) -> Dict[str, object]:
    raw_probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    cal_probs = apply_platt_probs(logits, platt_a, platt_b)
    preds = (cal_probs >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "model": name,
        "platt_a": round(platt_a, 5),
        "platt_b": round(platt_b, 5),
        "threshold": round(thresh, 4),
        "roc_auc": round(float(roc_auc_score(labels, cal_probs)), 4),
        "pr_auc": round(float(average_precision_score(labels, cal_probs)), 4),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "specificity": round(float(tn / max(1, tn + fp)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(labels, preds)), 4),
        "brier": round(float(brier_score_loss(labels, cal_probs)), 4),
        "ece_raw": round(compute_ece(raw_probs, labels), 4),
        "ece_cal": round(compute_ece(cal_probs, labels), 4),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def build_epoch_samples(
    pos_idx: np.ndarray,
    item_cat_id: np.ndarray,
    pool_by_cat: Dict[int, np.ndarray],
    all_train_rows: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pos, max_len = pos_idx.shape
    neg_idx = pos_idx.copy()
    neg_item_y = np.zeros((n_pos, max_len), dtype=np.float32)

    is_single_swap = rng.random(n_pos) < 0.50

    for i in range(n_pos):
        row = pos_idx[i]
        valid_positions = np.where(row > 0)[0]
        n_valid = len(valid_positions)
        if n_valid < 2:
            continue

        if is_single_swap[i]:
            neg_item_y[i, valid_positions] = 1.0
            swap_pos = int(valid_positions[rng.integers(0, n_valid)])
            orig_item = int(row[swap_pos])
            cat = int(item_cat_id[orig_item])
            pool = pool_by_cat.get(cat, all_train_rows)
            cand = int(pool[rng.integers(0, len(pool))])
            if cand == orig_item and len(pool) > 1:
                cand = int(pool[rng.integers(0, len(pool))])
            neg_idx[i, swap_pos] = cand
            neg_item_y[i, swap_pos] = 0.0
        else:
            for pos_j in valid_positions:
                orig_item = int(row[pos_j])
                cat = int(item_cat_id[orig_item])
                pool = pool_by_cat.get(cat, all_train_rows)
                cand = int(pool[rng.integers(0, len(pool))])
                if cand == orig_item and len(pool) > 1:
                    cand = int(pool[rng.integers(0, len(pool))])
                neg_idx[i, pos_j] = cand

    pos_item_y = (pos_idx > 0).astype(np.float32)
    all_idx = np.concatenate([pos_idx, neg_idx], axis=0)
    all_y = np.concatenate(
        [np.ones(n_pos, dtype=np.float32), np.zeros(n_pos, dtype=np.float32)], axis=0
    )
    all_item_y = np.concatenate([pos_item_y, neg_item_y], axis=0)

    perm = rng.permutation(len(all_y))
    return all_idx[perm], all_y[perm], all_item_y[perm]


def sample_modality_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    B, L = pad_mask.shape
    device = pad_mask.device
    u = torch.rand(B, device=device)
    has_vis = torch.zeros((B, L), dtype=torch.bool, device=device)

    full_vis = (u >= 0.45) & (u < 0.90)
    has_vis[full_vis] = True

    partial_idx = torch.where(u >= 0.90)[0]
    if len(partial_idx) > 0:
        has_vis[partial_idx] = True
        valid_counts = (~pad_mask[partial_idx]).sum(dim=1).clamp(min=1)
        drop_pos = (
            torch.rand(len(partial_idx), device=device) * valid_counts.float()
        ).long()
        has_vis[partial_idx, drop_pos] = False

    return has_vis & (~pad_mask)


class FeatureStoreGPU:
    def __init__(self, ds, item_emb_np: np.ndarray, device: torch.device) -> None:
        self.device = device
        self.emb = torch.tensor(item_emb_np, dtype=torch.float16, device=device)
        self.color = torch.tensor(
            ds["item_color_feats"], dtype=torch.float32, device=device
        )
        self.cat = torch.tensor(ds["item_cat_id"], dtype=torch.long, device=device)
        self.sem = torch.tensor(ds["item_sem_cat"], dtype=torch.long, device=device)
        self.form = torch.tensor(
            ds["item_formality"], dtype=torch.float32, device=device
        )

    def gather(self, b_idx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        b_idx = b_idx.to(self.device, non_blocking=True)
        pad_mask = b_idx == 0
        vis = self.emb[b_idx].float()
        col = self.color[b_idx]
        cat = self.cat[b_idx]
        sem = self.sem[b_idx]
        frm = self.form[b_idx]
        return vis, col, cat, sem, frm, pad_mask


@torch.inference_mode()
def predict_compatibility_logits(
    model: HueFitNetV2,
    store: FeatureStoreGPU,
    idx_arr: np.ndarray,
    use_visual: bool,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(idx_arr, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    outs: List[np.ndarray] = []
    for (b_idx,) in loader:
        vis, col, cat, sem, frm, pm = store.gather(b_idx)
        hv = (~pm) if use_visual else torch.zeros_like(pm)
        o_logit, _ = model(vis, col, cat, sem, frm, pm, hv)
        outs.append(o_logit.cpu().numpy())
    return np.concatenate(outs)


@torch.inference_mode()
def evaluate_fitb(
    model: HueFitNetV2,
    store: FeatureStoreGPU,
    q_arr: np.ndarray,
    a_arr: np.ndarray,
    gt_arr: np.ndarray,
    use_visual: bool,
    batch_size: int = 256,
) -> float:
    model.eval()
    n_q = len(gt_arr)
    correct = 0

    for start in range(0, n_q, batch_size):
        end = min(n_q, start + batch_size)
        bq = q_arr[start:end]
        ba = a_arr[start:end]
        bgt = gt_arr[start:end]
        bs = len(bgt)

        cand_seqs = np.zeros((bs, 4, bq.shape[1] + 1), dtype=np.int64)
        cand_seqs[:, :, 0] = ba
        cand_seqs[:, :, 1:] = bq[:, None, :]
        flat_idx = torch.tensor(cand_seqs.reshape(bs * 4, -1), dtype=torch.long)

        vis, col, cat, sem, frm, pm = store.gather(flat_idx)
        hv = (~pm) if use_visual else torch.zeros_like(pm)
        o_logit, i_logits = model(vis, col, cat, sem, frm, pm, hv)
        scores = (o_logit + 0.35 * i_logits[:, 0]).view(bs, 4).cpu().numpy()
        preds = np.argmax(scores, axis=1)
        correct += int((preds == bgt).sum())

    return round(correct / max(1, n_q), 4)


@torch.inference_mode()
def extract_tabular_summary(
    store: FeatureStoreGPU,
    idx_arr: np.ndarray,
    use_visual: bool,
    batch_size: int = 512,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(idx_arr, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    feats: List[np.ndarray] = []
    for (b_idx,) in loader:
        vis, col, _, _, frm, pm = store.gather(b_idx)
        hv = (~pm) if use_visual else torch.zeros_like(pm)
        s = HueFitNetV2._compute_set_summary(vis, col, frm, pm, hv)
        feats.append(s.cpu().numpy())
    return np.concatenate(feats, axis=0)


def run_lightgbm_baselines(
    store: FeatureStoreGPU,
    idx_train: np.ndarray,
    y_train: np.ndarray,
    idx_valid: np.ndarray,
    y_valid: np.ndarray,
    idx_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for mode_key, use_vis, label in [
        ("lgbm_multimodal", True, "LightGBM Baseline (Multimodal)"),
        ("lgbm_color_only", False, "LightGBM Baseline (Color-Only)"),
    ]:
        X_tr = extract_tabular_summary(store, idx_train, use_visual=use_vis)
        X_va = extract_tabular_summary(store, idx_valid, use_visual=use_vis)
        X_te = extract_tabular_summary(store, idx_test, use_visual=use_vis)

        clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1,
        )
        clf.fit(
            X_tr,
            y_train,
            eval_set=[(X_va, y_valid)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        va_probs = np.clip(clf.predict_proba(X_va)[:, 1], 1e-6, 1.0 - 1e-6)
        te_probs = np.clip(clf.predict_proba(X_te)[:, 1], 1e-6, 1.0 - 1e-6)
        va_logits = np.log(va_probs / (1.0 - va_probs))
        te_logits = np.log(te_probs / (1.0 - te_probs))

        a_opt, b_opt = fit_platt_scaling(va_logits, y_valid)
        va_cal = apply_platt_probs(va_logits, a_opt, b_opt)
        th_opt = tune_threshold(va_cal, y_valid)
        results[mode_key] = compute_metrics(
            label, te_logits, y_test, a_opt, b_opt, th_opt
        )
    return results


@torch.inference_mode()
def run_color_swap_sensitivity_test(
    model: HueFitNetV2,
    device: torch.device,
    platt_a: float,
    platt_b: float,
    n_samples: int = 500,
) -> Dict[str, object]:
    model.eval()
    cat_t = np.load(ARTIFACTS_DIR / "catalog_tensors.npz")
    n_total = len(cat_t["slot_color_feats"])
    n = min(n_samples, n_total)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(n_total, size=n, replace=False)

    col_orig = cat_t["slot_color_feats"][sel].copy()
    cats = torch.tensor(cat_t["slot_cat_ids"][sel], dtype=torch.long, device=device)
    sems = torch.tensor(cat_t["slot_sem_cats"][sel], dtype=torch.long, device=device)
    frms = torch.tensor(
        cat_t["slot_formality"][sel], dtype=torch.float32, device=device
    )
    pm = torch.tensor(~cat_t["slot_mask"][sel], dtype=torch.bool, device=device)
    hv = torch.zeros_like(pm)
    vis_zero = torch.zeros((n, 4, model.visual_dim), dtype=torch.float32, device=device)

    orig_logits, _ = model(
        vis_zero,
        torch.tensor(col_orig, dtype=torch.float32, device=device),
        cats,
        sems,
        frms,
        pm,
        hv,
    )
    orig_probs = apply_platt_probs(orig_logits.cpu().numpy(), platt_a, platt_b) * 100.0

    clash_results: Dict[str, Dict[str, object]] = {}
    for cname, hx in [
        ("Neon Lime", "#00FF00"),
        ("Neon Magenta", "#FF00FF"),
        ("Safety Orange", "#FF6600"),
    ]:
        ok = hex_to_oklab(hx)
        swapped_vec = encode_garment_color(ok, ok, 1.0)
        col_swap = col_orig.copy()
        col_swap[:, 0, :] = swapped_vec[None, :]
        swap_logits, _ = model(
            vis_zero,
            torch.tensor(col_swap, dtype=torch.float32, device=device),
            cats,
            sems,
            frms,
            pm,
            hv,
        )
        swap_probs = (
            apply_platt_probs(swap_logits.cpu().numpy(), platt_a, platt_b) * 100.0
        )
        delta = orig_probs - swap_probs
        clash_results[cname] = {
            "hex": hx,
            "mean_orig_pct": round(float(orig_probs.mean()), 2),
            "mean_swapped_pct": round(float(swap_probs.mean()), 2),
            "mean_drop_pp": round(float(delta.mean()), 2),
            "median_drop_pp": round(float(np.median(delta)), 2),
            "pct_penalized": round(float((delta > 0).mean() * 100.0), 1),
        }
    return {"n_outfits_tested": n, "clashes": clash_results}


def save_eval_figure(
    hist: Dict[str, object],
    y_test: np.ndarray,
    probs_multi: np.ndarray,
    probs_color: np.ndarray,
    m_multi: Dict[str, object],
    m_color: Dict[str, object],
    m_lgb_multi: Dict[str, object],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))

    axes[0, 0].plot(
        hist["val_auc_multi"],
        label="Val AUC (Multimodal)",
        color="#10b981",
        linewidth=2.2,
    )
    axes[0, 0].plot(
        hist["val_auc_color"],
        label="Val AUC (Color-Only)",
        color="#3b82f6",
        linewidth=2.2,
    )
    axes[0, 0].axvline(
        hist["best_epoch"] - 1,
        color="#f59e0b",
        linestyle="--",
        label=f"Best Ep {hist['best_epoch']}",
    )
    axes[0, 0].set_title("HueFitNet v2 Validation ROC-AUC Progression")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.25)

    fpr_m, tpr_m, _ = roc_curve(y_test, probs_multi)
    fpr_c, tpr_c, _ = roc_curve(y_test, probs_color)
    axes[0, 1].plot(
        fpr_m,
        tpr_m,
        color="#10b981",
        linewidth=2.3,
        label=f"HueFitNet v2 Multimodal (AUC={m_multi['roc_auc']:.4f})",
    )
    axes[0, 1].plot(
        fpr_c,
        tpr_c,
        color="#3b82f6",
        linewidth=2.1,
        label=f"HueFitNet v2 Color-Only (AUC={m_color['roc_auc']:.4f})",
    )
    axes[0, 1].plot(
        [0, 1],
        [0, 1],
        color="#64748b",
        linestyle="--",
        label=f"LightGBM Multi AUC={m_lgb_multi['roc_auc']:.4f}",
    )
    axes[0, 1].set_title(f"Unseen Disjoint Test ROC ({len(y_test):,} Outfits)")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(True, alpha=0.25)

    bins = np.linspace(0.0, 1.0, 11)
    for probs, col_hex, lbl in [
        (probs_multi, "#10b981", f"Multimodal (ECE={m_multi['ece_cal']:.4f})"),
        (probs_color, "#3b82f6", f"Color-Only (ECE={m_color['ece_cal']:.4f})"),
    ]:
        xs, ys = [], []
        for i in range(10):
            mask = (probs >= bins[i]) & (
                probs <= bins[i + 1] if i == 9 else probs < bins[i + 1]
            )
            if mask.sum() >= 20:
                xs.append(float(probs[mask].mean()))
                ys.append(float(y_test[mask].mean()))
        axes[1, 0].plot(xs, ys, marker="o", linewidth=2, color=col_hex, label=lbl)
    axes[1, 0].plot(
        [0, 1], [0, 1], linestyle="--", color="#64748b", label="Perfect Calibration"
    )
    axes[1, 0].set_title("Post-Platt Calibration Reliability Diagram")
    axes[1, 0].set_xlabel("Mean Predicted Probability")
    axes[1, 0].set_ylabel("Empirical Positive Fraction")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(True, alpha=0.25)

    cm = np.array(m_multi["cm"], dtype=np.int64)
    cm_pct = cm / np.maximum(1, cm.sum(axis=1, keepdims=True)) * 100.0
    axes[1, 1].imshow(cm, cmap="Blues")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(["Incompatible", "Compatible"])
    axes[1, 1].set_yticklabels(["Incompatible", "Compatible"])
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")
    axes[1, 1].set_title(f"Multimodal Confusion Matrix (@ {m_multi['threshold']:.2f})")
    for r in range(2):
        for c in range(2):
            axes[1, 1].text(
                c,
                r,
                f"{cm[r, c]:,}\n({cm_pct[r, c]:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[r, c] > cm.max() * 0.55 else "#0f172a",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HueFitNet v2 Training & Platt Calibration"
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Skip training and apply 2-parameter Platt calibration to existing artifacts/huefit_v2.pt.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    set_seed(SEED)
    device = get_device()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    mode_tag = (
        "Platt Calibration Refresh"
        if args.calibrate_only
        else "Training, Calibration & Full Evaluation"
    )
    print(f"=== HueFit v2 — Step 4: {mode_tag} ({gpu_name}) ===")

    ds = np.load(ARTIFACTS_DIR / "training_dataset.npz")
    item_emb_np = np.load(ARTIFACTS_DIR / "item_embeddings.npy")
    with open(ARTIFACTS_DIR / "item_mappings.json", "r", encoding="utf-8") as fh:
        mappings = json.load(fh)

    num_cats = int(ds["num_cats"])
    num_sem_cats = int(ds["num_sem_cats"])
    idx_train, y_train = ds["idx_train"], ds["y_train"]
    idx_valid, y_valid = ds["idx_valid"], ds["y_valid"]
    idx_test, y_test = ds["idx_test"], ds["y_test"]
    item_cat_id = ds["item_cat_id"]

    store = FeatureStoreGPU(ds, item_emb_np, device)
    ckpt_path = ARTIFACTS_DIR / "huefit_v2.pt"
    rep_path = ARTIFACTS_DIR / "evaluation_report.json"

    if args.calibrate_only:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = HueFitNetV2(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        with open(rep_path, "r", encoding="utf-8") as fh:
            prev_report = json.load(fh)
        hist = prev_report["history"]
        lgbm_metrics = {
            "lgbm_multimodal": prev_report["test_compatibility"]["lgbm_multimodal"],
            "lgbm_color_only": prev_report["test_compatibility"]["lgbm_color_only"],
        }
        fitb_multi = prev_report["test_fitb"]["multimodal_top1"]
        fitb_color = prev_report["test_fitb"]["color_only_top1"]
        print(
            f"Loaded existing checkpoint from {ckpt_path.name} (Best Epoch {hist['best_epoch']})"
        )
    else:
        pos_train_idx = idx_train[y_train == 1.0]
        pool_by_cat = {
            int(k): np.asarray(v, dtype=np.int32)
            for k, v in mappings["train_pool_by_cat"].items()
            if len(v) > 0
        }
        all_train_rows = np.concatenate(list(pool_by_cat.values()))

        print(
            f"Splits -> Train Pos: {len(pos_train_idx):,} (dynamic 1:1 negs/ep) | "
            f"Valid: {len(y_valid):,} | Test: {len(y_test):,} | FITB Test: {len(ds['fitb_gt_test']):,}"
        )

        model = HueFitNetV2(
            visual_dim=item_emb_np.shape[1],
            color_dim=13,
            num_cats=num_cats,
            num_sem_cats=num_sem_cats,
            d_model=160,
            nhead=4,
            num_layers=2,
            dropout=0.25,
            visual_dropout=0.35,
            visual_noise_std=0.02,
        ).to(device)
        print(
            f"HueFitNetV2 trainable parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        epochs = 28
        warmup_epochs = 2
        batch_size = 256
        label_smooth = 0.03
        item_loss_weight = 0.35

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        def lr_lambda(ep: int) -> float:
            if ep < warmup_epochs:
                return float(ep + 1) / float(warmup_epochs)
            prog = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        rng = np.random.default_rng(SEED)
        hist = {
            "train_loss": [],
            "val_auc_multi": [],
            "val_auc_color": [],
            "best_epoch": 1,
        }
        best_combined_auc = 0.0
        best_state = None
        patience = 7
        patience_left = patience

        for epoch in range(1, epochs + 1):
            ep_idx, ep_y, ep_item_y = build_epoch_samples(
                pos_train_idx, item_cat_id, pool_by_cat, all_train_rows, rng
            )
            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(ep_idx, dtype=torch.long),
                    torch.tensor(ep_y, dtype=torch.float32),
                    torch.tensor(ep_item_y, dtype=torch.float32),
                ),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )

            model.train()
            tr_loss_sum = 0.0
            n_seen = 0
            for b_idx, b_y, b_iy in train_loader:
                vis, col, cat, sem, frm, pm = store.gather(b_idx)
                b_y = b_y.to(device, non_blocking=True)
                b_iy = b_iy.to(device, non_blocking=True)
                hv = sample_modality_mask(pm)

                y_sm = b_y * (1.0 - label_smooth) + 0.5 * label_smooth
                iy_sm = b_iy * (1.0 - label_smooth) + 0.5 * label_smooth

                optimizer.zero_grad(set_to_none=True)
                o_logit, i_logits = model(vis, col, cat, sem, frm, pm, hv)
                loss_o = F.binary_cross_entropy_with_logits(o_logit, y_sm)
                valid_f = (~pm).float()
                loss_i = (
                    F.binary_cross_entropy_with_logits(
                        i_logits, iy_sm, reduction="none"
                    )
                    * valid_f
                ).sum() / valid_f.sum().clamp(min=1.0)
                loss = loss_o + item_loss_weight * loss_i

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                tr_loss_sum += loss.item() * len(b_y)
                n_seen += len(b_y)

            scheduler.step()
            tr_loss = tr_loss_sum / max(1, n_seen)

            val_logits_multi = predict_compatibility_logits(
                model, store, idx_valid, use_visual=True
            )
            val_logits_color = predict_compatibility_logits(
                model, store, idx_valid, use_visual=False
            )
            val_auc_multi = float(roc_auc_score(y_valid, val_logits_multi))
            val_auc_color = float(roc_auc_score(y_valid, val_logits_color))
            combined = 0.60 * val_auc_multi + 0.40 * val_auc_color

            hist["train_loss"].append(round(tr_loss, 5))
            hist["val_auc_multi"].append(round(val_auc_multi, 5))
            hist["val_auc_color"].append(round(val_auc_color, 5))

            if combined > best_combined_auc + 1e-4:
                best_combined_auc = combined
                hist["best_epoch"] = epoch
                patience_left = patience
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d}/{epochs} | LR: {lr_now:.5f} | Train Loss: {tr_loss:.4f} | "
                f"Val AUC Multi: {val_auc_multi:.4f} | Val AUC Color: {val_auc_color:.4f}"
            )
            if patience_left == 0:
                print(
                    f"Early stopping at Epoch {epoch} (Best combined Val AUC at Epoch {hist['best_epoch']})"
                )
                break

        model.load_state_dict(best_state)
        model.to(device).eval()
        lgbm_metrics = run_lightgbm_baselines(
            store, idx_train, y_train, idx_valid, y_valid, idx_test, y_test
        )
        fitb_multi = evaluate_fitb(
            model,
            store,
            ds["fitb_q_test"],
            ds["fitb_a_test"],
            ds["fitb_gt_test"],
            use_visual=True,
        )
        fitb_color = evaluate_fitb(
            model,
            store,
            ds["fitb_q_test"],
            ds["fitb_a_test"],
            ds["fitb_gt_test"],
            use_visual=False,
        )

    print(
        "\nFitting 2-parameter Platt calibration (a * logit + b) strictly on validation split ..."
    )
    val_logits_multi = predict_compatibility_logits(
        model, store, idx_valid, use_visual=True
    )
    val_logits_color = predict_compatibility_logits(
        model, store, idx_valid, use_visual=False
    )

    a_multi, b_multi = fit_platt_scaling(val_logits_multi, y_valid)
    a_color, b_color = fit_platt_scaling(val_logits_color, y_valid)

    val_probs_multi = apply_platt_probs(val_logits_multi, a_multi, b_multi)
    val_probs_color = apply_platt_probs(val_logits_color, a_color, b_color)
    thresh_multi = tune_threshold(val_probs_multi, y_valid)
    thresh_color = tune_threshold(val_probs_color, y_valid)

    torch.save(
        {
            "state_dict": best_state,
            "config": model.get_config(),
            "platt_a_multimodal": a_multi,
            "platt_b_multimodal": b_multi,
            "platt_a_color": a_color,
            "platt_b_color": b_color,
            "temp_multimodal": round(1.0 / max(1e-6, a_multi), 4),
            "temp_color": round(1.0 / max(1e-6, a_color), 4),
            "thresh_multimodal": thresh_multi,
            "thresh_color": thresh_color,
            "best_epoch": int(hist["best_epoch"]),
        },
        ckpt_path,
    )
    print(
        f"Updated checkpoint -> {ckpt_path} | "
        f"Multi(a={a_multi}, b={b_multi}, th={thresh_multi}) | "
        f"Color(a={a_color}, b={b_color}, th={thresh_color})"
    )

    print("\n=== Unseen Disjoint Test Evaluation (30,290 Outfits) ===")
    test_logits_multi = predict_compatibility_logits(
        model, store, idx_test, use_visual=True
    )
    test_logits_color = predict_compatibility_logits(
        model, store, idx_test, use_visual=False
    )

    m_multi = compute_metrics(
        "HueFitNet v2 (Multimodal)",
        test_logits_multi,
        y_test,
        a_multi,
        b_multi,
        thresh_multi,
    )
    m_color = compute_metrics(
        "HueFitNet v2 (Color-Only)",
        test_logits_color,
        y_test,
        a_color,
        b_color,
        thresh_color,
    )

    print("\nModel Comparison (Unseen Polyvore Disjoint Test, n=30,290):")
    header = f"{'Model':<32s} | {'ROC-AUC':>7s} | {'PR-AUC':>7s} | {'Acc':>7s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'ECE':>6s}"
    print(header)
    print("-" * len(header))
    for m in [
        m_multi,
        m_color,
        lgbm_metrics["lgbm_multimodal"],
        lgbm_metrics["lgbm_color_only"],
    ]:
        print(
            f"{m['model']:<32s} | {m['roc_auc']:7.4f} | {m['pr_auc']:7.4f} | "
            f"{m['accuracy']*100:6.2f}% | {m['precision']:6.4f} | {m['recall']:6.4f} | "
            f"{m['f1']:6.4f} | {m['ece_cal']:6.4f}"
        )

    print(
        "\n=== Fill-In-The-Blank (FITB) Evaluation (15,145 Unseen Test Questions, Random=25.0%) ==="
    )
    print(f"  FITB Top-1 Accuracy (Multimodal) : {fitb_multi * 100:.2f}%")
    print(f"  FITB Top-1 Accuracy (Color-Only) : {fitb_color * 100:.2f}%")

    print(
        "\n=== Color-Swap Sensitivity Audit (Calibrated Color-Wheel Mode, 500 Real Outfits) ==="
    )
    swap_audit = run_color_swap_sensitivity_test(
        model, device, a_color, b_color, n_samples=500
    )
    for cname, cr in swap_audit["clashes"].items():
        print(
            f"  Swap Top -> {cname:<14s} ({cr['hex']}) | "
            f"Orig: {cr['mean_orig_pct']:5.2f}% -> Swapped: {cr['mean_swapped_pct']:5.2f}% | "
            f"Mean drop: {cr['mean_drop_pp']:+6.2f} pp | Penalized: {cr['pct_penalized']:.1f}%"
        )

    test_probs_multi = apply_platt_probs(test_logits_multi, a_multi, b_multi)
    test_probs_color = apply_platt_probs(test_logits_color, a_color, b_color)
    fig_path = ARTIFACTS_DIR / "training_eval_v2.png"
    save_eval_figure(
        hist,
        y_test,
        test_probs_multi,
        test_probs_color,
        m_multi,
        m_color,
        lgbm_metrics["lgbm_multimodal"],
        fig_path,
    )

    report = {
        "hardware": gpu_name,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "history": hist,
        "test_compatibility": {
            "huefit_v2_multimodal": m_multi,
            "huefit_v2_color_only": m_color,
            **lgbm_metrics,
        },
        "test_fitb": {
            "multimodal_top1": fitb_multi,
            "color_only_top1": fitb_color,
            "random_baseline": 0.25,
            "num_questions": int(len(ds["fitb_gt_test"])),
        },
        "color_swap_sensitivity": swap_audit,
    }
    with open(rep_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nSaved evaluation report -> {rep_path}")
    print(f"Saved diagnostic plots  -> {fig_path}")
    print(f"Total runtime           : {(time.perf_counter() - t0):.1f} sec")


if __name__ == "__main__":
    main()
