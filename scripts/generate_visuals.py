import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ARTIFACTS_DIR, get_device

FIGURES_DIR = ROOT_DIR / "docs" / "figures"

BG = "#F5F3EE"
CARD = "#FFFFFF"
BORDER = "#D6D0C4"
INK = "#161616"
MUTED = "#6E6960"
NAVY = "#1D3557"
FOREST = "#1E4231"
CAMEL = "#C19A6B"
TERRA = "#C25934"
BURGUNDY = "#6E2632"
SLATE = "#64748B"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": CARD,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#E5E0D6",
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": True,
            "legend.facecolor": CARD,
            "legend.edgecolor": BORDER,
        }
    )


def _compute_exact_test_probs() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ds_path = ARTIFACTS_DIR / "training_dataset.npz"
    emb_path = ARTIFACTS_DIR / "item_embeddings.npy"
    ckpt_path = ARTIFACTS_DIR / "huefit_v2.pt"
    if not (ds_path.exists() and emb_path.exists() and ckpt_path.exists()):
        return None

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.model import HueFitNetV2

    device = get_device()
    ds = np.load(ds_path)
    emb_np = np.load(emb_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = HueFitNetV2(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    emb_t = torch.tensor(emb_np, dtype=torch.float16, device=device)
    col_t = torch.tensor(ds["item_color_feats"], dtype=torch.float32, device=device)
    cat_t = torch.tensor(ds["item_cat_id"], dtype=torch.long, device=device)
    sem_t = torch.tensor(ds["item_sem_cat"], dtype=torch.long, device=device)
    frm_t = torch.tensor(ds["item_formality"], dtype=torch.float32, device=device)

    idx_test = ds["idx_test"]
    y_test = ds["y_test"]

    a_m = float(ckpt.get("platt_a_multimodal", 1.0))
    b_m = float(ckpt.get("platt_b_multimodal", 0.0))
    a_c = float(ckpt.get("platt_a_color", 1.0))
    b_c = float(ckpt.get("platt_b_color", 0.0))

    loader = DataLoader(
        TensorDataset(torch.tensor(idx_test, dtype=torch.long)),
        batch_size=512,
        shuffle=False,
    )
    logits_m, logits_c = [], []

    with torch.inference_mode():
        for (b_idx,) in loader:
            b_idx = b_idx.to(device)
            pm = b_idx == 0
            vis = emb_t[b_idx].float()
            col = col_t[b_idx]
            cat = cat_t[b_idx]
            sem = sem_t[b_idx]
            frm = frm_t[b_idx]

            om, _ = model(vis, col, cat, sem, frm, pm, ~pm)
            oc, _ = model(vis, col, cat, sem, frm, pm, torch.zeros_like(pm))
            logits_m.append(om.cpu().numpy())
            logits_c.append(oc.cpu().numpy())

    lm = np.concatenate(logits_m)
    lc = np.concatenate(logits_c)
    pm_cal = 1.0 / (1.0 + np.exp(-np.clip(a_m * lm + b_m, -30.0, 30.0)))
    pc_cal = 1.0 / (1.0 + np.exp(-np.clip(a_c * lc + b_c, -30.0, 30.0)))
    return y_test, pm_cal, pc_cal


def _draw_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    facecolor=CARD,
    edgecolor=INK,
    title_color=INK,
    sub_color=MUTED,
    lw=1.4,
):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.5,
        color=sub_color,
    )


def _draw_arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.6,head_width=0.35", color=color, lw=1.6
        ),
    )


def fig_01_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.94,
        "HueFit — Multimodal & Perceptual Color AI Architecture",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.03,
        0.895,
        "Variable-Length Set Transformer (661,850 params) trained on Polyvore Outfits Disjoint with Whole-Outfit Modality Dropout",
        fontsize=10.5,
        color=MUTED,
    )

    _draw_box(
        ax,
        0.03,
        0.62,
        0.20,
        0.19,
        "Photo Input Path",
        "SegFormer-B2 (256x256)\nBiRefNet_lite + Flood-Fill\nMarqo-FashionSigLIP (768-d)",
        facecolor=CARD,
        edgecolor=NAVY,
        lw=1.8,
    )
    _draw_box(
        ax,
        0.03,
        0.36,
        0.20,
        0.19,
        "Color Studio Path",
        "2D HSV -> Perceptual Oklab\nChroma-Gated Hue (13-d)\nZero Visual Leakage (hv = 0)",
        facecolor=CARD,
        edgecolor=FOREST,
        lw=1.8,
    )
    _draw_box(
        ax,
        0.03,
        0.10,
        0.20,
        0.19,
        "Training Strategy",
        "16,995 Pos + Dynamic 1:1 Neg\n50% All-Swap / 50% 1-Swap\n45% Color / 45% Multi / 10% Mix",
        facecolor="#ECE8E0",
        edgecolor=BORDER,
        lw=1.4,
    )

    _draw_box(
        ax,
        0.29,
        0.62,
        0.21,
        0.19,
        "Visual Bottleneck",
        "768 -> 128 + LayerNorm\nDropout(0.35) + Noise(σ=0.02)\nStrictly Gated by has_visual",
        facecolor=CARD,
        edgecolor=NAVY,
    )
    _draw_box(
        ax,
        0.29,
        0.36,
        0.21,
        0.19,
        "Color & Metadata",
        "Color MLP (13 -> 64 -> 64)\nFine Cat (24) + Sem Cat (16)\nFormality + Mode Gate (16)",
        facecolor=CARD,
        edgecolor=FOREST,
    )
    _draw_box(
        ax,
        0.29,
        0.10,
        0.21,
        0.19,
        "Set Summary Skip (20-d)",
        "12 Oklab Pairwise Stats\n5 Gated Cosine Stats\n3 Formality & Size Stats",
        facecolor=CARD,
        edgecolor=CAMEL,
    )

    _draw_box(
        ax,
        0.56,
        0.38,
        0.18,
        0.43,
        "HueFitNetV2\nSet Transformer",
        "[CLS] + 2-12 Garment Tokens\n2 Encoder Layers (d=160)\n4 Attention Heads · FFN 320\n\nCLS + Mean + Max Pooling",
        facecolor=NAVY,
        edgecolor=NAVY,
        title_color="#F5F3EE",
        sub_color="#DCE5F2",
        lw=1.8,
    )
    _draw_box(
        ax,
        0.56,
        0.10,
        0.18,
        0.20,
        "Platt Calibration",
        "2-Param L-BFGS on Valid\nSeparate Multi & Color\nAligned 0-100% Scale",
        facecolor="#ECE8E0",
        edgecolor=BORDER,
    )

    _draw_box(
        ax,
        0.79,
        0.62,
        0.18,
        0.19,
        "Outfit Head",
        "ROC-AUC: 0.9005 (Multi)\nROC-AUC: 0.8350 (Color)\nCalibrated Cohesion %",
        facecolor=FOREST,
        edgecolor=FOREST,
        title_color="#F5F3EE",
        sub_color="#D8EDE2",
        lw=1.6,
    )
    _draw_box(
        ax,
        0.79,
        0.36,
        0.18,
        0.19,
        "1-Item MISFIT Head",
        "FITB Top-1: 60.79% (Multi)\nFITB Top-1: 50.96% (Color)\nPowers Live Outfit Audit",
        facecolor=TERRA,
        edgecolor=TERRA,
        title_color="#F5F3EE",
        sub_color="#FAE3DA",
        lw=1.6,
    )
    _draw_box(
        ax,
        0.79,
        0.10,
        0.18,
        0.19,
        "Live Retrieval Engine",
        "5,318 Clean Real Outfits\n80 Oklab + 40 Neutral Pool\nSwapped-Oklab MMR (~40 ms)",
        facecolor=CARD,
        edgecolor=INK,
        lw=1.6,
    )

    _draw_arrow(ax, 0.23, 0.715, 0.29, 0.715, NAVY)
    _draw_arrow(ax, 0.23, 0.455, 0.29, 0.455, FOREST)
    _draw_arrow(ax, 0.23, 0.43, 0.29, 0.22, CAMEL)
    _draw_arrow(ax, 0.50, 0.715, 0.56, 0.66, NAVY)
    _draw_arrow(ax, 0.50, 0.455, 0.56, 0.53, FOREST)
    _draw_arrow(ax, 0.50, 0.195, 0.79, 0.69, CAMEL)
    _draw_arrow(ax, 0.74, 0.67, 0.79, 0.715, FOREST)
    _draw_arrow(ax, 0.74, 0.50, 0.79, 0.455, TERRA)
    _draw_arrow(ax, 0.65, 0.38, 0.65, 0.30, MUTED)
    _draw_arrow(ax, 0.74, 0.195, 0.79, 0.195, INK)

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=300, facecolor=BG)
    plt.close(fig)


def fig_02_training_dynamics(report: Dict, out_path: Path) -> None:
    hist = report["history"]
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    best_ep = int(hist["best_epoch"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    fig.suptitle(
        "HueFitNetV2 Training Dynamics (28 Epochs · AdamW + Cosine Annealing)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax1.plot(
        epochs,
        hist["train_loss"],
        color=TERRA,
        lw=2.4,
        marker="o",
        markersize=4,
        label="Joint BCE Loss (Outfit + 0.35·Item)",
    )
    ax1.axvline(
        best_ep,
        color=CAMEL,
        ls="--",
        lw=1.6,
        label=f"Best Checkpoint (Epoch {best_ep})",
    )
    ax1.set_title("Training Objective Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (Label Smoothing = 0.03)")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(
        epochs,
        hist["val_auc_multi"],
        color=NAVY,
        lw=2.5,
        marker="o",
        markersize=4,
        label=f"Validation AUC — Multimodal (Peak {max(hist['val_auc_multi']):.4f})",
    )
    ax2.plot(
        epochs,
        hist["val_auc_color"],
        color=FOREST,
        lw=2.5,
        marker="s",
        markersize=4,
        label=f"Validation AUC — Color-Only (Peak {max(hist['val_auc_color']):.4f})",
    )
    ax2.axvline(
        best_ep,
        color=CAMEL,
        ls="--",
        lw=1.6,
        label=f"Selected Checkpoint (Epoch {best_ep})",
    )
    ax2.set_title("Dual-Mode Validation ROC-AUC Progression")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation ROC-AUC")
    ax2.set_ylim(0.64, 0.95)
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, facecolor=BG)
    plt.close(fig)


def fig_03_roc_pr_calibration(
    report: Dict,
    test_probs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    m_m = report["test_compatibility"]["huefit_v2_multimodal"]
    m_c = report["test_compatibility"]["huefit_v2_color_only"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=300)
    fig.suptitle(
        "Unseen Polyvore Disjoint Test Curves (n = 30,290 Outfits)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    if test_probs is not None:
        y_test, pm, pc = test_probs
        fpr_m, tpr_m, _ = roc_curve(y_test, pm)
        fpr_c, tpr_c, _ = roc_curve(y_test, pc)
        prec_m, rec_m, _ = precision_recall_curve(y_test, pm)
        prec_c, rec_c, _ = precision_recall_curve(y_test, pc)

        axes[0].plot(
            fpr_m,
            tpr_m,
            color=NAVY,
            lw=2.4,
            label=f"Multimodal (AUC = {m_m['roc_auc']:.4f})",
        )
        axes[0].plot(
            fpr_c,
            tpr_c,
            color=FOREST,
            lw=2.4,
            label=f"Color-Only (AUC = {m_c['roc_auc']:.4f})",
        )
        axes[0].plot(
            [0, 1], [0, 1], ls="--", color=SLATE, lw=1.2, label="Random (0.5000)"
        )
        axes[0].fill_between(fpr_m, tpr_m, alpha=0.08, color=NAVY)
        axes[0].set_title("ROC Curve")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc="lower right", fontsize=9)
        axes[0].grid(True)

        axes[1].plot(
            rec_m,
            prec_m,
            color=NAVY,
            lw=2.4,
            label=f"Multimodal (PR-AUC = {m_m['pr_auc']:.4f})",
        )
        axes[1].plot(
            rec_c,
            prec_c,
            color=FOREST,
            lw=2.4,
            label=f"Color-Only (PR-AUC = {m_c['pr_auc']:.4f})",
        )
        axes[1].axhline(
            0.50, ls="--", color=SLATE, lw=1.2, label="Balanced Prior (0.50)"
        )
        axes[1].set_title("Precision-Recall Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_ylim(0.45, 1.02)
        axes[1].legend(loc="lower left", fontsize=9)
        axes[1].grid(True)

        bins = np.linspace(0.0, 1.0, 11)
        for probs, col, lbl in [
            (pm, NAVY, f"Multimodal (ECE = {m_m['ece_cal']:.4f})"),
            (pc, FOREST, f"Color-Only (ECE = {m_c['ece_cal']:.4f})"),
        ]:
            xs, ys = [], []
            for i in range(10):
                mask = (probs >= bins[i]) & (
                    probs <= bins[i + 1] if i == 9 else probs < bins[i + 1]
                )
                if mask.sum() >= 20:
                    xs.append(float(probs[mask].mean()))
                    ys.append(float(y_test[mask].mean()))
            axes[2].plot(xs, ys, marker="o", lw=2.2, color=col, label=lbl)
        axes[2].plot(
            [0, 1], [0, 1], ls="--", color=SLATE, lw=1.2, label="Perfect Calibration"
        )
        axes[2].set_title("Post-Platt Reliability Diagram")
        axes[2].set_xlabel("Mean Predicted Probability")
        axes[2].set_ylabel("Empirical Positive Rate")
        axes[2].legend(loc="upper left", fontsize=9)
        axes[2].grid(True)
    else:
        for ax in axes:
            ax.axis("off")
        axes[1].text(
            0.5,
            0.5,
            f"Multimodal AUC={m_m['roc_auc']:.4f} | Color AUC={m_c['roc_auc']:.4f}\n"
            "(Keep training_dataset.npz + item_embeddings.npy to render exact ROC/PR curves)",
            ha="center",
            va="center",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300, facecolor=BG)
    plt.close(fig)


def fig_04_model_comparison_fitb(report: Dict, out_path: Path) -> None:
    tc = report["test_compatibility"]
    models = [
        ("HueFitNetV2\n(Multimodal)", tc["huefit_v2_multimodal"], NAVY),
        ("HueFitNetV2\n(Color-Only)", tc["huefit_v2_color_only"], FOREST),
        ("LightGBM\n(Multimodal)", tc["lgbm_multimodal"], CAMEL),
        ("LightGBM\n(Color-Only)", tc["lgbm_color_only"], "#9CA3AF"),
    ]
    metric_keys = [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
    ]

    fig = plt.figure(figsize=(15, 5.6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 0.05, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    fig.suptitle(
        "Model Benchmark on Unseen Polyvore Disjoint Splits",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(metric_keys))
    w = 0.19
    for i, (label, m, col) in enumerate(models):
        vals = [m[k] * 100.0 for k, _ in metric_keys]
        bars = ax1.bar(
            x + (i - 1.5) * w,
            vals,
            width=w,
            color=col,
            label=label.replace("\n", " "),
            edgecolor=BG,
            linewidth=0.6,
        )
        if i in (0, 1):
            for b, v in zip(bars, vals):
                ax1.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.7,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color=INK,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in metric_keys], fontweight="bold")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(55, 98)
    ax1.set_title("Compatibility Classification (n = 30,290 Outfits)")
    ax1.grid(True, axis="y")
    ax1.legend(loc="lower right", ncol=2, fontsize=8.5)

    fitb = report["test_fitb"]
    fitb_labels = [
        "HueFitNetV2\nMultimodal",
        "HueFitNetV2\nColor-Only",
        "Random\nBaseline",
    ]
    fitb_vals = [
        fitb["multimodal_top1"] * 100.0,
        fitb["color_only_top1"] * 100.0,
        fitb["random_baseline"] * 100.0,
    ]
    fitb_cols = [NAVY, FOREST, "#9CA3AF"]
    bars2 = ax2.bar(fitb_labels, fitb_vals, width=0.52, color=fitb_cols, edgecolor=BG)
    for b, v in zip(bars2, fitb_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 1.4,
            f"{v:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=INK,
        )
    ax2.axhline(25.0, ls="--", color=SLATE, lw=1.1)
    ax2.set_ylim(0, 74)
    ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.set_title(f"Fill-In-The-Blank (n = {fitb['num_questions']:,})")
    ax2.grid(True, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300, facecolor=BG)
    plt.close(fig)


def fig_05_confusion_and_sensitivity(report: Dict, out_path: Path) -> None:
    m_m = report["test_compatibility"]["huefit_v2_multimodal"]
    m_c = report["test_compatibility"]["huefit_v2_color_only"]
    swap = report["color_swap_sensitivity"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=300)
    fig.suptitle(
        "Confusion Matrices & Color-Swap Sensitivity Audit",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for ax, m, title, cmap in [
        (
            axes[0],
            m_m,
            f"Multimodal Confusion Matrix\n(Acc {m_m['accuracy']*100:.2f}% · F1 {m_m['f1']:.4f})",
            "Blues",
        ),
        (
            axes[1],
            m_c,
            f"Color-Only Confusion Matrix\n(Acc {m_c['accuracy']*100:.2f}% · F1 {m_c['f1']:.4f})",
            "Greens",
        ),
    ]:
        cm = np.array(m["cm"], dtype=np.int64)
        cm_pct = cm / np.maximum(1, cm.sum(axis=1, keepdims=True)) * 100.0
        ax.imshow(cm, cmap=cmap)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Incompatible", "Compatible"])
        ax.set_yticklabels(["Incompatible", "Compatible"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Ground Truth")
        ax.set_title(title, fontsize=10.5)
        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    f"{cm[r, c]:,}\n({cm_pct[r, c]:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > cm.max() * 0.52 else INK,
                    fontsize=10,
                    fontweight="bold",
                )

    clashes = swap["clashes"]
    names = list(clashes.keys())
    orig_vals = [clashes[k]["mean_orig_pct"] for k in names]
    swap_vals = [clashes[k]["mean_swapped_pct"] for k in names]
    drops = [clashes[k]["mean_drop_pp"] for k in names]
    pen = [clashes[k]["pct_penalized"] for k in names]

    y_pos = np.arange(len(names))
    h = 0.34
    axes[2].barh(
        y_pos - h / 2, orig_vals, height=h, color=FOREST, label="Original Real Outfit"
    )
    axes[2].barh(
        y_pos + h / 2,
        swap_vals,
        height=h,
        color=BURGUNDY,
        label="After Clashing Top Swap",
    )
    for i in range(len(names)):
        axes[2].text(
            orig_vals[i] + 1.2,
            y_pos[i] - h / 2,
            f"{orig_vals[i]:.1f}%",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=FOREST,
        )
        axes[2].text(
            swap_vals[i] + 1.2,
            y_pos[i] + h / 2,
            f"{swap_vals[i]:.1f}%  (-{drops[i]:.1f} pp · {pen[i]:.0f}%↓)",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=BURGUNDY,
        )
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(names, fontweight="bold")
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 96)
    axes[2].set_xlabel("Mean Calibrated Compatibility (%)")
    axes[2].set_title(
        f"Color-Swap Sensitivity (n = {swap['n_outfits_tested']})", fontsize=10.5
    )
    axes[2].grid(True, axis="x")
    axes[2].legend(loc="lower right", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=300, facecolor=BG)
    plt.close(fig)


def main() -> None:
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rep_path = ARTIFACTS_DIR / "evaluation_report.json"
    if not rep_path.exists():
        raise FileNotFoundError(f"Missing {rep_path}. Run scripts/train.py first.")

    with open(rep_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)

    print("=== HueFit — Generating High-Resolution Presentation Visuals (300 DPI) ===")
    test_probs = _compute_exact_test_probs()
    if test_probs is not None:
        print(
            f"  [OK] Computed exact test probability curves across {len(test_probs[0]):,} unseen outfits."
        )
    else:
        print(
            "  [WARN] training_dataset.npz not found; ROC/PR plot will use summary fallback."
        )

    outputs = [
        ("01_architecture_pipeline.png", lambda p: fig_01_architecture(p)),
        ("02_training_dynamics.png", lambda p: fig_02_training_dynamics(report, p)),
        (
            "03_roc_pr_calibration.png",
            lambda p: fig_03_roc_pr_calibration(report, test_probs, p),
        ),
        (
            "04_model_comparison_fitb.png",
            lambda p: fig_04_model_comparison_fitb(report, p),
        ),
        (
            "05_confusion_and_sensitivity.png",
            lambda p: fig_05_confusion_and_sensitivity(report, p),
        ),
    ]

    for fname, fn in outputs:
        out_path = FIGURES_DIR / fname
        fn(out_path)
        print(f"  [OK] Saved -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    print(f"\nAll 5 presentation visuals ready in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
