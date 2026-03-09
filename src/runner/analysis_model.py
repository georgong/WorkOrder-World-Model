from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("predictions JSON must be a list of records")
    return data


def plot_prediction_vs_ground_truth(
    data: List[Dict[str, Any]],
    out_path: Path,
    *,
    figsize: tuple = (14, 5),
    alpha: float = 0.3,
    sample_cap: int = 50000,
) -> None:
    """Scatter: x=y_true, y=y_pred for each model; draw y=x reference line."""
    y_true = np.array([r["y"] for r in data], dtype=np.float64)
    n = len(y_true)
    if n > sample_cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_cap, replace=False)
        y_true = y_true[idx]

    models = []
    if any("pred_GraphSAGE" in r for r in data):
        models.append(("pred_GraphSAGE", "GraphSAGE", "C0"))
    if any("pred_MLP" in r for r in data):
        models.append(("pred_MLP", "MLP", "C1"))
    if any("pred_LightGBM" in r for r in data):
        models.append(("pred_LightGBM", "LightGBM", "C2"))

    if not models:
        raise ValueError("No pred_* keys found in predictions JSON")

    num_plots = len(models)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, squeeze=(num_plots != 1))
    if num_plots == 1:
        axes = [axes]

    y_min_global = min(r["y"] for r in data)
    y_max_global = max(r["y"] for r in data)
    for ax, (key, label, color) in zip(axes, models):
        pred = np.array([r[key] for r in data], dtype=np.float64)
        if n > sample_cap:
            pred = pred[idx]
        ax.scatter(y_true, pred, alpha=alpha, s=5, c=color, label=label)
        lim_lo = min(y_true.min(), pred.min(), y_min_global)
        lim_hi = max(y_true.max(), pred.max(), y_max_global)
        margin = (lim_hi - lim_lo) * 0.02 or 1.0
        ax.plot([lim_lo - margin, lim_hi + margin], [lim_lo - margin, lim_hi + margin], "k--", lw=1, label="y = x")
        ax.set_xlabel("y_true (Ground Truth)")
        ax.set_ylabel("y_pred")
        ax.set_title(f"{label}: Prediction vs Ground Truth")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved Prediction vs Ground Truth -> {out_path}")


def _draw_pair_row(
    hard: List[Dict[str, Any]],
    pred_key: str,
    model_label: str,
    fail_label: str,
    color: str,
    ax_hist: Any,
    ax_scatter: Any,
) -> None:
    """Draw one row: when fail_label fails, show model_label's error dist + pred vs y."""
    if not hard:
        ax_hist.text(0.5, 0.5, "No hard cases", ha="center", va="center", transform=ax_hist.transAxes)
        ax_hist.set_title(f"When {fail_label} fails: {model_label} error")
        ax_scatter.text(0.5, 0.5, "No hard cases", ha="center", va="center", transform=ax_scatter.transAxes)
        ax_scatter.set_title(f"When {fail_label} fails: {model_label} pred vs truth")
        return
    y = np.array([r["y"] for r in hard], dtype=np.float64)
    pred = np.array([r[pred_key] for r in hard], dtype=np.float64)
    err = np.abs(pred - y)
    mean_e = float(np.mean(err))
    median_e = float(np.median(err))
    ax_hist.hist(err, bins=50, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax_hist.axvline(median_e, color="red", linestyle="--", lw=2, label=f"median = {median_e:.3f}")
    ax_hist.axvline(mean_e, color="orange", linestyle=":", lw=2, label=f"mean = {mean_e:.3f}")
    ax_hist.set_xlabel(f"|error_{model_label}|")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"When {fail_label} fails: {model_label} error distribution")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    ax_scatter.scatter(y, pred, alpha=0.5, s=10, c=color, label=model_label)
    lim_lo = min(y.min(), pred.min())
    lim_hi = max(y.max(), pred.max())
    margin = (lim_hi - lim_lo) * 0.02 or 1.0
    ax_scatter.plot([lim_lo - margin, lim_hi + margin], [lim_lo - margin, lim_hi + margin], "k--", lw=1, label="y = x")
    ax_scatter.set_xlabel("y_true")
    ax_scatter.set_ylabel(f"pred_{model_label}")
    ax_scatter.set_title(f"When {fail_label} fails: {model_label} pred vs truth")
    ax_scatter.legend()
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.grid(alpha=0.3)


def hard_case_analysis(
    data: List[Dict[str, Any]],
    threshold: float,
    out_path: Path,
    *,
    figsize: tuple = (12, 10),
) -> None:
    """
    Hard-case analysis: one-to-one pairwise. 3 figures (A vs B), each 2x2:
    - Row 1: when A fails → B's error dist + B's pred vs truth
    - Row 2: when B fails → A's error dist + A's pred vs truth
    Pairs: GraphSAGE vs LightGBM, LightGBM vs MLP, GraphSAGE vs MLP.
    """
    has_lgb = any("pred_LightGBM" in r for r in data)
    stem = out_path.stem
    suffix = out_path.suffix
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    hard_sage = [r for r in data if abs(r["y"] - r["pred_GraphSAGE"]) > threshold]
    hard_mlp = [r for r in data if abs(r["y"] - r["pred_MLP"]) > threshold]
    hard_lgb = [r for r in data if abs(r["y"] - r["pred_LightGBM"]) > threshold] if has_lgb else []

    print(f"[Hard-case] When GraphSAGE fails: count = {len(hard_sage)}")
    print(f"[Hard-case] When MLP fails: count = {len(hard_mlp)}")
    if has_lgb:
        print(f"[Hard-case] When LightGBM fails: count = {len(hard_lgb)}")

    # Figure 1: GraphSAGE vs LightGBM — Row1 when GraphSAGE fails → LightGBM; Row2 when LightGBM fails → GraphSAGE
    if has_lgb:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        _draw_pair_row(hard_sage, "pred_LightGBM", "LightGBM", "GraphSAGE", "C2", axes[0, 0], axes[0, 1])
        _draw_pair_row(hard_lgb, "pred_GraphSAGE", "GraphSAGE", "LightGBM", "C0", axes[1, 0], axes[1, 1])
        plt.suptitle("Hard-case: GraphSAGE vs LightGBM", fontsize=14)
        plt.tight_layout()
        p = parent / f"{stem}_GraphSAGE_vs_LightGBM{suffix}"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[ok] saved Hard-case (GraphSAGE vs LightGBM) -> {p}")

    # Figure 2: LightGBM vs MLP — Row1 when LightGBM fails → MLP; Row2 when MLP fails → LightGBM
    if has_lgb:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        _draw_pair_row(hard_lgb, "pred_MLP", "MLP", "LightGBM", "C1", axes[0, 0], axes[0, 1])
        _draw_pair_row(hard_mlp, "pred_LightGBM", "LightGBM", "MLP", "C2", axes[1, 0], axes[1, 1])
        plt.suptitle("Hard-case: LightGBM vs MLP", fontsize=14)
        plt.tight_layout()
        p = parent / f"{stem}_LightGBM_vs_MLP{suffix}"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[ok] saved Hard-case (LightGBM vs MLP) -> {p}")

    # Figure 3: GraphSAGE vs MLP — Row1 when GraphSAGE fails → MLP; Row2 when MLP fails → GraphSAGE
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    _draw_pair_row(hard_sage, "pred_MLP", "MLP", "GraphSAGE", "C1", axes[0, 0], axes[0, 1])
    _draw_pair_row(hard_mlp, "pred_GraphSAGE", "GraphSAGE", "MLP", "C0", axes[1, 0], axes[1, 1])
    plt.suptitle("Hard-case: GraphSAGE vs MLP", fontsize=14)
    plt.tight_layout()
    p = parent / f"{stem}_GraphSAGE_vs_MLP{suffix}"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved Hard-case (GraphSAGE vs MLP) -> {p}")


def compute_metrics_from_data(data: List[Dict[str, Any]], pred_key: str, beta: float = 1.0) -> Dict[str, float]:
    """Compute MAE, Smooth L1, MSE, RMSE from predictions list."""
    y_true = np.array([r["y"] for r in data], dtype=np.float64)
    y_pred = np.array([r[pred_key] for r in data], dtype=np.float64)
    n = len(y_true)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = np.sqrt(mse)
    diff = np.abs(err)
    smoothl1 = float(np.mean(np.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)))
    return {"mae": mae, "smoothl1": smoothl1, "mse": mse, "rmse": rmse}


def plot_metrics_bar(
    data: List[Dict[str, Any]],
    out_path: Path,
    *,
    beta: float = 1.0,
    figsize: tuple = (10, 5),
) -> None:
    """Bar plot: MAE, Smooth L1, MSE, RMSE per model (from predictions JSON)."""
    models = []
    if any("pred_GraphSAGE" in r for r in data):
        models.append(("pred_GraphSAGE", "GraphSAGE", "C0"))
    if any("pred_MLP" in r for r in data):
        models.append(("pred_MLP", "MLP", "C1"))
    if any("pred_LightGBM" in r for r in data):
        models.append(("pred_LightGBM", "LightGBM", "C2"))
    if not models:
        raise ValueError("No pred_* keys found in predictions JSON")

    metric_keys = ["mae", "smoothl1", "rmse"]
    metric_labels = ["MAE", "Smooth L1", "RMSE"]
    # GraphSAGE: full opacity to highlight; others: more transparent
    model_metrics = [
        (label, compute_metrics_from_data(data, key, beta=beta), color, 1.0 if label == "GraphSAGE" else 0.65)
        for key, label, color in models
    ]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(metric_keys))
    width = 0.8 / len(models)
    for i, (label, metrics, color, alpha) in enumerate(model_metrics):
        vals = [metrics[k] for k in metric_keys]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=alpha)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Error")
    ax.set_title("Comparison of Regression Metrics by Model")
    ax.legend(prop={"size": 20})
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved metrics bar plot -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze compare_model predictions JSON: Prediction vs Ground Truth scatter + Hard-case analysis."
    )
    ap.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON from compare_model")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="Hard-case: |error_MLP| and |error_LightGBM| > this (default: 3)")
    ap.add_argument("--out_scatter", type=str, default="runs/analysis_model/pred_vs_truth.png",
                    help="Output path for Prediction vs Ground Truth figure")
    ap.add_argument("--out_hardcase", type=str, default="runs/analysis_model/hard_case_analysis.png",
                    help="Output path for Hard-case analysis figure")
    ap.add_argument("--out_metrics", type=str, default="runs/analysis_model/metrics_bar.png",
                    help="Output path for MAE / Smooth L1 / MSE / RMSE bar plot")
    ap.add_argument("--beta", type=float, default=1.0, help="Smooth L1 beta (default 1.0)")
    ap.add_argument("--sample_cap", type=int, default=50000,
                    help="Max points to plot in scatter (subsample if larger, default 50000)")
    args = ap.parse_args()

    path = Path(args.predictions)
    if not path.exists():
        raise FileNotFoundError(f"predictions file not found: {path}")

    print(f"[info] Loading {path}...")
    data = load_predictions(path)
    print(f"[info] Loaded {len(data)} records")

    sns.set_theme(style="whitegrid")

    plot_prediction_vs_ground_truth(
        data,
        Path(args.out_scatter),
        sample_cap=args.sample_cap,
    )

    hard_case_analysis(data, args.threshold, Path(args.out_hardcase))

    plot_metrics_bar(data, Path(args.out_metrics), beta=args.beta)

    # Print MAE, Smooth L1, RMSE for each model
    models = []
    if any("pred_GraphSAGE" in r for r in data):
        models.append(("pred_GraphSAGE", "GraphSAGE"))
    if any("pred_MLP" in r for r in data):
        models.append(("pred_MLP", "MLP"))
    if any("pred_LightGBM" in r for r in data):
        models.append(("pred_LightGBM", "LightGBM"))
    if models:
        print("\n--- MAE, Smooth L1, RMSE (all samples) ---")
        for key, label in models:
            m = compute_metrics_from_data(data, key, beta=args.beta)
            print(f"  {label}: MAE = {m['mae']:.4f}, Smooth L1 = {m['smoothl1']:.4f}, RMSE = {m['rmse']:.4f}")


if __name__ == "__main__":
    main()
