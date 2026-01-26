# training_data_eda.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from torch_geometric.data import HeteroData


def _to_float(x: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point():
        return x.float()
    return x


@torch.no_grad()
def feature_stats(
    x: torch.Tensor,
    *,
    max_rows: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-feature mean/var/std on CPU tensor x [N, F].
    Optionally subsample rows for speed/memory stability.
    """
    assert x.dim() == 2, f"x must be 2D, got {x.shape}"
    x = _to_float(x).cpu()

    N, F = x.shape

    if max_rows is not None and N > max_rows:
        # deterministic-ish subsample: take evenly spaced indices
        idx = torch.linspace(0, N - 1, steps=max_rows).long()
        x = x.index_select(0, idx)
        N = x.shape[0]

    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)

    # some robust summaries
    abs_mean = mean.abs()
    abs_max = x.abs().max(dim=0).values

    return {
        "N": torch.tensor(N),
        "F": torch.tensor(F),
        "mean": mean,
        "var": var,
        "std": std,
        "abs_mean": abs_mean,
        "abs_max": abs_max,
    }


def summarize_feature_scale(name: str, stats: Dict[str, torch.Tensor], attr_name=None, topk: int = 10) -> str:
    mean = stats["mean"]
    std = stats["std"]
    var = stats["var"]
    abs_max = stats["abs_max"]
    abs_mean = stats["abs_mean"]
    N = int(stats["N"].item())
    F = int(stats["F"].item())

    def fname(i: int) -> str:
        if attr_name is None:
            return f"f{i}"
        if i < len(attr_name):
            return attr_name[i]
        return f"f{i}"

    std_min = float(std.min().item())
    std_max = float(std.max().item())
    absmax_max = float(abs_max.max().item())
    absmean_max = float(abs_mean.max().item())

    std_ratio = (std_max / max(std_min, 1e-12)) if F > 0 else float("nan")

    top_std_vals, top_std_idx = torch.topk(std, k=min(topk, F), largest=True)
    bot_std_vals, bot_std_idx = torch.topk(std, k=min(topk, F), largest=False)

    lines = []
    lines.append(f"[{name}] x shape: N={N}, F={F}")
    lines.append(f"  std range: min={std_min:.3e}, max={std_max:.3e}, ratio(max/min)={std_ratio:.3e}")
    lines.append(f"  abs(mean) max={absmean_max:.3e} | abs(x) max over features={absmax_max:.3e}")

    lines.append("  top std features (name | idx | std | mean | abs_max):")
    for i in range(top_std_idx.numel()):
        j = int(top_std_idx[i].item())
        lines.append(
            f"    {fname(j):<30} | {j:4d} | std={float(std[j]):.3e} | "
            f"mean={float(mean[j]):.3e} | abs_max={float(abs_max[j]):.3e}"
        )

    lines.append("  bottom std features (name | idx | std | mean | abs_max):")
    for i in range(bot_std_idx.numel()):
        j = int(bot_std_idx[i].item())
        lines.append(
            f"    {fname(j):<30} | {j:4d} | std={float(std[j]):.3e} | "
            f"mean={float(mean[j]):.3e} | abs_max={float(abs_max[j]):.3e}"
        )

    if std_ratio > 1e4:
        lines.append("  [RED FLAG] std ratio > 1e4: extreme scale mismatch between features.")

    if absmax_max > 1e6:
        lines.append("  [RED FLAG] abs_max > 1e6: likely timestamp / ID / money-like feature.")

    return "\n".join(lines)


@torch.no_grad()
def target_stats(y: torch.Tensor, *, max_rows: Optional[int] = None, eps: float = 1e-12) -> Dict[str, float]:
    y = _to_float(y).flatten().cpu()
    N = y.numel()
    if max_rows is not None and N > max_rows:
        idx = torch.linspace(0, N - 1, steps=max_rows).long()
        y = y.index_select(0, idx)
        N = y.numel()

    mean = float(y.mean().item())
    var = float(y.var(unbiased=False).item())
    std = float((var + eps) ** 0.5)
    mn = float(y.min().item())
    mx = float(y.max().item())

    # some quantiles for distribution shape
    qs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    qv = torch.quantile(y, qs)

    return {
        "N": N,
        "mean": mean,
        "var": var,
        "std": std,
        "min": mn,
        "max": mx,
        "q": {float(qs[i]): float(qv[i]) for i in range(len(qs))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, default="data/graph/sdge.pt")
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--out", type=str, default="", help="optional: write report to a txt file")
    ap.add_argument("--max_rows", type=int, default=200000, help="subsample rows per node type for faster stats")
    ap.add_argument("--topk", type=int, default=10, help="top-k features to print by std")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    assert pt_path.exists(), f"File not found: {pt_path}"

    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData), "pt must contain a torch_geometric.data.HeteroData"

    target = args.target
    assert target in data.node_types, f"target node type {target!r} not in {data.node_types}"
    assert hasattr(data[target], "y"), f"{target}.y missing"

    lines = []
    lines.append("=" * 80)
    lines.append("HeteroData Training Data EDA Report")
    lines.append("=" * 80)
    lines.append(f"pt: {str(pt_path)}")
    lines.append(f"node_types: {data.node_types}")
    lines.append(f"edge_types: {data.edge_types}")
    lines.append("")

    # 1) Feature scale per node type
    lines.append("=" * 80)
    lines.append("1) Feature mean/variance scale (per node type)")
    lines.append("=" * 80)

    for nt in data.node_types:
        if not hasattr(data[nt], "x"):
            continue
        x = data[nt].x
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            lines.append(f"[{nt}] x is not a 2D tensor, skip.")
            continue

        attr_names = None
        if hasattr(data[nt], "attr_name"):
            attr_names = data[nt].attr_name

        stats = feature_stats(x, max_rows=args.max_rows)
        lines.append(
            summarize_feature_scale(
                f"node_type={nt}",
                stats,
                attr_name=attr_names,
                topk=args.topk,
            )
        )
        lines.append("")

    # 2) Target label distribution
    lines.append("=" * 80)
    lines.append("2) Target y distribution (mean/variance + quantiles)")
    lines.append("=" * 80)

    y = data[target].y
    tstats = target_stats(y, max_rows=None)
    lines.append(f"[target={target}] N={tstats['N']}")
    lines.append(f"  mean={tstats['mean']:.6g} | var={tstats['var']:.6g} | std={tstats['std']:.6g}")
    lines.append(f"  min={tstats['min']:.6g} | max={tstats['max']:.6g}")
    lines.append("  quantiles:")
    for q in sorted(tstats["q"].keys()):
        lines.append(f"    q{q:>5.2f}: {tstats['q'][q]:.6g}")

    # quick red flags for y
    if tstats["std"] > 10.0 and (abs(tstats["mean"]) < 1e-3 or abs(tstats["mean"]) / max(tstats["std"], 1e-12) < 0.1):
        lines.append("  [note] y has large std relative to mean. Check if y needs normalization/log-transform.")
    if tstats["max"] > 1e6:
        lines.append("  [RED FLAG] y max > 1e6. If you expected small values, labels might be wrong-scale.")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
