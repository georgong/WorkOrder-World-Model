from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    import pandas as pd
except ImportError:
    pd = None  # 你不装 pandas 也能跑，只是 heatmap 用 dict 返回


@dataclass(frozen=True)
class MetaPathResult:
    metapath: Tuple[str, str, str]
    missing_edges: bool
    missing_legs: List[Tuple[str, str]]  # e.g. [("assignments","engineers")]
    frac_A_reach_C_via_B: float
    num_A_reachable: int
    num_A_total: int


def _load_graph(path: str | Path):
    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]
    return obj


def compute_type_connectivity_heatmap(
    data,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    """
    For each directed pair (src_type -> dst_type):
      ratio = (#src nodes having at least one outgoing edge to dst_type) / (#src nodes)

    Returns:
      ratio_map[src][dst] = float
      count_map[src][dst] = int (#src nodes connected)
    """
    node_types = list(data.node_types)
    num_nodes = {t: int(data[t].num_nodes) for t in node_types}

    # Only allocate masks for pairs that actually appear in edge_types
    masks: Dict[Tuple[str, str], torch.Tensor] = {}

    for etype in data.edge_types:
        src_type, _, dst_type = etype
        store = data[etype]
        if not hasattr(store, "edge_index"):
            continue
        ei = store.edge_index
        if not isinstance(ei, torch.Tensor) or ei.numel() == 0:
            continue

        key = (src_type, dst_type)
        if key not in masks:
            masks[key] = torch.zeros(num_nodes[src_type], dtype=torch.bool)

        src_idx = ei[0].to(torch.long)
        masks[key][src_idx] = True

    # Build full maps (include zeros for missing pairs)
    ratio_map: Dict[str, Dict[str, float]] = {s: {d: 0.0 for d in node_types} for s in node_types}
    count_map: Dict[str, Dict[str, int]] = {s: {d: 0 for d in node_types} for s in node_types}

    for (s, d), m in masks.items():
        cnt = int(m.sum().item())
        denom = max(1, num_nodes[s])
        ratio_map[s][d] = cnt / denom
        count_map[s][d] = cnt

    return ratio_map, count_map


def _has_any_edge_type(data, src_type: str, dst_type: str) -> bool:
    for etype in data.edge_types:
        s, _, d = etype
        if s == src_type and d == dst_type:
            store = data[etype]
            if hasattr(store, "edge_index") and getattr(store, "edge_index").numel() > 0:
                return True
    return False


def compute_second_order_connectivity(
    data,
    metapaths: List[Tuple[str, str, str]],
) -> List[MetaPathResult]:
    """
    For metapath A-B-C:
      frac = (#A nodes that have an edge to some B that has an edge to some C) / (#A nodes)

    Streaming, no giant joins:
      - active_B = B nodes that connect to C
      - reachable_A = A nodes that connect to active_B
    """
    results: List[MetaPathResult] = []

    node_types = set(data.node_types)
    num_nodes = {t: int(data[t].num_nodes) for t in data.node_types}

    for (A, B, C) in metapaths:
        missing = []
        if A not in node_types or B not in node_types or C not in node_types:
            # treat as missing legs
            if A not in node_types:
                missing.append((A, B))
            if B not in node_types:
                missing.append((A, B))
                missing.append((B, C))
            if C not in node_types:
                missing.append((B, C))

            results.append(
                MetaPathResult(
                    metapath=(A, B, C),
                    missing_edges=True,
                    missing_legs=missing,
                    frac_A_reach_C_via_B=0.0,
                    num_A_reachable=0,
                    num_A_total=num_nodes.get(A, 0),
                )
            )
            continue

        # check existence of edges A->B and B->C (any relation name)
        has_AB = _has_any_edge_type(data, A, B)
        has_BC = _has_any_edge_type(data, B, C)
        if not has_AB:
            missing.append((A, B))
        if not has_BC:
            missing.append((B, C))

        if missing:
            results.append(
                MetaPathResult(
                    metapath=(A, B, C),
                    missing_edges=True,
                    missing_legs=missing,
                    frac_A_reach_C_via_B=0.0,
                    num_A_reachable=0,
                    num_A_total=num_nodes[A],
                )
            )
            continue

        # 1) active_B: B nodes that have ANY outgoing edge to C
        active_B = torch.zeros(num_nodes[B], dtype=torch.bool)
        for etype in data.edge_types:
            s, _, d = etype
            if s != B or d != C:
                continue
            ei = data[etype].edge_index
            if not isinstance(ei, torch.Tensor) or ei.numel() == 0:
                continue
            b_idx = ei[0].to(torch.long)
            active_B[b_idx] = True

        # 2) reachable_A: A nodes that connect to active_B
        reachable_A = torch.zeros(num_nodes[A], dtype=torch.bool)
        for etype in data.edge_types:
            s, _, d = etype
            if s != A or d != B:
                continue
            ei = data[etype].edge_index
            if not isinstance(ei, torch.Tensor) or ei.numel() == 0:
                continue

            a_idx = ei[0].to(torch.long)
            b_idx = ei[1].to(torch.long)
            # keep edges whose b is active
            keep = active_B[b_idx]
            if keep.any():
                reachable_A[a_idx[keep]] = True

        num_reach = int(reachable_A.sum().item())
        denom = max(1, num_nodes[A])
        frac = num_reach / denom

        results.append(
            MetaPathResult(
                metapath=(A, B, C),
                missing_edges=False,
                missing_legs=[],
                frac_A_reach_C_via_B=frac,
                num_A_reachable=num_reach,
                num_A_total=num_nodes[A],
            )
        )

    return results


def analyze_graph_connectivity(
    graph_path: str | Path = "data/graph/sdge.pt",
    metapaths: Optional[List[Tuple[str, str, str]]] = None,
    save_csv: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Main entry:
      - load graph
      - compute type-to-type heatmap ratios
      - compute second-order metapath connectivity

    Returns a dict with:
      - heatmap_ratio (dict or DataFrame)
      - heatmap_count (dict or DataFrame)
      - metapath_results (list of dict)
    """
    data = _load_graph(graph_path)

    ratio_map, count_map = compute_type_connectivity_heatmap(data)

    if metapaths is None:
        # Put your common metapaths here
        metapaths = [
            ("tasks", "assignments", "engineers"),
            ("tasks", "assignments", "tasks"),
            ("tasks", "districts", "tasks"),
            ("tasks", "task_types", "tasks"),
            ("tasks", "task_statuses", "tasks"),
        ]

    meta_res = compute_second_order_connectivity(data, metapaths)
    meta_res_dicts = [
        {
            "metapath": r.metapath,
            "missing_edges": r.missing_edges,
            "missing_legs": r.missing_legs,
            "frac_A_reach_C_via_B": r.frac_A_reach_C_via_B,
            "num_A_reachable": r.num_A_reachable,
            "num_A_total": r.num_A_total,
        }
        for r in meta_res
    ]

    out: Dict[str, Any] = {
        "heatmap_ratio": ratio_map,
        "heatmap_count": count_map,
        "metapath_results": meta_res_dicts,
    }

    # Optional: convert to DataFrame (nice for heatmap plotting)
    if pd is not None:
        out["heatmap_ratio_df"] = pd.DataFrame(ratio_map).T
        out["heatmap_count_df"] = pd.DataFrame(count_map).T

        if save_csv is not None:
            save_csv = Path(save_csv)
            save_csv.parent.mkdir(parents=True, exist_ok=True)
            out["heatmap_ratio_df"].to_csv(save_csv.with_suffix(".ratio.csv"))
            out["heatmap_count_df"].to_csv(save_csv.with_suffix(".count.csv"))

    return out



res = analyze_graph_connectivity(
    "data/graph/sdge.pt",
    metapaths=[("tasks","assignments","engineers"), ("tasks","assignments","tasks")],
    save_csv="data/analysis/connectivity"
)

# 如果装了 pandas：
print(res["heatmap_ratio_df"].round(4))
print(res["metapath_results"])