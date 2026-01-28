# prune_graph_and_persist.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch_geometric.data import HeteroData


@torch.no_grad()
def compute_node_degree_all(data: HeteroData) -> Dict[str, torch.Tensor]:
    """
    Degree per node type, counting ALL incident edges (in + out) across all edge types.
    """
    deg = {nt: torch.zeros(data[nt].num_nodes, dtype=torch.long) for nt in data.node_types}

    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if not isinstance(ei, torch.Tensor) or ei.numel() == 0:
            continue
        deg[src] += torch.bincount(ei[0], minlength=data[src].num_nodes)
        deg[dst] += torch.bincount(ei[1], minlength=data[dst].num_nodes)

    return deg


def _stash_non_tensor_fields(data: HeteroData) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[Tuple[str, str, str], str], Dict[str, Any]]]:
    """
    Stash non-tensor fields from node/edge stores so we can reattach after subgraph.
    Returns:
      node_extra[(node_type, "node")][key] = value
      edge_extra[(edge_type, "edge")][key] = value
    """
    node_extra: Dict[Tuple[str, str], Dict[str, Any]] = {}
    edge_extra: Dict[Tuple[Tuple[str, str, str], str], Dict[str, Any]] = {}

    for nt in data.node_types:
        store = data[nt]
        extras = {}
        for key in list(store.keys()):
            v = store[key]
            if isinstance(v, torch.Tensor):
                continue
            extras[key] = v
            del store[key]
        if extras:
            node_extra[(nt, "node")] = extras

    for et in data.edge_types:
        store = data[et]
        extras = {}
        for key in list(store.keys()):
            v = store[key]
            if isinstance(v, torch.Tensor):
                continue
            extras[key] = v
            del store[key]
        if extras:
            edge_extra[(et, "edge")] = extras

    return node_extra, edge_extra


def _reattach_non_tensor_fields(
    data_new: HeteroData,
    *,
    node_extra: Dict[Tuple[str, str], Dict[str, Any]],
    edge_extra: Dict[Tuple[Tuple[str, str, str], str], Dict[str, Any]],
    node_keep_idx: Dict[str, torch.Tensor],
    edge_keep_mask: Dict[Tuple[str, str, str], torch.Tensor],
) -> None:
    """
    Reattach stashed non-tensor fields after subgraph, slicing them if possible.
    - For node fields: if list/tuple length == old_num_nodes, slice by node_keep_idx
    - For edge fields: if list/tuple length == old_num_edges, slice by edge_keep_mask
    """
    # nodes
    for (nt, _kind), extras in node_extra.items():
        if nt not in data_new.node_types:
            continue
        keep = node_keep_idx[nt].cpu().tolist()
        for key, v in extras.items():
            if isinstance(v, (list, tuple)):
                # slice if length matches original
                try:
                    sliced = [v[i] for i in keep]
                    data_new[nt][key] = sliced
                except Exception:
                    # fallback: keep as-is (better than crashing)
                    data_new[nt][key] = v
            else:
                data_new[nt][key] = v

    # edges
    for (et, _kind), extras in edge_extra.items():
        if et not in data_new.edge_types:
            continue
        mask = edge_keep_mask[et]
        keep_e = mask.nonzero(as_tuple=False).view(-1).cpu().tolist()
        for key, v in extras.items():
            if isinstance(v, (list, tuple)):
                try:
                    sliced = [v[i] for i in keep_e]
                    data_new[et][key] = sliced
                except Exception:
                    data_new[et][key] = v
            else:
                data_new[et][key] = v


@torch.no_grad()
def prune_isolated_nodes(data: HeteroData, *, min_degree: int = 1) -> HeteroData:
    """
    Prune nodes with incident degree < min_degree for EVERY node type.
    Keeps edge-induced subgraph and reindexes nodes.
    """
    if not hasattr(data, "subgraph"):
        raise RuntimeError("HeteroData.subgraph() not found. Upgrade torch-geometric.")

    # Stash non-tensor fields so subgraph won't choke on list/str
    node_extra, edge_extra = _stash_non_tensor_fields(data)

    # Compute degree and build node_dict
    deg = compute_node_degree_all(data)
    node_dict: Dict[str, torch.Tensor] = {}
    for nt in data.node_types:
        keep_mask = deg[nt] >= min_degree
        keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)
        # If a node type becomes empty, keep 1 node to avoid weird downstream failures.
        # (You can set min_degree=0 if you want to keep everything.)
        if keep_idx.numel() == 0 and data[nt].num_nodes > 0:
            keep_idx = torch.tensor([0], dtype=torch.long)
        node_dict[nt] = keep_idx

    # For reattaching node extras we need keep idx per type
    node_keep_idx = {nt: node_dict[nt].clone() for nt in data.node_types}

    # Also need edge keep mask for reattaching edge extras:
    # subgraph keeps only edges whose src/dst are kept.
    # We'll compute it manually on old graph before subgraph.
    edge_keep_mask: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if not isinstance(ei, torch.Tensor) or ei.numel() == 0:
            edge_keep_mask[(src, rel, dst)] = torch.zeros(0, dtype=torch.bool)
            continue
        src_keep = torch.zeros(data[src].num_nodes, dtype=torch.bool)
        dst_keep = torch.zeros(data[dst].num_nodes, dtype=torch.bool)
        src_keep[node_dict[src]] = True
        dst_keep[node_dict[dst]] = True
        mask = src_keep[ei[0]] & dst_keep[ei[1]]
        edge_keep_mask[(src, rel, dst)] = mask

    # Prune
    data_new: HeteroData = data.subgraph(node_dict)

    # Reattach non-tensor fields (attr_name/node_ids/etc.)
    _reattach_non_tensor_fields(
        data_new,
        node_extra=node_extra,
        edge_extra=edge_extra,
        node_keep_idx=node_keep_idx,
        edge_keep_mask=edge_keep_mask,
    )

    return data_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt", type=str, default="data/graph/sdge.pt")
    ap.add_argument("--out_pt", type=str, default="data/graph/sdge_pruned.pt")
    ap.add_argument("--min_degree", type=int, default=1, help="Prune nodes with incident degree < min_degree")
    args = ap.parse_args()

    in_path = Path(args.in_pt)
    out_path = Path(args.out_pt)
    assert in_path.exists(), f"File not found: {in_path}"

    data: HeteroData = torch.load(in_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData), "Input must be a torch_geometric.data.HeteroData"

    print("================================================================================")
    print("[before] node counts")
    for nt in data.node_types:
        print(f"  {nt:16s}: {data[nt].num_nodes}")
    print("[before] edge counts")
    for et in data.edge_types:
        ei = data[et].edge_index
        m = int(ei.size(1)) if isinstance(ei, torch.Tensor) else 0
        print(f"  {str(et):40s}: {m}")
    print("================================================================================")

    data2 = prune_isolated_nodes(data, min_degree=args.min_degree)

    print("================================================================================")
    print("[after] node counts")
    for nt in data2.node_types:
        print(f"  {nt:16s}: {data2[nt].num_nodes}")
    print("[after] edge counts")
    for et in data2.edge_types:
        ei = data2[et].edge_index
        m = int(ei.size(1)) if isinstance(ei, torch.Tensor) else 0
        print(f"  {str(et):40s}: {m}")
    print("================================================================================")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data2, out_path)
    print(f"[ok] saved pruned graph -> {out_path}")


if __name__ == "__main__":
    main()
