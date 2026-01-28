# interpret_server/app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


# ----------------------------
# config (edit these paths)
# ----------------------------
GRAPH_PT = Path("data/graph/sdge.pt")
INTERPRET_DIR = Path("runs/interpret")
STATIC_DIR = Path("interpret_server/static")

DEFAULT_TARGET = "assignments"
DEFAULT_LAYERS = 2
DEFAULT_NUM_NEIGHBORS = 10

# how many feats to send to frontend per node (sidebar bar chart)
DEFAULT_NODE_FEATS_K = 20


app = FastAPI(title="SDGE Interpret Viewer")

# Serve static files (index.html, js, etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load graph once
if not GRAPH_PT.exists():
    raise FileNotFoundError(f"Graph pt not found: {GRAPH_PT}")

DATA: HeteroData = torch.load(GRAPH_PT, map_location="cpu", weights_only=False)
if not isinstance(DATA, HeteroData):
    raise TypeError("Loaded graph is not HeteroData")


# ----------------------------
# utils
# ----------------------------
def list_interpret_files() -> List[str]:
    if not INTERPRET_DIR.exists():
        return []
    return sorted([p.name for p in INTERPRET_DIR.glob("*.json")])


def load_interpret_json(fname: str) -> Dict[str, Any]:
    p = INTERPRET_DIR / fname
    if not p.exists():
        raise FileNotFoundError(f"Interpret json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def sanitize_for_neighbor_loader(data: HeteroData) -> HeteroData:
    # NeighborLoader cannot handle non-tensor fields
    for nt in data.node_types:
        store = data[nt]
        for key in list(store.keys()):
            if isinstance(store[key], torch.Tensor):
                continue
            del store[key]
    for et in data.edge_types:
        store = data[et]
        for key in list(store.keys()):
            if isinstance(store[key], torch.Tensor):
                continue
            del store[key]
    return data


def build_subgraph_batch(
    *,
    seed_idx: int,
    target: str,
    layers: int,
    num_neighbors: int,
) -> HeteroData:
    data = DATA.clone()
    data = sanitize_for_neighbor_loader(data)

    seed = torch.tensor([int(seed_idx)], dtype=torch.long)
    num_neighbors_dict = {et: [int(num_neighbors)] * int(layers) for et in data.edge_types}

    loader = NeighborLoader(
        data,
        input_nodes=(target, seed),
        num_neighbors=num_neighbors_dict,
        batch_size=1,
        shuffle=False,
    )
    return next(iter(loader))


def node_key(nt: str, orig: int) -> str:
    return f"{nt}:{orig}"


def label_from_batch(batch: HeteroData, nt: str, local_idx: int) -> Tuple[int, str]:
    """
    Returns (orig_id, label_str like 'assignments[orig=194]')
    """
    store = batch[nt]
    if hasattr(store, "n_id") and isinstance(store.n_id, torch.Tensor):
        orig = int(store.n_id[local_idx].item())
        return orig, f"{nt}[orig={orig}]"
    return local_idx, f"{nt}[local={local_idx}]"


def parse_top_nodes(summary: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Your interpret json format:
      summary["top_nodes"][nt] = [[label, signed_value], [label, signed_value], ...]
    Return:
      nt -> label -> {"value": signed, "abs": abs(signed)}
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    top_nodes = summary.get("top_nodes", {}) or {}

    for nt, arr in top_nodes.items():
        d: Dict[str, Dict[str, float]] = {}
        if not isinstance(arr, list):
            out[nt] = d
            continue

        for item in arr:
            # item is usually [label, value]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                label = str(item[0])
                v = float(item[1])
                d[label] = {"value": v, "abs": abs(v)}
            elif isinstance(item, dict) and "node" in item:
                # tolerate dict format too
                label = str(item["node"])
                v = float(item.get("value", 0.0))
                d[label] = {"value": v, "abs": abs(v)}
        out[nt] = d

    return out


def parse_top_feats(summary: Dict[str, Any]) -> Dict[str, List[Dict[str, float]]]:
    """
    Your interpret json format:
      summary["top_feats"][nt] = [[feat_name, signed_value], ...]
    Normalize to:
      nt -> [{"feat":..., "value":..., "abs":...}, ...]
    """
    out: Dict[str, List[Dict[str, float]]] = {}
    top_feats = summary.get("top_feats", {}) or {}

    for nt, arr in top_feats.items():
        rows: List[Dict[str, float]] = []
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    feat = str(item[0])
                    v = float(item[1])
                    rows.append({"feat": feat, "value": v, "abs": abs(v)})
                elif isinstance(item, dict) and "feat" in item:
                    feat = str(item["feat"])
                    v = float(item.get("value", 0.0))
                    rows.append({"feat": feat, "value": v, "abs": abs(v)})
        # sort by abs desc
        rows.sort(key=lambda x: float(x["abs"]), reverse=True)
        out[nt] = rows

    return out


# ----------------------------
# routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home() -> str:
    p = STATIC_DIR / "index.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Missing {p}")
    return p.read_text(encoding="utf-8")


@app.get("/api/seeds")
def api_seeds() -> Dict[str, Any]:
    return {"files": list_interpret_files()}


@app.get("/api/view")
def api_view(
    file: str = Query(..., description="interpret json file name in runs/interpret"),
    target: str = Query(DEFAULT_TARGET),
    layers: int = Query(DEFAULT_LAYERS, ge=1, le=6),
    num_neighbors: int = Query(DEFAULT_NUM_NEIGHBORS, ge=1, le=200),
    node_feats_k: int = Query(DEFAULT_NODE_FEATS_K, ge=1, le=100),
) -> Dict[str, Any]:
    interpret = load_interpret_json(file)
    summary = interpret.get("summary", {}) or {}

    seed_idx = int(interpret.get("seed_idx", summary.get("seed_idx", -1)))
    if seed_idx < 0:
        raise HTTPException(status_code=400, detail="seed_idx missing in interpret json")

    if target not in DATA.node_types:
        raise HTTPException(status_code=400, detail=f"target '{target}' not in graph node_types")

    batch = build_subgraph_batch(
        seed_idx=seed_idx,
        target=target,
        layers=layers,
        num_neighbors=num_neighbors,
    )

    # parsed maps
    top_nodes_map = parse_top_nodes(summary)
    top_feats_map = parse_top_feats(summary)

    # build nodes
    nodes: List[Dict[str, Any]] = []
    node_feats: Dict[str, List[Dict[str, float]]] = {}  # node_id -> feats list

    for nt in batch.node_types:
        store = batch[nt]
        n = int(store.num_nodes) if hasattr(store, "num_nodes") else int(store.x.size(0))
        for li in range(n):
            orig, label = label_from_batch(batch, nt, li)
            nid = node_key(nt, orig)

            contrib = top_nodes_map.get(nt, {}).get(label, {"value": 0.0, "abs": 0.0})

            nodes.append(
                {
                    "id": nid,
                    "type": nt,
                    "orig": orig,
                    "label": label,
                    "value": float(contrib["value"]),
                    "abs": float(contrib["abs"]),
                }
            )

            # Fallback: give each node the type-level feats (until you implement per-node feats)
            feats = top_feats_map.get(nt, [])[: int(node_feats_k)]
            node_feats[nid] = feats

    # edges
    edges: List[Dict[str, Any]] = []
    for (src_t, rel, dst_t), store in batch.edge_items():
        if "edge_index" not in store:
            continue
        ei = store["edge_index"]
        if not isinstance(ei, torch.Tensor) or ei.numel() == 0:
            continue

        src_nid = batch[src_t].n_id if hasattr(batch[src_t], "n_id") else None
        dst_nid = batch[dst_t].n_id if hasattr(batch[dst_t], "n_id") else None

        for j in range(ei.size(1)):
            s_local = int(ei[0, j].item())
            d_local = int(ei[1, j].item())

            s_orig = int(src_nid[s_local].item()) if src_nid is not None else s_local
            d_orig = int(dst_nid[d_local].item()) if dst_nid is not None else d_local

            edges.append(
                {
                    "source": node_key(src_t, s_orig),
                    "target": node_key(dst_t, d_orig),
                    "etype": f"{src_t}__{rel}__{dst_t}",
                }
            )

    pred = float(summary.get("pred", interpret.get("pred", 0.0)))

    return {
        "file": file,
        "seed_idx": seed_idx,
        "pred": pred,
        "type_mass": summary.get("type_mass", []),
        "top_feats": top_feats_map,      # normalized [{feat,value,abs}]
        "node_feats": node_feats,        # node_id -> feats (currently type-level fallback)
        "nodes": nodes,
        "edges": edges,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("interpret_server.app:app", host="127.0.0.1", port=8000, reload=True)
