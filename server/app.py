from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .utils import build_graph_summary, get_node_payload

GRAPH_PATH = Path("data/graph/sdge.pt")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SDGE HeteroGraph Server", version="0.5")

DATA = None


@app.on_event("startup")
def _startup():
    global DATA
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH.resolve()}")

    obj = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]

    DATA = obj
    assert hasattr(DATA, "node_types") and hasattr(DATA, "edge_types"), "Loaded object not HeteroData-like"

    print(
        f"[startup] loaded graph: "
        f"node_types={list(DATA.node_types)} "
        f"edge_types={len(DATA.edge_types)}"
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/graph/summary")
def graph_summary(
    node_limit: int = Query(2000, ge=50, le=8000),
    edge_limit: int = Query(8000, ge=50, le=40000),
    seed: int = Query(0, ge=0, le=10_000_000),
    use_pos_if_exists: bool = Query(True),

    # assignment-anchored triad knobs
    anchor_assignments: int = Query(1500, ge=10, le=20000, description="Target number of 'good' assignments"),
    probe_be_edges: int = Query(20000, ge=100, le=200000, description="How many assignment->engineer edges to probe"),
    probe_ab_edges: int = Query(40000, ge=100, le=400000, description="How many tasks->assignment edges to probe"),

    cap_be: int = Query(6000, ge=10, le=40000, description="Max edges assignments->engineers in final subgraph"),
    cap_ab: int = Query(6000, ge=10, le=40000, description="Max edges tasks->assignments in final subgraph"),

    include_extra: bool = Query(True),
    extra_cap_per_rel: int = Query(2500, ge=10, le=40000),
):
    return build_graph_summary(
        DATA,
        node_limit=node_limit,
        edge_limit=edge_limit,
        seed=seed,
        use_pos_if_exists=use_pos_if_exists,
        anchor_assignments=anchor_assignments,
        probe_be_edges=probe_be_edges,
        probe_ab_edges=probe_ab_edges,
        cap_be=cap_be,
        cap_ab=cap_ab,
        include_extra=include_extra,
        extra_cap_per_rel=extra_cap_per_rel,
    )


@app.get("/node/{ntype}/{nid}")
def node_detail(
    ntype: str,
    nid: int,
    include: Optional[str] = Query(None, description="Comma-separated fields to include (default: filtered)"),
    include_x: bool = Query(False, description="If true, allow sending x if it is small enough"),
):
    fields = None
    if include:
        fields = [s.strip() for s in include.split(",") if s.strip()]
    return get_node_payload(DATA, ntype=ntype, nid=nid, fields=fields, include_x=include_x)

@app.get("/graph/ego/{ntype}/{nid}")
def graph_ego(
    ntype: str,
    nid: int,
    hops: int = Query(2, ge=1, le=4),
    max_nodes: int = Query(800, ge=50, le=5000),
    max_edges: int = Query(2000, ge=50, le=20000),
    per_hop_edge_cap: int = Query(3000, ge=50, le=50000),
    seed: int = Query(0, ge=0, le=10_000_000),
    use_pos_if_exists: bool = Query(True),
):
    """
    Return an ego (n-hop) subgraph centered at (ntype, nid).
    """
    from .utils import build_ego_summary

    return build_ego_summary(
        DATA,
        center_type=ntype,
        center_id=nid,
        hops=hops,
        max_nodes=max_nodes,
        max_edges=max_edges,
        per_hop_edge_cap=per_hop_edge_cap,
        seed=seed,
        use_pos_if_exists=use_pos_if_exists,
    )