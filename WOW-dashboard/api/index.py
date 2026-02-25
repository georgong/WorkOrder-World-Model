"""
FastAPI backend for the Capstone Dashboard.
Vercel Python Serverless Runtime entry point.

Endpoints:
  GET  /api/health    → health check
  POST /api/predict   → upload 7 dataset CSVs, build graph, run GNN inference
"""
from __future__ import annotations

import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Graph-building pipeline ─────────────────────────────────────────────────
from .inference.graph_inference_api import (
    EXPECTED_FILES,
    DEFAULT_CONFIG_PATH,
    DEFAULT_UPLOAD_ROOT,
    build_graph as build_graph_from_dir,
    validate_data_dir,
)

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="WorkOrder Risk Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded globals ─────────────────────────────────────────────────────
_device = torch.device("cpu")

MODEL_VERSION = "v1"

# ── Demo data generation ────────────────────────────────────────────────────
DEMO_DISTRICTS = ["North", "South", "East", "West", "Central", "Downtown", "Suburb-A", "Suburb-B"]
DEMO_DEPARTMENTS = ["Electrical", "Plumbing", "HVAC", "Structural", "General Maintenance", "Landscaping"]
DEMO_ENGINEERS = [f"ENG-{i:03d}" for i in range(1, 16)]
DEMO_TASK_TYPES = ["Repair", "Inspection", "Installation", "Replacement", "Emergency"]
DEMO_STATUSES = ["Scheduled", "In Progress", "Pending Parts", "On Hold"]

NUM_DEMO_ASSIGNMENTS = 200


def _generate_demo_records() -> List[Dict[str, Any]]:
    """Generate a list of realistic-looking assignment records."""
    rng = np.random.default_rng(seed=2024)
    records = []
    for i in range(NUM_DEMO_ASSIGNMENTS):
        district = rng.choice(DEMO_DISTRICTS)
        department = rng.choice(DEMO_DEPARTMENTS)
        engineer = rng.choice(DEMO_ENGINEERS)
        task_type = rng.choice(DEMO_TASK_TYPES)
        status = rng.choice(DEMO_STATUSES)

        # Base duration depends on task type
        base_hours = {
            "Repair": 4.0, "Inspection": 1.5, "Installation": 6.0,
            "Replacement": 5.0, "Emergency": 3.0,
        }[task_type]
        duration = max(0.5, base_hours + rng.normal(0, base_hours * 0.4))

        records.append({
            "assignment_id": f"WO-{2024_0000 + i:08d}",
            "district": district,
            "department": department,
            "engineer_id": engineer,
            "task_type": task_type,
            "status": status,
            "duration": round(duration, 2),
        })
    return records


def _generate_demo_predictions(records: List[Dict[str, Any]]) -> np.ndarray:
    """Produce synthetic predicted completion hours that correlate with record attributes."""
    rng = np.random.default_rng(seed=2025)
    preds = []
    for r in records:
        base = r["duration"]
        # Add district-based noise (some districts are "harder")
        district_offset = {"Downtown": 2.0, "Central": 1.5, "East": 1.0}.get(r["district"], 0.0)
        # Emergency tasks often take longer than estimated
        emergency_bump = 2.5 if r["task_type"] == "Emergency" else 0.0
        noise = rng.normal(0, 1.0)
        pred = max(0.3, base + district_offset + emergency_bump + noise)
        preds.append(pred)
    return np.array(preds)


# ── Pydantic schemas ────────────────────────────────────────────────────────
class ScheduleMetrics(BaseModel):
    overall_risk_score: float
    workload_imbalance_score: float
    total_assignments: int
    avg_predicted_hours: float
    median_predicted_hours: float
    most_overloaded_engineer: str
    highest_risk_district: str
    highest_risk_department: str


class AssignmentPrediction(BaseModel):
    assignment_id: str
    pred_completion_hours: float
    risk_score: float
    top_factors: List[str]


class ChartData(BaseModel):
    risk_histogram: List[Dict[str, Any]]
    risk_by_district: List[Dict[str, Any]]
    workload_by_engineer: List[Dict[str, Any]]
    risk_by_department: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    schedule_metrics: ScheduleMetrics
    assignment_predictions: List[AssignmentPrediction]
    charts: ChartData
    metadata: Dict[str, Any]

class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    node_types: List[str]
    edge_types: List[str]


# ── Model loading ───────────────────────────────────────────────────────────
_ASSETS_DIR = Path(__file__).parent / "assets"


def _find_asset(name: str) -> Path:
    """Search common locations for an asset file."""
    candidates = [
        _ASSETS_DIR / name,
        Path(__file__).parent.parent / "data" / "graph" / name,
        Path("data/graph") / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find asset '{name}' in any of {[str(c) for c in candidates]}")


def _load_model_for_graph(graph):
    """
    Load the trained model, rebuilt to match the metadata of a freshly-built *graph*.
    """
    ckpt_path = _find_asset("model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    in_dims = {}
    for nt in graph.node_types:
        if hasattr(graph[nt], "x") and graph[nt].x is not None:
            in_dims[nt] = graph[nt].x.shape[1]
        else:
            in_dims[nt] = 1

    from api.inference.model import build_model
    model = build_model(
        metadata=graph.metadata(),
        in_dims=in_dims,
        hidden_dim=ckpt.get("args", {}).get("hidden", 64),
        num_layers=ckpt.get("args", {}).get("layers", 2),
        target="assignments",
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(_device)
    return model


# ── Helpers ─────────────────────────────────────────────────────────────────
def _compute_risk_scores(predictions: np.ndarray, threshold_hours: float = 8.0) -> np.ndarray:
    scale = threshold_hours * 0.3
    risk = 1.0 / (1.0 + np.exp(-(predictions - threshold_hours) / max(scale, 0.1)))
    return np.clip(risk, 0.0, 1.0)


def _compute_workload_imbalance(engineer_counts: Dict[str, int]) -> float:
    if not engineer_counts or len(engineer_counts) <= 1:
        return 0.0
    counts = np.array(list(engineer_counts.values()), dtype=float)
    if counts.sum() == 0:
        return 0.0
    counts_sorted = np.sort(counts)
    n = len(counts_sorted)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * counts_sorted)) / (n * np.sum(counts_sorted)) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def _compute_congestion(district_counts: Dict[str, int], threshold: int = 10) -> float:
    if not district_counts:
        return 0.0
    congested = sum(1 for c in district_counts.values() if c > threshold)
    return congested / len(district_counts)


def _build_histogram(values: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    counts, edges = np.histogram(values, bins=bins)
    result = []
    for i in range(len(counts)):
        result.append({
            "bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
            "count": int(counts[i]),
            "binStart": float(edges[i]),
            "binEnd": float(edges[i+1]),
        })
    return result


def _aggregate_by_key(records: List[Dict], key: str, value_key: str) -> List[Dict[str, Any]]:
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        k = r.get(key, "unknown")
        v = r.get(value_key, 0.0)
        groups[str(k)].append(v)
    return [
        {"name": k, "avg_risk": float(np.mean(vs)), "count": len(vs)}
        for k, vs in sorted(groups.items())
    ]


def _save_upload_to_disk(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in upload.file:
            f.write(chunk)


def _build_result_from_predictions(
    preds: np.ndarray,
    records: List[Dict],
) -> Dict:
    n = len(records)
    risk_scores = _compute_risk_scores(preds)

    assignment_preds = []
    engineer_counts: Dict[str, int] = {}
    district_counts: Dict[str, int] = {}
    for i, r in enumerate(records):
        aid = r.get("assignment_id", r.get("ASSIGNMENT_ID", f"A{i:04d}"))
        factors = []
        if preds[i] > 8.0:
            factors.append("long predicted duration")
        if risk_scores[i] > 0.7:
            factors.append("high risk zone")
        if r.get("district"):
            factors.append(f"district: {r.get('district')}")
        if r.get("department"):
            factors.append(f"department: {r.get('department')}")
        if not factors:
            factors.append("normal workload")

        assignment_preds.append({
            "assignment_id": str(aid),
            "pred_completion_hours": round(float(preds[i]), 2),
            "risk_score": round(float(risk_scores[i]), 4),
            "top_factors": factors[:3],
        })

        if r.get("engineer_id"):
            eng = r["engineer_id"]
            engineer_counts[eng] = engineer_counts.get(eng, 0) + 1
        
        if r.get("district"):
            dist = r["district"]
            district_counts[dist] = district_counts.get(dist, 0) + 1

    # Most overloaded engineer
    if engineer_counts:
        most_overloaded_engineer = max(engineer_counts.items(), key=lambda x: x[1])[0]
    else:
        most_overloaded_engineer = "N/A"

    # Highest risk district
    risk_by_district = _aggregate_by_key(
        [{"district": r.get('district'), "risk": s}
         for r, s in zip(records, risk_scores) if r.get("district")],
        "district", "risk"
    )
    if risk_by_district:
        highest_risk_district = max(risk_by_district, key=lambda x: x["avg_risk"])["name"]
    else:
        highest_risk_district = "N/A"

    # Highest risk department
    risk_by_department = _aggregate_by_key(
        [{"department": r.get("department"), "risk": s}
         for r, s in zip(records, risk_scores) if r.get("department")],
        "department", "risk"
    )
    if risk_by_department:
        highest_risk_department = max(risk_by_department, key=lambda x: x["avg_risk"])["name"]
    else:
        highest_risk_department = "N/A"

    schedule_metrics = {
        "overall_risk_score": round(float(np.mean(risk_scores)), 4),
        "workload_imbalance_score": round(_compute_workload_imbalance(engineer_counts), 4),
        "total_assignments": n,
        "avg_predicted_hours": round(float(np.mean(preds)), 2),
        "median_predicted_hours": round(float(np.median(preds)), 2),
        "most_overloaded_engineer": most_overloaded_engineer,
        "highest_risk_district": highest_risk_district,
        "highest_risk_department": highest_risk_department,
    }

    # Remove keys that are not in the schema
    schedule_metrics = {k: v for k, v in schedule_metrics.items() if k in ScheduleMetrics.model_fields}

    charts = {
        "risk_histogram": _build_histogram(risk_scores, bins=10),
        "risk_by_district": _aggregate_by_key(
            [{"district": r.get("district"), "risk": s}
             for r, s in zip(records, risk_scores) if r.get("district")],
            "district", "risk"
        ),
        "workload_by_engineer": [
            {"name": k, "assignments": v}
            for k, v in sorted(engineer_counts.items(), key=lambda x: -x[1])[:20]
        ],
        "risk_by_department": _aggregate_by_key(
            [{"department": r.get("department"), "risk": s}
             for r, s in zip(records, risk_scores) if r.get("department")],
            "department", "risk"
        ),
    }

    return {
        "schedule_metrics": schedule_metrics,
        "assignment_predictions": assignment_preds,
        "charts": charts,
    }


# ── Demo inference (fallback when model assets unavailable) ─────────────────
def _run_demo_inference(records: List[Dict]) -> Dict:
    np.random.seed(42)
    preds = []
    for r in records:
        base = 2.0 + np.random.exponential(3.0)
        if r.get("duration"):
            try:
                base = float(r["duration"]) + np.random.normal(0, 0.5)
            except (ValueError, TypeError):
                pass
        preds.append(max(0.1, base))
    preds = np.array(preds)
    return _build_result_from_predictions(preds, records)


# ── Model inference on a freshly-built graph ────────────────────────────────
@torch.no_grad()
def _run_graph_inference(graph, records: List[Dict]) -> Dict:
    """
    Run GNN inference on a freshly-built HeteroData graph.
    Falls back to demo inference if model checkpoint is not found.
    """
    try:
        model = _load_model_for_graph(graph)
    except FileNotFoundError:
        print("[API] Model checkpoint not found — falling back to demo inference")
        return _run_demo_inference(records)

    data = graph.to(_device)
    out = model(data)

    if isinstance(out, dict):
        raw = out.get("pred")
        if raw is None:
            raw = out.get("assignments")
        if raw is None:
            raw = next(iter(out.values()))
        preds_all = raw.cpu().numpy()
    else:
        preds_all = out.cpu().numpy()

    preds_all = preds_all.ravel()

    n = len(records)
    if n <= len(preds_all):
        preds = preds_all[:n]
    else:
        preds = np.concatenate([
            preds_all,
            np.full(n - len(preds_all), np.median(preds_all)),
        ])

    return _build_result_from_predictions(preds, records)

def _build_complete_records_from_graph(graph) -> List[Dict]:
    """
    Generate complete assignment records using only the graph structure and relationships.
    """
    assignments = graph["assignments"]
    node_ids = assignments.node_ids
    records = []

    # Build lookup maps for relationships
    def build_edge_map(src_type, rel, dst_type):
        edge_key = (src_type, rel, dst_type)
        if edge_key in graph.edge_types:
            edge_index = graph[edge_key].edge_index
            return {int(src): int(dst) for src, dst in zip(edge_index[0], edge_index[1])}
        return {}

    # Assignment relationships
    task_map = build_edge_map("assignments", "relates_to", "tasks")
    engineer_map = build_edge_map("assignments", "relates_to", "engineers")

    # Task relationships
    status_map = build_edge_map("tasks", "relates_to", "task_statuses")
    type_map = build_edge_map("tasks", "relates_to", "task_types")
    district_map = build_edge_map("tasks", "relates_to", "districts")
    department_map = build_edge_map("tasks", "relates_to", "departments")

    # Helper to get attribute from node store
    def get_attr(store, idx):
        return store['node_ids'][idx]

    for idx in range(assignments.num_nodes):
        record = {
            "assignment_id": str(node_ids[idx]),
            "duration": float(assignments.y[idx]) if hasattr(assignments, "y") else None,
        }

        # Engineer info
        eng_idx = engineer_map.get(idx)
        if eng_idx is not None:
            print()
            record["engineer_id"] = get_attr(graph["engineers"], eng_idx)

        # Task info
        task_idx = task_map.get(idx)
        if task_idx is not None:
            record["task_id"] = str(task_idx)
            # Task status
            status_idx = status_map.get(task_idx)
            if status_idx is not None:
                record["status"] = get_attr(graph["task_statuses"], status_idx)
            # Task type
            type_idx = type_map.get(task_idx)
            if type_idx is not None:
                record["task_type"] = get_attr(graph["task_types"], type_idx)
            # District
            district_idx = district_map.get(task_idx)
            if district_idx is not None:
                record["district"] = get_attr(graph["districts"], district_idx)
            # Department
            department_idx = department_map.get(task_idx)
            if department_idx is not None:
                record["department"] = get_attr(graph["departments"], department_idx)

        records.append(record)

    return records

# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/api/demo", response_model=PredictResponse)
async def demo():
    """
    Demo endpoint — returns realistic synthetic predictions without requiring
    file uploads or a trained model.  Useful for frontend development and demos.
    """
    t0 = time.time()
    request_id = str(uuid.uuid4())[:8]

    records = _generate_demo_records()
    preds = _generate_demo_predictions(records)
    result = _build_result_from_predictions(preds, records)

    elapsed_ms = int((time.time() - t0) * 1000)

    return PredictResponse(
        schedule_metrics=ScheduleMetrics(**result["schedule_metrics"]),
        assignment_predictions=[
            AssignmentPrediction(**p) for p in result["assignment_predictions"]
        ],
        charts=ChartData(**result["charts"]),
        metadata={
            "runtime_ms": elapsed_ms,
            "model_version": MODEL_VERSION,
            "request_id": request_id,
            "session_id": "demo",
            "mode": "demo",
            "num_assignments": NUM_DEMO_ASSIGNMENTS,
            "note": "Synthetic data — no real model inference was performed.",
        },
    )


@app.post("/api/predict")
async def predict(
    W6ASSIGNMENTS: UploadFile = File(...),
    W6DEPARTMENT: UploadFile = File(...),
    W6DISTRICTS: UploadFile = File(...),
    W6ENGINEERS: UploadFile = File(...),
    W6TASK_STATUSES: UploadFile = File(...),
    W6TASK_TYPES: UploadFile = File(...),
    W6TASKS: UploadFile = File(...),
    config_path: str = Query(DEFAULT_CONFIG_PATH, description="Path to graph YAML config"),
):
    """
    End-to-end prediction pipeline:
      1. Accept 7 required CSV file uploads
      2. Build a HeteroData graph via GraphBuilder
      3. Load the trained GNN model
      4. Run inference on the freshly-built graph
      5. Return schedule-level risk metrics + per-assignment predictions
    """
    t0 = time.time()
    request_id = str(uuid.uuid4())[:8]

    uploads = {
        "W6ASSIGNMENTS.csv": W6ASSIGNMENTS,
        "W6DEPARTMENT.csv": W6DEPARTMENT,
        "W6DISTRICTS.csv": W6DISTRICTS,
        "W6ENGINEERS.csv": W6ENGINEERS,
        "W6TASK_STATUSES.csv": W6TASK_STATUSES,
        "W6TASK_TYPES.csv": W6TASK_TYPES,
        "W6TASKS.csv": W6TASKS,
    }

    session_id = uuid.uuid4().hex[:12]
    staging_dir = Path(DEFAULT_UPLOAD_ROOT) / session_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        for canonical_name, upload_file in uploads.items():
            _save_upload_to_disk(upload_file, staging_dir / canonical_name)

        missing = validate_data_dir(staging_dir)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required CSV file(s): {missing}. "
                       f"Expected all of: {EXPECTED_FILES}",
            )

        # Build graph
        graph = build_graph_from_dir(staging_dir, config_path=config_path, save_path="data/graph/last_graph.pt")
        records = _build_complete_records_from_graph(graph)

        # Extract assignment-level records from the graph
        # assign_node_ids = (
            # graph["assignments"].node_ids
            # if hasattr(graph["assignments"], "node_ids")
            # else list(range(graph["assignments"].num_nodes))
        # )
        # records = [{"assignment_id": str(aid)} for aid in assign_node_ids]

        # Run model inference on the freshly-built graph
        result = _run_graph_inference(graph, records)

        elapsed_ms = int((time.time() - t0) * 1000)

        # Graph summary for metadata
        node_summary = {}
        for ntype in graph.node_types:
            ns = graph[ntype]
            node_summary[ntype] = {
                "num_nodes": int(ns.num_nodes),
                "feature_dim": int(ns.x.shape[1]) if hasattr(ns, "x") and ns.x is not None else 0,
            }

        edge_summary = {}
        for etype in graph.edge_types:
            ei = graph[etype].edge_index
            edge_summary[str(etype)] = {"num_edges": int(ei.shape[1])}

        return PredictResponse(
            schedule_metrics=ScheduleMetrics(**result["schedule_metrics"]),
            assignment_predictions=[
                AssignmentPrediction(**p) for p in result["assignment_predictions"]
            ],
            charts=ChartData(**result["charts"]),
            metadata={
                "runtime_ms": elapsed_ms,
                "model_version": MODEL_VERSION,
                "request_id": request_id,
                "session_id": session_id,
                "mode": "graph",
                "graph_nodes": node_summary,
                "graph_edges": edge_summary,
            },
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph build / inference error: {str(e)}")
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

# ── Graph serialization for visualization ──────────────────────────────────
def serialize_graph(graph, max_nodes=300):
    """
    Serialize the constructed graph to a node/edge list for visualization.
    If the graph is too large, sample a subgraph centered on assignments.
    """
    import random
    # Collect all nodes
    nodes = []
    node_id_map = {}  # (type, idx) -> global id
    node_types = list(graph.node_types)
    node_count = 0
    for ntype in node_types:
        ns = graph[ntype]
        for i in range(ns.num_nodes):
            node_id = f"{ntype}:{getattr(ns, 'node_ids', [str(i)]*ns.num_nodes)[i]}"
            node = {
                "id": node_id,
                "type": ntype,
                "label": str(getattr(ns, 'node_ids', [str(i)]*ns.num_nodes)[i]),
            }
            nodes.append(node)
            node_id_map[(ntype, i)] = node_id
            node_count += 1

    # If too many nodes, sample a subgraph centered on assignments
    if node_count > max_nodes and "assignments" in node_types:
        # Sample a subset of assignments
        assign_ns = graph["assignments"]
        num_assign = assign_ns.num_nodes
        sample_size = min(max_nodes // 3, num_assign)
        sample_idxs = set(random.sample(range(num_assign), sample_size))
        keep_nodes = set()
        for idx in sample_idxs:
            keep_nodes.add(node_id_map[("assignments", idx)])
        # Add neighbors via edges
        for etype in graph.edge_types:
            ei = graph[etype].edge_index
            src_type, _, dst_type = etype
            for src, dst in zip(ei[0], ei[1]):
                src_id = node_id_map.get((src_type, int(src)))
                dst_id = node_id_map.get((dst_type, int(dst)))
                if src_type == "assignments" and src_id in keep_nodes:
                    keep_nodes.add(dst_id)
                if dst_type == "assignments" and dst_id in keep_nodes:
                    keep_nodes.add(src_id)
        # Filter nodes
        nodes = [n for n in nodes if n["id"] in keep_nodes]

    # Collect all edges
    edges = []
    for etype in graph.edge_types:
        ei = graph[etype].edge_index
        src_type, rel, dst_type = etype
        for src, dst in zip(ei[0], ei[1]):
            src_id = node_id_map.get((src_type, int(src)))
            dst_id = node_id_map.get((dst_type, int(dst)))
            if src_id is not None and dst_id is not None:
                edges.append({
                    "source": src_id,
                    "target": dst_id,
                    "type": rel,
                    "source_type": src_type,
                    "target_type": dst_type,
                })
    return {"nodes": nodes, "edges": edges, "node_types": node_types, "edge_types": [str(e) for e in graph.edge_types]}

# ── Graph API endpoint ─────────────────────────────────────────────────────
@app.post("/api/graph")
async def get_graph(
    W6ASSIGNMENTS: UploadFile = File(...),
    W6DEPARTMENT: UploadFile = File(...),
    W6DISTRICTS: UploadFile = File(...),
    W6ENGINEERS: UploadFile = File(...),
    W6TASK_STATUSES: UploadFile = File(...),
    W6TASK_TYPES: UploadFile = File(...),
    W6TASKS: UploadFile = File(...),
    config_path: str = Query(DEFAULT_CONFIG_PATH, description="Path to graph YAML config"),
    max_nodes: int = 300,
):
    """
    Returns the constructed graph (nodes and edges) for visualization.
    If the graph is too large, samples a subgraph.
    """
    session_id = uuid.uuid4().hex[:12]
    staging_dir = Path(DEFAULT_UPLOAD_ROOT) / session_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    uploads = {
        "W6ASSIGNMENTS.csv": W6ASSIGNMENTS,
        "W6DEPARTMENT.csv": W6DEPARTMENT,
        "W6DISTRICTS.csv": W6DISTRICTS,
        "W6ENGINEERS.csv": W6ENGINEERS,
        "W6TASK_STATUSES.csv": W6TASK_STATUSES,
        "W6TASK_TYPES.csv": W6TASK_TYPES,
        "W6TASKS.csv": W6TASKS,
    }
    try:
        for canonical_name, upload_file in uploads.items():
            _save_upload_to_disk(upload_file, staging_dir / canonical_name)
        missing = validate_data_dir(staging_dir)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required CSV file(s): {missing}. "
                       f"Expected all of: {EXPECTED_FILES}",
            )
        graph = build_graph_from_dir(staging_dir, config_path=config_path, save_path=None)
        graph_json = serialize_graph(graph, max_nodes=max_nodes)
        return GraphResponse(
            nodes=graph_json['nodes'], 
            edges=graph_json['edges'],
            node_types=graph_json['node_types'],
            edge_types=graph_json['edge_types']
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph build/serialization error: {str(e)}")
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)