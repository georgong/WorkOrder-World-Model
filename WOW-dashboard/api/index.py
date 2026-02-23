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


# ── Pydantic schemas ────────────────────────────────────────────────────────
class ScheduleMetrics(BaseModel):
    overall_risk_score: float
    expected_overdue_rate: float
    workload_imbalance_score: float
    congestion_score: float
    total_assignments: int
    avg_predicted_hours: float
    median_predicted_hours: float


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


# ── Model loading ───────────────────────────────────────────────────────────
def _find_asset(name: str) -> Path:
    """Search common locations for an asset file."""
    candidates = [
        Path(__file__).parent / "assets" / name,
        Path(__file__).parent.parent / "data" / "graph" / name,
        Path("data/graph") / name,
        Path("api/assets") / name,
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
    for i, r in enumerate(records):
        aid = r.get("assignment_id", r.get("ASSIGNMENT_ID", f"A{i:04d}"))
        factors = []
        if preds[i] > 8.0:
            factors.append("long predicted duration")
        if risk_scores[i] > 0.7:
            factors.append("high risk zone")
        if r.get("district", r.get("DISTRICT")):
            factors.append(f"district: {r.get('district', r.get('DISTRICT'))}")
        if r.get("department", r.get("DEPARTMENT")):
            factors.append(f"department: {r.get('department', r.get('DEPARTMENT'))}")
        if not factors:
            factors.append("normal workload")

        assignment_preds.append({
            "assignment_id": str(aid),
            "pred_completion_hours": round(float(preds[i]), 2),
            "risk_score": round(float(risk_scores[i]), 4),
            "top_factors": factors[:3],
        })

    engineer_counts: Dict[str, int] = {}
    district_counts: Dict[str, int] = {}
    for r in records:
        eng = str(r.get("engineer_id", r.get("ASSIGNEDENGINEERS", "unknown")))
        dist = str(r.get("district", r.get("DISTRICT", "unknown")))
        engineer_counts[eng] = engineer_counts.get(eng, 0) + 1
        district_counts[dist] = district_counts.get(dist, 0) + 1

    overdue_rate = float(np.mean(risk_scores > 0.5))

    schedule_metrics = {
        "overall_risk_score": round(float(np.mean(risk_scores)), 4),
        "expected_overdue_rate": round(overdue_rate, 4),
        "workload_imbalance_score": round(_compute_workload_imbalance(engineer_counts), 4),
        "congestion_score": round(_compute_congestion(district_counts), 4),
        "total_assignments": n,
        "avg_predicted_hours": round(float(np.mean(preds)), 2),
        "median_predicted_hours": round(float(np.median(preds)), 2),
    }

    charts = {
        "risk_histogram": _build_histogram(risk_scores, bins=10),
        "risk_by_district": _aggregate_by_key(
            [{"district": r.get("district", r.get("DISTRICT", "unknown")), "risk": s}
             for r, s in zip(records, risk_scores)],
            "district", "risk"
        ),
        "workload_by_engineer": [
            {"name": k, "assignments": v}
            for k, v in sorted(engineer_counts.items(), key=lambda x: -x[1])[:20]
        ],
        "risk_by_department": _aggregate_by_key(
            [{"department": r.get("department", r.get("DEPARTMENT", "unknown")), "risk": s}
             for r, s in zip(records, risk_scores)],
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
        preds_all = (out.get("pred") or out.get("assignments") or next(iter(out.values()))).cpu().numpy()
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


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


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
        graph = build_graph_from_dir(staging_dir, config_path=config_path)

        # Extract assignment-level records from the graph
        assign_node_ids = (
            graph["assignments"].node_ids
            if hasattr(graph["assignments"], "node_ids")
            else list(range(graph["assignments"].num_nodes))
        )
        records = [{"assignment_id": str(aid)} for aid in assign_node_ids]

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
