"""
API entry point:
  1. Accept uploaded CSV files (W6ASSIGNMENTS.csv, W6TASKS.csv, etc.)
  2. Build a HeteroData graph via GraphBuilder
  3. Run inference with a trained model
  4. Return predictions
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch_geometric.data import HeteroData

from .structure_graph_builder import GraphBuilder 

# Expected file stems that the graph builder knows how to load
EXPECTED_FILES = [
    "W6ASSIGNMENTS.csv",
    "W6DEPARTMENT.csv",
    "W6DISTRICTS.csv",
    "W6ENGINEERS.csv",
    "W6TASK_STATUSES.csv",
    "W6TASK_TYPES.csv",
    "W6TASKS.csv",
]

# Default paths — override via function args or env vars
DEFAULT_CONFIG_PATH = os.environ.get(
    "GRAPH_YAML_PATH",
    os.path.join(os.path.dirname(__file__), "..", "assets", "graph.yaml"),
)
DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "assets", "model.pt"),
)
DEFAULT_UPLOAD_ROOT = os.environ.get("UPLOAD_ROOT", "data/uploads")


# ------------------------------------------------------------------
# 1. Validate & stage uploaded files
# ------------------------------------------------------------------
def validate_data_dir(data_dir: str | Path) -> List[str]:
    """Return list of missing files (empty list = all present)."""
    data_dir = Path(data_dir)
    return [f for f in EXPECTED_FILES if not (data_dir / f).exists()]


def stage_uploaded_files(
    file_map: Dict[str, str | Path],
    *,
    upload_root: str | Path = DEFAULT_UPLOAD_ROOT,
) -> Path:
    """
    Copy / move uploaded CSV files into a unique working directory.

    Parameters
    ----------
    file_map : dict
        Mapping of canonical filename (e.g. "W6ASSIGNMENTS.csv") to the
        actual path on disk where the upload was saved.
    upload_root : str | Path
        Parent directory under which a unique session folder is created.

    Returns
    -------
    Path
        The session data directory containing all staged CSVs.
    """
    session_id = uuid.uuid4().hex[:12]
    data_dir = Path(upload_root) / session_id
    data_dir.mkdir(parents=True, exist_ok=True)

    for canonical_name, src_path in file_map.items():
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"Uploaded file not found: {src}")
        dst = data_dir / canonical_name
        shutil.copy2(src, dst)

    missing = validate_data_dir(data_dir)
    if missing:
        raise FileNotFoundError(
            f"Missing required CSV file(s) in upload: {missing}. "
            f"Expected: {EXPECTED_FILES}"
        )

    return data_dir


# ------------------------------------------------------------------
# 2. Build graph from staged data
# ------------------------------------------------------------------
def build_graph(
    data_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    save_path: Optional[str | Path] = None,
) -> HeteroData:
    """
    Run the full GraphBuilder pipeline on files in *data_dir*.

    Parameters
    ----------
    data_dir : path
        Directory containing the uploaded CSV files.
    config_path : path
        Path to the graph YAML config (``configs/graph.yaml``).
    save_path : path, optional
        If given, persist the built graph to this ``.pt`` file.

    Returns
    -------
    HeteroData
    """
    data_dir = Path(data_dir)
    missing = validate_data_dir(data_dir)
    if missing:
        raise FileNotFoundError(
            f"Cannot build graph — missing files in {data_dir}: {missing}"
        )

    with open(config_path, "r") as f:
        schema = yaml.safe_load(f)

    gb = GraphBuilder(yaml=schema, data_dir=str(data_dir))
    graph = gb.build()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, save_path)
        print(f"[graph_inference_api] Graph saved to {save_path}")

    return graph


# ------------------------------------------------------------------
# 3. Load model and run inference
# ------------------------------------------------------------------
def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load a trained PyTorch (Geometric) model from disk.

    The checkpoint is expected to be either:
      - A plain ``state_dict`` (requires you to instantiate the model class first), or
      - A full model saved via ``torch.save(model, path)``.

    Adjust the logic below to match your actual saving convention.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.eval()
        return checkpoint

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raise NotImplementedError(
            "Checkpoint is a state_dict. Instantiate your model class in "
            "load_model() and load the weights."
        )

    raise TypeError(
        f"Unrecognised checkpoint format ({type(checkpoint)}). "
        "Expected a torch.nn.Module or a dict with 'model_state_dict'."
    )


def run_inference(
    graph: HeteroData,
    model: torch.nn.Module,
    *,
    device: str = "cpu",
    target_node_type: str = "assignments",
) -> Dict[str, Any]:
    """
    Run a forward pass on the built graph and return predictions.

    Parameters
    ----------
    graph : HeteroData
    model : torch.nn.Module
    device : str
    target_node_type : str
        The node type whose predictions we care about.

    Returns
    -------
    dict
        ``{"node_ids": [...], "predictions": Tensor}``
    """
    graph = graph.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(graph)

    if isinstance(out, dict):
        preds = out.get(target_node_type, out)
    elif isinstance(out, torch.Tensor):
        preds = out
    else:
        preds = out

    node_ids = (
        graph[target_node_type].node_ids
        if hasattr(graph[target_node_type], "node_ids")
        else list(range(graph[target_node_type].num_nodes))
    )

    return {
        "node_ids": node_ids,
        "predictions": preds,
    }


# ------------------------------------------------------------------
# 4. End-to-end convenience function
# ------------------------------------------------------------------
def predict_from_uploads(
    file_map: Dict[str, str | Path],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str = "cpu",
    target_node_type: str = "assignments",
    save_graph_path: Optional[str | Path] = None,
    cleanup: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: stage files → build graph → load model → inference.

    Parameters
    ----------
    file_map : dict
        ``{"W6ASSIGNMENTS.csv": "/tmp/upload_abc123.csv", ...}``
    config_path : path
        YAML config for graph builder.
    model_path : path
        Trained model checkpoint.
    device : str
    target_node_type : str
    save_graph_path : path, optional
        Persist the intermediate graph for debugging.
    cleanup : bool
        Remove the staged upload directory after inference.

    Returns
    -------
    dict  with keys ``node_ids`` and ``predictions``.
    """
    # 1. Stage
    data_dir = stage_uploaded_files(file_map)
    print(f"[predict_from_uploads] Staged files in {data_dir}")

    try:
        # 2. Build graph
        graph = build_graph(
            data_dir,
            config_path=config_path,
            save_path=save_graph_path,
        )

        # 3. Load model & infer
        model = load_model(model_path, device=device)
        return run_inference(
            graph, model, device=device, target_node_type=target_node_type
        )

    finally:
        if cleanup and data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"[predict_from_uploads] Cleaned up {data_dir}")


# ------------------------------------------------------------------
# 5. Build-only convenience (graph without inference)
# ------------------------------------------------------------------
def build_graph_from_uploads(
    file_map: Dict[str, str | Path],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    save_graph_path: str | Path = "data/graph/uploaded_graph.pt",
    cleanup: bool = False,
) -> HeteroData:
    """
    Stage uploaded files and build a graph (no inference).

    Useful when you only need the ``.pt`` artifact.
    """
    data_dir = stage_uploaded_files(file_map)
    print(f"[build_graph_from_uploads] Staged files in {data_dir}")

    try:
        return build_graph(
            data_dir, config_path=config_path, save_path=save_graph_path
        )
    finally:
        if cleanup and data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)


