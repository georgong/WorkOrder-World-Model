# pca_weights.py
# Load checkpoint, run forward on dataset, collect hidden-layer activations for target nodes,
# run PCA and show results grouped by neighbor-derived labels (e.g. task type, engineer).
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

from tqdm import tqdm
import numpy as np

# Dimensionality reduction: PyTorch PCA (no sklearn required); optional t-SNE via sklearn
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# -------------------------
# metadata extraction (before sanitize)
# -------------------------
def extract_metadata_maps(
    data: HeteroData,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Any]], Dict[str, List[str]]]:
    attr_name_map: Dict[str, List[str]] = {}
    node_ids_map: Dict[str, List[Any]] = {}
    mask_cols_map: Dict[str, List[str]] = {}

    for nt in data.node_types:
        store = data[nt]
        if hasattr(store, "attr_name"):
            an = getattr(store, "attr_name")
            if isinstance(an, (list, tuple)):
                an = list(an)
                if hasattr(store, "x") and isinstance(store.x, torch.Tensor) and store.x.dim() == 2:
                    assert len(an) == int(store.x.size(1)), (
                        f"[attr_name mismatch] {nt}: len(attr_name)={len(an)} != x.shape[1]={int(store.x.size(1))}"
                    )
                attr_name_map[nt] = an
        if hasattr(store, "node_ids"):
            nid = getattr(store, "node_ids")
            if isinstance(nid, (list, tuple)):
                node_ids_map[nt] = list(nid)
            elif isinstance(nid, torch.Tensor) and nid.ndim == 1:
                node_ids_map[nt] = nid.detach().cpu().tolist()
        if hasattr(store, "mask_cols"):
            mc = getattr(store, "mask_cols")
            if isinstance(mc, (list, tuple)):
                mask_cols_map[nt] = list(mc)

    return attr_name_map, node_ids_map, mask_cols_map


def pick_device(s: str) -> torch.device:
    s = s.lower()
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


# -------------------------
# model that returns hidden layers (must match training SAGE)
# -------------------------
class HeteroSAGERegressorWithHidden(nn.Module):
    """Same as HeteroSAGERegressor but forward returns hidden layer activations for target node type."""

    def __init__(self, metadata, in_dims, hidden_dim=128, num_layers=2, target_node_type="assignments"):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.target_node_type = target_node_type

        self.in_proj = nn.ModuleDict({nt: Linear(in_dims[nt], hidden_dim) for nt in self.node_types})
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {et: SAGEConv((-1, -1), hidden_dim) for et in self.edge_types}
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types}))

        self.base = nn.Parameter(torch.tensor(0.0))
        self.out = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData, return_hidden: bool = True):
        x_dict = {nt: F.relu(self.in_proj[nt](data[nt].x)) for nt in self.node_types}
        hidden_layers: List[torch.Tensor] = [x_dict[self.target_node_type].detach()]

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(self.norms[i][k](v)) for k, v in x_dict.items()}
            hidden_layers.append(x_dict[self.target_node_type].detach())

        delta = self.out(x_dict[self.target_node_type]).squeeze(-1)
        pred = self.base + delta

        if return_hidden:
            return {"pred": pred, "hidden_layers": hidden_layers}
        return {"pred": pred}


# -------------------------
# graph prep (match training)
# -------------------------
@torch.no_grad()
def normalize_node_features_inplace(
    data: HeteroData,
    *,
    eps: float = 1e-6,
    drop_const: bool = True,
    const_std_thr: float = 1e-8,
) -> None:
    for nt in data.node_types:
        if not hasattr(data[nt], "x"):
            continue
        x = data[nt].x
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            continue
        if not x.is_floating_point():
            x = x.float()
        x = x.cpu()
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        std = torch.sqrt(var + eps)
        keep = torch.ones_like(std, dtype=torch.bool)
        if drop_const:
            keep = std > const_std_thr
            if keep.sum().item() == 0:
                keep[0] = True
        x2 = (x[:, keep] - mean[keep]) / std[keep]
        data[nt].x = x2
        if hasattr(data[nt], "attr_name"):
            an = data[nt].attr_name
            if isinstance(an, list) and len(an) == int(keep.numel()):
                data[nt].attr_name = [an[i] for i in range(len(an)) if bool(keep[i].item())]


def sanitize_for_neighbor_loader(data: HeteroData) -> HeteroData:
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


def ensure_all_node_types_have_x(data: HeteroData) -> Dict[str, int]:
    in_dims = {nt: data[nt].x.size(-1) for nt in data.node_types if hasattr(data[nt], "x")}
    for nt in data.node_types:
        if nt not in in_dims:
            in_dims[nt] = 1
            data[nt].x = torch.zeros((data[nt].num_nodes, 1), dtype=torch.float)
    return in_dims


def compute_target_degree(data: HeteroData, target: str, *, degree_mode: str = "in") -> torch.Tensor:
    N = data[target].num_nodes
    deg = torch.zeros(N, dtype=torch.long)
    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if degree_mode in ("in", "inout") and dst == target:
            deg += torch.bincount(ei[1], minlength=N)
        if degree_mode in ("out", "inout") and src == target:
            deg += torch.bincount(ei[0], minlength=N)
    return deg


def assignments_with_engineer_and_task_neighbors(
    data: HeteroData,
    target: str,
) -> torch.Tensor:
    """
    Return boolean mask of shape (data[target].num_nodes,) that is True for nodes
    that have at least one engineer neighbor AND at least one task neighbor.
    Use this to sample only assignments that can get both labels (faster, no wasted batches).
    """
    N = data[target].num_nodes
    has_engineer = torch.zeros(N, dtype=torch.bool)
    has_task = torch.zeros(N, dtype=torch.bool)
    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if src == target and dst == "engineers":
            has_engineer[ei[0]] = True
        elif src == "engineers" and dst == target:
            has_engineer[ei[1]] = True
        if src == target and dst == "tasks":
            has_task[ei[0]] = True
        elif src == "tasks" and dst == target:
            has_task[ei[1]] = True
    return has_engineer & has_task


# -------------------------
# checkpoint
# -------------------------
_epoch_re = re.compile(r"_epoch(\d+)\.pt$")
_fold_re = re.compile(r"_fold(\d+)_epoch(\d+)\.pt$")


def load_payload(ckpt_path: Path) -> dict:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" not in payload or "args" not in payload:
        raise ValueError(f"Bad checkpoint format: {ckpt_path}")
    return payload


def sort_ckpts(paths: List[Path]) -> List[Path]:
    def key(p: Path):
        m = _fold_re.search(p.name)
        if m:
            return (0, int(m.group(1)), int(m.group(2)))
        m = _epoch_re.search(p.name)
        if m:
            return (1, 0, int(m.group(1)))
        return (2, 0, 0)

    return sorted(paths, key=key)


# -------------------------
# neighbor-derived labels from batch
# -------------------------
def _find_edge_type(batch: HeteroData, src_nt: str, dst_nt: str) -> Optional[Tuple[str, str, str]]:
    for et in batch.edge_types:
        if et[0] == src_nt and et[2] == dst_nt:
            return et
    return None


def _fill_labels_from_direct_edge(
    batch: HeteroData,
    target: str,
    bs: int,
    other_nt: str,
    out: np.ndarray,
) -> None:
    """Fill out[seed_local] = global index of other_nt for (target <-> other_nt) edges. -1 if missing."""
    et = _find_edge_type(batch, target, other_nt)
    if et is None:
        et = _find_edge_type(batch, other_nt, target)
    if et is None or other_nt not in batch.node_types or not hasattr(batch[other_nt], "n_id"):
        return
    ei = batch[et].edge_index
    if ei.size(1) == 0:
        return
    n_id = batch[other_nt].n_id.cpu().numpy()
    src_is_target = et[0] == target
    for k in range(ei.size(1)):
        seed_local = int(ei[0, k].item()) if src_is_target else int(ei[1, k].item())
        other_local = int(ei[1, k].item()) if src_is_target else int(ei[0, k].item())
        if 0 <= seed_local < bs and 0 <= other_local < n_id.size:
            out[seed_local] = int(n_id[other_local])


def _fill_labels_via_intermediate(
    batch: HeteroData,
    bs: int,
    inter_nt: str,
    inter_labels: np.ndarray,
    dst_nt: str,
    out: np.ndarray,
) -> None:
    """Fill out[i] = global dst_nt index from (inter_nt, *, dst_nt) edges using inter_labels. -1 if missing."""
    if dst_nt not in batch.node_types or not hasattr(batch[dst_nt], "n_id"):
        return
    et = _find_edge_type(batch, inter_nt, dst_nt)
    if et is None:
        et = _find_edge_type(batch, dst_nt, inter_nt)
    if et is None or not hasattr(batch[inter_nt], "n_id"):
        return
    ei = batch[et].edge_index
    if ei.size(1) == 0:
        return
    inter_n_id = batch[inter_nt].n_id.cpu().numpy()
    dst_n_id = batch[dst_nt].n_id.cpu().numpy()
    # map inter_global -> dst_global
    inter_global_to_dst_global: Dict[int, int] = {}
    src_is_inter = et[0] == inter_nt
    for k in range(ei.size(1)):
        i_local = int(ei[0, k].item()) if src_is_inter else int(ei[1, k].item())
        d_local = int(ei[1, k].item()) if src_is_inter else int(ei[0, k].item())
        if i_local < inter_n_id.size and d_local < dst_n_id.size:
            inter_global_to_dst_global[int(inter_n_id[i_local])] = int(dst_n_id[d_local])
    for i in range(bs):
        if inter_labels[i] >= 0:
            out[i] = inter_global_to_dst_global.get(int(inter_labels[i]), -1)


@torch.no_grad()
def get_seed_neighbor_labels(
    batch: HeteroData,
    target: str,
    node_ids_map: Dict[str, List[Any]],
) -> Dict[str, np.ndarray]:
    """
    For each seed (first batch_size nodes of target), get neighbor-based labels.
    Returns dict keyed by neighbor node type: engineers, task_types, tasks, districts, departments, regions, etc.
    Each value is (batch_size,) int64 array of global node index, or -1 if not available.
    """
    bs = int(batch[target].batch_size)
    labels: Dict[str, np.ndarray] = {}

    # Direct: target <-> engineers
    engineer_labels = np.full(bs, -1, dtype=np.int64)
    _fill_labels_from_direct_edge(batch, target, bs, "engineers", engineer_labels)
    labels["engineers"] = engineer_labels

    # Direct: target <-> tasks
    task_labels = np.full(bs, -1, dtype=np.int64)
    _fill_labels_from_direct_edge(batch, target, bs, "tasks", task_labels)
    labels["tasks"] = task_labels

    # Via tasks: task -> task_types, districts, departments, regions
    for dst_nt in ["task_types", "districts", "departments", "regions"]:
        arr = np.full(bs, -1, dtype=np.int64)
        _fill_labels_via_intermediate(batch, bs, "tasks", task_labels, dst_nt, arr)
        labels[dst_nt] = arr

    # Via engineers: engineer -> departments (if not already set from task)
    if "departments" in batch.node_types and hasattr(batch["departments"], "n_id"):
        dept_from_eng = np.full(bs, -1, dtype=np.int64)
        _fill_labels_via_intermediate(batch, bs, "engineers", engineer_labels, "departments", dept_from_eng)
        # fill only where we don't have from task
        for i in range(bs):
            if labels["departments"][i] < 0 and dept_from_eng[i] >= 0:
                labels["departments"][i] = dept_from_eng[i]

    return labels


def _build_full_graph_lookups(data: HeteroData) -> Dict[Tuple[str, str], Dict[int, int]]:
    """
    Build (src_nt, dst_nt) -> {src_global_idx: dst_global_idx} from full graph edges.
    E.g. ("tasks", "task_types"), ("tasks", "districts"), ("engineers", "departments").
    """
    out: Dict[Tuple[str, str], Dict[int, int]] = {}
    for (src, _rel, dst) in data.edge_types:
        key = (src, dst)
        if key not in out:
            out[key] = {}
        ei = data[(src, _rel, dst)].edge_index
        if ei is None or ei.numel() == 0:
            continue
        for j in range(ei.size(1)):
            a, b = int(ei[0, j].item()), int(ei[1, j].item())
            out[key][a] = b
    return out


def enrich_labels_from_graph(
    data: HeteroData,
    labels_dict: Dict[str, np.ndarray],
) -> None:
    """
    Fill task_types, districts, departments, regions from full-graph edges when batch didn't have those nodes.
    Modifies labels_dict in place. Uses labels_dict["tasks"] and labels_dict["engineers"] as sources.
    """
    lookups = _build_full_graph_lookups(data)
    n = labels_dict["tasks"].shape[0] if "tasks" in labels_dict else 0
    if n == 0:
        return

    for dst_nt in ["task_types", "districts", "departments", "regions"]:
        if dst_nt not in labels_dict:
            continue
        arr = labels_dict[dst_nt]
        # From tasks
        key_t = ("tasks", dst_nt)
        if key_t in lookups and "tasks" in labels_dict:
            m = lookups[key_t]
            task_arr = labels_dict["tasks"]
            for i in range(n):
                if arr[i] < 0 and task_arr[i] >= 0:
                    arr[i] = m.get(int(task_arr[i]), -1)
        # From engineers (only for departments)
        if dst_nt == "departments":
            key_e = ("engineers", "departments")
            if key_e in lookups and "engineers" in labels_dict:
                m = lookups[key_e]
                eng_arr = labels_dict["engineers"]
                for i in range(n):
                    if arr[i] < 0 and eng_arr[i] >= 0:
                        arr[i] = m.get(int(eng_arr[i]), -1)


# -------------------------
# collect hidden activations + labels
# -------------------------
@torch.no_grad()
def collect_hidden_and_labels(
    model: nn.Module,
    loader: NeighborLoader,
    target: str,
    node_ids_map: Dict[str, List[Any]],
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
      hidden_per_layer: list of (N, hidden_dim) arrays, one per layer
      labels_dict: dict of (N,) int64 arrays keyed by neighbor type (engineers, task_types, districts, departments, etc.)
    """
    hidden_per_layer: Optional[List[List[torch.Tensor]]] = None
    labels_lists: Dict[str, List[int]] = {}
    n_total = 0
    print("[info] Collect Hidden weights:")

    for batch in tqdm(loader, desc="collect hidden"):
        batch = batch.to(device)
        out = model(batch, return_hidden=True)
        layers = out["hidden_layers"]
        bs = int(batch[target].batch_size)
        if bs == 0:
            continue

        lab = get_seed_neighbor_labels(batch, target, node_ids_map)
        for k, arr in lab.items():
            if k not in labels_lists:
                labels_lists[k] = []
            labels_lists[k].extend(arr[:bs].tolist())

        if hidden_per_layer is None:
            hidden_per_layer = [[] for _ in range(len(layers))]
        for i, h in enumerate(layers):
            hidden_per_layer[i].append(h[:bs].cpu().float().numpy())

        n_total += bs
        if max_samples is not None and n_total >= max_samples:
            break

    if hidden_per_layer is None:
        return [], {k: np.array([], dtype=np.int64) for k in labels_lists}

    hidden_arrays = [np.concatenate(parts, axis=0) for parts in hidden_per_layer]
    n = hidden_arrays[0].shape[0]
    if max_samples is not None and n > max_samples:
        hidden_arrays = [a[:max_samples] for a in hidden_arrays]
        n = max_samples
        labels_dict = {k: np.array(v[:max_samples], dtype=np.int64) for k, v in labels_lists.items()}
    else:
        labels_dict = {k: np.array(v, dtype=np.int64) for k, v in labels_lists.items()}

    return hidden_arrays, labels_dict


# Default neighbor types to group by (order preserved for plots)
# All of these are shown in the UI; ones with no labels in batch will show "(no data)".
DEFAULT_GROUP_BY = ["engineers", "task_types", "districts", "departments", "regions", "tasks"]

# Max points to embed per view in Plotly HTML (keeps file small and dropdown responsive)
MAX_POINTS_PLOTLY_EMBED = 12000


# -------------------------
# Dimensionality reduction (PyTorch PCA + optional t-SNE)
# -------------------------
def pca_torch(
    X: np.ndarray,
    n_components: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    PCA via PyTorch SVD. X: (n, d). Returns (coords (n, k), info dict).
    Can run on GPU for large matrices.
    """
    if device is None:
        device = torch.device("cpu")
    X_ = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.from_numpy(X_).to(device)
    t = t - t.mean(dim=0)
    U, S, _ = torch.linalg.svd(t, full_matrices=False)
    k = min(n_components, S.shape[0], U.shape[1])
    coords = (U[:, :k] * S[:k]).cpu().numpy()
    var = (S * S).cpu().numpy()
    total_var = float(var.sum())
    explained = (var[:k] / total_var).tolist() if total_var > 0 else [0.0] * k
    return coords, {"explained_variance_ratio": explained, "n_components": k, "n_samples": int(t.shape[0])}


def tsne_reduce(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """t-SNE via sklearn (CPU). Returns (coords (n, 2), info dict)."""
    if not HAS_TSNE:
        raise RuntimeError("sklearn not installed; install scikit-learn for t-SNE")
    X_ = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n_comp = min(n_components, 2, X_.shape[0], X_.shape[1])
    tsne = TSNE(n_components=n_comp, perplexity=perplexity, random_state=random_state, **kwargs)
    coords = tsne.fit_transform(X_)
    return coords, {"n_components": coords.shape[1], "n_samples": coords.shape[0], "method": "tsne"}


def reduce_dim(
    X: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce X (n, d) to (n, n_components). method in ('pca', 'tsne').
    Returns (coords, info_dict). PCA uses PyTorch (GPU-capable); t-SNE uses sklearn.
    """
    if method.lower() == "pca":
        return pca_torch(X, n_components=n_components, device=device)
    if method.lower() == "tsne":
        return tsne_reduce(X, n_components=n_components, **kwargs)
    raise ValueError(f"Unknown method {method!r}. Use 'pca' or 'tsne'.")


def _subsample_for_pca(
    hidden_arrays: List[np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    max_samples: int,
    seed: int = 42,
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    """Subsample to max_samples (random) so PCA/plotting stay efficient. Returns new arrays and dict."""
    n = hidden_arrays[0].shape[0]
    if n <= max_samples:
        return hidden_arrays, labels_dict
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    idx = np.sort(idx)  # preserve rough order
    hidden_out = [h[idx] for h in hidden_arrays]
    labels_out = {k: v[idx] for k, v in labels_dict.items()}
    return hidden_out, labels_out


def _resolve_plot_keys(
    labels_dict: Dict[str, np.ndarray],
    group_by: Optional[List[str]] = None,
    require_valid: bool = False,
) -> List[str]:
    """Return ordered list of group keys to show. If require_valid, only include keys with at least one non-negative label."""
    if group_by is not None:
        candidates = [k for k in group_by if k in labels_dict]
    else:
        seen = set()
        candidates = []
        for k in DEFAULT_GROUP_BY + list(labels_dict.keys()):
            if k in labels_dict and k not in seen:
                seen.add(k)
                candidates.append(k)
    if require_valid:
        return [k for k in candidates if (labels_dict[k] >= 0).any()]
    return candidates


# -------------------------
# Reduce + plot (PCA or t-SNE)
# -------------------------
def _axis_labels(method: str) -> Tuple[str, str]:
    if method.lower() == "tsne":
        return "t-SNE 1", "t-SNE 2"
    return "PC1", "PC2"


def run_pca_and_plot(
    hidden_arrays: List[np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    out_dir: Path,
    n_components: int = 2,
    layer_names: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
    method: str = "pca",
    device: Optional[torch.device] = None,
    tsne_perplexity: float = 30.0,
    tsne_random_state: int = 42,
) -> Dict[str, Any]:
    """
    method: 'pca' (PyTorch) or 'tsne' (sklearn). Only keys with valid labels get a static PNG.
    """
    results: Dict[str, Any] = {"layers": [], "pca": {}, "group_by": [], "method": method}
    if method.lower() == "tsne" and not HAS_TSNE:
        results["error"] = "sklearn not installed; install scikit-learn for t-SNE"
        return results
    if not HAS_MATPLOTLIB:
        results["plot_error"] = "matplotlib not installed; skipping plots"

    plot_keys = _resolve_plot_keys(labels_dict, group_by, require_valid=False)
    results["group_by"] = plot_keys
    results["group_coverage"] = {k: int((labels_dict[k] >= 0).sum()) for k in plot_keys}
    xlabel, ylabel = _axis_labels(method)

    out_dir.mkdir(parents=True, exist_ok=True)

    reduce_kw = {}
    if method.lower() == "tsne":
        reduce_kw = {"perplexity": tsne_perplexity, "random_state": tsne_random_state}

    for layer_idx, H in enumerate(hidden_arrays):
        name = (layer_names[layer_idx]) if layer_names and layer_idx < len(layer_names) else f"layer_{layer_idx}"
        results["layers"].append(name)

        coords, info = reduce_dim(
            H,
            method=method,
            n_components=n_components,
            device=device,
            **reduce_kw,
        )
        results["pca"][name] = {**info}

        if not HAS_MATPLOTLIB or coords.shape[1] < 2:
            continue

        dim1, dim2 = coords[:, 0], coords[:, 1]

        for key in plot_keys:
            if key not in labels_dict:
                continue
            lab = labels_dict[key]
            valid = lab >= 0
            if not valid.any():
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(dim1[valid], dim2[valid], c=lab[valid], cmap="tab20", alpha=0.6, s=10)
            plt.colorbar(scatter, ax=ax, label=f"{key} (global idx)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Hidden {name} — {method.upper()} colored by {key}")
            fig.tight_layout()
            safe_key = key.replace(" ", "_")
            fig.savefig(out_dir / f"{method}_{name}_by_{safe_key}.png", dpi=120)
            plt.close(fig)

    return results


# -------------------------
# Interactive Plotly (single HTML with dropdown)
# -------------------------
def run_pca_plotly(
    hidden_arrays: List[np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    out_dir: Path,
    n_components: int = 2,
    layer_names: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
    method: str = "pca",
    device: Optional[torch.device] = None,
    tsne_perplexity: float = 30.0,
    tsne_random_state: int = 42,
) -> Optional[Path]:
    """
    Build one interactive HTML with a dropdown to switch (layer × group_by).
    method: 'pca' (PyTorch) or 'tsne' (sklearn).
    """
    if not HAS_PLOTLY:
        return None
    if method.lower() == "tsne" and not HAS_TSNE:
        return None

    plot_keys = _resolve_plot_keys(labels_dict, group_by, require_valid=False)
    if not plot_keys:
        return None

    reduce_kw = {}
    if method.lower() == "tsne":
        reduce_kw = {"perplexity": tsne_perplexity, "random_state": tsne_random_state}

    layer_coords: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for layer_idx, H in enumerate(hidden_arrays):
        name = (layer_names[layer_idx]) if layer_names and layer_idx < len(layer_names) else f"layer_{layer_idx}"
        coords, _ = reduce_dim(H, method=method, n_components=n_components, device=device, **reduce_kw)
        if coords.shape[1] < 2:
            continue
        layer_coords.append((name, coords[:, 0], coords[:, 1]))

    if not layer_coords:
        return None

    # Build views: one per (layer, key). Cap points per view for Plotly HTML size/performance.
    views: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, List[str], bool]] = []
    for name, pc1, pc2 in layer_coords:
        for key in plot_keys:
            if key not in labels_dict:
                continue
            lab = labels_dict[key]
            valid = lab >= 0
            has_data = valid.any()
            if has_data:
                x, y = pc1[valid], pc2[valid]
                c = lab[valid]
                hover = [f"idx={i}<br>{key}={int(lab[i])}" for i in range(len(lab)) if valid[i]]
                # Subsample for embedding so HTML and dropdown stay fast
                m = len(x)
                if m > MAX_POINTS_PLOTLY_EMBED:
                    idx = np.random.default_rng(42).choice(m, size=MAX_POINTS_PLOTLY_EMBED, replace=False)
                    idx = np.sort(idx)
                    x, y, c = x[idx], y[idx], c[idx]
                    hover = [hover[i] for i in idx]
                views.append((name, key, x, y, c, hover, True))
            else:
                views.append((name, key, np.array([]), np.array([]), np.array([]), [], False))

    if not views:
        return None

    # Default: first view that has data, else first view (empty)
    default_idx = 0
    for i, v in enumerate(views):
        if v[6]:  # has_data
            default_idx = i
            break
    name0, key0, x0, y0, c0, hover0, has_data0 = views[default_idx]
    suffix0 = " (no data)" if not has_data0 else ""
    trace = go.Scattergl(
        x=x0.tolist(),
        y=y0.tolist(),
        mode="markers",
        marker=dict(
            size=4,
            color=c0.tolist() if has_data0 else [],
            colorscale="Viridis",
            opacity=0.7,
            colorbar=dict(title=key0),
        ),
        text=hover0,
        hoverinfo="text",
        name="",
    )
    xlabel, ylabel = _axis_labels(method)
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=dict(text=f"{method.upper()} — Layer: {name0} | Group: {key0}{suffix0}", x=0.5, xanchor="center"),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
        height=600,
        margin=dict(t=60),
    )

    buttons: List[Dict[str, Any]] = []
    for i, (name, key, x, y, c, hover, has_data) in enumerate(views):
        suffix = " (no data)" if not has_data else ""
        buttons.append(
            dict(
                label=f"{name} × {key}{suffix}",
                method="update",
                args=[
                    {
                        "x": [x.tolist()],
                        "y": [y.tolist()],
                        "marker": {
                            "color": [c.tolist()] if has_data else [],
                            "colorscale": "Viridis",
                            "opacity": 0.7,
                            "colorbar": {"title": key},
                        },
                        "text": [hover],
                    },
                    {"title": {"text": f"{method.upper()} — Layer: {name} | Group: {key}{suffix}", "x": 0.5, "xanchor": "center"}},
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                active=0,
                x=0.01,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                showactive=True,
            )
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{method.lower()}_interactive.html"
    fig.write_html(str(html_path), config=dict(responsive=True))
    return html_path


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="PCA of checkpoint model hidden layers on dataset, grouped by neighbor labels (task type, engineer)."
    )
    ap.add_argument("--pt", type=str, required=True, help="Path to graph .pt (HeteroData)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt (or directory for latest)")
    ap.add_argument("--target", type=str, default=None, help="Target node type (default: from checkpoint)")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Which split to use")
    ap.add_argument("--max_samples", type=int, default=None, help="Cap number of samples collected from loader")
    ap.add_argument(
        "--max_pca_samples",
        type=int,
        default=25000,
        help="Cap samples used for PCA and plotting (subsample if collected more). Reduces memory and time.",
    )
    ap.add_argument("--batch_size", type=int, default=256, help="Default: from checkpoint (eval_batch_size) or 256")
    ap.add_argument("--num_neighbors", type=int, default=5, help="Default: from checkpoint (matches train_kfold) or 5")
    ap.add_argument("--n_components", type=int, default=2)
    ap.add_argument(
        "--last_layer_only",
        action="store_true",
        help="Use only the last hidden layer for PCA/t-SNE (instead of all layers).",
    )
    ap.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Dimensionality reduction: pca (PyTorch, GPU-capable) or tsne (sklearn, CPU).",
    )
    ap.add_argument("--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity (only if --method tsne).")
    ap.add_argument("--tsne_random_state", type=int, default=42, help="t-SNE random state for reproducibility.")
    ap.add_argument("--out_dir", type=str, default="runs/pca_weights")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--min_degree", type=int, default=1)
    ap.add_argument("--degree_mode", type=str, default="in")
    ap.add_argument(
        "--group_by",
        type=str,
        default=None,
        help="Comma-separated neighbor types to plot (e.g. engineers,task_types,districts,departments,regions). Default: auto from data",
    )
    ap.add_argument(
        "--plotly",
        action="store_true",
        help="Build interactive Plotly HTML (pca_interactive.html) with dropdown to switch layer × group",
    )
    ap.add_argument(
        "--open",
        action="store_true",
        help="Open the Plotly HTML in the default browser when used with --plotly",
    )
    args = ap.parse_args()

    device = pick_device(args.device)
    pt_path = Path(args.pt)
    ckpt_path = Path(args.ckpt)
    assert pt_path.exists(), f"Graph file not found: {pt_path}"

    if ckpt_path.is_dir():
        ckpts = sort_ckpts(list(ckpt_path.glob("*.pt")))
        if not ckpts:
            raise SystemExit(f"No .pt checkpoints in {ckpt_path}")
        ckpt_path = ckpts[-1]
        print(f"[info] using checkpoint: {ckpt_path}")
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    payload = load_payload(ckpt_path)
    ckpt_args = payload["args"]
    target = args.target or ckpt_args["target"]
    layers = int(ckpt_args.get("layers", 2))
    hidden = int(ckpt_args.get("hidden", 128))

    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData)

    attr_name_map, node_ids_map, _ = extract_metadata_maps(data)
    assert target in data.node_types
    assert hasattr(data[target], "y")
    data[target].y = data[target].y.float()

    deg = compute_target_degree(data, target, degree_mode=ckpt_args.get("degree_mode", args.degree_mode))
    min_deg = int(ckpt_args.get("min_degree", args.min_degree))
    kept = (deg >= min_deg).nonzero(as_tuple=False).view(-1)
    # Only sample assignments that have both engineer and task neighbors (faster, no wasted batches)
    has_both = assignments_with_engineer_and_task_neighbors(data, target)
    kept = kept[has_both[kept]]
    if kept.numel() == 0:
        raise SystemExit("No nodes after degree filter (need assignments with both engineer and task neighbors)")
    print(f"[info] seeds with engineer+task neighbors: {kept.numel()}")

    # split
    g = torch.Generator().manual_seed(int(ckpt_args.get("seed", args.seed)))
    perm = kept[torch.randperm(kept.numel(), generator=g)]
    n_train = int(kept.numel() * float(ckpt_args.get("train_ratio", args.train_ratio)))
    n_val = int(kept.numel() * float(ckpt_args.get("val_ratio", args.val_ratio)))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    eval_idx = idx_map[args.split]

    normalize_node_features_inplace(data, drop_const=True)
    in_dims = ensure_all_node_types_have_x(data)
    data = sanitize_for_neighbor_loader(data)

    # Match train_kfold sampling: use checkpoint num_neighbors / eval_batch_size when available
    num_neighbors_val = args.num_neighbors
    if num_neighbors_val is None:
        num_neighbors_val = int(ckpt_args.get("num_neighbors", 5))
        print(f"[info] num_neighbors={num_neighbors_val} (from checkpoint, matches train_kfold)")
    batch_size_val = args.batch_size
    if batch_size_val is None:
        batch_size_val = int(ckpt_args.get("eval_batch_size", 256))
        print(f"[info] batch_size={batch_size_val} (from checkpoint eval_batch_size)")
    num_neighbors = {et: [num_neighbors_val] * layers for et in data.edge_types}
    loader = NeighborLoader(
        data,
        input_nodes=(target, eval_idx),
        num_neighbors=num_neighbors,
        batch_size=batch_size_val,
        shuffle=False,
    )

    model = HeteroSAGERegressorWithHidden(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=hidden,
        num_layers=layers,
        target_node_type=target,
    ).to(device)
    state = payload["model_state"]
    # allow loading from strict HeteroSAGERegressor (no hidden_layers in state)
    model.load_state_dict(state, strict=False)
    model.eval()

    group_by_list = None
    if args.group_by:
        group_by_list = [s.strip() for s in args.group_by.split(",") if s.strip()]

    print(f"[info] collecting hidden states on {args.split} ({eval_idx.numel()} seeds), max_samples={args.max_samples}")
    hidden_arrays, labels_dict = collect_hidden_and_labels(
        model, loader, target, node_ids_map, device, max_samples=args.max_samples
    )

    n = hidden_arrays[0].shape[0] if hidden_arrays else 0
    # Enrich from full graph so task_types, districts, departments, regions get filled even when batch didn't sample those nodes
    enrich_labels_from_graph(data, labels_dict)
    coverage = {k: int((labels_dict[k] >= 0).sum()) for k in labels_dict}
    print(f"[info] collected n={n} samples, {len(hidden_arrays)} hidden layers")
    print(f"[info] label coverage: " + ", ".join(f"{k}={coverage[k]}/{n}" for k in sorted(labels_dict.keys())))

    # Subsample for PCA/plotting when large (saves memory and time)
    n_collected = n
    max_pca = getattr(args, "max_pca_samples", 1000)
    if n > max_pca:
        print(f"[info] subsampling {n} -> {max_pca} for PCA/plotting (--max_pca_samples)")
        hidden_arrays, labels_dict = _subsample_for_pca(hidden_arrays, labels_dict, max_pca, seed=args.seed)
        n = max_pca

    out_dir = Path(args.out_dir)
    layer_names = [f"after_in_proj"] + [f"after_conv_{i}" for i in range(len(hidden_arrays) - 1)]
    if getattr(args, "last_layer_only", False):
        hidden_arrays = [hidden_arrays[-1]]
        layer_names = [layer_names[-1]]
        print(f"[info] using last hidden layer only: {layer_names[0]}")
    results = run_pca_and_plot(
        hidden_arrays,
        labels_dict,
        out_dir,
        n_components=args.n_components,
        layer_names=layer_names,
        group_by=group_by_list,
        method=args.method,
        device=device,
        tsne_perplexity=args.tsne_perplexity,
        tsne_random_state=args.tsne_random_state,
    )

    import json
    summary_path = out_dir / "pca_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    n_labels = {k: int((v >= 0).sum()) for k, v in labels_dict.items()}
    summary = {
        "ckpt": str(ckpt_path),
        "target": target,
        "split": args.split,
        "method": args.method,
        "n_samples": n,
        "n_collected": n_collected,
        "n_labels_per_group": n_labels,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[ok] {args.method.upper()} summary -> {summary_path}")
    print(f"[ok] plots -> {out_dir}")

    if args.plotly and HAS_PLOTLY:
        html_path = run_pca_plotly(
            hidden_arrays,
            labels_dict,
            out_dir,
            n_components=args.n_components,
            layer_names=layer_names,
            group_by=group_by_list,
            method=args.method,
            device=device,
            tsne_perplexity=args.tsne_perplexity,
            tsne_random_state=args.tsne_random_state,
        )
        if html_path is not None:
            print(f"[ok] interactive -> {html_path}")
            if args.open:
                import webbrowser
                webbrowser.open(html_path.resolve().as_uri())
        else:
            print("[warn] no Plotly view generated (no valid layer×group combinations?)")


if __name__ == "__main__":
    main()
