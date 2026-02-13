# src/runner/analyze_embeddings.py
"""
Analyze and compare RGCN vs GraphSAGE model performance,
and investigate district/department embedded representations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from src.model.gnn import HeteroSAGERegressor, HeteroGCN

# -------------------------
# device
# -------------------------
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
# normalization (same as training)
# -------------------------
@torch.no_grad()
def normalize_node_features_inplace(
    data: HeteroData, *, eps: float = 1e-6, drop_const: bool = True, const_std_thr: float = 1e-8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    stats = {}
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
        stats[nt] = {"mean": mean[keep], "std": std[keep], "keep_mask": keep}
        if hasattr(data[nt], "attr_name"):
            an = data[nt].attr_name
            if isinstance(an, list) and len(an) == int(keep.numel()):
                data[nt].attr_name = [an[i] for i in range(len(an)) if bool(keep[i].item())]
    return stats


def sanitize_for_neighbor_loader(data: HeteroData) -> HeteroData:
    for nt in data.node_types:
        store = data[nt]
        for key in list(store.keys()):
            v = store[key]
            if isinstance(v, torch.Tensor):
                continue
            del store[key]
    for et in data.edge_types:
        store = data[et]
        for key in list(store.keys()):
            v = store[key]
            if isinstance(v, torch.Tensor):
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


def split_indices(idx, *, seed, train_ratio, val_ratio):
    N = idx.numel()
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(N, generator=g)]
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


# -------------------------
# evaluation
# -------------------------
@torch.no_grad()
def eval_model(model, loader, target, device):
    model.eval()
    se_sum = ae_sum = sl1_sum = 0.0
    n = 0
    is_raw_gcn = isinstance(model, HeteroGCN)
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = batch.to(device)
        if is_raw_gcn:
            x_dict = model(batch)
            # No learned head — just use mean of target embeddings as prediction
            pred = x_dict[target].mean(dim=-1)
        else:
            pred = model(batch)["pred"]
        y = batch[target].y.float()
        bs = int(batch[target].batch_size)
        p, t = pred[:bs], y[:bs]
        se_sum += F.mse_loss(p, t, reduction="sum").item()
        ae_sum += F.l1_loss(p, t, reduction="sum").item()
        sl1_sum += F.smooth_l1_loss(p, t, beta=1.0, reduction="sum").item()
        n += bs
    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "smoothl1": float("nan")}
    mse = se_sum / n
    return {"mse": mse, "mae": ae_sum / n, "rmse": mse ** 0.5, "smoothl1": sl1_sum / n}


# -------------------------
# embedding analysis
# -------------------------
@torch.no_grad()
def extract_embeddings(model, data, device, target, num_neighbors=3, layers=2, batch_size=256):
    """Extract embeddings for all node types from a trained model."""
    model.eval()
    data = data.to(device)

    if isinstance(model, HeteroGCN):
        # HeteroGCN.forward returns x_dict directly
        x_dict = model(data)
        return {k: v.cpu() for k, v in x_dict.items()}

    if not hasattr(model, "get_embeddings"):
        # Fallback: run forward pass internals manually for HeteroSAGERegressor
        x_dict = {nt: F.relu(model.in_proj[nt](data[nt].x)) for nt in model.node_types}
        for i, conv in enumerate(model.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(model.norms[i][k](v)) for k, v in x_dict.items()}
        return {k: v.cpu() for k, v in x_dict.items()}

    # For small graphs, run forward pass on full graph
    if data[target].num_nodes < 50000:
        emb = model.get_embeddings(data)
        return {k: v.cpu() for k, v in emb.items()}

    # For large graphs, use mini-batching
    all_idx = torch.arange(data[target].num_nodes)
    num_neighbors_dict = {et: [num_neighbors] * layers for et in data.edge_types}
    loader = NeighborLoader(
        data, input_nodes=(target, all_idx),
        num_neighbors=num_neighbors_dict, batch_size=batch_size, shuffle=False,
    )
    embeddings = defaultdict(list)
    for batch in tqdm(loader, desc="extracting embeddings"):
        batch = batch.to(device)
        emb = model.get_embeddings(batch)
        for nt, v in emb.items():
            embeddings[nt].append(v.cpu())
    return {k: torch.cat(v, dim=0) for k, v in embeddings.items()}

def analyze_district_department_embeddings(
    embeddings: Dict[str, torch.Tensor],
    data: HeteroData,
):
    """Analyze whether district and department embeddings form meaningful clusters."""
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import pdist, squareform

    lines = []
    lines.append("=" * 80)
    lines.append("District & Department Embedding Analysis")
    lines.append("=" * 80)

    for ntype in ["districts", "departments"]:
        if ntype not in embeddings:
            lines.append(f"\n[{ntype}] No embeddings found, skipping.")
            continue

        emb = embeddings[ntype].numpy()
        n = emb.shape[0]
        lines.append(f"\n[{ntype}] Embedding shape: {emb.shape}")

        if n < 3:
            lines.append(f"  Too few nodes ({n}) for clustering analysis.")
            continue

        # 1. Pairwise cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = cosine_similarity(emb)
        lines.append(f"  Cosine similarity: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")
        lines.append(f"    min={cos_sim.min():.4f}, max={cos_sim.max():.4f}")

        # 2. K-Means clustering
        max_k = min(n - 1, 10)
        best_k, best_sil = 2, -1
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(emb)
            sil = silhouette_score(emb, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        lines.append(f"  Best K-Means: k={best_k}, silhouette={best_sil:.4f}")

        # 3. Variance explained by clusters
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(emb)
        total_var = np.var(emb, axis=0).sum()
        within_var = sum(np.var(emb[labels == c], axis=0).sum() for c in range(best_k)) / best_k
        var_explained = 1 - within_var / total_var if total_var > 0 else 0
        lines.append(f"  Variance explained by {best_k} clusters: {var_explained:.4f}")

        # 4. Per-cluster sizes
        from collections import Counter
        cluster_counts = Counter(labels)
        lines.append(f"  Cluster sizes: {dict(sorted(cluster_counts.items()))}")

        # 5. Top-3 most similar pairs
        np.fill_diagonal(cos_sim, -1)
        flat_idx = np.argsort(cos_sim.ravel())[::-1]
        lines.append(f"  Top-3 most similar pairs:")
        seen = set()
        count = 0
        for idx in flat_idx:
            i, j = divmod(idx, n)
            if (min(i, j), max(i, j)) in seen:
                continue
            seen.add((min(i, j), max(i, j)))
            lines.append(f"    {ntype}[{i}] ↔ {ntype}[{j}]: cosine={cos_sim[i, j]:.4f}")
            count += 1
            if count >= 3:
                break

        # 6. Top-3 most dissimilar pairs
        np.fill_diagonal(cos_sim, 2)
        flat_idx = np.argsort(cos_sim.ravel())
        lines.append(f"  Top-3 most dissimilar pairs:")
        seen = set()
        count = 0
        for idx in flat_idx:
            i, j = divmod(idx, n)
            if i == j:
                continue
            if (min(i, j), max(i, j)) in seen:
                continue
            seen.add((min(i, j), max(i, j)))
            lines.append(f"    {ntype}[{i}] ↔ {ntype}[{j}]: cosine={cos_sim[i, j]:.4f}")
            count += 1
            if count >= 3:
                break

    return "\n".join(lines)


def analyze_district_department_combinations(
    embeddings: Dict[str, torch.Tensor],
    data: HeteroData,
):
    """Analyze whether combinations of district+department form meaningful representations."""
    lines = []
    lines.append("=" * 80)
    lines.append("District × Department Combination Analysis")
    lines.append("=" * 80)

    if "districts" not in embeddings or "departments" not in embeddings:
        lines.append("Missing district or department embeddings.")
        return "\n".join(lines)

    dist_emb = embeddings["districts"]
    dept_emb = embeddings["departments"]

    lines.append(f"Districts: {dist_emb.shape[0]} nodes, embedding dim={dist_emb.shape[1]}")
    lines.append(f"Departments: {dept_emb.shape[0]} nodes, embedding dim={dept_emb.shape[1]}")

    # Find which tasks connect to which districts and departments
    task_to_district = {}
    task_to_department = {}

    for etype in data.edge_types:
        src, rel, dst = etype
        ei = data[etype].edge_index
        if src == "tasks" and dst == "districts":
            for i in range(ei.shape[1]):
                task_to_district[ei[0, i].item()] = ei[1, i].item()
        elif src == "tasks" and dst == "departments":
            for i in range(ei.shape[1]):
                task_to_department[ei[0, i].item()] = ei[1, i].item()

    # Build district-department combinations
    combo_to_tasks = defaultdict(list)
    for t_id in range(data["tasks"].num_nodes):
        d = task_to_district.get(t_id)
        p = task_to_department.get(t_id)
        if d is not None and p is not None:
            combo_to_tasks[(d, p)].append(t_id)

    lines.append(f"\nUnique (district, department) combinations: {len(combo_to_tasks)}")
    lines.append(f"Tasks with both district and department: {sum(len(v) for v in combo_to_tasks.values())}")

    # Analyze combination embeddings (concatenation of district + department embeddings)
    combo_keys = list(combo_to_tasks.keys())
    if len(combo_keys) < 3:
        lines.append("Too few combinations for analysis.")
        return "\n".join(lines)

    combo_embs = []
    combo_labels = []
    combo_sizes = []
    for d, p in combo_keys:
        if d < dist_emb.shape[0] and p < dept_emb.shape[0]:
            combined = torch.cat([dist_emb[d], dept_emb[p]], dim=0)
            combo_embs.append(combined)
            combo_labels.append(f"dist{d}_dept{p}")
            combo_sizes.append(len(combo_to_tasks[(d, p)]))

    if len(combo_embs) < 3:
        lines.append("Too few valid combinations for analysis.")
        return "\n".join(lines)

    combo_embs = torch.stack(combo_embs).numpy()
    lines.append(f"Valid combinations for analysis: {len(combo_embs)}")
    lines.append(f"Combined embedding dim: {combo_embs.shape[1]}")

    # Cosine similarity of combination embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = cosine_similarity(combo_embs)
    lines.append(f"\nCombination cosine similarity: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")

    # Clustering of combinations
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    max_k = min(len(combo_embs) - 1, 10)
    best_k, best_sil = 2, -1
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(combo_embs)
        sil = silhouette_score(combo_embs, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    lines.append(f"Best K-Means for combinations: k={best_k}, silhouette={best_sil:.4f}")

    # Show cluster composition
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(combo_embs)
    lines.append(f"\nCluster composition (top combinations per cluster):")
    for c in range(best_k):
        members = [(combo_labels[i], combo_sizes[i]) for i in range(len(labels)) if labels[i] == c]
        members.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"  Cluster {c}: {len(members)} combinations")
        for label, size in members[:5]:
            lines.append(f"    {label}: {size} tasks")

    # Correlation between combination embedding distance and task count similarity
    from scipy.stats import pearsonr
    combo_sizes_arr = np.array(combo_sizes, dtype=float)
    n = len(combo_embs)
    dists = []
    size_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(1 - cos_sim[i, j])
            size_diffs.append(abs(combo_sizes_arr[i] - combo_sizes_arr[j]))
    if len(dists) > 2:
        corr, pval = pearsonr(dists, size_diffs)
        lines.append(f"\nCorrelation between embedding distance and task count difference:")
        lines.append(f"  Pearson r={corr:.4f}, p={pval:.2g}")

    return "\n".join(lines)


# -------------------------
# load checkpoint and build model
# -------------------------
def load_model_from_ckpt(ckpt_path: str, data: HeteroData, in_dims: Dict[str, int], device: torch.device):
    """Load a model from a checkpoint file, auto-detecting model type."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = payload.get("args", {})
    
    target = ckpt_args.get("target", "assignments")
    hidden = int(ckpt_args.get("hidden", 64))
    layers = int(ckpt_args.get("layers", 2))
    arch = ckpt_args.get("arch", "sage")
    
    if arch == "rgcn" or "rgcn" in Path(ckpt_path).name.lower():
        # Load the original HeteroGCN with the original weights
        model = HeteroGCN(data.metadata(), hidden, hidden)
        model.load_state_dict(payload["model_state"], strict=False)
        model_name = "RGCN"
    else:
        model = HeteroSAGERegressor(
            metadata=data.metadata(), in_dims=in_dims,
            hidden_dim=hidden, num_layers=layers, target_node_type=target,
        )
        model.load_state_dict(payload["model_state"], strict=False)
        model_name = "GraphSAGE"
    
    model = model.to(device)
    model.eval()
    
    return model, model_name, payload, ckpt_args


# -------------------------
# model comparison (from checkpoints only)
# -------------------------
def compare_models_from_ckpts(
    data: HeteroData,
    in_dims: Dict[str, int],
    target: str,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    device: torch.device,
    sage_ckpt_path: str,
    rgcn_ckpt_path: str,
    batch_size: int = 256,
    num_neighbors: int = 3,
    layers: int = 2,
    splits: List[str] = None,
):
    """Compare pre-trained GraphSAGE and RGCN models from checkpoints."""
    if splits is None:
        splits = ["train", "val", "test", "all"]

    lines = []
    lines.append("=" * 80)
    lines.append("Model Comparison: GraphSAGE vs RGCN (from checkpoints)")
    lines.append(f"  Splits evaluated: {', '.join(splits)}")
    lines.append("=" * 80)

    num_neighbors_dict = {et: [num_neighbors] * layers for et in data.edge_types}
    
    # Build loaders only for requested splits
    loaders = {}
    if "train" in splits:
        loaders["train"] = NeighborLoader(
            data, input_nodes=(target, train_idx),
            num_neighbors=num_neighbors_dict, batch_size=batch_size, shuffle=False,
        )
    if "val" in splits:
        loaders["val"] = NeighborLoader(
            data, input_nodes=(target, val_idx),
            num_neighbors=num_neighbors_dict, batch_size=batch_size, shuffle=False,
        )
    if "test" in splits:
        loaders["test"] = NeighborLoader(
            data, input_nodes=(target, test_idx),
            num_neighbors=num_neighbors_dict, batch_size=batch_size, shuffle=False,
        )
    if "all" in splits:
        all_idx = torch.arange(data[target].num_nodes)
        loaders["all"] = NeighborLoader(
            data, input_nodes=(target, all_idx),
            num_neighbors=num_neighbors_dict, batch_size=batch_size, shuffle=False,
        )

    results = {}
    models = {}

    # --- GraphSAGE ---
    lines.append(f"\n[GraphSAGE] Loading checkpoint: {sage_ckpt_path}")
    sage_model, sage_name, sage_payload, ckpt_args = load_model_from_ckpt(sage_ckpt_path, data, in_dims, device)
    sage_epoch = sage_payload.get("epoch", "?")
    lines.append(f"  Checkpoint epoch: {sage_epoch}")
    
    target = ckpt_args.get("target", "assignments")
    seed = int(ckpt_args.get("seed", 42))
    train_ratio = float(ckpt_args.get("train_ratio", 0.8))
    val_ratio = float(ckpt_args.get("val_ratio", 0.1))
    min_degree = int(ckpt_args.get("min_degree", 1))
    degree_mode = ckpt_args.get("degree_mode", "in")
    layers = int(ckpt_args.get("layers", 2))

    print(f"[info] config from checkpoint: target={target}, seed={seed}, "
          f"train_ratio={train_ratio}, val_ratio={val_ratio}, "
          f"min_degree={min_degree}, degree_mode={degree_mode}, layers={layers}")

    results["sage"] = {}
    for split in splits:
        results["sage"][split] = eval_model(sage_model, loaders[split], target, device)
        r = results["sage"][split]
        lines.append(f"  {split.capitalize():>5}: RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}, SmoothL1={r['smoothl1']:.4f}")
    models["sage"] = sage_model

    # --- RGCN ---
    lines.append(f"\n[RGCN] Loading checkpoint: {rgcn_ckpt_path}")
    rgcn_model, rgcn_name, rgcn_payload, _ = load_model_from_ckpt(rgcn_ckpt_path, data, in_dims, device)
    rgcn_epoch = rgcn_payload.get("epoch", "?")
    lines.append(f"  Checkpoint epoch: {rgcn_epoch}")
    
    results["rgcn"] = {}
    for split in splits:
        results["rgcn"][split] = eval_model(rgcn_model, loaders[split], target, device)
        r = results["rgcn"][split]
        lines.append(f"  {split.capitalize():>5}: RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}, SmoothL1={r['smoothl1']:.4f}")
    models["rgcn"] = rgcn_model

    # --- Comparison Table ---
    lines.append("\n" + "-" * 100)
    lines.append("Comparison Summary")
    lines.append("-" * 100)
    header = f"{'Metric':<12}"
    for split in splits:
        header += f" {'SAGE '+split:>12} {'RGCN '+split:>12}"
    lines.append(header)
    lines.append("-" * 100)
    for m in ["rmse", "mae", "smoothl1"]:
        row = f"{m:<12}"
        for split in splits:
            row += f" {results['sage'][split][m]:>12.4f} {results['rgcn'][split][m]:>12.4f}"
        lines.append(row)
    
    # --- Winner per metric ---
    lines.append("\n" + "-" * 60)
    lines.append("Winner (lower is better)")
    lines.append("-" * 60)
    for split in splits:
        for m in ["rmse", "mae", "smoothl1"]:
            s = results["sage"][split][m]
            r = results["rgcn"][split][m]
            winner = "GraphSAGE" if s < r else "RGCN" if r < s else "Tie"
            diff = abs(s - r)
            pct = diff / max(min(s, r), 1e-8) * 100
            lines.append(f"  {split:>5} {m:<12}: {winner:<12} (diff={diff:.4f}, {pct:.2f}%)")

    return "\n".join(lines), models["sage"], models["rgcn"], results


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True, help="Path to graph .pt")
    ap.add_argument("--sage_ckpt", type=str, default="runs/checkpoints/kfold-sage-fold00_fold00_epoch002.pt",
                     help="Path to GraphSAGE checkpoint")
    ap.add_argument("--rgcn_ckpt", type=str, default="runs/checkpoints/kfold-rgcn-fold00_fold00_epoch002.pt",
                     help="Path to RGCN checkpoint")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_neighbors", type=int, default=3)
    ap.add_argument("--out", type=str, default="runs/analysis/embedding_analysis.txt")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "all"],
                     choices=["train", "val", "test", "all"],
                     help="Which splits to evaluate (default: train val test all)")
    ap.add_argument("--skip_eval", action="store_true",
                     help="Skip model evaluation and only perform embedding analysis")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[info] device={device}")

    # Peek into the sage checkpoint to get training config
    sage_payload = torch.load(args.sage_ckpt, map_location="cpu", weights_only=False)
    ckpt_args = sage_payload.get("args", {})

    target = ckpt_args.get("target", "assignments")
    seed = int(ckpt_args.get("seed", 42))
    train_ratio = float(ckpt_args.get("train_ratio", 0.8))
    val_ratio = float(ckpt_args.get("val_ratio", 0.1))
    min_degree = int(ckpt_args.get("min_degree", 1))
    degree_mode = ckpt_args.get("degree_mode", "in")
    layers = int(ckpt_args.get("layers", 2))

    print(f"[info] config from checkpoint: target={target}, seed={seed}, "
          f"train_ratio={train_ratio}, val_ratio={val_ratio}, "
          f"min_degree={min_degree}, degree_mode={degree_mode}, layers={layers}")

    # Load data
    data: HeteroData = torch.load(args.pt, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData)
    data[target].y = data[target].y.float()

    # Degree filter and split (using checkpoint config)
    deg = compute_target_degree(data, target, degree_mode=degree_mode)
    kept = (deg >= min_degree).nonzero(as_tuple=False).view(-1)
    train_idx, val_idx, test_idx = split_indices(
        kept, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio,
    )

    # Normalize and sanitize
    normalize_node_features_inplace(data, drop_const=True)
    in_dims = ensure_all_node_types_have_x(data)
    data = sanitize_for_neighbor_loader(data)

    report_lines = []

    if args.skip_eval:
        # Load models without evaluation
        print("\n[info] Skipping evaluation, loading models for embedding extraction only...")
        sage_model, _, _, _ = load_model_from_ckpt(args.sage_ckpt, data, in_dims, device)
        rgcn_model, _, _, _ = load_model_from_ckpt(args.rgcn_ckpt, data, in_dims, device)
    else:
        # 1. Compare models from checkpoints (no retraining)
        comparison_report, sage_model, rgcn_model, results = compare_models_from_ckpts(
            data, in_dims, target, train_idx, val_idx, test_idx, device,
            sage_ckpt_path=args.sage_ckpt,
            rgcn_ckpt_path=args.rgcn_ckpt,
            batch_size=args.batch_size,
            num_neighbors=args.num_neighbors,
            layers=layers,
            splits=args.splits,
        )
        report_lines.append(comparison_report)

    # 2. Extract embeddings from both models
    print("\n[info] Extracting GraphSAGE embeddings...")
    sage_emb = extract_embeddings(
        sage_model, data, device, target,
        num_neighbors=args.num_neighbors, layers=layers, batch_size=args.batch_size,
    )

    print("[info] Extracting RGCN embeddings...")
    rgcn_emb = extract_embeddings(
        rgcn_model, data, device, target,
        num_neighbors=args.num_neighbors, layers=layers, batch_size=args.batch_size,
    )

    # 3. Analyze district/department embeddings for both models
    report_lines.append("\n\n--- GraphSAGE Embeddings ---")
    report_lines.append(analyze_district_department_embeddings(sage_emb, data))
    report_lines.append(analyze_district_department_combinations(sage_emb, data))

    report_lines.append("\n\n--- RGCN Embeddings ---")
    report_lines.append(analyze_district_department_embeddings(rgcn_emb, data))
    report_lines.append(analyze_district_department_combinations(rgcn_emb, data))

    # Save report
    report = "\n".join(report_lines)
    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    # python -m src.runner.analyze_embeddings --pt data/graph/sdge.pt --sage_ckpt runs/checkpoints/kfold-sage-fold00_fold00_epoch002.pt --rgcn_ckpt runs/checkpoints/kfold-rgcn-fold00_fold00_epoch002.pt --out runs/analysis/embedding_analysis.txt --split val test
    main()