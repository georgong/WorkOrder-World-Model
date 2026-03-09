# tsne_weights.py — t-SNE of last hidden layer, grouped by engineer / task.
# Same pipeline as pca_weights but uses t-SNE instead of PCA.
# Only samples assignments that have both engineer and task neighbors.
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

from tqdm import tqdm


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
# graph prep (match training)
# -------------------------
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


def assignments_with_engineer_and_task(data: HeteroData, target: str) -> torch.Tensor:
    """Boolean mask: True for nodes that have both an engineer and a task neighbor."""
    N = data[target].num_nodes
    has_eng = torch.zeros(N, dtype=torch.bool)
    has_task = torch.zeros(N, dtype=torch.bool)
    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if src == target and dst == "engineers":
            has_eng[ei[0]] = True
        elif src == "engineers" and dst == target:
            has_eng[ei[1]] = True
        if src == target and dst == "tasks":
            has_task[ei[0]] = True
        elif src == "tasks" and dst == target:
            has_task[ei[1]] = True
    return has_eng & has_task


@torch.no_grad()
def normalize_node_features_inplace(data: HeteroData, *, eps: float = 1e-6, drop_const: bool = True, const_std_thr: float = 1e-8) -> None:
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
        std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
        keep = (std > const_std_thr) if drop_const else torch.ones_like(std, dtype=torch.bool)
        if keep.sum().item() == 0:
            keep[0] = True
        data[nt].x = (x[:, keep] - mean[keep]) / std[keep]
        if hasattr(data[nt], "attr_name") and isinstance(data[nt].attr_name, list) and len(data[nt].attr_name) == keep.numel():
            data[nt].attr_name = [data[nt].attr_name[i] for i in range(len(data[nt].attr_name)) if keep[i].item()]


def sanitize_for_neighbor_loader(data: HeteroData) -> None:
    for nt in data.node_types:
        for key in list(data[nt].keys()):
            if not isinstance(data[nt][key], torch.Tensor):
                del data[nt][key]
    for et in data.edge_types:
        for key in list(data[et].keys()):
            if not isinstance(data[et][key], torch.Tensor):
                del data[et][key]


def ensure_all_node_types_have_x(data: HeteroData) -> Dict[str, int]:
    in_dims = {nt: data[nt].x.size(-1) for nt in data.node_types if hasattr(data[nt], "x")}
    for nt in data.node_types:
        if nt not in in_dims:
            in_dims[nt] = 1
            data[nt].x = torch.zeros((data[nt].num_nodes, 1), dtype=torch.float)
    return in_dims


# -------------------------
# model: SAGE that returns last hidden (target nodes only)
# -------------------------
class HeteroSAGERegressorLastHidden(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=128, num_layers=2, target_node_type="assignments"):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.target_node_type = target_node_type
        self.in_proj = nn.ModuleDict({nt: Linear(in_dims[nt], hidden_dim) for nt in self.node_types})
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in self.edge_types}, aggr="mean"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types}))
        self.out = Linear(hidden_dim, 1)
        self.base = nn.Parameter(torch.tensor(0.0))

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        x_dict = {nt: F.relu(self.in_proj[nt](data[nt].x)) for nt in self.node_types}
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(self.norms[i][k](v)) for k, v in x_dict.items()}
        last_hidden = x_dict[self.target_node_type]
        pred = self.base + self.out(last_hidden).squeeze(-1)
        return last_hidden, pred


# -------------------------
# labels from batch (engineer, task for seeds only)
# -------------------------
def _edge_type(batch: HeteroData, src: str, dst: str) -> Optional[Tuple[str, str, str]]:
    for et in batch.edge_types:
        if et[0] == src and et[2] == dst:
            return et
        if et[0] == dst and et[2] == src:
            return et
    return None


@torch.no_grad()
def get_batch_labels(batch: HeteroData, target: str, bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (engineer_idx, task_idx) each [bs] with -1 where missing."""
    eng = torch.full((bs,), -1, dtype=torch.long, device=batch[target].x.device)
    task = torch.full((bs,), -1, dtype=torch.long, device=batch[target].x.device)
    # target -> engineers
    et = _edge_type(batch, target, "engineers")
    if et and "engineers" in batch.node_types and hasattr(batch["engineers"], "n_id"):
        ei = batch[et].edge_index
        src_is_target = et[0] == target
        for k in range(ei.size(1)):
            i = ei[0, k].item() if src_is_target else ei[1, k].item()
            j = ei[1, k].item() if src_is_target else ei[0, k].item()
            if 0 <= i < bs:
                eng[i] = batch["engineers"].n_id[j].item()
    # target -> tasks
    et = _edge_type(batch, target, "tasks")
    if et and "tasks" in batch.node_types and hasattr(batch["tasks"], "n_id"):
        ei = batch[et].edge_index
        src_is_target = et[0] == target
        for k in range(ei.size(1)):
            i = ei[0, k].item() if src_is_target else ei[1, k].item()
            j = ei[1, k].item() if src_is_target else ei[0, k].item()
            if 0 <= i < bs:
                task[i] = batch["tasks"].n_id[j].item()
    return eng, task


# -------------------------
# collect last hidden + labels (PyTorch only, stop at max_samples)
# -------------------------
@torch.no_grad()
def collect_last_hidden_and_labels(
    model: HeteroSAGERegressorLastHidden,
    loader: NeighborLoader,
    target: str,
    device: torch.device,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (hidden [N, D], engineer_labels [N], task_labels [N], assignment_ids [N]) on CPU."""
    hidden_list: List[torch.Tensor] = []
    eng_list: List[torch.Tensor] = []
    task_list: List[torch.Tensor] = []
    nid_list: List[torch.Tensor] = []
    n = 0
    for batch in tqdm(loader, desc="collect", leave=False):
        batch = batch.to(device)
        h, _ = model(batch)
        bs = batch[target].batch_size
        if bs == 0:
            continue
        h_seed = h[:bs]
        eng, task = get_batch_labels(batch, target, bs)
        # global node ids for the seed (assignment) nodes
        a_nid = batch[target].n_id[:bs] if hasattr(batch[target], "n_id") else torch.arange(bs, device=batch[target].x.device)
        hidden_list.append(h_seed.cpu())
        eng_list.append(eng.cpu())
        task_list.append(task.cpu())
        nid_list.append(a_nid.cpu())
        n += bs
        if n >= max_samples:
            break
    H = torch.cat(hidden_list, dim=0)[:max_samples]
    E = torch.cat(eng_list, dim=0)[:max_samples]
    T = torch.cat(task_list, dim=0)[:max_samples]
    A_ids = torch.cat(nid_list, dim=0)[:max_samples]
    return H, E, T, A_ids


# -------------------------
# Full-graph lookups: task_type, district, department from task/engineer
# -------------------------
def build_neighbor_lookups(data: HeteroData) -> Dict[Tuple[str, str], Dict[int, int]]:
    """(src_nt, dst_nt) -> {src_global_idx: dst_global_idx} from full graph edges."""
    out: Dict[Tuple[str, str], Dict[int, int]] = {}
    for (src, _rel, dst) in data.edge_types:
        key = (src, dst)
        if key not in out:
            out[key] = {}
        ei = data[(src, _rel, dst)].edge_index
        for j in range(ei.size(1)):
            a, b = int(ei[0, j].item()), int(ei[1, j].item())
            out[key][a] = b
    return out


def get_neighbors_with_features(
    data: HeteroData,
    target: str,
    node_id: int,
    attr_name_by_type: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """For one assignment node_id, return list of neighbors with their x and attr_list (from .attr_name) corresponding to x."""
    out: List[Dict[str, Any]] = []
    for (src, _rel, dst) in data.edge_types:
        ei = data[(src, _rel, dst)].edge_index
        if src == target:
            mask = ei[0] == node_id
            neighbor_ids = ei[1, mask]
            other_type = dst
        elif dst == target:
            mask = ei[1] == node_id
            neighbor_ids = ei[0, mask]
            other_type = src
        else:
            continue
        for j in range(neighbor_ids.size(0)):
            nid = int(neighbor_ids[j].item())
            entry: Dict[str, Any] = {"node_type": other_type, "node_id": nid}
            if hasattr(data[other_type], "x") and isinstance(data[other_type].x, torch.Tensor):
                entry["x"] = [float(v) for v in data[other_type].x[nid].tolist()]
            # attr_list for this node type, corresponding to x (use .attr_name from graph or saved lookup)
            an = None
            if hasattr(data[other_type], "attr_name"):
                an = data[other_type].attr_name
            if an is None and attr_name_by_type is not None:
                an = attr_name_by_type.get(other_type)
            if an is not None:
                entry["attr_list"] = an if isinstance(an, list) else (an.tolist() if hasattr(an, "tolist") else list(an))
            out.append(entry)
    return out


def enrich_labels_from_graph(
    data: HeteroData,
    E: torch.Tensor,
    T: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """From engineer E and task T (each [N], -1 = missing), fill task_type, district, department [N] via full graph."""
    N = E.size(0)
    task_type = torch.full((N,), -1, dtype=torch.long)
    district = torch.full((N,), -1, dtype=torch.long)
    department = torch.full((N,), -1, dtype=torch.long)
    lookups = build_neighbor_lookups(data)
    # From task -> task_types, districts, departments
    for key, m in lookups.items():
        src, dst = key
        if src != "tasks":
            continue
        if dst == "task_types":
            for i in range(N):
                if T[i] >= 0:
                    task_type[i] = m.get(int(T[i].item()), -1)
        elif dst == "districts":
            for i in range(N):
                if T[i] >= 0:
                    district[i] = m.get(int(T[i].item()), -1)
        elif dst == "departments":
            for i in range(N):
                if T[i] >= 0:
                    department[i] = m.get(int(T[i].item()), -1)
    # Department from engineer where still missing
    if ("engineers", "departments") in lookups:
        m = lookups["engineers", "departments"]
        for i in range(N):
            if department[i] < 0 and E[i] >= 0:
                department[i] = m.get(int(E[i].item()), -1)
    return task_type, district, department


# -------------------------
# PCA + t-SNE (sklearn) — optional PCA to reduce memory before t-SNE
# -------------------------
def tsne_fit(
    X: torch.Tensor,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = 1,
    pca_dims: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """X: [N, D]. Returns (coords [N, n_components], info). Uses n_jobs=1 by default to avoid segfaults.
    If pca_dims is set, reduce to that many components first to save memory."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    X_np = X.numpy().astype("float64")
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
    if pca_dims is not None and pca_dims > 0 and X_np.shape[1] > pca_dims:
        pca = PCA(n_components=min(pca_dims, X_np.shape[0], X_np.shape[1]), random_state=random_state)
        X_np = pca.fit_transform(X_np)
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, max(1, X_np.shape[0] // 3)),
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    coords_np = tsne.fit_transform(X_np)
    coords = torch.from_numpy(coords_np).float()
    info = {
        "n_components": n_components,
        "perplexity": float(tsne.perplexity),
        "max_iter": max_iter,
        "pca_dims": pca_dims,
    }
    return coords, info


# -------------------------
# checkpoint
# -------------------------
_fold_re = re.compile(r"_fold(\d+)_epoch(\d+)\.pt$")


def load_ckpt(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt or "args" not in ckpt:
        raise ValueError(f"Invalid checkpoint: {path}")
    return ckpt


def sort_ckpts(paths: List[Path]) -> List[Path]:
    def key(p: Path):
        m = _fold_re.search(p.name)
        return (0, int(m.group(1)), int(m.group(2))) if m else (1, 0, 0)
    return sorted(paths, key=key)


# -------------------------
# plot (matplotlib optional)
# -------------------------
def save_scatter_png(
    coords: torch.Tensor,
    labels: torch.Tensor,
    out_path: Path,
    title: str,
    label_name: str,
    xlabel: str = "t-SNE 1",
    ylabel: str = "t-SNE 2",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    valid = labels >= 0
    if valid.sum() == 0:
        return
    x = coords[valid, 0].numpy()
    y = coords[valid, 1].numpy()
    c = labels[valid].numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c=c, cmap="tab20", alpha=0.6, s=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(ax.collections[0], ax=ax, label=label_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="t-SNE of last hidden layer (assignments with engineer+task neighbors only).")
    ap.add_argument("--pt", type=str, required=True, help="Graph .pt")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt or dir")
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--max_samples", type=int, default=10000, help="Max samples to collect (faster).")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_neighbors", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="runs/tsne_weights")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    ap.add_argument("--max_iter", type=int, default=1000, help="t-SNE iterations (sklearn max_iter).")
    ap.add_argument("--n_jobs", type=int, default=1, help="t-SNE n_jobs (default 1 to avoid segfaults).")
    ap.add_argument("--tsne_subsample", type=int, default=2000, help="Max points to use for t-SNE (subsample if larger; 0 = use all). Reduces CPU memory.")
    ap.add_argument("--pca_dims", type=int, default=10, help="PCA components before t-SNE (0 = no PCA). Reduces memory and time.")
    args = ap.parse_args()

    device = pick_device(args.device)
    pt_path = Path(args.pt).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    if ckpt_path.is_dir():
        ckpts = sort_ckpts(list(ckpt_path.glob("*.pt")))
        ckpt_path = ckpts[-1].resolve() if ckpts else ckpt_path
    elif not ckpt_path.exists() and ckpt_path.suffix == ".pt" and str(ckpt_path).endswith(".pt.pt"):
        # try stripping duplicate .pt (e.g. ...epoch002.pt.pt -> ...epoch002.pt)
        alt = ckpt_path.parent / ckpt_path.stem
        if alt.exists():
            ckpt_path = alt.resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"--pt path does not exist: {pt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--ckpt path does not exist: {ckpt_path}")

    payload = load_ckpt(ckpt_path)
    ckpt_args = payload["args"]
    target = args.target or ckpt_args.get("target", "assignments")
    layers = int(ckpt_args.get("layers", 2))
    hidden = int(ckpt_args.get("hidden", 128))

    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData)
    data[target].y = data[target].y.float()

    deg = compute_target_degree(data, target, degree_mode=ckpt_args.get("degree_mode", "in"))
    kept = (deg >= int(ckpt_args.get("min_degree", 1))).nonzero(as_tuple=False).squeeze(-1)
    has_both = assignments_with_engineer_and_task(data, target)
    kept = kept[has_both[kept]]
    if kept.numel() == 0:
        raise SystemExit("No assignments with both engineer and task neighbors.")
    print(f"[info] seeds (engineer+task): {kept.numel()}")

    g = torch.Generator().manual_seed(ckpt_args.get("seed", args.seed))
    perm = kept[torch.randperm(kept.numel(), generator=g)]
    n_train = int(kept.numel() * ckpt_args.get("train_ratio", args.train_ratio))
    n_val = int(kept.numel() * ckpt_args.get("val_ratio", args.val_ratio))
    train_idx, val_idx, test_idx = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    eval_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[args.split]

    normalize_node_features_inplace(data, drop_const=True)
    in_dims = ensure_all_node_types_have_x(data)
    # Save .attr_name per node type before sanitize (sanitize removes non-tensor keys)
    attr_name_by_type: Dict[str, List[str]] = {}
    for nt in data.node_types:
        if hasattr(data[nt], "attr_name") and isinstance(data[nt].attr_name, list):
            attr_name_by_type[nt] = list(data[nt].attr_name)
    sanitize_for_neighbor_loader(data)

    num_neighbors = {et: [ckpt_args.get("num_neighbors", args.num_neighbors)] * layers for et in data.edge_types}
    loader = NeighborLoader(
        data,
        input_nodes=(target, eval_idx),
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = HeteroSAGERegressorLastHidden(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=hidden,
        num_layers=layers,
        target_node_type=target,
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()

    print(f"[info] collecting last hidden, max_samples={args.max_samples}")
    H, E, T, A_ids = collect_last_hidden_and_labels(model, loader, target, device, args.max_samples)
    n = H.size(0)
    print(f"[info] collected n={n}")

    # Subsample for t-SNE to reduce CPU memory (avoid OOM/segfault)
    if args.tsne_subsample > 0 and n > args.tsne_subsample:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n, generator=g)
        keep = perm[: args.tsne_subsample]
        H = H[keep]
        E = E[keep]
        T = T[keep]
        A_ids = A_ids[keep]
        n = H.size(0)
        print(f"[info] subsampled to n={n} for t-SNE (--tsne_subsample)")

    # Enrich task_type, district, department from full graph (via task / engineer)
    task_type, district, department = enrich_labels_from_graph(data, E, T)

    coords, tsne_info = tsne_fit(
        H,
        n_components=2,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        pca_dims=args.pca_dims if args.pca_dims > 0 else None,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_scatter_png(coords, E, out_dir / "tsne_by_engineer.png", "Last hidden — by engineer", "engineer")
    save_scatter_png(coords, T, out_dir / "tsne_by_task.png", "Last hidden — by task", "task")
    save_scatter_png(coords, task_type, out_dir / "tsne_by_task_type.png", "Last hidden — by task type", "task_type")
    save_scatter_png(coords, district, out_dir / "tsne_by_district.png", "Last hidden — by district", "district")
    save_scatter_png(coords, department, out_dir / "tsne_by_department.png", "Last hidden — by department", "department")

    summary = {
        "ckpt": str(ckpt_path),
        "target": target,
        "split": args.split,
        "n_samples": n,
        "tsne": tsne_info,
        "label_coverage": {
            "engineer": int((E >= 0).sum().item()),
            "task": int((T >= 0).sum().item()),
            "task_type": int((task_type >= 0).sum().item()),
            "district": int((district >= 0).sum().item()),
            "department": int((department >= 0).sum().item()),
        },
    }
    (out_dir / "tsne_summary.json").write_text(json.dumps(summary, indent=2))

    # Detailed JSON: each node's x, y, attr_list (from .attr_name) corresponding to x, attributes, neighbors, and id_map
    target_attr_list = attr_name_by_type.get(target)
    id_map: Dict[str, List[float]] = {}
    nodes_list: List[Dict[str, Any]] = []
    for i in range(n):
        a_id = int(A_ids[i].item())
        x, y = float(coords[i, 0].item()), float(coords[i, 1].item())
        id_map[str(a_id)] = [x, y]
        # Node feature vector and attr_list so that x[i] corresponds to attr_list[i]
        node_x = data[target].x[a_id].tolist() if hasattr(data[target], "x") and isinstance(data[target].x, torch.Tensor) else None
        attributes = {
            "engineer_id": int(E[i].item()) if E[i] >= 0 else None,
            "task_id": int(T[i].item()) if T[i] >= 0 else None,
            "task_type_id": int(task_type[i].item()) if task_type[i] >= 0 else None,
            "district_id": int(district[i].item()) if district[i] >= 0 else None,
            "department_id": int(department[i].item()) if department[i] >= 0 else None,
            "y": float(data[target].y[a_id].item()) if hasattr(data[target], "y") and isinstance(data[target].y, torch.Tensor) else None,
        }
        neighbors = get_neighbors_with_features(data, target, a_id, attr_name_by_type=attr_name_by_type)
        node_entry: Dict[str, Any] = {
            "id": a_id,
            "x": x,
            "y": y,
            "attributes": attributes,
            "neighbors": neighbors,
        }
        if target_attr_list is not None:
            node_entry["attr_list"] = target_attr_list
        if node_x is not None:
            node_entry["x_features"] = [float(v) for v in node_x]
        nodes_list.append(node_entry)
    tsne_nodes = {
        "target_node_type": target,
        "id_map": id_map,
        "nodes": nodes_list,
    }
    (out_dir / "tsne_nodes.json").write_text(json.dumps(tsne_nodes, indent=2))
    print(f"[ok] -> {out_dir} (tsne_summary.json, tsne_nodes.json)")


if __name__ == "__main__":
    main()
