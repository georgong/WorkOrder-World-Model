# pca_weights.py — Fast PCA of last hidden layer, grouped by engineer / task.
# Only samples assignments that have both engineer and task neighbors.
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (hidden [N, D], engineer_labels [N], task_labels [N]) on CPU."""
    hidden_list: List[torch.Tensor] = []
    eng_list: List[torch.Tensor] = []
    task_list: List[torch.Tensor] = []
    n = 0
    for batch in tqdm(loader, desc="collect", leave=False):
        batch = batch.to(device)
        h, _ = model(batch)
        bs = batch[target].batch_size
        if bs == 0:
            continue
        h_seed = h[:bs]
        eng, task = get_batch_labels(batch, target, bs)
        hidden_list.append(h_seed.cpu())
        eng_list.append(eng.cpu())
        task_list.append(task.cpu())
        n += bs
        if n >= max_samples:
            break
    H = torch.cat(hidden_list, dim=0)[:max_samples]
    E = torch.cat(eng_list, dim=0)[:max_samples]
    T = torch.cat(task_list, dim=0)[:max_samples]
    return H, E, T


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
# PyTorch PCA (SVD)
# -------------------------
def pca_torch(X: torch.Tensor, n_components: int = 2, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """X: [N, D]. Returns (coords [N, k], info)."""
    if device is None:
        device = X.device
    X = X.to(device).float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.mean(dim=0)
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    k = min(n_components, S.size(0), U.size(1))
    coords = (U[:, :k] * S[:k]).cpu()
    var = S * S
    total = var.sum().item()
    explained = (var[:k] / total).tolist() if total > 0 else [0.0] * k
    return coords, {"explained_variance_ratio": explained, "n_components": k}


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
    x, y = coords[valid, 0].numpy(), coords[valid, 1].numpy()
    c = labels[valid].numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c=c, cmap="tab20", alpha=0.6, s=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.colorbar(ax.collections[0], ax=ax, label=label_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="PCA of last hidden layer (assignments with engineer+task neighbors only).")
    ap.add_argument("--pt", type=str, required=True, help="Graph .pt")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt or dir")
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--max_samples", type=int, default=10000, help="Max samples to collect (faster).")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_neighbors", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="runs/pca_weights")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    device = pick_device(args.device)
    pt_path = Path(args.pt)
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpts = sort_ckpts(list(ckpt_path.glob("*.pt")))
        ckpt_path = ckpts[-1] if ckpts else ckpt_path
    assert pt_path.exists() and ckpt_path.exists(), "Missing --pt or --ckpt"

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
    H, E, T = collect_last_hidden_and_labels(model, loader, target, device, args.max_samples)
    n = H.size(0)
    print(f"[info] collected n={n}")

    coords, pca_info = pca_torch(H, n_components=2, device=device)
    # Enrich task_type, district, department from full graph (via task / engineer)
    task_type, district, department = enrich_labels_from_graph(data, E, T)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_scatter_png(coords, E, out_dir / "pca_by_engineer.png", "Last hidden — by engineer", "engineer")
    save_scatter_png(coords, T, out_dir / "pca_by_task.png", "Last hidden — by task", "task")
    save_scatter_png(coords, task_type, out_dir / "pca_by_task_type.png", "Last hidden — by task type", "task_type")
    save_scatter_png(coords, district, out_dir / "pca_by_district.png", "Last hidden — by district", "district")
    save_scatter_png(coords, department, out_dir / "pca_by_department.png", "Last hidden — by department", "department")

    summary = {
        "ckpt": str(ckpt_path),
        "target": target,
        "split": args.split,
        "n_samples": n,
        "explained_variance_ratio": pca_info["explained_variance_ratio"],
        "label_coverage": {
            "engineer": int((E >= 0).sum().item()),
            "task": int((T >= 0).sum().item()),
            "task_type": int((task_type >= 0).sum().item()),
            "district": int((district >= 0).sum().item()),
            "department": int((department >= 0).sum().item()),
        },
    }
    import json
    (out_dir / "pca_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] -> {out_dir}")


if __name__ == "__main__":
    main()
