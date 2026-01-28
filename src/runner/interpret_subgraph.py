# interpret_subgraph.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


# -------------------------
# metadata extraction (before sanitize)
# -------------------------
def extract_metadata_maps(
    data: HeteroData,
) -> tuple[dict[str, list[str]], dict[str, list[Any]], dict[str, list[str]]]:
    """
    Returns:
      attr_name_map[nt] = list[str] (feature names)
      node_ids_map[nt]  = list[Any] (raw ids, optional)
      mask_cols_map[nt] = list[str] (masked feature names, optional)
    Also validates attr_name length vs x.shape[1] when both exist.
    """
    attr_name_map: dict[str, list[str]] = {}
    node_ids_map: dict[str, list[Any]] = {}
    mask_cols_map: dict[str, list[str]] = {}

    for nt in data.node_types:
        store = data[nt]

        # attr_name
        if hasattr(store, "attr_name"):
            an = getattr(store, "attr_name")
            if isinstance(an, (list, tuple)):
                an = list(an)
                if hasattr(store, "x") and isinstance(store.x, torch.Tensor) and store.x.dim() == 2:
                    assert len(an) == int(store.x.size(1)), (
                        f"[attr_name mismatch] {nt}: len(attr_name)={len(an)} "
                        f"!= x.shape[1]={int(store.x.size(1))}"
                    )
                attr_name_map[nt] = an

        # node_ids (optional)
        if hasattr(store, "node_ids"):
            nid = getattr(store, "node_ids")
            if isinstance(nid, (list, tuple)):
                node_ids_map[nt] = list(nid)
            elif isinstance(nid, torch.Tensor) and nid.ndim == 1:
                node_ids_map[nt] = nid.detach().cpu().tolist()

        # mask_cols (optional)
        if hasattr(store, "mask_cols"):
            mc = getattr(store, "mask_cols")
            if isinstance(mc, (list, tuple)):
                mask_cols_map[nt] = list(mc)

    return attr_name_map, node_ids_map, mask_cols_map


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
# model (must match training)
# -------------------------
class HeteroSAGERegressor(nn.Module):
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

    def forward(self, data: HeteroData):
        x_dict = {nt: F.relu(self.in_proj[nt](data[nt].x)) for nt in self.node_types}

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(self.norms[i][k](v)) for k, v in x_dict.items()}

        delta = self.out(x_dict[self.target_node_type]).squeeze(-1)
        pred = self.base + delta
        return {"pred": pred}


# -------------------------
# graph prep (match training-ish)
# -------------------------
@torch.no_grad()
def normalize_node_features_inplace(
    data: HeteroData,
    *,
    eps: float = 1e-6,
    drop_const: bool = True,
    const_std_thr: float = 1e-8,
) -> None:
    """
    IMPORTANT: this changes x dim (drops const cols). If you want names, you must
    also filter attr_name accordingly (done here).
    """
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
    """
    NeighborLoader chokes on non-tensor fields inside storages (list, str, etc).
    So we delete them, after we extracted maps.
    """
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


def load_ckpt(ckpt_path: Path) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in payload or "args" not in payload:
        raise ValueError(f"Bad checkpoint format: {ckpt_path}")
    return payload


# -------------------------
# attribution helpers
# -------------------------
def _topk_abs(items: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    return sorted(items, key=lambda x: abs(float(x[1])), reverse=True)[:k]


def _node_label(batch: HeteroData, nt: str, local_idx: int) -> str:
    """
    NeighborLoader puts original indices in batch[nt].n_id (global index in that node type).
    """
    store = batch[nt]
    if hasattr(store, "n_id") and isinstance(store.n_id, torch.Tensor):
        if 0 <= local_idx < store.n_id.numel():
            return f"{nt}[orig={int(store.n_id[local_idx].item())}]"
    return f"{nt}[local={local_idx}]"


def gradxinput_attrib(
    model: nn.Module,
    batch: HeteroData,
    *,
    device: torch.device,
    explain_node_types: List[str] | None,
) -> Dict[str, Any]:
    """
    Computes grad * input for each node type:
      - signed: g*x
      - abs: abs(g*x)

    Returns per-type:
      node_signed: [N]    sum over feats of (g*x)
      node_abs:    [N]    sum over feats of abs(g*x)
      feat_signed: [F]    sum over nodes of (g*x)
      feat_abs:    [F]    sum over nodes of abs(g*x)
    """
    model.eval()
    batch = batch.to(device)

    nts = explain_node_types if explain_node_types else list(batch.node_types)

    # enable grads on selected x
    for nt in nts:
        if hasattr(batch[nt], "x") and isinstance(batch[nt].x, torch.Tensor):
            batch[nt].x = batch[nt].x.detach()
            batch[nt].x.requires_grad_(True)

    pred_all = model(batch)["pred"]
    if pred_all.numel() < 1:
        raise ValueError("No predictions in batch. Did NeighborLoader produce seeds?")

    pred0 = pred_all[0]
    pred0.backward()

    out: Dict[str, Any] = {"pred": float(pred0.detach().cpu().item()), "per_type": {}}

    for nt in nts:
        if not hasattr(batch[nt], "x") or not isinstance(batch[nt].x, torch.Tensor):
            continue
        x = batch[nt].x
        g = x.grad
        if g is None:
            continue

        imp = (g * x)  # [N,F] signed
        node_signed = imp.sum(dim=1).detach().cpu()
        node_abs = imp.abs().sum(dim=1).detach().cpu()
        feat_signed = imp.sum(dim=0).detach().cpu()
        feat_abs = imp.abs().sum(dim=0).detach().cpu()

        out["per_type"][nt] = {
            "num_nodes": int(x.size(0)),
            "num_feats": int(x.size(1)),
            "node_signed": node_signed.tolist(),
            "node_abs": node_abs.tolist(),
            "feat_signed": feat_signed.tolist(),
            "feat_abs": feat_abs.tolist(),
        }

    return out


def integrated_gradients_attrib(
    model: nn.Module,
    batch: HeteroData,
    *,
    device: torch.device,
    explain_node_types: List[str] | None,
    steps: int = 32,
) -> Dict[str, Any]:
    """
    IG with zero baseline per node-type x.

    For each nt:
      avg_grad = mean over steps of grad at interpolated inputs
      imp_signed = (x - x0) * avg_grad
      imp_abs    = abs(imp_signed)

    Returns same per-type fields as gradxinput.
    """
    model.eval()
    batch = batch.to(device)

    nts = explain_node_types if explain_node_types else list(batch.node_types)

    x_orig: Dict[str, torch.Tensor] = {}
    for nt in nts:
        if hasattr(batch[nt], "x") and isinstance(batch[nt].x, torch.Tensor):
            x_orig[nt] = batch[nt].x.detach()

    base: Dict[str, torch.Tensor] = {nt: torch.zeros_like(x_orig[nt]) for nt in x_orig}
    grad_sum: Dict[str, torch.Tensor] = {nt: torch.zeros_like(x_orig[nt]) for nt in x_orig}

    for s in range(1, steps + 1):
        alpha = float(s) / float(steps)

        for nt in x_orig:
            x_interp = base[nt] + alpha * (x_orig[nt] - base[nt])
            batch[nt].x = x_interp.detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        pred = model(batch)["pred"][0]
        pred.backward()

        for nt in x_orig:
            g = batch[nt].x.grad
            if g is not None:
                grad_sum[nt] += g.detach()

    # compute final pred on original inputs (restore)
    for nt in x_orig:
        batch[nt].x = x_orig[nt]
    pred0 = float(model(batch)["pred"][0].detach().cpu().item())

    out: Dict[str, Any] = {"pred": pred0, "per_type": {}}

    for nt in x_orig:
        avg_grad = grad_sum[nt] / float(steps)
        imp = (x_orig[nt] - base[nt]) * avg_grad  # signed [N,F]

        node_signed = imp.sum(dim=1).detach().cpu()
        node_abs = imp.abs().sum(dim=1).detach().cpu()
        feat_signed = imp.sum(dim=0).detach().cpu()
        feat_abs = imp.abs().sum(dim=0).detach().cpu()

        out["per_type"][nt] = {
            "num_nodes": int(x_orig[nt].size(0)),
            "num_feats": int(x_orig[nt].size(1)),
            "steps": int(steps),
            "node_signed": node_signed.tolist(),
            "node_abs": node_abs.tolist(),
            "feat_signed": feat_signed.tolist(),
            "feat_abs": feat_abs.tolist(),
        }

    return out


@torch.no_grad()
def edge_type_occlusion(model: nn.Module, batch: HeteroData, *, device: torch.device) -> Dict[str, Any]:
    model.eval()
    batch = batch.to(device)

    full_pred = float(model(batch)["pred"][0].detach().cpu().item())
    results: List[Tuple[str, float, float]] = []

    for et in batch.edge_types:
        b2 = batch.clone()
        if hasattr(b2[et], "edge_index") and isinstance(b2[et].edge_index, torch.Tensor):
            b2[et].edge_index = b2[et].edge_index[:, :0]
        p2 = float(model(b2)["pred"][0].detach().cpu().item())
        delta = full_pred - p2
        results.append((str(et), delta, p2))

    results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
    return {
        "pred": full_pred,
        "edge_type_effects": [
            {"edge_type": et, "delta": float(delta), "pred_without": float(p_wo)}
            for et, delta, p_wo in results
        ],
    }


# -------------------------
# summarization (good for UI)
# -------------------------
def summarize_attrib(
    batch: HeteroData,
    attrib: Dict[str, Any],
    *,
    attr_name_map: dict[str, list[str]] | None = None,
    topk_nodes: int = 10,
    topk_feats: int = 10,
) -> Dict[str, Any]:
    per_type = attrib.get("per_type", {})

    type_mass: List[Dict[str, Any]] = []
    top_nodes: Dict[str, List[Dict[str, float | str]]] = {}
    top_feats: Dict[str, List[Dict[str, float | str]]] = {}

    # build mass using ABS so ratio is sane
    mass_abs_list: List[Tuple[str, float]] = []

    for nt, info in per_type.items():
        node_signed = torch.tensor(info["node_signed"], dtype=torch.float) if "node_signed" in info else None
        node_abs = torch.tensor(info["node_abs"], dtype=torch.float) if "node_abs" in info else None
        feat_signed = torch.tensor(info["feat_signed"], dtype=torch.float) if "feat_signed" in info else None
        feat_abs = torch.tensor(info["feat_abs"], dtype=torch.float) if "feat_abs" in info else None

        if node_abs is None or feat_abs is None:
            continue

        mass_abs = float(node_abs.sum().item()) if node_abs.numel() else 0.0
        mass_signed = float(node_signed.sum().item()) if (node_signed is not None and node_signed.numel()) else 0.0
        mass_abs_list.append((nt, mass_abs))

        # nodes topk: by abs, show signed + abs
        nodes_payload: List[Dict[str, float | str]] = []
        for i in range(int(node_abs.numel())):
            label = _node_label(batch, nt, i)
            v_signed = float(node_signed[i].item()) if node_signed is not None else 0.0
            v_abs = float(node_abs[i].item())
            nodes_payload.append({"node": label, "value": v_signed, "abs": v_abs})
        nodes_payload = sorted(nodes_payload, key=lambda d: float(d["abs"]), reverse=True)[:topk_nodes]
        top_nodes[nt] = nodes_payload

        # feats topk: by abs, show signed + abs (and names)
        feat_names = None
        if attr_name_map is not None and nt in attr_name_map and len(attr_name_map[nt]) == int(feat_abs.numel()):
            feat_names = attr_name_map[nt]

        feats_payload: List[Dict[str, float | str]] = []
        for j in range(int(feat_abs.numel())):
            name = feat_names[j] if feat_names else f"feat_{j}"
            v_signed = float(feat_signed[j].item()) if feat_signed is not None else 0.0
            v_abs = float(feat_abs[j].item())
            feats_payload.append({"feat": name, "value": v_signed, "abs": v_abs})
        feats_payload = sorted(feats_payload, key=lambda d: float(d["abs"]), reverse=True)[:topk_feats]
        top_feats[nt] = feats_payload

        type_mass.append(
            {
                "node_type": nt,
                "mass_abs": mass_abs,
                "mass_signed": mass_signed,
            }
        )

    total_abs = sum(v for _, v in mass_abs_list) + 1e-12
    # add ratios + sort by abs
    type_mass = sorted(type_mass, key=lambda d: float(d["mass_abs"]), reverse=True)
    for d in type_mass:
        d["ratio"] = float(d["mass_abs"] / total_abs)

    return {
        "pred": attrib.get("pred"),
        "type_mass": type_mass,
        "top_nodes": top_nodes,
        "top_feats": top_feats,
    }


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--target", type=str, default=None, help="Override target node type (else from ckpt args)")
    ap.add_argument("--seed_idx", type=int, required=True, help="Seed node index in ORIGINAL target node space")
    ap.add_argument("--layers", type=int, default=None, help="Override num layers (else from ckpt args)")
    ap.add_argument("--hidden", type=int, default=None, help="Override hidden dim (else from ckpt args)")
    ap.add_argument("--num_neighbors", type=int, default=10)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--method", type=str, default="gradxinput",
                    choices=["gradxinput", "ig", "edge_type_occlusion"])
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--topk_nodes", type=int, default=10)
    ap.add_argument("--topk_feats", type=int, default=10)

    ap.add_argument("--out", type=str, default="runs/interpret/interpret.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[info] device={device}")

    pt_path = Path(args.pt)
    ckpt_path = Path(args.ckpt)
    assert pt_path.exists(), f"Graph file not found: {pt_path}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    payload = load_ckpt(ckpt_path)
    ckpt_args = payload["args"]

    target = args.target or ckpt_args["target"]
    layers = int(args.layers if args.layers is not None else ckpt_args["layers"])
    hidden = int(args.hidden if args.hidden is not None else ckpt_args["hidden"])

    # load graph (keep metadata first)
    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData)

    # extract maps BEFORE sanitize (since sanitize deletes non-tensors)
    attr_name_map, node_ids_map, mask_cols_map = extract_metadata_maps(data)

    assert target in data.node_types, f"target {target} not in node_types"
    assert hasattr(data[target], "x"), f"data[{target}].x missing"
    assert hasattr(data[target], "y"), f"data[{target}].y missing"

    data[target].y = data[target].y.float()

    # match training normalization
    normalize_node_features_inplace(data, drop_const=True)

    # after normalization, attr_name in data is updated; update map accordingly if present
    # (if you extracted before normalization, and normalization dropped cols, your map would mismatch)
    # easiest: re-extract attr_name_map from normalized data (still contains attr_name lists)
    attr_name_map2, _, _ = extract_metadata_maps(data)
    # merge: prefer normalized names when available
    for nt, names in attr_name_map2.items():
        attr_name_map[nt] = names

    in_dims = ensure_all_node_types_have_x(data)
    data = sanitize_for_neighbor_loader(data)

    # build loader for ONE seed, bs=1
    seed = torch.tensor([int(args.seed_idx)], dtype=torch.long)
    num_neighbors = {et: [int(args.num_neighbors)] * layers for et in data.edge_types}

    loader = NeighborLoader(
        data,
        input_nodes=(target, seed),
        num_neighbors=num_neighbors,
        batch_size=1,
        shuffle=False,
    )

    # model
    model = HeteroSAGERegressor(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=hidden,
        num_layers=layers,
        target_node_type=target,
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    # fetch single batch
    batch = next(iter(loader))

    # run method
    if args.method == "gradxinput":
        attrib = gradxinput_attrib(model, batch, device=device, explain_node_types=None)
        summary = summarize_attrib(
            batch, attrib, attr_name_map=attr_name_map, topk_nodes=args.topk_nodes, topk_feats=args.topk_feats
        )
        out = {
            "method": "gradxinput",
            "seed_idx": int(args.seed_idx),
            "pred": attrib["pred"],
            "summary": summary,
            "meta": {"node_ids_map": node_ids_map, "mask_cols_map": mask_cols_map},
        }

    elif args.method == "ig":
        attrib = integrated_gradients_attrib(
            model, batch, device=device, explain_node_types=None, steps=int(args.ig_steps)
        )
        summary = summarize_attrib(
            batch, attrib, attr_name_map=attr_name_map, topk_nodes=args.topk_nodes, topk_feats=args.topk_feats
        )
        out = {
            "method": "ig",
            "ig_steps": int(args.ig_steps),
            "seed_idx": int(args.seed_idx),
            "pred": attrib["pred"],
            "summary": summary,
            "meta": {"node_ids_map": node_ids_map, "mask_cols_map": mask_cols_map},
        }

    else:  # edge_type_occlusion
        occ = edge_type_occlusion(model, batch, device=device)
        out = {
            "method": "edge_type_occlusion",
            "seed_idx": int(args.seed_idx),
            "pred": occ["pred"],
            "edge_type_effects": occ["edge_type_effects"][: max(1, args.topk_nodes)],
            "meta": {"node_ids_map": node_ids_map, "mask_cols_map": mask_cols_map},
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
