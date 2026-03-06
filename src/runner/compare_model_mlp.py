# compare_model_mlp.py
# 加载 GNN 与 MLP 两个模型，在所有样本上得到预测，并记录为 {id: {model1_pred, model2_pred, y}}。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from src.runner.eval import (
    _ckpt_args_to_namespace,
    ensure_all_node_types_have_x,
    load_payload,
    make_loader,
    normalize_node_features_inplace,
    sanitize_for_neighbor_loader,
)
from src.runner.mlp_baseline import (
    MLPRegressor,
    batch_to_tabular_per_seed,
    pick_device,
)
from src.runner.train_kfold import build_model


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


def infer_mlp_config(state_dict: Dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """从 MLP state_dict 推断 d_in, hidden, depth。"""
    # net.0.weight -> (hidden, d_in), net.3.weight -> (hidden, hidden), ... 最后 net.X.weight -> (1, hidden)
    keys = [k for k in state_dict if k.startswith("net.") and k.endswith(".weight")]
    if not keys:
        raise ValueError("state_dict 中未找到 net.*.weight")
    # 按模块下标排序
    def idx(s: str) -> int:
        # "net.0.weight" -> 0, "net.3.weight" -> 3
        return int(s.split(".")[1])
    keys_sorted = sorted(keys, key=idx)
    first_w = state_dict[keys_sorted[0]]
    hidden, d_in = int(first_w.size(0)), int(first_w.size(1))
    # Linear 数量: 每个 block 2 个 + 最后 1 个 => 2*depth + 1
    num_linear = len(keys_sorted)
    depth = (num_linear - 1) // 2
    return d_in, hidden, depth


def main():
    ap = argparse.ArgumentParser(description="在所有样本上对比 GNN 与 MLP 预测，输出 {id: model1_pred, model2_pred, y}")
    ap.add_argument("--pt", type=str, required=True, help="图数据 .pt (HeteroData)")
    ap.add_argument("--model_ckpt", type=str, required=True, help="GNN 模型 checkpoint .pt (train_kfold 格式)")
    ap.add_argument("--mlp_ckpt", type=str, required=True, help="MLP 模型 checkpoint .pt (piecewise 单 fold)")
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_neighbors", type=int, default=5)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", type=str, default="compare_model_mlp.json", help="输出 JSON 路径")
    ap.add_argument("--min_degree", type=int, default=1, help="只保留 degree >= 的节点")
    args = ap.parse_args()

    device = pick_device(args.device)
    pt_path = Path(args.pt)
    model_ckpt_path = Path(args.model_ckpt)
    mlp_ckpt_path = Path(args.mlp_ckpt)
    if not pt_path.exists():
        raise FileNotFoundError(f"图文件不存在: {pt_path}")
    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"GNN checkpoint 不存在: {model_ckpt_path}")
    if not mlp_ckpt_path.exists():
        raise FileNotFoundError(f"MLP checkpoint 不存在: {mlp_ckpt_path}")

    # 加载图
    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError("加载的不是 HeteroData")
    target = args.target
    if target not in data.node_types:
        raise ValueError(f"target {target!r} 不在 node_types 中")
    if not hasattr(data[target], "y") or data[target].y is None:
        raise ValueError(f"data[{target}].y 不存在，无法对比")
    data[target].y = data[target].y.view(-1).float()
    y = data[target].y
    finite = torch.isfinite(y)
    full_idx = torch.nonzero(finite, as_tuple=False).view(-1).long()
    if not finite.all():
        print(f"[warn] 存在非有限 y，仅使用 {full_idx.numel()}/{y.numel()} 个样本")

    # 与 train_kfold / eval 对齐：degree 过滤
    deg = compute_target_degree(data, target, degree_mode="in")
    kept = (deg >= args.min_degree).nonzero(as_tuple=False).view(-1)
    full_idx = torch.tensor(sorted(set(full_idx.tolist()) & set(kept.tolist())), dtype=torch.long)
    if full_idx.numel() == 0:
        raise ValueError("degree 过滤后无样本")

    # 归一化与预处理（与训练一致）
    normalize_node_features_inplace(data, drop_const=True)
    in_dims = ensure_all_node_types_have_x(data)
    data = sanitize_for_neighbor_loader(data)

    neighbor_types = ["engineers", "tasks", "task_types", "districts", "departments"]

    # 先读 GNN 配置，保证 loader 与 GNN 训练时一致
    payload = load_payload(model_ckpt_path)
    ckpt_args = payload["args"]
    gnn_layers = int(ckpt_args.get("layers", args.layers))
    gnn_num_neighbors = int(ckpt_args.get("num_neighbors", args.num_neighbors))

    # 构建全量 loader（GNN 和 MLP 共用同一批 batch，保证 id 一致）
    num_neighbors = {et: [gnn_num_neighbors] * gnn_layers for et in data.edge_types}
    loader = NeighborLoader(
        data,
        input_nodes=(target, full_idx),
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 加载 GNN
    ckpt_ns = _ckpt_args_to_namespace(ckpt_args)
    model = build_model(ckpt_ns, data, in_dims, target).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    # 加载 MLP（从 state_dict 推断结构）
    mlp_state = torch.load(mlp_ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(mlp_state, dict):
        raise ValueError("MLP checkpoint 应为 state_dict")
    d_in, hidden, depth = infer_mlp_config(mlp_state)
    mlp = MLPRegressor(d_in=d_in, hidden=hidden, depth=depth, dropout=0.1).to(device)
    mlp.load_state_dict(mlp_state, strict=True)
    mlp.eval()

    # 逐 batch 收集 id -> (model1_pred, model2_pred, y)
    results: Dict[int, Dict[str, float]] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="compare"):
            batch = batch.to(device)
            bs = int(batch[target].batch_size)
            n_id = batch[target].n_id[:bs]
            y_b = batch[target].y[:bs].cpu()

            # GNN 预测
            pred_gnn = model(batch)["pred"][:bs].cpu()

            # MLP 表型特征 + 预测
            X, _ = batch_to_tabular_per_seed(
                batch, target=target, neighbor_types=neighbor_types, in_dims=in_dims
            )
            pred_mlp = mlp(X[:bs]).cpu()

            for i in range(bs):
                idx = int(n_id[i].item())
                results[idx] = {
                    "model1_pred": float(pred_gnn[i].item()),
                    "model2_pred": float(pred_mlp[i].item()),
                    "y": float(y_b[i].item()),
                }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 按 id 排序后写出，便于阅读
    out_obj = {str(k): v for k, v in sorted(results.items())}
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False))
    print(f"[ok] 共 {len(results)} 条，已写入 {out_path}")


if __name__ == "__main__":
    main()
