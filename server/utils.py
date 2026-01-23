from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple
import torch

MAX_ELEMS_TO_SEND = 80


def _to_jsonable(v: Any, *, allow_large: bool = False) -> Any:
    if isinstance(v, torch.Tensor):
        if v.numel() == 0:
            return []
        if v.ndim == 0:
            return v.item()
        if (v.numel() <= MAX_ELEMS_TO_SEND) or allow_large:
            return v.detach().cpu().tolist()
        return {"__tensor__": True, "shape": list(v.shape), "dtype": str(v.dtype), "numel": int(v.numel())}

    if isinstance(v, (int, float, str, bool)) or v is None:
        return v

    if isinstance(v, (list, tuple)):
        if len(v) <= MAX_ELEMS_TO_SEND:
            return [_to_jsonable(x) for x in v]
        return {"__list__": True, "len": len(v)}

    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}

    return str(v)


def _node_label(store, ntype: str, nid: int) -> str:
    for key in ["name", "NAME", "W6KEY", "id", "ID", "task_id", "engineer_id"]:
        if hasattr(store, key):
            v = getattr(store, key)
            if isinstance(v, torch.Tensor) and v.ndim == 1 and nid < v.shape[0]:
                try:
                    return str(_to_jsonable(v[nid]))
                except Exception:
                    pass
    return f"{ntype}:{nid}"


def _get_pos(store, nid: int) -> Optional[Tuple[float, float]]:
    if hasattr(store, "pos"):
        pos = getattr(store, "pos")
        if isinstance(pos, torch.Tensor) and pos.ndim == 2 and pos.shape[1] >= 2 and nid < pos.shape[0]:
            return float(pos[nid, 0].item()), float(pos[nid, 1].item())
    return None


def _fallback_layout(n: int, seed: int) -> List[Tuple[float, float]]:
    rnd = random.Random(seed)
    coords = []
    r = 320.0
    for i in range(n):
        ang = 2 * math.pi * (i / max(1, n))
        x = r * math.cos(ang) + rnd.uniform(-25, 25)
        y = r * math.sin(ang) + rnd.uniform(-25, 25)
        coords.append((x, y))
    return coords


def _find_etypes(data, src: str, dst: str) -> List[Tuple[str, str, str]]:
    out = []
    for et in data.edge_types:
        s, r, d = et
        if s == src and d == dst:
            store = data[et]
            if hasattr(store, "edge_index") and isinstance(store.edge_index, torch.Tensor) and store.edge_index.numel() > 0:
                out.append(et)
    return out


def _sample_pairs_from_etype(data, etype: Tuple[str, str, str], k: int, rnd: random.Random) -> List[Tuple[int, int]]:
    ei = data[etype].edge_index
    E = int(ei.shape[1])
    if E <= 0 or k <= 0:
        return []
    kk = min(k, E)
    cols = rnd.sample(range(E), kk) if E > kk else list(range(E))
    src = ei[0, cols].to(torch.long).detach().cpu().tolist()
    dst = ei[1, cols].to(torch.long).detach().cpu().tolist()
    return list(zip(src, dst))


def build_graph_summary(
    data,
    *,
    node_limit: int,
    edge_limit: int,
    seed: int,
    use_pos_if_exists: bool = True,

    # assignment-anchored triad knobs
    anchor_assignments: int = 1500,
    probe_be_edges: int = 20000,
    probe_ab_edges: int = 40000,
    cap_be: int = 6000,
    cap_ab: int = 6000,

    include_extra: bool = True,
    extra_cap_per_rel: int = 2500,
) -> Dict[str, Any]:
    """
    Assignment-anchored triad sampling:
      - find assignments that have BOTH (tasks -> assignments) AND (assignments -> engineers)
      - then emit T->A and A->E edges around those assignments
      - optionally attach task -> {districts, departments, task_statuses, task_types}

    Fixed output schema: {meta,nodes,edges}
    """
    rnd = random.Random(seed)

    needed = {"tasks", "assignments", "engineers"}
    if not needed.issubset(set(data.node_types)):
        return {"meta": {"error": f"Missing node types: {sorted(list(needed - set(data.node_types)))}"}, "nodes": [], "edges": []}

    et_be = _find_etypes(data, "assignments", "engineers")
    et_ab = _find_etypes(data, "tasks", "assignments")
    if not et_be or not et_ab:
        return {
            "meta": {
                "error": "Missing required edge legs.",
                "missing_tasks_to_assignments": (not et_ab),
                "missing_assignments_to_engineers": (not et_be),
            },
            "nodes": [],
            "edges": [],
        }

    # ---- 1) Probe BE to get candidate assignments that reach engineers
    be_pairs: List[Tuple[int, int, str]] = []  # (assignment, engineer, rel)
    cand_A_from_BE: set = set()

    per = max(1, probe_be_edges // max(1, len(et_be)))
    for et in et_be:
        pairs = _sample_pairs_from_etype(data, et, k=per, rnd=rnd)
        for a, e in pairs:
            cand_A_from_BE.add(a)
            be_pairs.append((a, e, et[1]))
        if len(be_pairs) >= probe_be_edges:
            break

    if not cand_A_from_BE:
        return {"meta": {"error": "No BE pairs sampled. Increase probe_be_edges."}, "nodes": [], "edges": []}

    # ---- 2) Probe AB to find which assignments are touched by tasks
    ab_pairs: List[Tuple[int, int, str]] = []  # (task, assignment, rel)
    cand_A_from_AB: set = set()

    per = max(1, probe_ab_edges // max(1, len(et_ab)))
    for et in et_ab:
        pairs = _sample_pairs_from_etype(data, et, k=per, rnd=rnd)
        for t, a in pairs:
            cand_A_from_AB.add(a)
            ab_pairs.append((t, a, et[1]))
        if len(ab_pairs) >= probe_ab_edges:
            break

    if not cand_A_from_AB:
        return {"meta": {"error": "No AB pairs sampled. Increase probe_ab_edges."}, "nodes": [], "edges": []}

    # ---- 3) Good assignments = intersection
    good_A = list(cand_A_from_BE.intersection(cand_A_from_AB))
    if not good_A:
        return {
            "meta": {
                "error": "No 'good' assignments found in probes (need A with both T->A and A->E).",
                "hint": "Increase probe_ab_edges and probe_be_edges.",
                "cand_BE": len(cand_A_from_BE),
                "cand_AB": len(cand_A_from_AB),
            },
            "nodes": [],
            "edges": [],
        }

    if len(good_A) > anchor_assignments:
        good_A = rnd.sample(good_A, anchor_assignments)

    good_A_set = set(good_A)

    # ---- 4) Build final subgraph edges, filtered to good assignments
    edges_out: List[Dict[str, Any]] = []
    picked: Dict[str, set] = {"tasks": set(), "assignments": set(), "engineers": set()}

    def add_node(ntype: str, nid: int) -> None:
        picked.setdefault(ntype, set()).add(int(nid))

    def add_edge(src_type: str, src: int, rel: str, dst_type: str, dst: int) -> None:
        edges_out.append({"src_type": src_type, "src": int(src), "rel": rel, "dst_type": dst_type, "dst": int(dst)})

    # Emit AB edges: tasks -> assignments where assignment in good set
    rnd.shuffle(ab_pairs)
    ab_kept = 0
    for t, a, rel in ab_pairs:
        if a in good_A_set:
            add_edge("tasks", t, rel, "assignments", a)
            add_node("tasks", t)
            add_node("assignments", a)
            ab_kept += 1
            if ab_kept >= cap_ab or len(edges_out) >= edge_limit:
                break

    # Emit BE edges: assignments -> engineers where assignment in good set
    rnd.shuffle(be_pairs)
    be_kept = 0
    for a, e, rel in be_pairs:
        if a in good_A_set:
            add_edge("assignments", a, rel, "engineers", e)
            add_node("assignments", a)
            add_node("engineers", e)
            be_kept += 1
            if be_kept >= cap_be or len(edges_out) >= edge_limit:
                break

    # Guarantee we have triad-ish content
    if len(picked.get("tasks", set())) == 0 or len(picked.get("assignments", set())) == 0 or len(picked.get("engineers", set())) == 0:
        return {
            "meta": {
                "error": "Triad sampling produced empty layer (unexpected).",
                "tasks": len(picked.get("tasks", set())),
                "assignments": len(picked.get("assignments", set())),
                "engineers": len(picked.get("engineers", set())),
                "good_assignments": len(good_A_set),
                "note": "Try increasing cap_ab/cap_be or probe sizes.",
            },
            "nodes": [],
            "edges": [],
        }

    # ---- 5) Optionally attach task -> extras
    if include_extra and len(edges_out) < edge_limit and len(picked["tasks"]) > 0:
        task_ids = list(picked["tasks"])
        task_tensor = torch.tensor(task_ids, dtype=torch.long)

        extras = [("tasks", "districts"), ("tasks", "departments"), ("tasks", "task_statuses"), ("tasks", "task_types")]
        for s_type, d_type in extras:
            if d_type not in data.node_types:
                continue
            et_list = _find_etypes(data, s_type, d_type)
            if not et_list:
                continue

            cap = min(extra_cap_per_rel, edge_limit - len(edges_out))
            if cap <= 0:
                break

            for et in et_list:
                if len(edges_out) >= edge_limit or cap <= 0:
                    break
                ei = data[et].edge_index
                src = ei[0].to(torch.long)
                dst = ei[1].to(torch.long)
                mask = torch.isin(src, task_tensor)
                idx = torch.nonzero(mask, as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                idx_list = idx.detach().cpu().tolist()
                rnd.shuffle(idx_list)

                take = min(cap, len(idx_list))
                for j in idx_list[:take]:
                    t = int(src[j].item())
                    x = int(dst[j].item())
                    add_edge("tasks", t, et[1], d_type, x)
                    add_node(d_type, x)
                    cap -= 1
                    if cap <= 0 or len(edges_out) >= edge_limit:
                        break

    # ---- 6) Trim nodes to node_limit (keep core types first)
    core_order = ["tasks", "assignments", "engineers"]
    all_nodes: List[Tuple[str, int]] = [(t, i) for t, ids in picked.items() for i in ids]

    if len(all_nodes) > node_limit:
        keep: List[Tuple[str, int]] = []
        keep_set = set()
        for t in core_order:
            for i in picked.get(t, set()):
                p = (t, i)
                keep.append(p)
                keep_set.add(p)
                if len(keep) >= node_limit:
                    break
            if len(keep) >= node_limit:
                break

        if len(keep) < node_limit:
            others = [p for p in all_nodes if p not in keep_set]
            need = node_limit - len(keep)
            if others and need > 0:
                others = rnd.sample(others, min(need, len(others)))
                keep.extend(others)

        picked2: Dict[str, set] = {}
        for t, i in keep:
            picked2.setdefault(t, set()).add(i)
        picked = picked2

    # ---- 7) Filter edges to kept nodes
    edges_filtered: List[Dict[str, Any]] = []
    for e in edges_out:
        if (e["src"] in picked.get(e["src_type"], set())) and (e["dst"] in picked.get(e["dst_type"], set())):
            edges_filtered.append(e)
            if len(edges_filtered) >= edge_limit:
                break

    # ---- 8) Build node payload
    flat_nodes = [(t, i) for t, ids in picked.items() for i in ids]
    fallback = _fallback_layout(len(flat_nodes), seed=seed)

    nodes_out: List[Dict[str, Any]] = []
    for idx, (ntype, nid) in enumerate(flat_nodes):
        store = data[ntype]
        pos = _get_pos(store, nid) if use_pos_if_exists else None
        if pos is None:
            pos = fallback[idx]
        nodes_out.append({
            "type": ntype,
            "id": int(nid),
            "label": _node_label(store, ntype, nid),
            "x": float(pos[0]),
            "y": float(pos[1]),
        })

    return {
        "meta": {
            "mode": "assignment_anchored_triad",
            "seed": seed,
            "node_limit": node_limit,
            "edge_limit": edge_limit,
            "probes": {"probe_be_edges": probe_be_edges, "probe_ab_edges": probe_ab_edges},
            "caps": {"cap_ab": cap_ab, "cap_be": cap_be, "extra_cap_per_rel": extra_cap_per_rel},
            "good_assignments": len(good_A_set),
            "nodes_returned": len(nodes_out),
            "edges_returned": len(edges_filtered),
            "layer_sizes": {
                "tasks": len(picked.get("tasks", set())),
                "assignments": len(picked.get("assignments", set())),
                "engineers": len(picked.get("engineers", set())),
            },
        },
        "nodes": nodes_out,
        "edges": edges_filtered,
    }


def get_node_payload(
    data,
    *,
    ntype: str,
    nid: int,
    fields: Optional[List[str]],
    include_x: bool,
) -> Dict[str, Any]:
    if ntype not in data.node_types:
        return {"error": f"Unknown node type: {ntype}"}

    store = data[ntype]
    n = int(store.num_nodes)
    if nid < 0 or nid >= n:
        return {"error": f"Node id out of range: {nid} (0..{n-1})"}

    keys = [k for k in store.keys() if not str(k).startswith("_")]

    def default_keep(k: str) -> bool:
        if k in {"edge_index"}:
            return False
        if k in {"pos"}:
            return False
        if k == "x":
            return include_x
        return True

    if fields is None:
        chosen = [str(k) for k in keys if default_keep(str(k))]
    else:
        wanted = set(fields)
        chosen = [str(k) for k in keys if str(k) in wanted]

    attrs: Dict[str, Any] = {}
    for k in chosen:
        v = getattr(store, k)
        try:
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n:
                attrs[k] = _to_jsonable(v[nid], allow_large=False)
            else:
                attrs[k] = _to_jsonable(v, allow_large=False)
        except Exception as e:
            attrs[k] = f"<error reading field: {e}>"

    return {
        "type": ntype,
        "id": int(nid),
        "label": _node_label(store, ntype, nid),
        "attr_names": sorted(list(attrs.keys())),
        "attrs": attrs,
    }


def build_ego_summary(
    data,
    *,
    center_type: str,
    center_id: int,
    hops: int,
    max_nodes: int,
    max_edges: int,
    per_hop_edge_cap: int,
    seed: int,
    use_pos_if_exists: bool = True,
) -> Dict[str, Any]:
    import random
    rnd = random.Random(seed)

    if center_type not in data.node_types:
        return {"meta": {"error": f"Unknown node type: {center_type}"}, "nodes": [], "edges": []}

    if center_id < 0 or center_id >= int(data[center_type].num_nodes):
        return {"meta": {"error": f"center_id out of range for {center_type}"}, "nodes": [], "edges": []}

    # picked nodes by type
    picked: Dict[str, set] = {center_type: {int(center_id)}}
    edges_out: List[Dict[str, Any]] = []

    # frontier: list of (type, id)
    frontier: Dict[str, List[int]] = {center_type: [int(center_id)]}

    def add_node(ntype: str, nid: int):
        picked.setdefault(ntype, set()).add(int(nid))

    def add_edge(src_type: str, src: int, rel: str, dst_type: str, dst: int):
        edges_out.append({"src_type": src_type, "src": int(src), "rel": rel, "dst_type": dst_type, "dst": int(dst)})

    # BFS hops
    for h in range(hops):
        if len(edges_out) >= max_edges:
            break

        new_frontier: Dict[str, set] = {}
        hop_edges = 0

        # For each edge type, expand both directions from current frontier
        for etype in data.edge_types:
            if hop_edges >= per_hop_edge_cap or len(edges_out) >= max_edges:
                break

            src_type, rel, dst_type = etype
            store = data[etype]
            if not hasattr(store, "edge_index"):
                continue
            ei = store.edge_index
            if not isinstance(ei, torch.Tensor) or ei.numel() == 0:
                continue

            src = ei[0].to(torch.long)
            dst = ei[1].to(torch.long)

            # ---- outgoing from frontier[src_type]
            src_front = frontier.get(src_type)
            if src_front:
                src_front_t = torch.tensor(src_front, dtype=torch.long)
                mask = torch.isin(src, src_front_t)
                idx = torch.nonzero(mask, as_tuple=False).view(-1)
                if idx.numel() > 0:
                    idx_list = idx.detach().cpu().tolist()
                    rnd.shuffle(idx_list)

                    # cap per etype per hop to avoid domination
                    take = min(len(idx_list), max(1, per_hop_edge_cap // max(1, len(data.edge_types))))
                    for j in idx_list[:take]:
                        s = int(src[j].item())
                        d = int(dst[j].item())
                        add_edge(src_type, s, rel, dst_type, d)
                        add_node(src_type, s)
                        add_node(dst_type, d)
                        new_frontier.setdefault(dst_type, set()).add(d)
                        hop_edges += 1
                        if hop_edges >= per_hop_edge_cap or len(edges_out) >= max_edges:
                            break

            if hop_edges >= per_hop_edge_cap or len(edges_out) >= max_edges:
                break

            # ---- incoming to frontier[dst_type]  (treat as undirected expansion)
            dst_front = frontier.get(dst_type)
            if dst_front:
                dst_front_t = torch.tensor(dst_front, dtype=torch.long)
                mask = torch.isin(dst, dst_front_t)
                idx = torch.nonzero(mask, as_tuple=False).view(-1)
                if idx.numel() > 0:
                    idx_list = idx.detach().cpu().tolist()
                    rnd.shuffle(idx_list)

                    take = min(len(idx_list), max(1, per_hop_edge_cap // max(1, len(data.edge_types))))
                    for j in idx_list[:take]:
                        s = int(src[j].item())
                        d = int(dst[j].item())
                        add_edge(src_type, s, rel, dst_type, d)
                        add_node(src_type, s)
                        add_node(dst_type, d)
                        new_frontier.setdefault(src_type, set()).add(s)
                        hop_edges += 1
                        if hop_edges >= per_hop_edge_cap or len(edges_out) >= max_edges:
                            break

        # build next frontier (convert set -> list)
        frontier = {t: list(ids) for t, ids in new_frontier.items()}

        # node cap early stop
        total_nodes = sum(len(v) for v in picked.values())
        if total_nodes >= max_nodes:
            break

        if not frontier:
            break

    # Trim nodes if exceeded (keep center first)
    flat_nodes = [(t, i) for t, ids in picked.items() for i in ids]
    if len(flat_nodes) > max_nodes:
        # keep center + random others
        center_pair = (center_type, int(center_id))
        others = [p for p in flat_nodes if p != center_pair]
        others = others[:]
        rnd.shuffle(others)
        keep = [center_pair] + others[: max_nodes - 1]

        picked2: Dict[str, set] = {}
        for t, i in keep:
            picked2.setdefault(t, set()).add(i)
        picked = picked2

    # Filter edges to kept nodes
    kept_edges: List[Dict[str, Any]] = []
    for e in edges_out:
        if e["src"] in picked.get(e["src_type"], set()) and e["dst"] in picked.get(e["dst_type"], set()):
            kept_edges.append(e)
            if len(kept_edges) >= max_edges:
                break

    # Node payload
    flat_nodes = [(t, i) for t, ids in picked.items() for i in ids]
    fallback = _fallback_layout(len(flat_nodes), seed=seed)

    nodes_out: List[Dict[str, Any]] = []
    for idx, (ntype, nid) in enumerate(flat_nodes):
        store = data[ntype]
        pos = _get_pos(store, nid) if use_pos_if_exists else None
        if pos is None:
            pos = fallback[idx]
        nodes_out.append(
            {
                "type": ntype,
                "id": int(nid),
                "label": _node_label(store, ntype, nid),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "is_center": (ntype == center_type and int(nid) == int(center_id)),
            }
        )

    return {
        "meta": {
            "mode": "ego",
            "center": {"type": center_type, "id": int(center_id)},
            "hops": hops,
            "seed": seed,
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "per_hop_edge_cap": per_hop_edge_cap,
            "nodes_returned": len(nodes_out),
            "edges_returned": len(kept_edges),
        },
        "nodes": nodes_out,
        "edges": kept_edges,
    }
