from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import torch
from torch_geometric.data import HeteroData


# -----------------------------
# IO / helpers
# -----------------------------
def load_graph(path: str | Path) -> HeteroData:
    g = torch.load(path,weights_only=False)
    if not isinstance(g, HeteroData):
        raise TypeError(f"Expected HeteroData, got {type(g)} from {path}")
    return g


def safe_node_ids(g: HeteroData, ntype: str) -> List[str]:
    if hasattr(g[ntype], "node_ids"):
        return [str(x) for x in g[ntype].node_ids]
    return [f"{ntype}:{i}" for i in range(g[ntype].num_nodes)]


def edge_types_between(g: HeteroData, a: str, b: str) -> List[Tuple[str, str, str]]:
    cands = []
    for et in g.edge_types:
        s, r, t = et
        if (s == a and t == b) or (s == b and t == a):
            cands.append(et)
    nonrev = [e for e in cands if not str(e[1]).startswith("rev_")]
    return nonrev if nonrev else cands


def neighbors_set(
    g: HeteroData,
    edge_types: List[Tuple[str, str, str]],
    src_type: str,
    dst_type: str,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    srcN = [set() for _ in range(g[src_type].num_nodes)]
    dstN = [set() for _ in range(g[dst_type].num_nodes)]

    for (s, r, t) in edge_types:
        ei = g[(s, r, t)].edge_index
        if ei.numel() == 0:
            continue
        a = ei[0].tolist()
        b = ei[1].tolist()

        if s == src_type and t == dst_type:
            for u, v in zip(a, b):
                if 0 <= u < len(srcN) and 0 <= v < len(dstN):
                    srcN[u].add(v)
                    dstN[v].add(u)
        elif s == dst_type and t == src_type:
            for u, v in zip(a, b):
                if 0 <= u < len(dstN) and 0 <= v < len(srcN):
                    dstN[u].add(v)
                    srcN[v].add(u)

    return srcN, dstN


# -----------------------------
# task_type inference (optional)
# -----------------------------
def infer_task_to_task_type(g: HeteroData) -> Dict[int, Optional[int]]:
    if "tasks" not in g.node_types or "task_types" not in g.node_types:
        return {}
    tt_edges = edge_types_between(g, "tasks", "task_types")
    if not tt_edges:
        return {}
    task_to_tt, _ = neighbors_set(g, tt_edges, "tasks", "task_types")
    out: Dict[int, Optional[int]] = {}
    for t_idx, tt_set in enumerate(task_to_tt):
        out[t_idx] = next(iter(tt_set)) if tt_set else None
    return out


# -----------------------------
# core verification
# -----------------------------
def compute_engineer_task_sets(g: HeteroData):
    """
    Build:
      e_to_tasks[e] = set(task_idx)
      t_to_engineers[t] = set(engineer_idx)
      t_to_tasks_assignments etc derived via assignments
    """
    for need in ["engineers", "assignments", "tasks"]:
        if need not in g.node_types:
            raise ValueError(f"Missing node type {need!r}. node_types={g.node_types}")

    ea_edges = edge_types_between(g, "engineers", "assignments")
    at_edges = edge_types_between(g, "assignments", "tasks")
    if not ea_edges:
        raise ValueError("No edge types found between engineers and assignments.")
    if not at_edges:
        raise ValueError("No edge types found between assignments and tasks.")

    # engineer -> assignments
    e_to_a, _ = neighbors_set(g, ea_edges, "engineers", "assignments")
    # assignment -> engineers
    _, a_to_e = neighbors_set(g, ea_edges, "engineers", "assignments")

    # assignment -> tasks
    a_to_t, _ = neighbors_set(g, at_edges, "assignments", "tasks")
    # task -> assignments
    _, t_to_a = neighbors_set(g, at_edges, "assignments", "tasks")

    # task -> engineers via assignments
    t_to_e: List[Set[int]] = [set() for _ in range(g["tasks"].num_nodes)]
    for t_idx, a_set in enumerate(t_to_a):
        es = set()
        for a in a_set:
            es.update(a_to_e[a])
        t_to_e[t_idx] = es

    # engineer -> tasks via assignments
    e_to_tasks: List[Set[int]] = [set() for _ in range(g["engineers"].num_nodes)]
    for e_idx, a_set in enumerate(e_to_a):
        ts = set()
        for a in a_set:
            ts.update(a_to_t[a])
        e_to_tasks[e_idx] = ts

    return e_to_tasks, t_to_e


def top_engineers_by_task_count(e_to_tasks: List[Set[int]], k: int) -> List[int]:
    counts = [(i, len(s)) for i, s in enumerate(e_to_tasks)]
    counts.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in counts[:k]]


def verify(
    g: HeteroData,
    topk: int = 20,
    partner_topk: int = 10,
    max_print_tasktypes: int = 15,
):
    eng_ids = safe_node_ids(g, "engineers")
    task_ids = safe_node_ids(g, "tasks")

    e_to_tasks, t_to_e = compute_engineer_task_sets(g)

    # optional task_type
    task_to_tt = infer_task_to_task_type(g)
    task_type_ids = safe_node_ids(g, "task_types") if "task_types" in g.node_types else []
    has_task_type = bool(task_to_tt) and bool(task_type_ids)

    elites = top_engineers_by_task_count(e_to_tasks, topk)

    print(f"Graph: engineers={len(eng_ids)}, tasks={len(task_ids)}")
    print(f"TopK elites (by distinct tasks): K={topk}")
    print("=" * 80)

    for rank, e_idx in enumerate(elites, start=1):
        e_name = eng_ids[e_idx]
        tasks = e_to_tasks[e_idx]
        tc = len(tasks)

        # (1) group size dist over this engineer's tasks
        size_dist = Counter()
        two_partner_counter = Counter()  # partner engineer id counts in size=2 tasks
        tasktype_dist = Counter()

        for t in tasks:
            workers = t_to_e[t]
            sz = len(workers)
            size_dist[sz] += 1

            # (2) partner diversity in size=2 tasks
            if sz == 2 and e_idx in workers:
                other = next(iter(workers - {e_idx}))
                two_partner_counter[eng_ids[other]] += 1

            # (3) task_type dist
            if has_task_type:
                tt = task_to_tt.get(t, None)
                if tt is None:
                    tasktype_dist["<unknown>"] += 1
                else:
                    if 0 <= tt < len(task_type_ids):
                        tasktype_dist[task_type_ids[tt]] += 1
                    else:
                        tasktype_dist["<out_of_range>"] += 1

        # compute share of size=2
        two = size_dist.get(2, 0)
        two_share = two / max(tc, 1)

        # partner stats
        unique_partners = len(two_partner_counter)
        top_partners = two_partner_counter.most_common(partner_topk)

        print(f"[rank {rank:>2}] engineer={e_name}")
        print(f"  distinct_task_count: {tc}")
        print(f"  worker_group_size_dist (top): {dict(size_dist.most_common(8))}")
        print(f"  share(group_size==2): {two_share:.3f}  ( {two} / {tc} )")
        print(f"  partners_in_size2_tasks: unique={unique_partners}")
        if top_partners:
            print("  top partners (count in size=2 tasks):")
            for p, c in top_partners:
                print(f"    - {p}: {c}")
        else:
            print("  top partners: <none> (no size=2 tasks or engineer not in them)")

        if has_task_type:
            print(f"  task_type_dist (top {max_print_tasktypes}):")
            for tt, c in tasktype_dist.most_common(max_print_tasktypes):
                print(f"    - {tt}: {c}")
        else:
            print("  task_type_dist: <unavailable> (no tasks->task_types path)")

        print("-" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=str, default="data/graph/sdge.pt")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--partner_topk", type=int, default=10)
    ap.add_argument("--max_tasktypes", type=int, default=15)
    args = ap.parse_args()

    g = load_graph(args.graph)
    verify(
        g,
        topk=args.topk,
        partner_topk=args.partner_topk,
        max_print_tasktypes=args.max_tasktypes,
    )


if __name__ == "__main__":
    main()
