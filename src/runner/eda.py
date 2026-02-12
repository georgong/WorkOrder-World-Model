# training_data_eda.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from torch_geometric.data import HeteroData


def _to_float(x: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point():
        return x.float()
    return x


@torch.no_grad()
def feature_stats(
    x: torch.Tensor,
    *,
    max_rows: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-feature mean/var/std on CPU tensor x [N, F].
    Optionally subsample rows for speed/memory stability.
    """
    assert x.dim() == 2, f"x must be 2D, got {x.shape}"
    x = _to_float(x).cpu()

    N, F = x.shape

    if max_rows is not None and N > max_rows:
        # deterministic-ish subsample: take evenly spaced indices
        idx = torch.linspace(0, N - 1, steps=max_rows).long()
        x = x.index_select(0, idx)
        N = x.shape[0]

    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)

    # some robust summaries
    abs_mean = mean.abs()
    abs_max = x.abs().max(dim=0).values

    return {
        "N": torch.tensor(N),
        "F": torch.tensor(F),
        "mean": mean,
        "var": var,
        "std": std,
        "abs_mean": abs_mean,
        "abs_max": abs_max,
    }


def summarize_feature_scale(name: str, stats: Dict[str, torch.Tensor], attr_name=None, topk: int = 10) -> str:
    mean = stats["mean"]
    std = stats["std"]
    var = stats["var"]
    abs_max = stats["abs_max"]
    abs_mean = stats["abs_mean"]
    N = int(stats["N"].item())
    F = int(stats["F"].item())

    def fname(i: int) -> str:
        if attr_name is None:
            return f"f{i}"
        if i < len(attr_name):
            return attr_name[i]
        return f"f{i}"

    std_min = float(std.min().item())
    std_max = float(std.max().item())
    absmax_max = float(abs_max.max().item())
    absmean_max = float(abs_mean.max().item())

    std_ratio = (std_max / max(std_min, 1e-12)) if F > 0 else float("nan")

    top_std_vals, top_std_idx = torch.topk(std, k=min(topk, F), largest=True)
    bot_std_vals, bot_std_idx = torch.topk(std, k=min(topk, F), largest=False)

    lines = []
    lines.append(f"[{name}] x shape: N={N}, F={F}")
    lines.append(f"  std range: min={std_min:.3e}, max={std_max:.3e}, ratio(max/min)={std_ratio:.3e}")
    lines.append(f"  abs(mean) max={absmean_max:.3e} | abs(x) max over features={absmax_max:.3e}")

    lines.append("  top std features (name | idx | std | mean | abs_max):")
    for i in range(top_std_idx.numel()):
        j = int(top_std_idx[i].item())
        lines.append(
            f"    {fname(j):<30} | {j:4d} | std={float(std[j]):.3e} | "
            f"mean={float(mean[j]):.3e} | abs_max={float(abs_max[j]):.3e}"
        )

    lines.append("  bottom std features (name | idx | std | mean | abs_max):")
    for i in range(bot_std_idx.numel()):
        j = int(bot_std_idx[i].item())
        lines.append(
            f"    {fname(j):<30} | {j:4d} | std={float(std[j]):.3e} | "
            f"mean={float(mean[j]):.3e} | abs_max={float(abs_max[j]):.3e}"
        )

    if std_ratio > 1e4:
        lines.append("  [RED FLAG] std ratio > 1e4: extreme scale mismatch between features.")

    if absmax_max > 1e6:
        lines.append("  [RED FLAG] abs_max > 1e6: likely timestamp / ID / money-like feature.")

    return "\n".join(lines)


@torch.no_grad()
def target_stats(y: torch.Tensor, *, max_rows: Optional[int] = None, eps: float = 1e-12) -> Dict[str, float]:
    y = _to_float(y).flatten().cpu()
    N = y.numel()
    if max_rows is not None and N > max_rows:
        idx = torch.linspace(0, N - 1, steps=max_rows).long()
        y = y.index_select(0, idx)
        N = y.numel()

    mean = float(y.mean().item())
    var = float(y.var(unbiased=False).item())
    std = float((var + eps) ** 0.5)
    mn = float(y.min().item())
    mx = float(y.max().item())

    # some quantiles for distribution shape
    qs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    qv = torch.quantile(y, qs)

    return {
        "N": N,
        "mean": mean,
        "var": var,
        "std": std,
        "min": mn,
        "max": mx,
        "q": {float(qs[i]): float(qv[i]) for i in range(len(qs))},
    }

def graph_structure_stats(data: HeteroData):
    """
    Print graph structure statistics:
      - Node/edge counts per type
      - Degree distributions
      - Edge attribute stats (if present)
      - Connected components, density, clustering (for homogeneous graphs)
    """
    from collections import Counter
    import numpy as np

    lines = []
    lines.append("=" * 80)
    lines.append("3) Graph Structure Statistics")
    lines.append("=" * 80)

    # Node counts
    for ntype in data.node_types:
        n = data[ntype].num_nodes
        lines.append(f"Node type '{ntype}': {n} nodes")

    # Edge counts and degree distributions
    for etype in data.edge_types:
        src, rel, dst = etype
        edge_index = data[etype].edge_index
        num_edges = edge_index.shape[1]
        lines.append(f"Edge type {etype}: {num_edges} edges")

        # Degree distributions
        src_deg = np.bincount(edge_index[0].cpu().numpy(), minlength=data[src].num_nodes)
        dst_deg = np.bincount(edge_index[1].cpu().numpy(), minlength=data[dst].num_nodes)
        lines.append(f"  {src} out-degree: mean={src_deg.mean():.2f}, std={src_deg.std():.2f}, max={src_deg.max()}, min={src_deg.min()}")
        lines.append(f"  {dst} in-degree: mean={dst_deg.mean():.2f}, std={dst_deg.std():.2f}, max={dst_deg.max()}, min={dst_deg.min()}")

        # Optional: print top nodes by degree
        topk = 5
        if len(src_deg) > 0:
            top_src = np.argsort(-src_deg)[:topk]
            lines.append(f"    Top {src} nodes by out-degree: " +
                         ", ".join([f"{i}({src_deg[i]})" for i in top_src]))
        if len(dst_deg) > 0:
            top_dst = np.argsort(-dst_deg)[:topk]
            lines.append(f"    Top {dst} nodes by in-degree: " +
                         ", ".join([f"{i}({dst_deg[i]})" for i in top_dst]))

        # Edge attribute stats (if present)
        for attr in ["weight", "timestamp", "status"]:
            if hasattr(data[etype], attr):
                vals = getattr(data[etype], attr)
                if isinstance(vals, torch.Tensor):
                    vals = vals.cpu().numpy()
                    lines.append(f"  Edge attr '{attr}': mean={vals.mean():.2f}, std={vals.std():.2f}, min={vals.min()}, max={vals.max()}")

    # Connected components and density for homogeneous graphs
    try:
        import networkx as nx
        # Only for simple graphs (e.g., engineer-task bipartite)
        # Here, build a simple undirected graph for the largest node type
        main_ntype = max(data.node_types, key=lambda n: data[n].num_nodes)
        G = nx.Graph()
        for etype in data.edge_types:
            src, rel, dst = etype
            if src == dst == main_ntype:
                edge_index = data[etype].edge_index
                edges = list(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))
                G.add_edges_from(edges)
        if G.number_of_nodes() > 0:
            ncc = nx.number_connected_components(G)
            dens = nx.density(G)
            lines.append(f"Undirected {main_ntype} graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            lines.append(f"  Connected components: {ncc}, density: {dens:.4f}")
            if G.number_of_nodes() < 10000:
                clust = nx.average_clustering(G)
                lines.append(f"  Average clustering coefficient: {clust:.4f}")
    except Exception as e:
        lines.append(f"[note] NetworkX graph stats skipped: {e}")

    return "\n".join(lines)


def missing_and_outlier_stats(data: HeteroData):
    """
    Check for missing values and outliers in node features.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("4) Missing Value and Outlier Check")
    lines.append("=" * 80)

    for ntype in data.node_types:
        node = data[ntype]
        if hasattr(node, "x") and isinstance(node.x, torch.Tensor):
            x = node.x
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            lines.append(f"[{ntype}.x] NaN count: {nan_count}, Inf count: {inf_count}")
            # Outlier: values > 5 std from mean
            mean = x.float().mean().item()
            std = x.float().std().item()
            outlier_count = ((x.float() - mean).abs() > 5 * std).sum().item()
            lines.append(f"  Outlier (>5 std from mean) count: {outlier_count}")
    return "\n".join(lines)


def build_collaboration_graph(data: HeteroData):
    """
    Build an engineer collaboration graph where edges represent co-work on tasks.
    Returns a NetworkX graph with engineer nodes.
    """
    import networkx as nx
    import numpy as np
    
    # Find engineer-assignment-task path
    eng_to_assign = {}
    assign_to_task = {}
    
    for etype in data.edge_types:
        src, rel, dst = etype
        if src == "engineers" and dst == "assignments":
            edge_index = data[etype].edge_index
            for i in range(edge_index.shape[1]):
                e = edge_index[0, i].item()
                a = edge_index[1, i].item()
                if e not in eng_to_assign:
                    eng_to_assign[e] = set()
                eng_to_assign[e].add(a)
        elif src == "assignments" and dst == "tasks":
            edge_index = data[etype].edge_index
            for i in range(edge_index.shape[1]):
                a = edge_index[0, i].item()
                t = edge_index[1, i].item()
                if a not in assign_to_task:
                    assign_to_task[a] = set()
                assign_to_task[a].add(t)
    
    # Build task -> engineers mapping
    task_to_eng = {}
    for e, assigns in eng_to_assign.items():
        for a in assigns:
            if a in assign_to_task:
                for t in assign_to_task[a]:
                    if t not in task_to_eng:
                        task_to_eng[t] = set()
                    task_to_eng[t].add(e)
    
    # Build engineer collaboration graph
    G = nx.Graph()
    num_engineers = data["engineers"].num_nodes
    G.add_nodes_from(range(num_engineers))
    
    for t, engineers in task_to_eng.items():
        engineers = list(engineers)
        for i in range(len(engineers)):
            for j in range(i+1, len(engineers)):
                if G.has_edge(engineers[i], engineers[j]):
                    G[engineers[i]][engineers[j]]['weight'] += 1
                else:
                    G.add_edge(engineers[i], engineers[j], weight=1)
    
    return G


def community_detection_stats(G):
    """
    Detect communities (clusters) in the engineer collaboration graph.
    """
    import networkx as nx

    lines = []
    lines.append("=" * 80)
    lines.append("5) Community Detection / Clustering")
    lines.append("=" * 80)

    try:
        if G.number_of_edges() == 0:
            lines.append("No collaboration graph edges found.")
            return "\n".join(lines)
        
        lines.append(f"Engineer collaboration graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        lines.append(f"Detected {len(comms)} communities.")
        sizes = sorted([len(c) for c in comms], reverse=True)
        lines.append(f"  Community sizes (top 10): {sizes[:10]}")
    except Exception as e:
        lines.append(f"Community detection failed: {e}")

    return "\n".join(lines)


def centrality_analysis(G):
    """
    Compute centrality measures for the engineer collaboration graph.
    """
    import networkx as nx

    lines = []
    lines.append("=" * 80)
    lines.append("6) Centrality & Influence Analysis")
    lines.append("=" * 80)

    try:
        if G.number_of_edges() == 0:
            lines.append("No collaboration graph edges found.")
            return "\n".join(lines)
        
        # Sample for speed if too large
        if G.number_of_nodes() > 5000:
            lines.append(f"Graph too large ({G.number_of_nodes()} nodes), computing degree centrality only.")
            deg = nx.degree_centrality(G)
            lines.append("Top 10 engineers by degree centrality:")
            for n, v in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"  Engineer {n}: {v:.4f}")
        else:
            deg = nx.degree_centrality(G)
            bet = nx.betweenness_centrality(G)
            clo = nx.closeness_centrality(G)
            
            lines.append("Top 10 engineers by degree centrality:")
            for n, v in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"  Engineer {n}: {v:.4f}")
            lines.append("Top 10 engineers by betweenness centrality:")
            for n, v in sorted(bet.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"  Engineer {n}: {v:.4f}")
            lines.append("Top 10 engineers by closeness centrality:")
            for n, v in sorted(clo.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"  Engineer {n}: {v:.4f}")
    except Exception as e:
        lines.append(f"Centrality computation failed: {e}")

    return "\n".join(lines)


def path_connectivity_analysis(G):
    """
    Analyze shortest paths and connectivity in the engineer collaboration graph.
    """
    import networkx as nx

    lines = []
    lines.append("=" * 80)
    lines.append("7) Path & Connectivity Analysis")
    lines.append("=" * 80)

    try:
        if G.number_of_edges() == 0:
            lines.append("No collaboration graph edges found.")
            return "\n".join(lines)
        
        if nx.is_connected(G):
            if G.number_of_nodes() > 5000:
                lines.append(f"Graph is connected with {G.number_of_nodes()} nodes (too large for detailed path analysis).")
            else:
                avg_path = nx.average_shortest_path_length(G)
                diam = nx.diameter(G)
                lines.append(f"Graph is connected. Avg shortest path: {avg_path:.2f}, diameter: {diam}")
        else:
            ncc = nx.number_connected_components(G)
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            lines.append(f"Graph is NOT connected. {ncc} components.")
            lines.append(f"Largest component: {subG.number_of_nodes()} nodes ({subG.number_of_nodes()/G.number_of_nodes()*100:.1f}%), {subG.number_of_edges()} edges")
            if subG.number_of_nodes() <= 5000:
                avg_path = nx.average_shortest_path_length(subG)
                diam = nx.diameter(subG)
                lines.append(f"  Avg shortest path: {avg_path:.2f}, diameter: {diam}")
    except Exception as e:
        lines.append(f"Path/connectivity computation failed: {e}")

    return "\n".join(lines)


def feature_structure_correlation(data: HeteroData):
    """
    Correlate node features (e.g., degree) with structural metrics.
    """
    import numpy as np

    lines = []
    lines.append("=" * 80)
    lines.append("9) Feature Correlation with Structure")
    lines.append("=" * 80)

    # Example: for each node type, correlate degree with first feature column (if exists)
    for ntype in data.node_types:
        node = data[ntype]
        if not hasattr(node, "x") or not isinstance(node.x, torch.Tensor):
            continue
        x = node.x
        if x.dim() != 2 or x.shape[1] == 0:
            continue
        feat = x[:, 0].cpu().numpy()
        # Compute degree (sum of in- and out-edges)
        deg = np.zeros(node.num_nodes, dtype=np.int32)
        for etype in data.edge_types:
            src, rel, dst = etype
            edge_index = data[etype].edge_index
            if src == ntype:
                src_idx = edge_index[0].cpu().numpy()
                for i in src_idx:
                    deg[i] += 1
            if dst == ntype:
                dst_idx = edge_index[1].cpu().numpy()
                for i in dst_idx:
                    deg[i] += 1
        if deg.max() == 0:
            continue
        # Pearson correlation
        from scipy.stats import pearsonr
        try:
            corr, pval = pearsonr(deg, feat)
            lines.append(f"[{ntype}] Degree-feature[0] correlation: r={corr:.3f}, p={pval:.2g}")
        except Exception as e:
            lines.append(f"[{ntype}] Correlation failed: {e}")
    return "\n".join(lines)

def heterogeneous_path_analysis(data: HeteroData):
    """
    Analyze heterogeneous paths that GNNs can leverage for context aggregation.
    Focus on multi-hop paths like: engineer → assignment → task → task_type
    """
    import numpy as np
    from collections import Counter, defaultdict
    
    lines = []
    lines.append("=" * 80)
    lines.append("8) Heterogeneous Path Analysis (Multi-hop Context)")
    lines.append("=" * 80)
    
    # Build mappings for heterogeneous paths
    eng_to_assign = defaultdict(set)
    assign_to_task = defaultdict(set)
    task_to_tasktype = defaultdict(set)
    
    for etype in data.edge_types:
        src, rel, dst = etype
        edge_index = data[etype].edge_index
        
        if src == "engineers" and dst == "assignments":
            for i in range(edge_index.shape[1]):
                e, a = edge_index[0, i].item(), edge_index[1, i].item()
                eng_to_assign[e].add(a)
                
        elif src == "assignments" and dst == "tasks":
            for i in range(edge_index.shape[1]):
                a, t = edge_index[0, i].item(), edge_index[1, i].item()
                assign_to_task[a].add(t)
                
        elif src == "tasks" and dst == "task_types":
            for i in range(edge_index.shape[1]):
                t, tt = edge_index[0, i].item(), edge_index[1, i].item()
                task_to_tasktype[t].add(tt)

    # Compute reachability: engineer → task_types (via assignments and tasks)
    eng_to_tasktype = defaultdict(set)
    eng_to_task = defaultdict(set)
    
    for e, assigns in eng_to_assign.items():
        for a in assigns:
            for t in assign_to_task[a]:
                eng_to_task[e].add(t)
                for tt in task_to_tasktype[t]:
                    eng_to_tasktype[e].add(tt)
    
    # Statistics
    lines.append(f"Total engineers: {data['engineers'].num_nodes}")
    lines.append(f"Engineers with reachable task_types: {len(eng_to_tasktype)}")
    
    tasktype_counts = [len(tts) for tts in eng_to_tasktype.values()]
    if tasktype_counts:
        lines.append(f"Task types per engineer: mean={np.mean(tasktype_counts):.2f}, "
                    f"median={np.median(tasktype_counts):.0f}, "
                    f"max={np.max(tasktype_counts)}, min={np.min(tasktype_counts)}")
    
    task_counts = [len(ts) for ts in eng_to_task.values()]
    if task_counts:
        lines.append(f"Tasks per engineer: mean={np.mean(task_counts):.2f}, "
                    f"median={np.median(task_counts):.0f}, "
                    f"max={np.max(task_counts)}, min={np.min(task_counts)}")
    
    # Path diversity: how many unique 2-hop and 3-hop neighbors?
    lines.append("\nPath diversity (what GNNs can aggregate):")
    lines.append("  1-hop: engineer → assignments")
    lines.append(f"    Avg neighbors: {np.mean([len(a) for a in eng_to_assign.values()]):.2f}")
    lines.append("  2-hop: engineer → assignment → tasks")
    lines.append(f"    Avg neighbors: {np.mean(task_counts) if task_counts else 0:.2f}")
    lines.append("  3-hop: engineer → assignment → task → task_types")
    lines.append(f"    Avg neighbors: {np.mean(tasktype_counts) if tasktype_counts else 0:.2f}")
        
    # Show example paths for a few engineers
    lines.append("\nExample heterogeneous paths (first 3 engineers with rich context):")
    count = 0
    for e, tts in sorted(eng_to_tasktype.items(), key=lambda x: len(x[1]), reverse=True)[:3]:
        count += 1
        tasks = eng_to_task[e]
        assigns = eng_to_assign[e]
        lines.append(f"  Engineer {e}:")
        lines.append(f"    → {len(assigns)} assignments → {len(tasks)} tasks → {len(tts)} task_types")
    
    return "\n".join(lines)


def bipartite_projection_analysis(data: HeteroData):
    """
    Analyze the engineer-task_type bipartite graph directly.
    This is denser than engineer-engineer collaboration and shows what GNNs can leverage.
    """
    import numpy as np
    import networkx as nx
    from collections import Counter
    
    lines = []
    lines.append("=" * 80)
    lines.append("10) Bipartite Projection: Engineer-TaskType Graph")
    lines.append("=" * 80)
    
    # Build engineer → task_type bipartite graph
    B = nx.Graph()
    eng_nodes = set()
    tt_nodes = set()
    
    # Find direct engineer → task_type edges (if they exist)
    direct_edges = 0
    for etype in data.edge_types:
        src, rel, dst = etype
        if (src == "engineers" and dst == "task_types") or (src == "task_types" and dst == "engineers"):
            edge_index = data[etype].edge_index
            direct_edges = edge_index.shape[1]
            for i in range(edge_index.shape[1]):
                if src == "engineers":
                    e, tt = edge_index[0, i].item(), edge_index[1, i].item()
                else:
                    tt, e = edge_index[0, i].item(), edge_index[1, i].item()
                B.add_edge(f"eng_{e}", f"tt_{tt}")
                eng_nodes.add(e)
                tt_nodes.add(tt)
    
    lines.append(f"Direct engineer-task_type edges: {direct_edges}")
    lines.append(f"Engineers connected to task_types: {len(eng_nodes)}")
    lines.append(f"Task_types connected to engineers: {len(tt_nodes)}")
    
    if direct_edges > 0:
        # Degree distributions in bipartite graph
        eng_degrees = []
        tt_degrees = []
        
        for node in B.nodes():
            deg = B.degree(node)
            if node.startswith("eng_"):
                eng_degrees.append(deg)
            else:
                tt_degrees.append(deg)
        
        if eng_degrees:
            lines.append(f"\nEngineer degree in bipartite graph (task_types per engineer):")
            lines.append(f"  mean={np.mean(eng_degrees):.2f}, median={np.median(eng_degrees):.0f}, "
                        f"max={np.max(eng_degrees)}, min={np.min(eng_degrees)}")
            lines.append(f"  Distribution: {Counter(eng_degrees).most_common(10)}")
        
        if tt_degrees:
            lines.append(f"\nTask_type degree in bipartite graph (engineers per task_type):")
            lines.append(f"  mean={np.mean(tt_degrees):.2f}, median={np.median(tt_degrees):.0f}, "
                        f"max={np.max(tt_degrees)}, min={np.min(tt_degrees)}")
            lines.append(f"  Distribution: {Counter(tt_degrees).most_common(10)}")
        
        # Compare density with engineer collaboration graph
        if len(eng_nodes) > 0 and len(tt_nodes) > 0:
            density = direct_edges / (len(eng_nodes) * len(tt_nodes))
            lines.append(f"\nBipartite graph density: {density:.6f}")
            lines.append(f"  (Much denser than engineer collaboration: {direct_edges} edges vs 490 collaboration edges)")
        
        # Specialization analysis
        lines.append("\nSpecialization insights:")
        specialized_engs = sum(1 for d in eng_degrees if d <= 3)
        generalist_engs = sum(1 for d in eng_degrees if d >= 10)
        lines.append(f"  Specialists (≤3 task_types): {specialized_engs} engineers ({specialized_engs/len(eng_degrees)*100:.1f}%)")
        lines.append(f"  Generalists (≥10 task_types): {generalist_engs} engineers ({generalist_engs/len(eng_degrees)*100:.1f}%)")
        
        # One-mode projection: engineer-engineer via shared task_types
        lines.append("\nOne-mode projection (engineer-engineer via shared task_types):")
        eng_projection = nx.bipartite.weighted_projected_graph(B, [f"eng_{e}" for e in eng_nodes])
        lines.append(f"  Projected graph: {eng_projection.number_of_nodes()} engineers, "
                    f"{eng_projection.number_of_edges()} edges")
        lines.append(f"  (Compare to direct collaboration: 490 edges)")
        
        if eng_projection.number_of_edges() > 0:
            # Weight distribution (number of shared task_types)
            weights = [d['weight'] for u, v, d in eng_projection.edges(data=True)]
            lines.append(f"  Shared task_types per edge: mean={np.mean(weights):.2f}, "
                        f"median={np.median(weights):.0f}, max={np.max(weights)}")
    else:
        lines.append("\n[Note] No direct engineer-task_type edges found.")
        lines.append("  GNNs will aggregate context via multi-hop paths (engineer→assignment→task→task_type)")
    
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, default="data/graph/sdge.pt")
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--out", type=str, default="", help="optional: write report to a txt file")
    ap.add_argument("--max_rows", type=int, default=200000, help="subsample rows per node type for faster stats")
    ap.add_argument("--topk", type=int, default=10, help="top-k features to print by std")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    assert pt_path.exists(), f"File not found: {pt_path}"

    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData), "pt must contain a torch_geometric.data.HeteroData"

    target = args.target
    assert target in data.node_types, f"target node type {target!r} not in {data.node_types}"
    assert hasattr(data[target], "y"), f"{target}.y missing"

    lines = []
    lines.append("=" * 80)
    lines.append("HeteroData Training Data EDA Report")
    lines.append("=" * 80)
    lines.append(f"pt: {str(pt_path)}")
    lines.append(f"node_types: {data.node_types}")
    lines.append(f"edge_types: {data.edge_types}")
    lines.append("")

    # 1) Feature scale per node type
    lines.append("=" * 80)
    lines.append("1) Feature mean/variance scale (per node type)")
    lines.append("=" * 80)

    for nt in data.node_types:
        if not hasattr(data[nt], "x"):
            continue
        x = data[nt].x
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            lines.append(f"[{nt}] x is not a 2D tensor, skip.")
            continue

        attr_names = None
        if hasattr(data[nt], "attr_name"):
            attr_names = data[nt].attr_name

        stats = feature_stats(x, max_rows=args.max_rows)
        lines.append(
            summarize_feature_scale(
                f"node_type={nt}",
                stats,
                attr_name=attr_names,
                topk=args.topk,
            )
        )
        lines.append("")

    # 2) Target label distribution
    lines.append("=" * 80)
    lines.append("2) Target y distribution (mean/variance + quantiles)")
    lines.append("=" * 80)

    y = data[target].y
    tstats = target_stats(y, max_rows=None)
    lines.append(f"[target={target}] N={tstats['N']}")
    lines.append(f"  mean={tstats['mean']:.6g} | var={tstats['var']:.6g} | std={tstats['std']:.6g}")
    lines.append(f"  min={tstats['min']:.6g} | max={tstats['max']:.6g}")
    lines.append("  quantiles:")
    for q in sorted(tstats["q"].keys()):
        lines.append(f"    q{q:>5.2f}: {tstats['q'][q]:.6g}")

    # quick red flags for y
    if tstats["std"] > 10.0 and (abs(tstats["mean"]) < 1e-3 or abs(tstats["mean"]) / max(tstats["std"], 1e-12) < 0.1):
        lines.append("  [note] y has large std relative to mean. Check if y needs normalization/log-transform.")
    if tstats["max"] > 1e6:
        lines.append("  [RED FLAG] y max > 1e6. If you expected small values, labels might be wrong-scale.")
    lines.append("")

    # 3) Graph structure stats
    lines.append(graph_structure_stats(data))
    # 4) Missing/outlier stats
    lines.append(missing_and_outlier_stats(data))
    
    # Build collaboration graph once for all subsequent analyses
    G = build_collaboration_graph(data)

    # 5) Community detection
    lines.append(community_detection_stats(G))
    # 6) Centrality analysis
    lines.append(centrality_analysis(G))
    # 7) Path/connectivity analysis
    lines.append(path_connectivity_analysis(G))
    # 8) Heterogeneous path analysis
    lines.append(heterogeneous_path_analysis(data))
    # 9) Correlate features with structure
    lines.append(feature_structure_correlation(data))
    # 10) Bipartite projection analysis
    lines.append(bipartite_projection_analysis(data))

    report = "\n".join(lines)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n[written] {out_path}")


if __name__ == "__main__":
    # python -m src.runner.eda
    main()
