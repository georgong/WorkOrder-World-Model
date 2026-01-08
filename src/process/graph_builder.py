"""
Minimal graph builder based on torch_geometric.data.HeteroData.

Design principles:
1. All datasets are first converted into nodes (id_map is frozen).
2. All edges are built strictly based on id_map (no node creation in edge builders).
3. YAML mappings are mechanically translated into mapping-based edges.
4. Keep it simple: no cleaning, no encoding, no feature engineering.
"""

from typing import Dict, Any, Tuple, List
import torch
import pandas as pd
from torch_geometric.data import HeteroData
import yaml

# -------------------------------------------------
# Utilities
# -------------------------------------------------

def infer_id_col(df: pd.DataFrame) -> str:
    """
    Infer a node id column.
    This is intentionally conservative.
    Override explicitly if this fails.
    """
    if "W6KEY" in df.columns:
        return "W6KEY"
    if "NAME" in df.columns:
        return "NAME"
    raise ValueError(
        f"Cannot infer id column automatically. Columns: {list(df.columns)[:10]}"
    )


# -------------------------------------------------
# Node building (Stage 1)
# -------------------------------------------------

class NodeBuilder:
    """
    Build node universe from all datasets.
    Only determines:
      - which node types exist
      - how many nodes per type
      - raw_id -> contiguous index mapping
    """

    def __init__(self, node_schema: Dict[str, Tuple[str, str]]):
        """
        node_schema:
          node_type -> (table_name, id_column)
        """
        self.node_schema = node_schema

    def build(self, datasets: Dict[str, pd.DataFrame]):
        id_maps: Dict[str, Dict[Any, int]] = {}
        node_counts: Dict[str, int] = {}

        for ntype, (table, id_col) in self.node_schema.items():
            ids = (
                datasets[table][id_col]
                .dropna()
                .drop_duplicates()
                .tolist()
            )
            id_maps[ntype] = {rid: i for i, rid in enumerate(ids)}
            node_counts[ntype] = len(ids)

        return id_maps, node_counts

    def attach_to_heterodata(self, data: HeteroData, node_counts: Dict[str, int]):
        """
        Register num_nodes for each node type in HeteroData.
        """
        for ntype, n in node_counts.items():
            data[ntype].num_nodes = int(n)


# -------------------------------------------------
# Edge building (Stage 2)
# -------------------------------------------------

class MappingEdgeBuilder:
    """
    Edge builder based on column equality (foreign-key style).
    """

    def __init__(
        self,
        etype: Tuple[str, str, str],
        table: str,
        src_col: str,
        dst_col: str,
        dedup: bool = True,
        device: str = "cpu",
    ):
        self.etype = etype          # (src_type, relation, dst_type)
        self.table = table          # source table
        self.src_col = src_col
        self.dst_col = dst_col
        self.dedup = dedup
        self.device = device

    def build(self, datasets, id_maps):
        src_type, _, dst_type = self.etype
        df = datasets[self.table]

        src_map = id_maps[src_type]
        dst_map = id_maps[dst_type]

        pairs = []
        for s, d in zip(df[self.src_col], df[self.dst_col]):
            if s in src_map and d in dst_map:
                pairs.append((src_map[s], dst_map[d]))

        if self.dedup and pairs:
            pairs = list(dict.fromkeys(pairs))

        edge_index = (
            torch.tensor(pairs, dtype=torch.long, device=self.device)
            .t()
            .contiguous()
            if pairs
            else torch.empty((2, 0), dtype=torch.long, device=self.device)
        )

        return {self.etype: edge_index}


# -------------------------------------------------
# YAML -> schema / builders
# -------------------------------------------------

def build_node_schema_from_yaml(
    cfg: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
    id_overrides: Dict[str, str] | None = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Convert YAML mappings into node_schema.
    Each table becomes one node type.
    """
    id_overrides = id_overrides or {}
    node_schema = {}

    for table in cfg["mapings"].keys():
        if table in id_overrides:
            node_schema[table] = (table, id_overrides[table])
        else:
            node_schema[table] = (table, infer_id_col(datasets[table]))

    return node_schema


def build_edge_builders_from_yaml(
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> List[MappingEdgeBuilder]:
    """
    Convert YAML mappings into mapping-based edge builders.
    """
    builders: List[MappingEdgeBuilder] = []

    for src_table, body in cfg["mappings"].items():
        links = body.get("links", {}) or {}
        for dst_table, rule in links.items():
            etype = (
                src_table,
                f"links_to_{dst_table}",
                dst_table,
            )
            builders.append(
                MappingEdgeBuilder(
                    etype=etype,
                    table=src_table,
                    src_col=rule["left_on"],
                    dst_col=rule["right_on"],
                    dedup=True,
                    device=device,
                )
            )

    return builders


# -------------------------------------------------
# Graph builder (orchestrator)
# -------------------------------------------------

class GraphBuilder:
    """
    Orchestrates:
      1. Node construction
      2. Edge construction
      3. HeteroData assembly
      4. Basic graph statistics
    """

    def __init__(self, node_builder: NodeBuilder, edge_builders: List[MappingEdgeBuilder]):
        self.node_builder = node_builder
        self.edge_builders = edge_builders

    def build(self, datasets: Dict[str, pd.DataFrame]):
        data = HeteroData()

        # ----- Stage 1: nodes -----
        id_maps, node_counts = self.node_builder.build(datasets)
        self.node_builder.attach_to_heterodata(data, node_counts)

        # ----- Stage 2: edges -----
        merged: Dict[Tuple[str, str, str], List[torch.Tensor]] = {}
        per_builder_stats = []

        for b in self.edge_builders:
            out = b.build(datasets, id_maps)
            for etype, ei in out.items():
                merged.setdefault(etype, []).append(ei)
                per_builder_stats.append({
                    "builder": b.__class__.__name__,
                    "etype": etype,
                    "edges": int(ei.size(1)),
                })

        edge_counts = {}
        for etype, chunks in merged.items():
            ei = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
            data[etype].edge_index = ei
            edge_counts[str(etype)] = int(ei.size(1))

        report = {
            "node_counts": {k: int(v) for k, v in node_counts.items()},
            "total_nodes": int(sum(node_counts.values())),
            "edge_counts": edge_counts,
            "total_edges": int(sum(edge_counts.values())),
            "per_builder": per_builder_stats,
        }

        # Stash for downstream inspection/debugging
        data._id_maps = id_maps
        data._report = report

        return data, report






