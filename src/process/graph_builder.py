from typing import Dict, Any, Tuple, List
import torch
import pandas as pd
from torch_geometric.data import HeteroData
import yaml

# -------------------------------------------------
# Utilities
# -------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging
import polars as pl
import torch
from torch_geometric.data import HeteroData


# --------- Aggregation hooks (pl.DataFrame -> pl.DataFrame(1 row)) ---------

NodeAggFn = Callable[[str, str, List[str], pl.DataFrame], pl.DataFrame]
# args: (node_name, primary_key, trait_cols, group_df) -> single-row df with trait cols
EdgeAggFn = Callable[[str, str, str, str, List[str], pl.DataFrame], pl.DataFrame]
# args: (src, dst, src_pk, dst_pk, edge_trait_cols, group_df) -> single-row df with edge traits


def _default_group_agg(primary_key: str, trait_cols: List[str], df: pl.DataFrame) -> pl.DataFrame:
    """
    Default aggregation for a group df (same primary key).
    Numeric -> mean, others -> first.
    Returns 1-row DataFrame with columns = trait_cols.
    """
    if not trait_cols:
        return pl.DataFrame()

    exprs = []
    schema = df.schema
    for c in trait_cols:
        dt = schema.get(c, None)
        if dt is None:
            # column missing in this group df (shouldn't happen if selection was correct)
            continue
        if dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                 pl.Float32, pl.Float64):
            exprs.append(pl.col(c).mean().alias(c))
        elif dt == pl.Boolean:
            # "any" is a reasonable default
            exprs.append(pl.col(c).any().alias(c))
        else:
            exprs.append(pl.col(c).first().alias(c))

    # If exprs empty, return empty df
    if not exprs:
        return pl.DataFrame()
    return df.select(exprs).head(1)


def default_node_agg_fn(node_name: str, primary_key: str, trait_cols: List[str], group_df: pl.DataFrame) -> pl.DataFrame:
    return _default_group_agg(primary_key, trait_cols, group_df)


def default_edge_agg_fn(
    src: str,
    dst: str,
    src_pk: str,
    dst_pk: str,
    edge_trait_cols: List[str],
    group_df: pl.DataFrame
) -> pl.DataFrame:
    # Same default: numeric mean, others first
    return _default_group_agg("__edge_pair__", edge_trait_cols, group_df)


# --------- Encoding hooks (pl.DataFrame -> torch.Tensor) ---------

def default_feature_encoder(df: pl.DataFrame, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert selected polars DataFrame into a float tensor.
    Default behavior:
      - keep only numeric + bool
      - cast bool -> int
      - null -> 0
    """
    if df.width == 0:
        return torch.empty((df.height, 0), dtype=dtype)

    keep = []
    for c, dt in df.schema.items():
        if dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                 pl.Float32, pl.Float64, pl.Boolean):
            keep.append(c)

    if not keep:
        return torch.empty((df.height, 0), dtype=dtype)

    x = df.select(
        [
            pl.col(c).cast(pl.Int64).fill_null(0) if df.schema[c] == pl.Boolean
            else pl.col(c).cast(pl.Float64).fill_null(0)
            for c in keep
        ]
    ).to_numpy()

    return torch.tensor(x, dtype=dtype)


# --------- Input spec helpers ---------

@dataclass
class NodeSpec:
    name: str
    data: pl.DataFrame
    key_cols: List[str]          # primary key is key_cols[0]
    node_trait_cols: List[str]
    edge_trait_cols: List[str]


class HeteroGraphBuilder:
    """
    Build torch_geometric HeteroData from:
      - node_mappings: list[{node_name: {data,key,node_trait,edge_trait}}]
      - edge_mappings: dict[src]['links'][dst] = {left_on,right_on}
    """

    def __init__(
        self,
        *,
        node_mappings: List[Dict[str, Dict[str, Any]]],
        edge_mappings: Dict[str, Dict[str, Any]],
        node_agg_fn: NodeAggFn = default_node_agg_fn,
        edge_agg_fn: EdgeAggFn = default_edge_agg_fn,
        node_encoder: Callable[[pl.DataFrame], torch.Tensor] = default_feature_encoder,
        edge_encoder: Callable[[pl.DataFrame], torch.Tensor] = default_feature_encoder,
        logger: Optional[logging.Logger] = None,
        debug: bool = False,
    ) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.debug = debug

        self.node_agg_fn = node_agg_fn
        self.edge_agg_fn = edge_agg_fn
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder

        self.node_specs = self._parse_node_mappings(node_mappings)
        self.edge_mappings = edge_mappings

        # built artifacts
        self.node_tables: Dict[str, pl.DataFrame] = {}        # node_name -> aggregated node table (pk + traits)
        self.node_id2idx: Dict[str, Dict[Any, int]] = {}      # node_name -> {pk_value: index}
        self.node_ids: Dict[str, List[Any]] = {}              # node_name -> list of pk values (index order)

    # ----------------- public API -----------------

    def build(self) -> Tuple[HeteroData, Dict[str, Any]]:
        """
        Returns:
          data: HeteroData
          meta: dict with mappings/columns etc.
        """
        self._build_nodes()
        data = HeteroData()
        meta: Dict[str, Any] = {"nodes": {}, "edges": {}}

        # populate node stores
        for node_name, node_df in self.node_tables.items():
            pk = self.node_specs[node_name].key_cols[0]
            # Node ids in index order
            ids = node_df.select(pk).to_series().to_list()
            self.node_ids[node_name] = ids
            self.node_id2idx[node_name] = {v: i for i, v in enumerate(ids)}

            # Encode features from trait cols (whatever exists after agg)
            trait_cols = [c for c in node_df.columns if c != pk]
            x = self.node_encoder(node_df.select(trait_cols)) if trait_cols else torch.empty((len(ids), 0))
            data[node_name].x = x
            # store original ids as metadata (Python list)
            data[node_name].node_ids = ids  # custom attribute, handy for debugging

            meta["nodes"][node_name] = {
                "primary_key": pk,
                "num_nodes": len(ids),
                "trait_cols": trait_cols,
            }

        # populate edges
        self._build_edges(data, meta)

        return data, meta

    # ----------------- internals -----------------

    def _parse_node_mappings(self, node_mappings: List[Dict[str, Dict[str, Any]]]) -> Dict[str, NodeSpec]:
        specs: Dict[str, NodeSpec] = {}
        for block in node_mappings:
            if len(block) != 1:
                raise ValueError(f"Each node mapping block must have exactly 1 node_name. Got keys: {list(block.keys())}")
            node_name, cfg = next(iter(block.items()))
            data = cfg["data"]
            if not isinstance(data, pl.DataFrame):
                raise TypeError(f"{node_name}.data must be polars DataFrame, got {type(data)}")

            key_cols = list(cfg.get("key", []))
            node_trait = list(cfg.get("node_trait", []))
            edge_trait = list(cfg.get("edge_trait", []))
            if not key_cols:
                raise ValueError(f"{node_name}.key must be non-empty; primary key = key[0].")

            specs[node_name] = NodeSpec(
                name=node_name,
                data=data,
                key_cols=key_cols,
                node_trait_cols=node_trait,
                edge_trait_cols=edge_trait,
            )
        return specs

    def _build_nodes(self) -> None:
        """
        Aggregate each dataset by primary key -> unique nodes.
        """
        for node_name, spec in self.node_specs.items():
            pk = spec.key_cols[0]
            df = spec.data

            # debug checks
            if self.debug:
                missing = [c for c in [pk, *spec.node_trait_cols, *spec.edge_trait_cols] if c not in df.columns]
                if missing:
                    raise KeyError(f"[{node_name}] missing columns in data: {missing}")

            # We only need pk + node_trait for node aggregation
            sel_cols = [pk] + [c for c in spec.node_trait_cols if c in df.columns]
            df_sel = df.select(sel_cols)

            # group and aggregate via map_groups
            # map_groups receives each group df; we return 1-row df with traits
            def _map_fn(group_df: pl.DataFrame) -> pl.DataFrame:
                out = self.node_agg_fn(node_name, pk, spec.node_trait_cols, group_df)
                # ensure single row
                if self.debug and out.height != 1:
                    raise ValueError(f"[{node_name}] node_agg_fn must return 1 row per group; got {out.height}")
                return out

            agg_trait = (
                df_sel
                .group_by(pk, maintain_order=True)
                .map_groups(_map_fn)
            )

            # attach pk column back (map_groups keeps group keys? no, so we reconstruct)
            # Actually, map_groups returns concatenated df without group key unless you include it.
            # We'll add pk inside by taking first pk from group and hstacking.
            def _map_fn_with_pk(group_df: pl.DataFrame) -> pl.DataFrame:
                pk_val = group_df.select(pk).to_series()[0]
                out = self.node_agg_fn(node_name, pk, spec.node_trait_cols, group_df)
                out = out.with_columns(pl.lit(pk_val).alias(pk))
                # reorder: pk first
                cols = [pk] + [c for c in out.columns if c != pk]
                return out.select(cols)

            node_df = (
                df_sel
                .group_by(pk, maintain_order=True)
                .map_groups(_map_fn_with_pk)
            )

            # Ensure uniqueness of pk
            if self.debug:
                n_unique = node_df.select(pk).n_unique()
                if n_unique != node_df.height:
                    raise ValueError(f"[{node_name}] aggregated node table has duplicate primary keys")

            self.node_tables[node_name] = node_df
            self.logger.info(f"Built nodes [{node_name}]: {node_df.height} nodes, traits={node_df.width-1}")

    def _build_edges(self, data: HeteroData, meta: Dict[str, Any]) -> None:
        """
        For each src->dst in edge_mappings, build edge_index and edge_attr.
        """
        for src, src_cfg in self.edge_mappings.items():
            links = src_cfg.get("links", {})
            if src not in self.node_specs:
                if self.debug:
                    raise KeyError(f"edge_mappings references src={src} not in node_mappings")
                self.logger.warning(f"Skip edges from unknown src={src}")
                continue

            for dst, join_cfg in links.items():
                if dst not in self.node_specs:
                    if self.debug:
                        raise KeyError(f"edge_mappings references dst={dst} not in node_mappings")
                    self.logger.warning(f"Skip edges to unknown dst={dst}")
                    continue

                left_on = join_cfg["left_on"]
                right_on = join_cfg["right_on"]

                src_spec = self.node_specs[src]
                dst_spec = self.node_specs[dst]
                src_pk = src_spec.key_cols[0]
                dst_pk = dst_spec.key_cols[0]

                src_df = src_spec.data
                dst_df = dst_spec.data

                # debug schema checks
                if self.debug:
                    for col, side in [(left_on, "src"), (src_pk, "src"), (right_on, "dst"), (dst_pk, "dst")]:
                        frame = src_df if side == "src" else dst_df
                        if col not in frame.columns:
                            raise KeyError(f"[{src}->{dst}] {side} missing column: {col}")

                # Build a mapping table for dst join key -> dst primary
                # If right_on == dst_pk, it's just identity mapping
                dst_map = dst_df.select([right_on, dst_pk]).unique(subset=[right_on])

                # pick edge-trait cols from both sides, prefix to avoid collision
                src_edge_cols = [c for c in src_spec.edge_trait_cols if c in src_df.columns]
                dst_edge_cols = [c for c in dst_spec.edge_trait_cols if c in dst_df.columns]

                src_keep = [src_pk, left_on] + src_edge_cols
                dst_keep = [right_on, dst_pk] + dst_edge_cols

                src_part = src_df.select(src_keep)
                dst_part = dst_df.select(dst_keep)

                # Join src foreign key -> dst join key
                joined = src_part.join(dst_part, left_on=left_on, right_on=right_on, how="inner")

                if joined.height == 0:
                    self.logger.warning(f"No edges for [{src}->{dst}] after join on {left_on}={right_on}")
                    continue

                # Rename trait cols with side prefixes
                rename_map = {}
                for c in src_edge_cols:
                    rename_map[c] = f"src__{c}"
                for c in dst_edge_cols:
                    rename_map[c] = f"dst__{c}"
                joined = joined.rename(rename_map)

                # Normalize to src_id, dst_id
                joined = joined.rename({src_pk: "src_id", dst_pk: "dst_id"})

                # Keep only nodes that exist in aggregated node tables
                # (this matters if some ids got dropped by node aggregation rules)
                src_allowed = set(self.node_id2idx.get(src, {}).keys()) if src in self.node_id2idx else None
                dst_allowed = set(self.node_id2idx.get(dst, {}).keys()) if dst in self.node_id2idx else None

                # node_id2idx is filled after nodes are loaded into data in build()
                # but we're called after that, so it's ok.

                if src_allowed is not None:
                    joined = joined.filter(pl.col("src_id").is_in(list(src_allowed)))
                if dst_allowed is not None:
                    joined = joined.filter(pl.col("dst_id").is_in(list(dst_allowed)))

                if joined.height == 0:
                    self.logger.warning(f"All edges filtered out for [{src}->{dst}] after restricting to built nodes")
                    continue

                # Aggregate edge traits per (src_id, dst_id)
                # edge trait columns are everything except src_id/dst_id/left_on
                ignore_cols = {"src_id", "dst_id", left_on}
                edge_trait_cols = [c for c in joined.columns if c not in ignore_cols]

                def _edge_map_fn(group_df: pl.DataFrame) -> pl.DataFrame:
                    out = self.edge_agg_fn(src, dst, src_pk, dst_pk, edge_trait_cols, group_df)
                    if self.debug and out.height != 1:
                        raise ValueError(f"[{src}->{dst}] edge_agg_fn must return 1 row per edge-pair group")
                    return out

                def _edge_map_fn_with_keys(group_df: pl.DataFrame) -> pl.DataFrame:
                    s = group_df.select("src_id").to_series()[0]
                    d = group_df.select("dst_id").to_series()[0]
                    out = self.edge_agg_fn(src, dst, src_pk, dst_pk, edge_trait_cols, group_df)
                    out = out.with_columns([pl.lit(s).alias("src_id"), pl.lit(d).alias("dst_id")])
                    cols = ["src_id", "dst_id"] + [c for c in out.columns if c not in ("src_id", "dst_id")]
                    return out.select(cols)

                edge_df = (
                    joined
                    .group_by(["src_id", "dst_id"], maintain_order=True)
                    .map_groups(_edge_map_fn_with_keys)
                )

                # Build edge_index with integer indices
                src_map = self.node_id2idx[src]
                dst_map = self.node_id2idx[dst]

                src_idx = torch.tensor([src_map[v] for v in edge_df["src_id"].to_list()], dtype=torch.long)
                dst_idx = torch.tensor([dst_map[v] for v in edge_df["dst_id"].to_list()], dtype=torch.long)

                edge_index = torch.stack([src_idx, dst_idx], dim=0)
                data[(src, "to", dst)].edge_index = edge_index

                # Edge attributes
                attr_cols = [c for c in edge_df.columns if c not in ("src_id", "dst_id")]
                edge_attr = self.edge_encoder(edge_df.select(attr_cols)) if attr_cols else torch.empty((edge_df.height, 0))
                data[(src, "to", dst)].edge_attr = edge_attr

                meta["edges"][f"{src}__to__{dst}"] = {
                    "left_on": left_on,
                    "right_on": right_on,
                    "num_edges": int(edge_df.height),
                    "attr_cols": attr_cols,
                }

                self.logger.info(
                    f"Built edges [{src} -> {dst}]: {edge_df.height} edges, attrs={len(attr_cols)}"
                )
