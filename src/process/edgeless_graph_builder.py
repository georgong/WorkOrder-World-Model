from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import yaml
from torch_geometric.data import HeteroData
import warnings
from .feature_engineering import (
    process_assignment_feature,
    process_districts_feature,
    process_engineer_feature,
    process_task_feature,
)
from .feature_schema import assignment_schema, district_schema, engineer_schema, task_schema


DTYPE_MAP = {
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "string": pl.Utf8,
    "datetime64[ns]": pl.Datetime("ns"),
}

NULL_VALUES = ["", " ", "NA", "N/A", "NULL", "null", "None"]

TYPE_DIR_MAP = {
    "tasks": "W6TASKS",
    "assignments": "W6ASSIGNMENTS",
    "engineers": "W6ENGINEERS",
    "districts": "W6DISTRICTS",
    "departments": "W6DEPARTMENT",
    "task_statuses": "W6TASK_STATUSES",
    "task_types": "W6TASK_TYPES",
    "equipment": "W6EQUIPMENT",
    "regions": "W6REGIONS",
}

# 你现在只给了这四个的 preprocessing
preprocessing_dict = {
    "tasks": {"func": process_task_feature, "schema": task_schema},
    "assignments": {"func": process_assignment_feature, "schema": assignment_schema},
    "districts": {"func": process_districts_feature, "schema": district_schema},
    "engineers": {"func": process_engineer_feature, "schema": engineer_schema},
    # departments / task_types / task_statuses / equipment / regions 目前没给 func，就先用“原始数值列”兜底
}


# -------------------------
# IO: load table
# -------------------------
def _resolve_files(data_dir: Path, base: str) -> List[Path]:
    shards = sorted(data_dir.glob(f"{base}-*.csv"))
    if shards:
        return shards
    single = data_dir / f"{base}.csv"
    if single.exists():
        return [single]
    return []


def load_table(schema: Dict[str, Any], df_type: str, data_dir: str = "data/raw") -> pl.DataFrame:
    assert df_type in TYPE_DIR_MAP, f"df_type must be one of {sorted(TYPE_DIR_MAP.keys())} but got {df_type!r}"

    data_dir_p = Path(data_dir)
    base = TYPE_DIR_MAP[df_type]
    files = _resolve_files(data_dir_p, base)
    if not files:
        raise FileNotFoundError(
            f"No files found for df_type={df_type!r} under {data_dir_p} "
            f"(expected {base}.csv or {base}-*.csv)"
        )

    ds = schema["datasets"][df_type]
    vars_meta = ds["variables"]
    cols = list(vars_meta.keys())

    pl_schema = {col: DTYPE_MAP.get((meta or {}).get("dtype", "string"), pl.Utf8) for col, meta in vars_meta.items()}

    dfs: List[pl.DataFrame] = []
    for f in files:
        df = pl.read_csv(
            f,
            columns=cols,
            schema_overrides=pl_schema,
            null_values=NULL_VALUES,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        # shards 可能缺列
        missing = [c for c in cols if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None).cast(pl_schema[c]).alias(c) for c in missing]).select(cols)
        else:
            df = df.select(cols)
        dfs.append(df)

    return pl.concat(dfs, how="vertical", rechunk=True)


# -------------------------
# Trait split (only used for fallback)
# -------------------------
def get_trait_cols(schema: Dict[str, Any], table_name: str) -> Tuple[List[str], List[str]]:
    """
    Returns: (node_cols, edge_cols) by trait_type.
    We are assuming "no edge_attr", but we still use this to filter node columns for fallback.
    """
    vars_cfg = schema["datasets"][table_name]["variables"]
    node_cols, edge_cols = [], []
    for col, info in vars_cfg.items():
        if (info or {}).get("trait_type") == "node":
            node_cols.append(col)
        elif (info or {}).get("trait_type") == "edge":
            edge_cols.append(col)
    # 去重保持顺序
    node_cols = list(dict.fromkeys(node_cols))
    edge_cols = list(dict.fromkeys(edge_cols))
    return node_cols, edge_cols

def warn_if_non_numeric(
    df: pl.DataFrame,
    name: str = "DataFrame",
    *,
    drop: bool = False,
) -> pl.DataFrame:
    """
    Warn if df contains non-numeric columns.

    Parameters
    ----------
    df : pl.DataFrame
    name : str
        Name shown in warning message.
    drop : bool
        If True, drop non-numeric columns and return a cleaned DataFrame.

    Returns
    -------
    pl.DataFrame
        Original df (if drop=False) or cleaned df (if drop=True).
    """

    numeric_types = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }

    bad_cols = [(c, t) for c, t in df.schema.items() if t not in numeric_types]

    if not bad_cols:
        return df

    preview = ", ".join(f"{c}:{t}" for c, t in bad_cols[:10])
    more = f" (+{len(bad_cols)-10} more)" if len(bad_cols) > 10 else ""

    warnings.warn(
        f"[{name}] Non-numeric columns detected: {preview}{more}",
        category=UserWarning,
        stacklevel=2,
    )

    if drop:
        drop_cols = [c for c, _ in bad_cols]
        return df.drop(drop_cols)

    return df

# -------------------------
# Tensor helpers
# -------------------------
def _to_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            x = x.float()
        return x

    if isinstance(x, pl.DataFrame):
        x = warn_if_non_numeric(x,drop = True)
        arr = x.to_numpy()
        return torch.tensor(arr, dtype=torch.float)

    if isinstance(x, np.ndarray):
        if x.dtype.kind not in {"f", "i", "u"}:
            raise TypeError(f"ndarray dtype not numeric: {x.dtype}")
        return torch.tensor(x, dtype=torch.float)

    raise TypeError(f"Unsupported feature type: {type(x)} (expected torch.Tensor / pl.DataFrame / np.ndarray)")


def _ensure_2d(t: torch.Tensor, name: str) -> torch.Tensor:
    assert t.ndim == 2, f"{name} must be 2D [N,F], got shape={tuple(t.shape)}"
    return t


# -------------------------
# Core: build node index mapping + aggregate features by entity_key
# -------------------------
def _build_node_index(df: pl.DataFrame, entity_key: str) -> Tuple[pl.DataFrame, List[Any]]:
    assert entity_key in df.columns, f"entity_key={entity_key!r} not in df.columns={df.columns}"
    keys = (
        df.select(entity_key)
        .drop_nulls()
        .unique()
        .sort(entity_key)
        .with_row_index("__idx")  # 0..N-1
    )
    node_ids = keys[entity_key].to_list()
    return keys, node_ids


def _aggregate_features_by_key(
    df: pl.DataFrame,
    *,
    entity_key: str,
    feat_tensor: torch.Tensor,
    keys_sorted: pl.DataFrame,
    agg: str = "mean",
) -> torch.Tensor:
    """
    df: original row-level table
    feat_tensor: row-level features [N_rows, F]
    keys_sorted: DataFrame with columns [__idx, entity_key] sorted by entity_key
    Returns: node-level features [N_nodes, F]
    """

    feat_tensor = _ensure_2d(feat_tensor, "feat_tensor")
    assert df.height == feat_tensor.shape[0], (
        f"Row mismatch: df.height={df.height} but feat_tensor.shape[0]={feat_tensor.shape[0]}"
    )

    # Build row -> node_idx mapping
    # (entity_key -> __idx) then map each row
    mapper = keys_sorted.select([entity_key, "__idx"])

    row_map = (
        df.select([entity_key])
        .with_row_index("__rid")
        .join(mapper, on=entity_key, how="inner")
        .select(["__rid", "__idx"])
        .sort("__rid")
    )
    assert row_map.height > 0, "No rows mapped to nodes. Check entity_key / nulls."

    rid = torch.tensor(row_map["__rid"].to_list(), dtype=torch.long)
    idx = torch.tensor(row_map["__idx"].to_list(), dtype=torch.long)

    # Filter feature rows to those that mapped
    X = feat_tensor[rid]  # [M, F]
    num_nodes = keys_sorted.height
    F = X.shape[1]

    if agg == "mean":
        out = torch.zeros((num_nodes, F), dtype=X.dtype)
        cnt = torch.zeros((num_nodes, 1), dtype=X.dtype)
        out.index_add_(0, idx, X)
        ones = torch.ones((idx.shape[0], 1), dtype=X.dtype)
        cnt.index_add_(0, idx, ones)
        cnt = torch.clamp(cnt, min=1.0)
        out = out / cnt
        return out

    if agg == "sum":
        out = torch.zeros((num_nodes, F), dtype=X.dtype)
        out.index_add_(0, idx, X)
        return out

    raise ValueError(f"Unsupported agg={agg!r} (use 'mean' or 'sum')")


# -------------------------
# Core: build edges (NO edge_attr)
# -------------------------
def build_edge_index_only(
    *,
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    left_entity_key: str,
    right_entity_key: str,
    left_join_col: str,
    right_join_col: str,
    left_keys_sorted: pl.DataFrame,   # [__idx, left_entity_key]
    right_keys_sorted: pl.DataFrame,  # [__idx, right_entity_key]
    keep_order_from: str = "left",
) -> torch.Tensor:
    """
    Edge construction rule:
      join(left[left_join_col]) == join(right[right_join_col])
    Then src = left[left_entity_key], dst = right[right_entity_key], map to node idx.

    Returns edge_index [2, E] long
    """

    for c in [left_entity_key, left_join_col]:
        assert c in left_df.columns, f"left_df missing col {c!r}. cols={left_df.columns}"
    for c in [right_entity_key, right_join_col]:
        assert c in right_df.columns, f"right_df missing col {c!r}. cols={right_df.columns}"

    # Build minimal join frames with non-colliding names
    if keep_order_from == "left":
        L = (
            left_df.select([
                pl.col(left_entity_key).alias("__src_key"),
                pl.col(left_join_col).alias("__join"),
            ])
            .with_row_index("__rid")
            .drop_nulls(["__src_key", "__join"])
        )
        R = (
            right_df.select([
                pl.col(right_entity_key).alias("__dst_key"),
                pl.col(right_join_col).alias("__join"),
            ])
            .drop_nulls(["__dst_key", "__join"])
        )
        J = L.join(R, on="__join", how="inner").sort("__rid")
    elif keep_order_from == "right":
        R = (
            right_df.select([
                pl.col(right_entity_key).alias("__dst_key"),
                pl.col(right_join_col).alias("__join"),
            ])
            .with_row_index("__rid")
            .drop_nulls(["__dst_key", "__join"])
        )
        L = (
            left_df.select([
                pl.col(left_entity_key).alias("__src_key"),
                pl.col(left_join_col).alias("__join"),
            ])
            .drop_nulls(["__src_key", "__join"])
        )
        J = R.join(L, on="__join", how="inner").sort("__rid")
    else:
        raise ValueError("keep_order_from must be 'left' or 'right'")

    assert J.height > 0, (
        f"Join produced 0 edges: left({left_join_col}) vs right({right_join_col}). "
        "Check value domains."
    )

    # Map src/dst keys -> node idx
    src_map = left_keys_sorted.select([
        pl.col(left_entity_key).alias("__src_key"),
        pl.col("__idx").alias("__src_idx"),
    ])
    dst_map = right_keys_sorted.select([
        pl.col(right_entity_key).alias("__dst_key"),
        pl.col("__idx").alias("__dst_idx"),
    ])

    E = (
        J.join(src_map, on="__src_key", how="inner")
         .join(dst_map, on="__dst_key", how="inner")
         .select(["__src_idx", "__dst_idx"])
    )

    assert E.height > 0, "After mapping keys->idx, edges are empty. Likely key mismatch / nulls."

    src = torch.tensor(E["__src_idx"].to_list(), dtype=torch.long)
    dst = torch.tensor(E["__dst_idx"].to_list(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


# -------------------------
# GraphBuilder
# -------------------------
class GraphBuilder:
    """
    Build a torch_geometric.data.HeteroData graph from:
      - schema["datasets"][node_type]["variables"][col] includes dtype/key/mask/trait_type
      - schema["mappings"][node_type] includes entity_key, name, links{dst_type:{left_on,right_on}}
    Assumption NOW: no edge_attr, everything is node_attr.
    """

    def __init__(self, *, yaml: Dict[str, Any], data_dir: str = "data/raw") -> None:
        self.yaml = yaml
        self.data_dir = data_dir
        self.logger = self.setup_logger()

        assert "datasets" in self.yaml, "YAML missing top-level 'datasets'"
        assert "mappings" in self.yaml, "YAML missing top-level 'mappings'"

        self.tables: Dict[str, pl.DataFrame] = {}
        self.data = HeteroData()

    def setup_logger(self) -> Logger:
        return Logger(__name__)

    def _load_all_tables(self) -> None:
        for t in self.yaml["mappings"].keys():
            assert t in self.yaml["datasets"], f"mappings contains {t!r} but datasets missing it"
            self.tables[t] = load_table(self.yaml, t, data_dir=self.data_dir)

    def _build_nodes_one(self, node_type: str) -> None:
        cfg = self.yaml["mappings"][node_type]
        entity_key = cfg["entity_key"]
        df = self.tables[node_type]

        # 1) build stable node index
        keys_sorted, node_ids = _build_node_index(df, entity_key)

        # 2) build row-level feature matrix
        feat_obj: Any = None

        if node_type in preprocessing_dict:
            fn = preprocessing_dict[node_type]["func"]
            sch = preprocessing_dict[node_type]["schema"]
            feat_obj = fn(df, sch)  # user-defined
        else:
            # fallback: use numeric "node" trait columns from yaml
            node_cols, _ = get_trait_cols(self.yaml, node_type)
            numeric_cols = [c for c in node_cols if c in df.columns and df.schema[c] in (pl.Int64, pl.Float64)]
            if not numeric_cols:
                # last fallback: dummy 1-dim feature
                feat_obj = torch.ones((df.height, 1), dtype=torch.float)
            else:
                feat_obj = df.select(numeric_cols).fill_null(0)

        X_row = _to_float_tensor(feat_obj)
        X_row = _ensure_2d(X_row, f"{node_type} row-features")

        # 3) aggregate row-level -> node-level by entity_key
        X_node = _aggregate_features_by_key(
            df,
            entity_key=entity_key,
            feat_tensor=X_row,
            keys_sorted=keys_sorted,
            agg="mean",
        )
        X_node = _ensure_2d(X_node, f"{node_type} node-features")

        # 4) store into HeteroData
        self.data[node_type].x = X_node
        self.data[node_type].node_ids = node_ids  # keep raw ids for reverse mapping/debug
        self.data[node_type].num_nodes = len(node_ids)

        # 5) store masks (which columns are hidden at prediction time) as metadata lists
        # (PyG doesn't care, but you do.)
        vars_cfg = self.yaml["datasets"][node_type]["variables"]
        mask_cols = [c for c, info in vars_cfg.items() if bool((info or {}).get("mask", False))]
        self.data[node_type].mask_cols = mask_cols

        # cache mapping table for edge building
        self.data[node_type].__key_index = keys_sorted  # polars df [__idx, entity_key]
        self.data[node_type].__entity_key = entity_key

    def _build_all_nodes(self) -> None:
        for node_type in self.yaml["mappings"].keys():
            assert node_type in self.tables, f"Missing loaded table for {node_type!r}"
            self._build_nodes_one(node_type)

    def _build_edges(self) -> None:
        for src_type, src_cfg in self.yaml["mappings"].items():
            links = src_cfg.get("links") or {}
            if not links:
                continue

            src_entity_key = src_cfg["entity_key"]
            src_df = self.tables[src_type]
            src_keys_sorted = self.data[src_type].__key_index

            for dst_type, link in links.items():
                assert dst_type in self.yaml["mappings"], f"{src_type}.links references unknown dst_type={dst_type!r}"

                dst_cfg = self.yaml["mappings"][dst_type]
                dst_entity_key = dst_cfg["entity_key"]
                dst_df = self.tables[dst_type]
                dst_keys_sorted = self.data[dst_type].__key_index

                left_on = link["left_on"]
                right_on = link["right_on"]
                edge_type = link.get("edge_type", "relates_to")

                edge_index = build_edge_index_only(
                    left_df=src_df,
                    right_df=dst_df,
                    left_entity_key=src_entity_key,
                    right_entity_key=dst_entity_key,
                    left_join_col=left_on,
                    right_join_col=right_on,
                    left_keys_sorted=src_keys_sorted,
                    right_keys_sorted=dst_keys_sorted,
                    keep_order_from="left",
                )

                rel = (src_type, edge_type, dst_type)
                self.data[rel].edge_index = edge_index

                # reverse edge (optional but usually helpful)
                rev_rel = (dst_type, f"rev_{edge_type}", src_type)
                self.data[rev_rel].edge_index = edge_index.flip(0)

    def build(self) -> HeteroData:
        self._load_all_tables()
        self._build_all_nodes()
        self._build_edges()

        # sanity checks
        for ntype in self.data.node_types:
            assert hasattr(self.data[ntype], "x"), f"Node type {ntype!r} missing .x"
            assert self.data[ntype].x.shape[0] == self.data[ntype].num_nodes, (
                f"{ntype}: x rows != num_nodes "
                f"({self.data[ntype].x.shape[0]} vs {self.data[ntype].num_nodes})"
            )

        for etype in self.data.edge_types:
            ei = self.data[etype].edge_index
            assert ei.ndim == 2 and ei.shape[0] == 2, f"Bad edge_index shape for {etype}: {tuple(ei.shape)}"
            src_type, _, dst_type = etype
            assert ei.max().item() < max(self.data[src_type].num_nodes, self.data[dst_type].num_nodes) + 10_000, (
                f"Edge index seems out of range for {etype}. "
                "This often means your entity_key mapping differs from table keys."
            )

        # clean temporary fields if you don't want them
        # (but keeping them is super useful for debugging)
        return self.data


# -------------------------
# Demo entry
# -------------------------
def _demo_test(schema: Dict[str, Any]) -> None:
    gb = GraphBuilder(yaml=schema, data_dir="data/raw")
    g = gb.build()
    print(g)

    # quick peek
    for ntype in g.node_types:
        print(ntype, g[ntype].x.shape, len(g[ntype].node_ids))

    for etype in g.edge_types[:10]:
        print(etype, g[etype].edge_index.shape)


if __name__ == "__main__":
    with open("configs/graph.yaml", "r") as f:
        schema = yaml.safe_load(f)

    _demo_test(schema)
