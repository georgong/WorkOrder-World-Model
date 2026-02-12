from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple

import os
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
    #process_equipments_feature,
)
from .feature_schema import assignment_schema, district_schema, engineer_schema, task_schema #equipment_schema
from ..process.utils.pipeline_logger import PipelineLogger


DTYPE_MAP = {
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "string": pl.Utf8,
    "datetime64[ns]": pl.Datetime("ns"),
}

PERFIX_MAP = {
    "tasks": "task_",
    "assignments": "assign_",
    "districts":"district_",
    "departments":"departments_",
    "task_statuses":"",
    "task_types":"",
    "regions":"",
    "engineers":"eng_"
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

preprocessing_dict = {
    "tasks": {"func": process_task_feature, "schema": task_schema},
    "assignments": {"func": process_assignment_feature, "schema": assignment_schema},
    "districts": {"func": process_districts_feature, "schema": district_schema},
    "engineers": {"func": process_engineer_feature, "schema": engineer_schema},
    # departments / task_types / task_statuses / equipment / regions 目前没给 func，就先用“原始数值列”兜底
}

# This is used in _build_edges_by_shared_edge_trait() to avoid memory explosion for very large groups
# MAX_NODES_PER_GROUP = 2000
# MAX_EDGES_PER_GROUP = 10000


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

def assert_no_nulls(df: pl.DataFrame) -> None:
    null_counts = df.select(pl.all().null_count())
    bad_cols = [c for c in null_counts.columns if null_counts[c][0] > 0]

    if bad_cols:
        raise ValueError(f"Null values detected in columns: {bad_cols}")

def filter_null_value(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.when(pl.col(c).is_null())
          .then(0)
          .otherwise(pl.col(c))
          .alias(c)
        for c, dtype in df.schema.items()
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64)
    ]) 






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
def _build_node_index(df: pl.DataFrame, entity_key: str, name:str) -> Tuple[pl.DataFrame, List[Any]]:
    assert entity_key in df.columns, f"entity_key={entity_key!r} not in df.columns={df.columns}"
    if entity_key != name:
        keys = (
            df.select([entity_key,name])
            .drop_nulls()
            .unique()
            .sort(entity_key)
            .with_row_index("__idx")  # 0..N-1
        )
    else:
        keys = (
            df.select([entity_key])
            .drop_nulls()
            .unique()
            .sort(entity_key)
            .with_row_index("__idx")  # 0..N-1
        ) 
    node_ids = keys[name].to_list()
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
def project_metapath_edges(
    data: HeteroData,
    *,
    rel1: tuple[str, str, str],  # (src, r1, mid1)
    rel2: tuple[str, str, str],  # (mid1, r2, mid2)
    rel3: tuple[str, str, str],  # (mid2, r3, dst)
    min_count: int = 1,          # 出现次数阈值，可选
) -> torch.Tensor:
    """
    Project rel1 ∘ rel2 ∘ rel3 into a direct edge_index from src -> dst
    using sparse matmul. Values become "number of distinct paths" (roughly).
    """
    try:
        from torch_sparse import SparseTensor
    except Exception as e:
        raise ImportError(
            "torch_sparse not available. Install PyG deps (torch-sparse) or switch to a join-based projection."
        ) from e

    src, _, mid1 = rel1
    _mid1, _, mid2 = rel2
    _mid2, _, dst = rel3

    assert mid1 == _mid1 and mid2 == _mid2, f"Bad metapath: {rel1} -> {rel2} -> {rel3}"

    ei1 = data[rel1].edge_index
    ei2 = data[rel2].edge_index
    ei3 = data[rel3].edge_index

    n_src = data[src].num_nodes
    n_mid1 = data[mid1].num_nodes
    n_mid2 = data[mid2].num_nodes
    n_dst = data[dst].num_nodes

    # Build SparseTensor adjacencies with value=1
    A1 = SparseTensor(row=ei1[0], col=ei1[1], sparse_sizes=(n_src, n_mid1))
    A2 = SparseTensor(row=ei2[0], col=ei2[1], sparse_sizes=(n_mid1, n_mid2))
    A3 = SparseTensor(row=ei3[0], col=ei3[1], sparse_sizes=(n_mid2, n_dst))

    P = (A1 @ A2) @ A3  # sparse matmul
    row, col, val = P.coo()  # val = 路径数（加和）

    if min_count > 1:
        keep = val >= min_count
        row, col = row[keep], col[keep]

    return torch.stack([row, col], dim=0)

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
        self.logger = PipelineLogger()

        assert "datasets" in self.yaml, "YAML missing top-level 'datasets'"
        assert "mappings" in self.yaml, "YAML missing top-level 'mappings'"

        self.tables: Dict[str, pl.DataFrame] = {}
        self.data = HeteroData()

        self.MAX_NODES_PER_GROUP = self.yaml.get("MAX_NODES_PER_GROUP", 2000)
        self.MAX_EDGES_PER_GROUP = self.yaml.get("MAX_EDGES_PER_GROUP", 10000)

    def _load_table(self, schema: Dict[str, Any], df_type: str, data_dir: str = "data/raw") -> pl.DataFrame:
        assert df_type in TYPE_DIR_MAP, f"df_type must be one of {sorted(TYPE_DIR_MAP.keys())} but got {df_type!r}"

        data_dir_p = Path(data_dir)
        base = TYPE_DIR_MAP[df_type]
        files = _resolve_files(data_dir_p, base)
        if not files:
            raise FileNotFoundError(f"No files found for {df_type=} under {data_dir_p} (expected {base}.csv or {base}-*.csv)")

        # schema path: schema["datasets"][df_type]["variables"]  (your earlier layout)
        vars_meta = schema["datasets"][df_type]["variables"]
        cols = list(vars_meta.keys())

        pl_schema = {
            col: DTYPE_MAP.get(meta.get("dtype", "string"), pl.Utf8)
            for col, meta in vars_meta.items()
        }

        dfs: List[pl.DataFrame] = []
        for f in files:
            df = pl.read_csv(
                f,
                columns=cols,                     # only keep known columns
                schema_overrides=pl_schema,        # enforce dtypes
                null_values=NULL_VALUES,
                ignore_errors=True,
                truncate_ragged_lines=True,
            )

            # ensure all requested columns exist (some shards may miss a column)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                df = df.with_columns([pl.lit(None).cast(pl_schema[c]).alias(c) for c in missing]).select(cols)
            else:
                df = df.select(cols)

            dfs.append(df)

        df_concat = pl.concat(dfs, how="vertical", rechunk=True)
        df_concat = self._preprocess_df(df_concat, vars_meta)
        if df_type == "assignments":
            df_concat = df_concat.with_columns(
                (pl.col("FINISHTIME") - pl.col("STARTTIME")).alias("COMPLETIONTIME")
            )

            df_concat = df_concat.drop("FINISHTIME")
            before = df_concat.height
            cap = df_concat.select(pl.col("COMPLETIONTIME").quantile(0.99)).item()

            df_concat = df_concat.with_columns(
                pl.when(pl.col("COMPLETIONTIME") > cap)
                .then(cap)
                .otherwise(pl.col("COMPLETIONTIME"))
                .alias("COMPLETIONTIME")
            )

            after = df_concat.height
            dropped = before - after
            print(f"[assignments] COMPLETIONTIME filter: dropped {dropped} / {before} rows ({dropped / max(before,1):.2%}); cap(q0.99)={cap}")
        return df_concat

    def _build_engineer_task_type_edges_from_graph(self) -> None:
        # engineers <-> task_types

        # 你得把这三个 rel 改成你自己图里真实存在的 edge_types
        # 最稳的办法：print(self.data.edge_types) 看一眼再填
        rel_ea = ("engineers", "relates_to", "assignments")
        rel_at = ("assignments", "relates_to", "tasks")
        rel_tt = ("tasks", "relates_to", "task_types")

        # 如果你实际是反向边，比如 ("assignments","rev_relates_to","engineers")
        # 就换成你想要的方向，或者直接用 flip(0)

        for rel in [rel_ea, rel_at, rel_tt]:
            assert rel in self.data.edge_types, f"Missing edge type {rel}. Existing: {self.data.edge_types}"

        edge_index = project_metapath_edges(
            self.data,
            rel1=rel_ea,
            rel2=rel_at,
            rel3=rel_tt,
            min_count=1,   # 你也可以设成 2/3，过滤掉“只出现一次”的弱关联
        )

        rel = ("engineers", "works_on_type", "task_types")
        self.data[rel].edge_index = edge_index
        self.data[("task_types", "rev_works_on_type", "engineers")].edge_index = edge_index.flip(0)

        print(f"Total edges built in _build_engineer_task_type_edges_from_graph(): {edge_index.shape[1]}")

        self.logger.log("graph_build_up", f"{rel}: edge_index={tuple(edge_index.shape)}")
        print(rel, edge_index.shape)

    def _preprocess_df(self, df_concat, vars_meta):
        '''
        Preprocess the dataframe by removing outliers based on the metadata provided.
        Support categorical, numeric, and datetime outlier types.
        
        Args:
            df_concat (pl.DataFrame): The input dataframe to preprocess.
            vars_meta (Dict[str, Dict[str, Any]]): Metadata containing outlier information for
                each column.
        Returns:
            pl.DataFrame: The preprocessed dataframe with outliers removed.
        '''

        for col, meta in vars_meta.items():
            outlier = meta.get("outlier", None)
            outlier_type = meta.get("outlier_type", None)
            if outlier is None or outlier_type is None:
                continue

            before_count = df_concat.height

            if outlier_type == "categorical":
                # Only keep rows where value is in outlier list
                include_values = [x for x in outlier if x not in [None, "None"]]
                df_concat = df_concat.filter(pl.col(col).is_in(include_values))
                after_count = df_concat.height
                print(f"Column '{col}' (categorical): kept {after_count} / {before_count} rows (filtered {before_count - after_count})")

            elif outlier_type == "numeric":
                lower, upper = outlier[0], outlier[1]
                try:
                    lower_val = float(lower) if lower not in [None, "None"] else None
                    upper_val = float(upper) if upper not in [None, "None"] else None
                except (ValueError, TypeError):
                    lower_val, upper_val = None, None

                if lower_val is not None:
                    df_concat = df_concat.filter(pl.col(col).is_null() | (pl.col(col) >= lower_val))
                if upper_val is not None:
                    df_concat = df_concat.filter(pl.col(col).is_null() | (pl.col(col) <= upper_val))

                after_count = df_concat.height
                self.logger.log("filter_row",f"Column '{col}' (numeric): kept {after_count} / {before_count} rows (filtered {before_count - after_count})")
                print(f"Column '{col}' (numeric): kept {after_count} / {before_count} rows (filtered {before_count - after_count})")
            
            elif outlier_type == "datetime":
                lower, upper = outlier[0], outlier[1]
                lower_val = pl.Series([lower]).str.to_datetime().to_list()[0] if lower not in [None, "None"] else None
                upper_val = pl.Series([upper]).str.to_datetime().to_list()[0] if upper not in [None, "None"] else None

                if lower_val is not None:
                    df_concat = df_concat.filter(pl.col(col).is_null() | (pl.col(col) >= lower_val))
                if upper_val is not None:
                    df_concat = df_concat.filter(pl.col(col).is_null() | (pl.col(col) <= upper_val))

                after_count = df_concat.height
                self.logger.log("filter_row",f"Column '{col}' (datetime): kept {after_count} / {before_count} rows (filtered {before_count - after_count})")
                print(f"Column '{col}' (datetime): kept {after_count} / {before_count} rows (filtered {before_count - after_count})")

            else:
                self.logger.log("filter_row",f"Unknown outlier_type {outlier_type!r} for column {col!r}. Please fix the yaml config file.")
                raise ValueError(f"Unknown outlier_type {outlier_type!r} for column {col!r}. Please fix the yaml config file.")

        return df_concat

    def setup_logger(self) -> Logger:
        return Logger(__name__)

    def _load_all_tables(self) -> None:
        for t in self.yaml["mappings"].keys():
            assert t in self.yaml["datasets"], f"mappings contains {t!r} but datasets missing it"
            self.tables[t] = self._load_table(self.yaml, t, data_dir=self.data_dir)

    def _build_nodes_one(self, node_type: str) -> None:
        cfg = self.yaml["mappings"][node_type]
        entity_key = cfg["entity_key"]
        name = cfg["name"]
        df = self.tables[node_type]

        # 1) build stable node index
        keys_sorted, node_ids = _build_node_index(df, entity_key, name)

        # 2) build row-level feature matrix
        feat_obj: Any = None

        if node_type in preprocessing_dict:
            fn = preprocessing_dict[node_type]["func"]
            sch = preprocessing_dict[node_type]["schema"]
            feat_obj = fn(df, sch)  # user-defined
            vars_cfg = self.yaml["datasets"][node_type]["variables"]
            key_cols = [PERFIX_MAP[node_type] + c for c, info in vars_cfg.items() if bool((info or {}).get("key", False))]
            feat_obj = feat_obj.drop(key_cols)
            feat_obj.select(pl.all().null_count())
            feat_obj = filter_null_value(feat_obj)
            assert_no_nulls(feat_obj)
        else:
            # fallback: use numeric "node" trait columns from yaml
            node_cols, _ = get_trait_cols(self.yaml, node_type)

            vars_cfg = self.yaml["datasets"][node_type]["variables"]
            key_cols = [c for c, info in vars_cfg.items() if bool((info or {}).get("key", False))]

            # 只在原 df 中的 numeric
            numeric_cols = [
                c for c in node_cols
                if c in df.columns and df.schema[c] in (pl.Int64, pl.Float64)
            ]

            # 去掉 key 列
            numeric_feat_cols = [c for c in numeric_cols if c not in key_cols]

            if not numeric_feat_cols:
                # 全是 key，或者根本没有数值特征
                feat_obj = pl.DataFrame({"__dummy__": pl.Series([1.0] * df.height)})
            else:
                feat_obj = df.select(numeric_feat_cols).fill_null(0)
                feat_obj = filter_null_value(feat_obj)
                assert_no_nulls(feat_obj)
        


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

        self.data[node_type].node_ids = node_ids  # keep raw ids for reverse mapping/debug
        self.data[node_type].num_nodes = len(node_ids)
        self.data[node_type].attr_name = feat_obj.columns
        attr_name = self.data[node_type].attr_name
        vars_cfg = self.yaml["datasets"][node_type]["variables"]
        mask_cols = [c for c, info in vars_cfg.items() if bool((info or {}).get("mask", False))]
        self.data[node_type].mask_cols = mask_cols

        ### if the assign_COMPLETIONTIME: target in the attr_name, select it and make it as y
        X = X_node
        ### Nan Value Check
        bad = torch.isnan(X) | torch.isinf(X)
        if bad.any():
            raise Exception(f"{node_type}: tensor has NaN/Inf")
        
        target_col = "assign_COMPLETIONTIME"

        if target_col in attr_name:
            k = attr_name.index(target_col)

            # 1. 拿出来当 y
            y = X[:, k].clone()

            # 2. 从 x 里删除这一列
            X_new = torch.cat([X[:, :k], X[:, k+1:]], dim=1)

            # 3. 更新 attr_name
            new_attr_name = attr_name[:k] + attr_name[k+1:]

            # 4. 写回
            self.data[node_type].x = X_new
            self.data[node_type].y = y
            self.data[node_type].attr_name = new_attr_name

        else:
            # 没有 target，就正常塞 x
            self.data[node_type].x = X

        # 5) store masks (which columns are hidden at prediction time) as metadata lists
        # (PyG doesn't care, but you do.)

        if node_type == "assignments":
            mask_cols.append("COMPLETIONTIME")

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
                # rev_rel = (dst_type, f"rev_{edge_type}", src_type)
                # self.data[rev_rel].edge_index = edge_index.flip(0)

    def _build_edges_by_shared_edge_trait(self):
        """
        For each node type, create edges between nodes that share the same value of an 'edge' trait variable.
        Supports flexible edge construction for datetime (weekday, month, day, week), categorical, and numeric.
        Uses the 'edge_construct' key in the variable's metadata to determine grouping logic.
        """
        print('Start building edges by shared edge trait...')

        # Define group extractors for different group types
        def group_weekday(col): return col.dt.weekday()
        def group_month(col): return col.dt.month()
        def group_day(col): return col.dt.day()
        def group_week(col): return col.dt.week()
        def group_identity(col): return col
        def group_top_n_categorical(col, top_n=10):
            top_vals = col.value_counts().head(top_n).select(pl.col(col.name)).to_series().to_list()
            return pl.when(col.is_in(top_vals)).then(col).otherwise(None)

        group_extractors = {
            "weekday": group_weekday,
            "month": group_month,
            "day": group_day,
            "week": group_week,
            "identity": group_identity,
            "categorical": group_identity,
            "top_n_categorical": lambda col: group_top_n_categorical(col=col, top_n=meta.get("top_n", 10)),
            # Add more as needed
        }
        
        for node_type, node_cfg in self.yaml["mappings"].items():
            df = self.tables[node_type]
            vars_cfg = self.yaml["datasets"][node_type]["variables"]
            entity_key = node_cfg["entity_key"]
            keys_sorted = self.data[node_type].__key_index

            for col, meta in vars_cfg.items():
                if meta.get("mask", True) or meta.get("trait_type") != "edge" or meta.get("edge_construct", None) is None:
                    continue

                edge_construct = meta.get("edge_construct", None)

                edge_group = meta.get("edge_group", "identity")  # default to identity
                extractor = group_extractors.get(edge_group, group_identity)
                group_vals = extractor(df[col]).alias("__group_val")
                group_label = edge_group + "_" + col

                # Add group value column
                df_with_group = df.with_columns(group_vals)
                # Drop null group values
                df_with_group = df_with_group.drop_nulls(["__group_val"])
                groups = df_with_group.group_by("__group_val").agg(pl.col(entity_key))

                id_to_idx = dict(zip(keys_sorted[entity_key].to_list(), keys_sorted["__idx"].to_list()))
                
                print('Building edges for node_type:', node_type, 'using column:', col, 'with edge_construct:', edge_construct)
                
                # Collect edges
                if edge_construct == "context_node":
                    CustomEdgeConstructor.build_shared_edges_context_node(
                        groups, entity_key, id_to_idx, node_type, group_label, data_store=self.data
                    )
                    ...
                elif edge_construct == "neighbor":
                    CustomEdgeConstructor.build_shared_edges_random_k_neighbors(
                        groups, entity_key, id_to_idx, node_type, group_label, k=meta.get("neighbor_k", 3), max_nodes_per_group=self.MAX_NODES_PER_GROUP, data_store=self.data
                    )
                else:
                    CustomEdgeConstructor.build_shared_edges_pairwise(
                        groups, entity_key, id_to_idx, node_type, group_label, max_edges_per_group=self.MAX_EDGES_PER_GROUP, data_store=self.data
                    )
                    ...

    def build(self) -> HeteroData:
        self._load_all_tables()
        self._build_all_nodes()
        self._build_edges()
        self._build_engineer_task_type_edges_from_graph()
        self._build_edges_by_shared_edge_trait()


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

        for ntype in self.data.node_types:
            node_storage = self.data[ntype]
            num_nodes = node_storage.num_nodes
            if hasattr(node_storage, "node_ids"):
                node_ids_len = len(node_storage.node_ids)
                self.logger.log(
                    "graph_build_up",
                    f"{ntype}: feature_shape={tuple(node_storage.x.shape)}, num_nodes={node_ids_len}"
                )
                print(ntype, node_storage.x.shape, node_ids_len)
            else:
                self.logger.log(
                    "graph_build_up",
                    f"{ntype}: feature_shape={tuple(node_storage.x.shape)}, num_nodes={num_nodes}"
                )
                print(ntype, node_storage.x.shape, num_nodes)
        
        for etype in self.data.edge_types[:10]:
            self.logger.log(
                "graph_build_up",
                f"{etype}: feature_shape={self.data[etype].edge_index.shape}"
            )
            print(etype, self.data[etype].edge_index.shape)

        total_edges = sum(self.data[etype].edge_index.shape[1] for etype in self.data.edge_types)
        print(f"Total number of edges: {total_edges}")

        self.logger.dump("pipeline_log.txt")
        return self.data
    
class CustomEdgeConstructor:
    @staticmethod
    def build_shared_edges_random_k_neighbors(
        groups,
        entity_key,
        id_to_idx,
        node_type,
        group_label,
        k,
        data_store,
        max_nodes_per_group,
        seed=42, # set seed for reproducibility
    ):
        src_indices = []
        dst_indices = []

        for group in groups.iter_rows(named=True):
            node_ids = group[entity_key]
            if not isinstance(node_ids, list):
                node_ids = [node_ids]

            idxs = np.array(
                [id_to_idx[nid] for nid in node_ids if nid in id_to_idx],
                dtype=np.int64,
            )

            n = len(idxs)
            if n < 2:
                continue

            if n > max_nodes_per_group:
                # Set rng for reproducibility
                rng = np.random.default_rng(seed)
                idxs = rng.choice(idxs, max_nodes_per_group, replace=False)
                n = len(idxs)

            rng = np.random.default_rng(seed ^ hash(group["__group_val"]) & 0xFFFFFFFF)

            for i in range(n):
                if n - 1 <= k:
                    neighbors = np.concatenate([idxs[:i], idxs[i+1:]])
                else:
                    choices = rng.choice(n - 1, k, replace=False)
                    neighbors = idxs[choices + (choices >= i)]

                src_indices.extend([idxs[i]] * len(neighbors))
                dst_indices.extend(neighbors.tolist())

        if not src_indices:
            return

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_type = f"on_{group_label}"
        rel = (node_type, edge_type, node_type)
        rev_rel = (node_type, f"rev_{edge_type}", node_type)

        data_store[rel].edge_index = edge_index
        data_store[rev_rel].edge_index = edge_index.flip(0)

        print(
            f"[random-neighbors] {node_type}: "
            f"k={k}, groups={len(groups)}, edges={edge_index.size(1)}"
        )

    @staticmethod
    def build_shared_edges_pairwise(
        groups,
        entity_key,
        id_to_idx,
        node_type,
        group_label,
        max_edges_per_group,
        data_store,
    ):
        import random
        src_indices = []
        dst_indices = []

        for group in groups.iter_rows(named=True):
            node_ids = group[entity_key]
            if not isinstance(node_ids, list):
                node_ids = [node_ids]

            idxs = [id_to_idx[nid] for nid in node_ids if nid in id_to_idx]
            n = len(idxs)
            if n < 2:
                continue

            max_possible = n * (n - 1) // 2
            num_edges = min(max_edges_per_group, max_possible)

            seen = set()
            while len(seen) < num_edges:
                i, j = random.sample(range(n), 2)
                a, b = idxs[min(i, j)], idxs[max(i, j)]
                seen.add((a, b))

            if seen:
                src, dst = zip(*seen)
                src_indices.extend(src)
                dst_indices.extend(dst)

        if not src_indices:
            return

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_type = f"on_{group_label}"
        rel = (node_type, edge_type, node_type)
        rev_rel = (node_type, f"rev_{edge_type}", node_type)

        data_store[rel].edge_index = edge_index
        data_store[rev_rel].edge_index = edge_index.flip(0)

        print(
            f"[pairwise] {node_type}: "
            f"edges={edge_index.size(1)}"
        )


    @staticmethod
    def build_shared_edges_context_node(
        groups,
        entity_key,
        id_to_idx,
        node_type,
        group_label,
        data_store,
    ):
        """
        Build context-node edges for a categorical variable.

        Example:
            task --on_weekday--> weekday_context
            weekday_context --rev_on_weekday--> task

        Assumptions:
        - groups has columns: ["__group_val", entity_key]
        - id_to_idx maps entity_key -> node index
        """

        # 1) Build context-node mapping (group_val -> context_idx)
        group_vals = [g["__group_val"] for g in groups.iter_rows(named=True)]
        unique_group_vals = sorted(set(group_vals))

        if len(unique_group_vals) == 0:
            return

        context_id_map = {val: i for i, val in enumerate(unique_group_vals)}
        num_context_nodes = len(unique_group_vals)

        # Use a SAFE, UNIQUE node type name
        context_node_type = f"{node_type}_{group_label}_context"

        # 2) Build (node, context) pairs
        rows = []
        for g in groups.iter_rows(named=True):
            gv = g["__group_val"]
            node_ids = g[entity_key]
            if not isinstance(node_ids, list):
                node_ids = [node_ids]
            for nid in node_ids:
                rows.append((nid, gv))

        if not rows:
            return

        df = pl.DataFrame(
            {
                "nid": [r[0] for r in rows],
                "group_val": [r[1] for r in rows],
            }
        )

        # 3) Vectorized mapping via joins (FAST, no Python lambdas)
        node_map_df = pl.DataFrame(
            {
                "nid": list(id_to_idx.keys()),
                "node_idx": list(id_to_idx.values()),
            }
        )

        context_map_df = pl.DataFrame(
            {
                "group_val": list(context_id_map.keys()),
                "context_idx": list(context_id_map.values()),
            }
        )

        df = (
            df
            .join(node_map_df, on="nid", how="inner")
            .join(context_map_df, on="group_val", how="inner")
        )

        if df.is_empty():
            return

        src_indices = df["node_idx"].to_numpy()
        dst_indices = df["context_idx"].to_numpy()

        # 4) Create edges (forward + reverse)
        edge_index = torch.tensor(
            [src_indices, dst_indices],
            dtype=torch.long,
        )

        edge_type = f"on_{group_label}"
        rel = (node_type, edge_type, context_node_type)
        rev_rel = (context_node_type, f"rev_{edge_type}", node_type)

        data_store[rel].edge_index = edge_index
        data_store[rev_rel].edge_index = edge_index.flip(0)

        # 5) Create context nodes
        # if not hasattr(data_store[context_node_type], "num_nodes"):
        data_store[context_node_type].num_nodes = num_context_nodes
        data_store[context_node_type].x = torch.zeros((num_context_nodes, 1), dtype=torch.float)

        # 6) Logging
        print(
            f"[context-node] {node_type} --{edge_type}--> {context_node_type} | "
            f"context_nodes={num_context_nodes}, edges={edge_index.size(1)}"
        )



def main(schema: Dict[str, Any]) -> None:
    gb = GraphBuilder(yaml=schema, data_dir="data/raw")
    g = gb.build()

    # quick peek

    # torch.save(g, "data/graph/sdge.pt")
    

    output_path = 'data/graph'
    graph_name = schema.get('graph_name', 'sdge.pt')
    os.makedirs(output_path, exist_ok=True)
    torch.save(g, output_path + '/' + graph_name)


if __name__ == "__main__":
    # How to run:
    # python -m src.process.structure_graph_builder
    with open("configs/graph.yaml", "r") as f:
        schema = yaml.safe_load(f)
    main(schema)
