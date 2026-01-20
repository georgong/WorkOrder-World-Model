from __future__ import annotations
from pathlib import Path
import polars as pl
from torch_geometric.data import HeteroData
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np

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


def _resolve_files(data_dir: Path, base: str) -> List[Path]:
    """
    Prefer sharded files: BASE-*.csv
    Fallback: BASE.csv
    """
    shards = sorted(data_dir.glob(f"{base}-*.csv"))
    if shards:
        return shards
    single = data_dir / f"{base}.csv"
    if single.exists():
        return [single]
    # last fallback: maybe user wrote lowercase or weird
    return []


def load_table(schema: Dict[str, Any], df_type: str, data_dir: str = "data/raw") -> pl.DataFrame:
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
    return df_concat


@dataclass
class NodeStore:
    node_type: str
    entity_key: str
    name_col: Optional[str]

    # entity ids in row order (index 0..N-1)
    ids: pl.Series
    id2idx: Dict[Any, int]

    # visualization labels aligned with ids (optional)
    names: Optional[pl.Series]

    # node features (trait_type==node)
    x: pl.DataFrame

    # columns whose values must be hidden at prediction time
    mask_cols: List[str]

    # per-node aggregated "edge-type" attributes (trait_type==edge)
    edge_attrs: pl.DataFrame


@dataclass
class EdgeStore:
    edge_type: Tuple[str, str, str]       # (src_type, rel, dst_type)
    edge_index: pl.DataFrame              # columns: ["src", "dst"] (Int64 indices)
    edge_attr: pl.DataFrame               # edge features aligned with edges (may be empty)


@dataclass
class Graph:
    nodes: Dict[str, NodeStore]
    edges: Dict[Tuple[str, str, str], EdgeStore]


def process_feature(graph_dict):
    """
    TODO: process node features and edge features from graph_dict
    """
    tasks_node,tasks_edge = graph_dict["tasks"]
    assignments_node, assignments_edge = graph_dict["assignments"]
    engineers_node, engineers_edge = graph_dict["engineers"]
    districts_node, districts_edge = graph_dict["districts"]
    departments_node, departments_edge = graph_dict["departments"]

    return 

## process key lists from two tables
def build_key_lists(
    *,
    left_df: pl.DataFrame,
    left_join_col: str,   # left table column used to join (e.g., tasks["DISTRICT"])
    right_df: pl.DataFrame,
    right_join_col: str,  # right table column used to join (e.g., districts["W6KEY"])
    left_entity_key: str, # primary key of left entity (e.g., tasks["W6KEY"])
    right_entity_key: str,# primary key of right entity (e.g., districts["W6KEY"])
    left_idx_name: str,
    right_idx_name: str,
    keep_order: bool = True,
) -> Tuple[torch.Tensor, pl.DataFrame, pl.DataFrame]:
    """
    Build edge_index aligned with node idx = row index of (sorted/deduped) node tables.

    Assumption:
      left_df and right_df are already deduped and sorted by their entity_key
      so that node idx == row order is stable.

    Returns:
      edge_index: LongTensor [2, E] where src=left_idx, dst=right_idx
      left_index_df:  [left_entity_key, left_idx_name]
      right_index_df: [right_entity_key, right_idx_name]
    """

    # 1) Build index tables: entity_key -> idx (row index)
    left_index_df = (
        left_df.select(left_entity_key)
               .drop_nulls()
               .unique(maintain_order=keep_order)
               .with_row_index(left_idx_name)
    )

    right_index_df = (
        right_df.select(right_entity_key)
                .drop_nulls()
                .unique(maintain_order=keep_order)
                .with_row_index(right_idx_name)
    )

    # 2) Normalize join tables (copy columns, avoid rename collision)
    # left: joined = left_df[left_join_col], left_id = left_df[left_entity_key]
    L = pl.DataFrame({
        "joined": left_df[left_join_col].to_list(),
        "left_id": left_df[left_entity_key].to_list(),
    }).drop_nulls(["joined", "left_id"])

    # right: joined = right_df[right_join_col], right_id = right_df[right_entity_key]
    R = pl.DataFrame({
        "joined": right_df[right_join_col].to_list(),
        "right_id": right_df[right_entity_key].to_list(),
    }).drop_nulls(["joined", "right_id"])

    # 3) Join on joined to get pairs of (left_entity_key, right_entity_key)
    J = L.join(R, on="joined", how="inner")
    if J.height == 0:
        raise ValueError("Join result is empty. Check your join columns overlap.")

    # 4) Map entity keys -> idx via join with index tables
    J = (
        J.join(left_index_df, left_on="left_id", right_on=left_entity_key, how="inner")
         .join(right_index_df, left_on="right_id", right_on=right_entity_key, how="inner")
    )

    if J.height == 0:
        raise ValueError("After mapping to idx, edge table is empty. Check index tables.")

    # 5) Build edge_index
    src = torch.tensor(J[left_idx_name].to_list(), dtype=torch.long)
    dst = torch.tensor(J[right_idx_name].to_list(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    return edge_index, left_index_df, right_index_df



## process edge index from key lists
def build_edge_index(
    *,
    left_df: pl.DataFrame,
    left_join_col: str,
    right_df: pl.DataFrame,
    right_join_col: str,
    left_entity_key: str,
    right_entity_key: str,
    keep_order: bool = True,
    return_maps: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, pl.DataFrame, pl.DataFrame]:
    """
    Build edge_index aligned with node idx spaces created from entity_key lists.
    Robust to column name collisions by explicitly reconstructing minimal frames.

    Returns:
      - edge_index: LongTensor [2, E]
      - optionally: (edge_index, left_k2i, right_k2i) where
          left_k2i:  [src_key, src_idx]
          right_k2i: [dst_key, dst_idx]
    """

    # 0) Defensive: require columns exist
    for c in (left_join_col, left_entity_key):
        if c not in left_df.columns:
            raise KeyError(f"left_df missing column {c!r}")
    for c in (right_join_col, right_entity_key):
        if c not in right_df.columns:
            raise KeyError(f"right_df missing column {c!r}")

    # 1) Rebuild minimal left/right frames to avoid rename collisions
    #    Use list copies so even if join_col == entity_key, it's still safe.
    L = pl.DataFrame({
        "left_joined": left_df[left_join_col].to_list(),
        "left_key": left_df[left_entity_key].to_list(),
    }).drop_nulls(["left_joined", "left_key"])

    R = pl.DataFrame({
        "right_joined": right_df[right_join_col].to_list(),
        "right_key": right_df[right_entity_key].to_list(),
    }).drop_nulls(["right_joined", "right_key"])

    # 2) Build key->idx maps (idx is row index, not key value)
    left_k2i = (
        pl.DataFrame({"src_key": L["left_key"].unique(maintain_order=keep_order).to_list()})
          .with_row_index("src_idx")
    )

    right_k2i = (
        pl.DataFrame({"dst_key": R["right_key"].unique(maintain_order=keep_order).to_list()})
          .with_row_index("dst_idx")
    )

    # 3) Join edges by joined value (FK match)
    #    Use explicit join keys to avoid any naming overlap.
    J = (
        L.join(R, left_on="left_joined", right_on="right_joined", how="inner")
         .select(["left_key", "right_key"])
         .rename({"left_key": "src_key", "right_key": "dst_key"})
    )

    if J.height == 0:
        raise ValueError("Join result is empty. Check your join columns overlap.")

    # 4) Map keys -> idx
    J = (
        J.join(left_k2i, on="src_key", how="inner")
         .join(right_k2i, on="dst_key", how="inner")
         .select(["src_idx", "dst_idx"])
    )

    if J.height == 0:
        raise ValueError("After mapping to idx, edge table is empty. Check key->idx maps.")

    edge_index = torch.tensor(
        [J["src_idx"].to_list(), J["dst_idx"].to_list()],
        dtype=torch.long
    )

    if return_maps:
        return edge_index, left_k2i, right_k2i
    return edge_index

# -------------------------
# Optional: table loader for sharded CSVs (W6TASKS-0.csv, W6TASKS-1.csv ...)
# -------------------------

DTYPE_MAP = {
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "string": pl.Utf8,
    "datetime64[ns]": pl.Datetime("ns"),
}


def load_sharded_table(
    *,
    df_type: str,
    schema_datasets: Dict[str, Any],
    data_dir: str = "data/raw",
    null_values: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Reads all shards like: data/raw/W6TASKS-*.csv (based on schema_datasets[df_type]["file"])
    and concatenates vertically.
    """
    null_values = null_values or ["", " ", "NA", "N/A", "NULL", "null", "None"]

    file_path = schema_datasets[df_type]["file"]  # e.g. data/raw/W6TASKS.csv
    stem = Path(file_path).name.replace(".csv", "")  # W6TASKS

    files = sorted(Path(data_dir).glob(f"{stem}-*.csv"))
    if not files:
        # fallback: single file
        files = [Path(file_path)]

    vars_meta = (schema_datasets[df_type] or {}).get("variables", {}) or {}
    cols = list(vars_meta.keys())

    pl_schema = {
        col: DTYPE_MAP.get((meta or {}).get("dtype", "string"), pl.Utf8)
        for col, meta in vars_meta.items()
        if isinstance(meta, dict)
    }

    dfs: List[pl.DataFrame] = []
    for f in files:
        df = pl.read_csv(
            f,
            columns=cols,
            schema_overrides=pl_schema,
            null_values=null_values,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        dfs.append(df)

    return pl.concat(dfs, how="vertical")


def split_tables_by_trait(
    schema: Dict[str, Any],
    table_dict: Dict[str, pl.DataFrame],
    *,
    include_key_cols_in_both: bool = True,
    strict: bool = False,
    treat_null_trait_as: Optional[str] = None,  # None | "node" | "edge"
) -> Dict[str, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Split each table into (node_df, edge_df) based on schema trait_type.

    Args:
        schema: dict loaded from YAML. Expected schema["datasets"][table_name]["variables"][col] -> info dict
        table_dict: {table_name: pl.DataFrame}
        include_key_cols_in_both: if True, columns with key:true are included in both node_df and edge_df.
        strict: if True, raise if schema refers to a column missing in the DataFrame.
        treat_null_trait_as: if trait_type is null/None/missing, optionally treat as "node" or "edge".
                             If None, those cols are ignored (unless key:true and include_key_cols_in_both).

    Returns:
        {table_name: (node_df, edge_df)}
    """

    if "datasets" not in schema:
        raise KeyError("schema missing top-level 'datasets'")

    out: Dict[str, Tuple[pl.DataFrame, pl.DataFrame]] = {}

    for table_name, df in table_dict.items():
        if table_name not in schema["datasets"]:
            if strict:
                raise KeyError(f"schema missing dataset entry for table {table_name!r}")
            # still return empty splits with original df columns ignored
            out[table_name] = (df.head(0), df.head(0))
            continue

        vars_cfg: Dict[str, Dict[str, Any]] = schema["datasets"][table_name].get("variables", {})
        if not isinstance(vars_cfg, dict):
            raise TypeError(f"schema['datasets'][{table_name!r}]['variables'] must be a dict")

        node_cols: List[str] = []
        edge_cols: List[str] = []

        for col, info in vars_cfg.items():
            if col not in df.columns:
                if strict:
                    raise KeyError(f"Table {table_name!r} missing column {col!r} required by schema")
                continue

            is_key = bool(info.get("key", False))
            trait = info.get("trait_type", None)

            if trait is None and treat_null_trait_as is not None:
                trait = treat_null_trait_as

            # Key columns: usually keep everywhere for join/traceability
            if is_key and include_key_cols_in_both:
                if col not in node_cols:
                    node_cols.append(col)
                if col not in edge_cols:
                    edge_cols.append(col)
                # do not "continue": a key col could still be explicitly node/edge, but we've already included it
                continue

            if trait == "node":
                node_cols.append(col)
            elif trait == "edge":
                edge_cols.append(col)
            else:
                # trait_type is null or unknown: ignore by default
                pass

        # Make sure we don't duplicate columns
        node_cols = list(dict.fromkeys(node_cols))
        edge_cols = list(dict.fromkeys(edge_cols))

        node_df = df.select(node_cols) if node_cols else df.select([])
        edge_df = df.select(edge_cols) if edge_cols else df.select([])

        out[table_name] = (node_df, edge_df)

    return out


if __name__ == "__main__":
    import yaml

    # Example usage
    with open("configs/graph.yaml", "r") as f:
        schema = yaml.safe_load(f)

    tasks = load_table(schema, "tasks")
    assignments = load_table(schema, "assignments")
    engineers = load_table(schema, "engineers")
    districts = load_table(schema, "districts")
    departments = load_table(schema, "departments")
    print(tasks.head())
    print(assignments.head())
    print(engineers.head())
    print(districts.head())
    print(departments.head())

    data = HeteroData()
    data['task'] = tasks
    data['assignment'] = assignments
    data['engineer'] = engineers
    data['district'] = districts
    data['department'] = departments

    data = HeteroData()
    data["task"]
     
    print("debug")

    print("debug   ")


    tasks = (
    tasks
    .group_by(schema["mappings"]["tasks"]["entity_key"])
    .agg(pl.all().first())
    .sort(schema["mappings"]["tasks"]["entity_key"])
    )

    assignments = (
        assignments
        .group_by(schema["mappings"]["assignments"]["entity_key"])
        .agg(pl.all().first())
        .sort(schema["mappings"]["assignments"]["entity_key"])
    )

    engineers = (
        engineers
        .group_by(schema["mappings"]["engineers"]["entity_key"])
        .agg(pl.all().first())
        .sort(schema["mappings"]["engineers"]["entity_key"])
    )

    districts = (
        districts
        .group_by(schema["mappings"]["districts"]["entity_key"])
        .agg(pl.all().first())
        .sort(schema["mappings"]["districts"]["entity_key"])
    )

    departments = (
        departments
        .group_by(schema["mappings"]["departments"]["entity_key"])
        .agg(pl.all().first())
        .sort(schema["mappings"]["departments"]["entity_key"])
    )

    
    
    
    
    



    edge_index= build_edge_index(
            left_df=tasks,
    left_join_col=schema["mappings"]["tasks"]["links"]["districts"]["left_on"],   # e.g. "DISTRICT"
    right_df=districts,
    right_join_col=schema["mappings"]["tasks"]["links"]["districts"]["right_on"], # e.g. "W6KEY"
    left_entity_key=schema["mappings"]["tasks"]["entity_key"],                    # e.g. "W6KEY"
    right_entity_key=schema["mappings"]["districts"]["entity_key"],               # e.g. "W6KEY"
)
    table_dict = {
        "tasks": tasks,
        "assignments": assignments,
        "engineers": engineers,
        "districts": districts,
        "departments": departments,
    }
    splits = split_tables_by_trait(schema, table_dict, include_key_cols_in_both=False)
    tasks_node, tasks_edge = splits["tasks"]
    assign_node, assign_edge = splits["assignments"]
    engineer_node, engineer_edge = splits["engineers"]
    district_node, district_edge = splits["districts"]
    department_node, department_edge = splits["departments"]
    print("debug 2")
    data = HeteroData()
    data["task"].x = torch.ones((tasks.shape[0], tasks.shape[1]), dtype=torch.float)
    data["district"].x = torch.ones((districts.shape[0], districts.shape[1]), dtype=torch.float)
    data[("task", "belongs_to", "district")].edge_index = edge_index
    print("debug 3")





    
