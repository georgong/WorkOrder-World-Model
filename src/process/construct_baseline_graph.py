from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import polars as pl
from torch_geometric.data import HeteroData

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl


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



    print("debug")

