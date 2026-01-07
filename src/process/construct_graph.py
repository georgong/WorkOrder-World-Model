import os
for i in os.walk("data/raw_data"):
    for root_dir, dirs, files in [i]:
        for file in files:
            if file.endswith(".csv"):
                print(os.path.join(root_dir, file)) #iterate through all csv files in raw_data folder

from pathlib import Path
from typing import Dict, List
import pandas as pd
from src.process.utils.filter_raw_data import drop_sparse_columns
from src.process.utils.convert_columns import convert_with_schema
from src.process.utils.convert_columns import remove_outliers_by_spec
import yaml
import polars as pl

def process_dfs(schema:dict, df_type:str) -> pl.DataFrame:
    assert df_type in ["tasks", "assignments", "engineers", "districts"], "type must be one of 'tasks', 'assignments', 'engineers', 'districts'"

    if df_type == "tasks":
        type_dir = "W6TASKS"
    elif df_type == "assignments":
        type_dir = "W6ASSIGNMENTS"
    elif df_type == "engineers":
        type_dir = "W6ENGINEERS"
    else:
        type_dir = "W6DISTRICTS"

    data_dir = Path("data/raw")
    files = sorted(data_dir.glob(f"{type_dir}-*.csv"))
    print(files)

    DTYPE_MAP = {
        "Int64": pl.Int64,
        "Float64": pl.Float64,
        "string": pl.Utf8,
        "datetime64[ns]": pl.Datetime("ns"),
    }

    vars_meta = schema["datasets"][df_type]["variables"]
    cols = list(vars_meta.keys())

    pl_schema = {
        col: DTYPE_MAP.get(meta["dtype"], pl.Utf8)   # unknown dtype -> Utf8
        for col, meta in vars_meta.items()
    }
    print(pl_schema)

    dfs = []
    for f in files:
        df = pl.read_csv(
            f,
            columns=cols,
            schema_overrides=pl_schema,
            null_values=["", " ", "NA", "N/A", "NULL", "null", "None"],
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        dfs.append(df)

    df_concat = pl.concat(dfs, how="vertical")
    print(df_concat)

    # Outlier filtering
    for col, meta in vars_meta.items():
        outlier = meta.get("outlier", [None, None])
        if outlier is not None and outlier != [None, None]:
            before_count = df_concat.height

            if isinstance(outlier[0], (int, float)) or isinstance(outlier[1], (int, float)):
                lower, upper = outlier
                if lower is not None:
                    df_concat = df_concat.filter(pl.col(col) >= lower)
                if upper is not None:
                    df_concat = df_concat.filter(pl.col(col) <= upper)
                after_count = df_concat.height
                print(f"Column '{col}': kept {after_count} / {before_count} rows (filtered {before_count - after_count})")

            elif isinstance(outlier, list) and any(isinstance(x, str) for x in outlier if x is not None):
                # For categorical outliers, treat as exclusion list
                before_count = df_concat.height
                df_concat = df_concat.filter(~pl.col(col).is_in([x for x in outlier if x is not None]))
                after_count = df_concat.height
                print(f"Column '{col}': kept {after_count} / {before_count} rows (filtered {before_count - after_count})")

    return df_concat


if __name__ == "__main__":

    with open("data/data.yaml", "r") as f:
        schema = yaml.safe_load(f)
    print(schema)

    task_df = process_dfs(schema, "tasks")
    assignment_df = process_dfs(schema, "assignments")
    # engineer_df = process_dfs(schema, "engineers")
    # district_df = process_dfs(schema, "districts")


