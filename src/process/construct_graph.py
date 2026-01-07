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

with open("data/data.yaml", "r") as f:
    schema = yaml.safe_load(f)
print(schema)
data_dir = Path("data/raw")

files = sorted(data_dir.glob("W6TASKS-*.csv"))
print(files)


DTYPE_MAP = {
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "string": pl.Utf8,
    "datetime64[ns]": pl.Datetime("ns"),
}

vars_meta = schema["datasets"]["tasks"]["variables"]
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

tasks = pl.concat(dfs, how="vertical")
print(tasks)
