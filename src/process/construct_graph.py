# import os
# for i in os.walk("data/raw"):
#     for root_dir, dirs, files in [i]:
#         for file in files:
#             if file.endswith(".csv"):
#                 print(os.path.join(root_dir, file)) #iterate through all csv files in raw_data folder

from pathlib import Path
from typing import Dict, List
import pandas as pd
from src.process.utils.filter_raw_data import drop_sparse_columns
# from src.process.utils.convert_columns import convert_with_schema
# from src.process.utils.convert_columns import remove_outliers_by_spec
import yaml
import polars as pl
from src.process.utils.inspect_relation import (inspect_task_assignment_relation, inspect_assignments_engineers)

def preprocess_dfs(
        schema: dict, 
        df_type: str, 
        drop_sparse: bool=True,
        min_non_na_ratio: float=0.1
    ) -> pl.DataFrame:
    '''
    Process raw CSV files of a given type into a single DataFrame 
    by converting types, removing outliers, and droping sparse columns
    
    Parameters
    ----
    schema : dict
        Schmema dictionary containing dataset and variable metadata.
    df_type : str
        Dataset type to process, one of "tasks", "assignments", "engineers", "districts".
    drop_sparse : bool
        Whether to drop sparse columns
    
    Return
    ----
    df_concat : pl.DataFrame
    '''
    assert df_type in ["tasks", "assignments", "engineers", "districts"], "type must be one of 'tasks', 'assignments', 'engineers', 'districts'"

    if df_type == "tasks":
        type_dir = "W6TASKS"
    elif df_type == "assignments":
        type_dir = "W6ASSIGNMENTS"
    elif df_type == "engineers":
        type_dir = "W6ENGINEERS"
    elif df_type == "districts":
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

    # --------------------------------------------------------------------
    # Outlier filtering
    # --------------------------------------------------------------------
    for col, meta in vars_meta.items():
        outlier = meta.get("outlier", ['None', 'None'])
        
        # If outlier is a list of two strings, try to interpret as numeric bounds
        lower, upper = outlier[0], outlier[1]
        numeric_bounds = True
        try:
            lower_val = float(lower) if lower not in [None, "None"] else None
            upper_val = float(upper) if upper not in [None, "None"] else None
        except (ValueError, TypeError):
            numeric_bounds = False

        before_count = df_concat.height

        if numeric_bounds and (lower_val is not None or upper_val is not None):
            if lower_val is not None:
                df_concat = df_concat.filter(pl.col(col) >= lower_val)
            if upper_val is not None:
                df_concat = df_concat.filter(pl.col(col) <= upper_val)
            after_count = df_concat.height
            print(f"Column '{col}': kept {after_count} / {before_count} rows (filtered {before_count - after_count})")

        elif isinstance(outlier, list) and any(x not in [None, "None"] for x in outlier):
            # Treat as categorical exclusion list
            exclude_values = [x for x in outlier if x not in [None, "None"]]
            df_concat = df_concat.filter(~pl.col(col).is_in(exclude_values))
            after_count = df_concat.height
            print(f"Column '{col}': kept {after_count} / {before_count} rows (filtered {before_count - after_count})")

    # --------------------------------------------------------------------
    # Sparse column dropping
    # --------------------------------------------------------------------
    if drop_sparse:
        # Convert Polars DataFrame to Pandas
        df_concat_pd = df_concat.to_pandas()
        df_concat_pd = drop_sparse_columns(
            df_concat_pd,
            min_non_na_ratio=min_non_na_ratio,
            inplace=False,
            verbose=True,
        )
        # Convert back to Polars DataFrame
        df_concat = pl.from_pandas(df_concat_pd)

    return df_concat

def parse_yaml(yaml_path: str) -> dict:
    '''
    Parse YAML schema file.
    '''
    with open(yaml_path, "r") as f:
        schema = yaml.safe_load(f)
    return schema 

if __name__ == "__main__":
    # How to run this file: 
    # python -m src.process.construct_graph

    schema = parse_yaml("data/data.yaml")

    # --------------------------------------------------------------------
    # Preprocess dataframes
    # --------------------------------------------------------------------
    task_df = preprocess_dfs(schema, "tasks")
    task_df = task_df.with_columns(
       (pl.col("SCHEDULEDFINISH") - pl.col("SCHEDULEDSTART")).alias("SCHEDULECOMPLETIONTIME")
    )
    
    assignment_df = preprocess_dfs(schema, "assignments")
    
    engineer_df = preprocess_dfs(schema, "engineers")
    
    district_df = preprocess_dfs(schema, "districts")
    district_df = district_df.with_columns(
       pl.col("POSTCODE").str.slice(0, 5).alias("POSTCODE")
    )
    
    # --------------------------------------------------------------------
    # Inspect relationships between tables
    # --------------------------------------------------------------------
    join_key_dict = schema["mappings"]
    keys = join_key_dict["tasks"]["links"]["assignments"]
    task_key_col   = keys["left_on"]   # 应该是 "W6KEY"
    assign_fk_col  = keys["right_on"]  # 应该是 "TASK"

    print(f"tasks.{task_key_col}  ↔  assignments.{assign_fk_col}")
    
    # 用你现在的表名，自己选：tasks / tasks_norm, assignments / assignments_norm
    task_assignment_relation_summary = inspect_task_assignment_relation(
        task_df.to_pandas(),           # 或 tasks_norm
        assignment_df.to_pandas(),     # 或 assignments_norm
        task_key_col=task_key_col,
        assign_fk_col=assign_fk_col,
    )

    assignemnt_engineer_relation_summary = inspect_assignments_engineers(
        assignments_df=assignment_df.to_pandas(),
        engineers_df=engineer_df.to_pandas(),
        left_key="ASSIGNEDENGINEERS",
        right_key="NAME",    # 如果你最后决定用 CREW 则改成 "CREW"
    )
