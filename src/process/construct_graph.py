# import os
# for i in os.walk("data/raw"):
#     for root_dir, dirs, files in [i]:
#         for file in files:
#             if file.endswith(".csv"):
#                 print(os.path.join(root_dir, file)) #iterate through all csv files in raw_data folder

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import yaml
import polars as pl

# Keep your original imports (do not change function names in this file).
# NOTE: We will override some imported function names below with pure-Polars implementations.
from src.process.utils.filter_raw_data import drop_sparse_columns as _drop_sparse_columns_pandas
from src.process.utils.inspect_relation import (
    inspect_task_assignment_relation as _inspect_task_assignment_relation_pandas,
    inspect_assignments_engineers as _inspect_assignments_engineers_pandas,
)
from src.process.feature_engineering import build_feature_table, clean_feat_by_keys
from src.process.feature_schema import (
    task_schema,
    engineer_schema,
    assignment_schema,
    district_schema,
    FeatureSchema,
)

# -------------------------------------------------
# Pure-Polars overrides (same function names, so your call sites stay unchanged)
# -------------------------------------------------

def drop_sparse_columns(
    df: pl.DataFrame,
    min_non_na_ratio: float = 0.1,
    inplace: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Polars version of drop_sparse_columns.
    Keeps columns whose non-null ratio >= min_non_na_ratio.

    We intentionally keep the same function name/signature as your pandas utility,
    so existing call sites remain unchanged.
    """
    if df.height == 0:
        return df

    n = df.height
    keep_cols: List[str] = []
    dropped: List[tuple[str, float]] = []

    for c in df.columns:
        non_null = df.select(pl.col(c).is_not_null().sum()).item()
        ratio = non_null / n
        if ratio >= min_non_na_ratio:
            keep_cols.append(c)
        else:
            dropped.append((c, ratio))

    if verbose and dropped:
        print(f"Dropping {len(dropped)} sparse columns (min_non_na_ratio={min_non_na_ratio}):")
        for c, r in sorted(dropped, key=lambda x: x[1]):
            print(f"  - {c}: {r:.4f}")

    # inplace kept for signature compatibility, but Polars is immutable anyway.
    return df.select(keep_cols)


def inspect_task_assignment_relation(
    tasks_df: pl.DataFrame,
    assignments_df: pl.DataFrame,
    task_key_col: str,
    assign_fk_col: str,
) -> dict:
    """
    Polars version of inspect_task_assignment_relation.
    Reports basic PK-FK sanity stats.
    """
    tasks_keys = tasks_df.select(pl.col(task_key_col).drop_nulls().unique())
    assigns_fk = assignments_df.select(pl.col(assign_fk_col).drop_nulls().alias("fk"))

    n_tasks_unique = tasks_keys.height
    n_assign_rows = assignments_df.height
    n_assign_fk_nonnull = assigns_fk.height

    # Orphan: FK exists in assignments but not in tasks PK set
    orphan = assigns_fk.join(
        tasks_keys.rename({task_key_col: "fk"}),
        on="fk",
        how="anti",
    )
    n_orphan_fk = orphan.height

    # Coverage: how many tasks appear at least once in assignments
    covered_tasks = tasks_keys.rename({task_key_col: "k"}).join(
        assigns_fk.rename({"fk": "k"}).unique(),
        on="k",
        how="inner",
    )
    n_covered_tasks = covered_tasks.height

    return {
        "n_tasks_unique": int(n_tasks_unique),
        "n_assignments_rows": int(n_assign_rows),
        "n_assignments_fk_nonnull": int(n_assign_fk_nonnull),
        "n_orphan_fk": int(n_orphan_fk),
        "task_coverage_ratio": (n_covered_tasks / n_tasks_unique) if n_tasks_unique else None,
    }


def inspect_assignments_engineers(
    assignments_df: pl.DataFrame,
    engineers_df: pl.DataFrame,
    left_key: str = "ASSIGNEDENGINEERS",
    right_key: str = "NAME",
    top_k: int = 10,
) -> dict:
    """
    Polars version of inspect_assignments_engineers.
    Reports:
      - how many non-null engineer keys appear in assignments
      - how many are orphans (not found in engineers table)
      - top engineer frequency table
    """
    eng_keys = engineers_df.select(pl.col(right_key).drop_nulls().unique())
    assign_eng = assignments_df.select(pl.col(left_key).drop_nulls().alias("k"))

    orphan = assign_eng.join(
        eng_keys.rename({right_key: "k"}),
        on="k",
        how="anti",
    )

    top_freq = (
        assignments_df
        .filter(pl.col(left_key).is_not_null())
        .group_by(left_key)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_k)
    )

    return {
        "n_assignments_eng_nonnull": int(assign_eng.height),
        "n_orphan_engineers": int(orphan.height),
        "top_engineers": top_freq,  # Polars DataFrame
    }


# -------------------------------------------------
# Main pipeline functions (keep names unchanged)
# -------------------------------------------------

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
    assert df_type in ["tasks", "assignments", "engineers", "districts"], \
        "type must be one of 'tasks', 'assignments', 'engineers', 'districts'"

    if df_type == "tasks":
        type_dir = "W6TASKS"
    elif df_type == "assignments":
        type_dir = "W6ASSIGNMENTS"
    elif df_type == "engineers":
        type_dir = "W6ENGINEERS"
    elif df_type == "districts":
        type_dir = "W6DISTRICTS"
    else:
        raise ValueError(f"Unknown df_type={df_type}")

    data_dir = Path("data/raw")
    # Prefer shards if exist (W6TASKS-*.csv), otherwise fall back to merged file (W6TASKS.csv)
    files = sorted(data_dir.glob(f"{type_dir}-*.csv"))
    if not files:
        single = data_dir / f"{type_dir}.csv"
        if single.exists():
            files = [single]

    print(files)

    if not files:
        raise RuntimeError(
            f"No input files found for {df_type!r}. Expected either "
            f"{data_dir}/{type_dir}-*.csv or {data_dir}/{type_dir}.csv"
        )

    DTYPE_MAP = {
        "Int64": pl.Int64,
        "Float64": pl.Float64,
        "string": pl.Utf8,
        "datetime64[ns]": pl.Datetime("ns"),
    }

    vars_meta = schema["datasets"][df_type]["variables"]
    cols = list(vars_meta.keys())

    pl_schema = {
        col: DTYPE_MAP.get(meta.get("dtype", "string"), pl.Utf8)  # unknown dtype -> Utf8
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

    df_concat = pl.concat(dfs, how="diagonal")
    print(df_concat)

    # --------------------------------------------------------------------
    # Outlier filtering
    # --------------------------------------------------------------------
    for col, meta in vars_meta.items():
        outlier = meta.get("outlier", ["None", "None"])

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
                df_concat = df_concat.filter(pl.col(col).is_null() | (pl.col(col) >= lower_val))
            if upper_val is not None:
                df_concat = df_concat.filter(pl.col(col).is_null() | (pl.col(col) <= upper_val))

            after_count = df_concat.height
            print(f"Column '{col}': kept {after_count} / {before_count} rows (filtered {before_count - after_count})")

        elif isinstance(outlier, list) and any(x not in [None, "None"] for x in outlier):
            # Treat as categorical exclusion list
            exclude_values = [x for x in outlier if x not in [None, "None"]]
            df_concat = df_concat.filter(~pl.col(col).is_in(exclude_values))
            after_count = df_concat.height
            print(f"Column '{col}': kept {after_count} / {before_count} rows (filtered {before_count - after_count})")

    # --------------------------------------------------------------------
    # Sparse column dropping (PURE POLARS now)
    # --------------------------------------------------------------------
    if drop_sparse:
        df_concat = drop_sparse_columns(
            df_concat,
            min_non_na_ratio=min_non_na_ratio,
            inplace=False,
            verbose=True,
        )

    return df_concat


def parse_yaml(yaml_path: str) -> dict:
    '''
    Parse YAML schema file.
    '''
    with open(yaml_path, "r") as f:
        schema = yaml.safe_load(f)
    return schema


def process_task_feature(
        task_df: pl.DataFrame,
        schema: FeatureSchema
    ) -> pl.DataFrame:
    '''
    Process raw task dataframe into task feature table

    For now, we use build_feature_table directly.
    In future, we may add more task-specific feature engineering here.
    '''
    task_feat = build_feature_table(
        task_df,
        key_col=schema.key_cols,
        category_cols=schema.category_feature,
        numeric_cols=schema.numeric_feature,
        time_cols=schema.time_feature,
        prefix="task",
        top_k_per_cat=30,
    )
    ...
    return task_feat


def process_engineer_feature(
        engineer_df: pl.DataFrame,
        schema: FeatureSchema
    ) -> pl.DataFrame:
    '''
    Process raw engineer dataframe into engineer feature table
    '''
    engineer_feat = build_feature_table(
        engineer_df,
        key_col=schema.key_cols,
        category_cols=schema.category_feature,
        numeric_cols=schema.numeric_feature,
        time_cols=schema.time_feature,
        prefix="eng",
        top_k_per_cat=30,
    )
    ...
    return engineer_feat


def process_assignment_feature(
        assignment_df: pl.DataFrame,
        schema: FeatureSchema
    ) -> pl.DataFrame:
    '''
    Process raw assignment dataframe into assignment feature table
    '''
    assignment_feat = build_feature_table(
        assignment_df,
        key_col=schema.key_cols,
        category_cols=schema.category_feature,
        numeric_cols=schema.numeric_feature,
        time_cols=schema.time_feature,
        prefix="assign",
        top_k_per_cat=30,
    )
    ...
    return assignment_feat


def process_district_feature(
        district_df: pl.DataFrame,
        schema: FeatureSchema
    ) -> pl.DataFrame:
    '''
    Process raw district dataframe into district feature table
    '''
    district_feat = build_feature_table(
        district_df,
        key_col=schema.key_cols,
        category_cols=schema.category_feature,
        numeric_cols=schema.numeric_feature,
        time_cols=schema.time_feature,
        prefix="district",
        top_k_per_cat=30,
    )
    ...
    return district_feat


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
        pl.col("POSTCODE").cast(pl.Utf8).str.slice(0, 5).alias("POSTCODE")
    )

    # --------------------------------------------------------------------
    # Inspect relationships between tables (PURE POLARS now)
    # --------------------------------------------------------------------
    join_key_dict = schema["mappings"]
    keys = join_key_dict["tasks"]["links"]["assignments"]
    task_key_col   = keys["left_on"]   # expected "W6KEY"
    assign_fk_col  = keys["right_on"]  # expected "TASK"

    print(f"tasks.{task_key_col}  ↔  assignments.{assign_fk_col}")

    task_assignment_relation_summary = inspect_task_assignment_relation(
        task_df,
        assignment_df,
        task_key_col=task_key_col,
        assign_fk_col=assign_fk_col,
    )
    print(task_assignment_relation_summary)

    assignemnt_engineer_relation_summary = inspect_assignments_engineers(
        assignments_df=assignment_df,
        engineers_df=engineer_df,
        left_key="ASSIGNEDENGINEERS",
        right_key="NAME",    # change to "CREW" if you decide to use CREW
        top_k=10,
    )
    print(assignemnt_engineer_relation_summary["top_engineers"])

    # --------------------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------------------
    task_feat = process_task_feature(task_df, task_schema)
    engineer_feat = process_engineer_feature(engineer_df, engineer_schema)
    assignment_feat = process_assignment_feature(assignment_df, assignment_schema)
    district_feat = process_district_feature(district_df, district_schema)

    task_feat_clean = clean_feat_by_keys(
        task_feat,
        key_cols=task_schema.key_cols,
        primary_key="W6KEY",
    )

    engineer_feat_clean = clean_feat_by_keys(
        engineer_feat,
        key_cols=engineer_schema.key_cols,
        primary_key="NAME",
    )

    assignment_feat_clean = clean_feat_by_keys(
        assignment_feat,
        key_cols=assignment_schema.key_cols,
        primary_key="TASK",
    )

    district_feat_clean = clean_feat_by_keys(
        district_feat,
        key_cols=district_schema.key_cols,
        primary_key="W6KEY",
    )

    # save (Parquet supports Duration/D datetime types safely)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    task_feat_clean.write_parquet(Path("data/processed/tasks_processed.parquet"))
    assignment_feat_clean.write_parquet(Path("data/processed/assignments_processed.parquet"))
    engineer_feat_clean.write_parquet(Path("data/processed/engineers_processed.parquet"))
    district_feat_clean.write_parquet(Path("data/processed/districts_processed.parquet"))
    print("Feature tables saved to data/processed/")
