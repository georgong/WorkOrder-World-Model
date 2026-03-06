from __future__ import annotations

from typing import List
import math
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

from .feature_schema import FeatureSchema

SCHEMA_DIR_DEFAULT = Path(__file__).parent.parent / "assets" / "feature_schemas"

def load_feature_schema(
    node_type: str,
    schema_dir: Path = SCHEMA_DIR_DEFAULT,
) -> Optional[Dict]:
    """Load a previously saved schema. Returns None if not found."""
    path = schema_dir / f"{node_type}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

def align_to_schema(
    df: pl.DataFrame,
    node_type: str,
    schema_dir: Path = SCHEMA_DIR_DEFAULT,
    key_cols: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Align *df* to the column schema saved during training:
      1. Add missing columns as 0 (Float64 for numeric / UInt8 for one-hot).
      2. Drop extra columns not in the schema.
      3. Reorder to match the training column order.

    If no saved schema is found, returns df unchanged (best-effort).

    Parameters
    ----------
    df : pl.DataFrame
        The feature table built on inference data.
    node_type : str
        Identifier matching what was used in save_feature_schema().
    schema_dir : Path
        Directory containing the saved JSON schemas.
    key_cols : list[str] | None
        Key/ID columns to always preserve regardless of schema.
    """
    schema = load_feature_schema(node_type, schema_dir)
    if schema is None:
        return df

    expected_cols: List[str] = schema["columns"]
    expected_dtypes: Dict[str, str] = schema.get("dtypes", {})

    # Columns to always keep (keys) even if not in schema
    preserve = set(key_cols or [])

    current_cols = set(df.columns)
    expected_set = set(expected_cols)

    # 1) Add missing columns with zeros
    missing = expected_set - current_cols
    if missing:
        fill_exprs = []
        for col in missing:
            dtype_str = expected_dtypes.get(col, "Float64")
            # One-hot columns are typically UInt8
            if "UInt8" in dtype_str or "Bool" in dtype_str:
                fill_exprs.append(pl.lit(0, dtype=pl.UInt8).alias(col))
            else:
                fill_exprs.append(pl.lit(0.0, dtype=pl.Float64).alias(col))
        df = df.with_columns(fill_exprs)

    # 2) Build final column order: expected columns + any preserve cols not already there
    final_cols = list(expected_cols)
    for c in preserve:
        if c in df.columns and c not in expected_set:
            final_cols.append(c)

    # Only select columns that actually exist now
    final_cols = [c for c in final_cols if c in df.columns]

    return df.select(final_cols)

def _add_time_features(
    df: pl.DataFrame,
    time_cols: List[str],
    prefix: str,
) -> pl.DataFrame:
    """
    For each time column:
      - If duration-like: add sec / hours / days numeric features
      - Else (datetime-like): parse to Datetime, extract year/month/day/hour,
        and add cyclic sin/cos for month/day/hour.

    Output column names:
      {prefix}_{col}_year
      {prefix}_{col}_month, _month_sin, _month_cos
      {prefix}_{col}_day,   _day_sin,   _day_cos
      {prefix}_{col}_hour,  _hour_sin,  _hour_cos

    For durations:
      {prefix}_{col}_sec, _hours, _days
    """
    exprs: List[pl.Expr] = []
    two_pi = 2.0 * math.pi
    

    def cyc(raw: pl.Expr, period: float, name: str) -> List[pl.Expr]:
        # raw is an expression (NOT a column reference string)
        radians = raw * (two_pi / period)
        return [
            radians.sin().alias(f"{name}_sin"),
            radians.cos().alias(f"{name}_cos"),
        ]

    for col in time_cols:
        if col not in df.columns:
            continue

        dtype = df.schema[col]
        base = f"{prefix}_{col}"

        # -------------------------
        # Duration branch
        # -------------------------
        if dtype == pl.Duration:
            sec_raw = pl.col(col).dt.total_seconds().cast(pl.Float64)
            exprs.extend([
                sec_raw.alias(f"{base}_sec"),
                (sec_raw / 3600.0).alias(f"{base}_hours"),
                (sec_raw / 86400.0).alias(f"{base}_days"),
            ])
            continue

        # -------------------------
        # Datetime branch
        # -------------------------
        dt = pl.col(col)
        if dtype == pl.Date:
            dt = dt.cast(pl.Datetime("ns"))
        elif dtype != pl.Datetime and dtype != pl.Datetime("ns"):
            # parse strings -> datetime
            dt = dt.cast(pl.Utf8).str.to_datetime(strict=False)

        # Use raw expressions for cyclic features (no referencing newly created cols)
        year_raw  = dt.dt.year().cast(pl.Float64)
        month_raw = dt.dt.month().cast(pl.Float64)
        day_raw   = dt.dt.day().cast(pl.Float64)
        hour_raw  = dt.dt.hour().cast(pl.Float64)

        # Base extracted components
        exprs.extend([
            year_raw.alias(f"{base}_year"),
            month_raw.alias(f"{base}_month"),
            day_raw.alias(f"{base}_day"),
            hour_raw.alias(f"{base}_hour"),
        ])

        # Cyclic encodings directly from raw expressions
        exprs.extend(cyc(month_raw, 12.0, f"{base}_month"))
        exprs.extend(cyc(day_raw,   31.0, f"{base}_day"))
        exprs.extend(cyc(hour_raw,  24.0, f"{base}_hour"))

    return df.with_columns(exprs) if exprs else df


def build_feature_table(
    df: pl.DataFrame,
    *,
    key_col: List[str],
    category_cols: List[str],
    numeric_cols: List[str],
    time_cols: List[str],
    prefix: str,
    top_k_per_cat: int = 30,
) -> pl.DataFrame:
    """
    Generic feature table builder (polars version).
    - numeric: cast to Float64 and rename with prefix
    - time: add derived time features (year/month/day/hour + sin/cos OR duration seconds/hours/days)
    - categorical: keep top-K values by frequency, compress others to "__OTHER__", then one-hot
    """
    df = df.clone()

    # 0) Rename all original columns with prefix
    rename_map = {c: f"{prefix}_{c}" for c in df.columns}
    df = df.rename(rename_map)

    # Update column names for downstream steps
    key_col_prefixed = [rename_map[c] for c in key_col if c in rename_map]
    category_cols_prefixed = [rename_map[c] for c in category_cols if c in rename_map]
    numeric_cols_prefixed = [rename_map[c] for c in numeric_cols if c in rename_map]
    time_cols_prefixed = [rename_map[c] for c in time_cols if c in rename_map]

    # 1) numeric columns: cast + rename with prefix
    num_exprs = []
    for c in numeric_cols_prefixed:
        num_exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
    if num_exprs:
        df = df.with_columns(num_exprs)

    # 2) time features
    # time_cols_exist = [c for c in time_cols if c in df.columns]
    # df = _add_time_features(df, time_cols_exist, prefix=prefix)

    # collect derived time cols by prefix pattern
    time_derived_cols: List[str] = []
    # for t in time_cols_exist:
        # base_prefix = f"{prefix}_{t}_"
        # time_derived_cols.extend([c for c in df.columns if c.startswith(base_prefix)])
    # time_derived_cols = sorted(set(time_derived_cols))

    # 3) categorical: top-k compression + one-hot
    for c in category_cols_prefixed:
        # Get top-k non-null categories (as strings)
        top_vals = (
            df.select(pl.col(c).cast(pl.Utf8))
              .to_series()
              .drop_nulls()
              .value_counts()
              .sort("count", descending=True)
              .head(top_k_per_cat)
              .get_column(c)
              .to_list()
        )
        top_set = set(top_vals)

        # Compress: null stays null; non-top becomes "__OTHER__"
        df = df.with_columns(
            pl.when(pl.col(c).is_null())
              .then(pl.lit(None, dtype=pl.Utf8))
              .when(pl.col(c).cast(pl.Utf8).is_in(list(top_set)))
              .then(pl.col(c).cast(pl.Utf8))
              .otherwise(pl.lit("__OTHER__", dtype=pl.Utf8))
              .alias(c)
        )

    # One-hot encode (no dummy for null by default in polars)
    if category_cols_prefixed:
        # Prefix each categorical column name in dummy output
        # Polars to_dummies uses col name + "_" + value; we first rename cols to include prefix.
        df_for_dummy = df.select(category_cols_prefixed)
        df_cat = df_for_dummy.to_dummies()  # UInt8 columns
    else:
        df_cat = pl.DataFrame()

    # 4) assemble base + dummy
    # base_cols = key_col + prefixed_numeric_cols + time_derived_cols
    # base_cols = [c for c in base_cols if c in df.columns]  # be tolerant

    # base_df = df.select(base_cols)

    feature_df = df.hstack(df_cat) if df_cat.width > 0 else df

    # Drop original categorical/time raw columns (keep only engineered)
    drop_cols = [c for c in (category_cols_prefixed + time_cols_prefixed) if c in feature_df.columns]
    if drop_cols:
        feature_df = feature_df.drop(drop_cols)

    return feature_df


def _non_empty_mask(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """
    Non-empty definition (polars expr version):
      - numeric: not null
      - others: cast to Utf8, not null AND length > 0
    """
    if dtype.is_numeric():
        return expr.is_not_null()
    else:
        s = expr.cast(pl.Utf8)
        return s.is_not_null() & (s.str.len_chars() > 0)


def clean_feat_by_keys(
    df: pl.DataFrame,
    *,
    key_cols: List[str],
    primary_key: str,
) -> pl.DataFrame:
    """
    Keep rows satisfying:
      1) primary_key is non-empty
      2) among other key_cols, at least one is non-empty

    Robust behavior:
      - If some key columns are missing, DO NOT raise.
      - Filter using the intersection of key_cols and df.columns.
      - If primary_key is missing, return df unchanged (can't validate rule 1).
      - If no other keys exist in df, fallback to rule 1 only.
    """
    if len(key_cols) <= 1:
        return df.clone()

    if primary_key not in key_cols:
        raise ValueError(f"primary_key {primary_key!r} not in key_cols")

    # Only use columns that actually exist in df
    existing = [c for c in key_cols if c in df.columns]

    # If primary key itself is missing, we cannot enforce rule (1)
    if primary_key not in existing:
        # Do not crash; just return df as-is
        return df.clone()

    schema = df.schema

    mask_primary = _non_empty_mask(pl.col(primary_key), schema[primary_key])

    other_keys = [c for c in existing if c != primary_key]
    if not other_keys:
        # Can't enforce rule (2). Fallback: keep rows satisfying rule (1) only.
        return df.filter(mask_primary)

    mask_any_other = None
    for c in other_keys:
        m = _non_empty_mask(pl.col(c), schema[c])
        mask_any_other = m if mask_any_other is None else (mask_any_other | m)

    keep = mask_primary & mask_any_other
    return df.filter(keep)


def process_task_feature(
        task_df: pl.DataFrame,
        schema: FeatureSchema
    ) -> pl.DataFrame:
    '''
    Process raw task dataframe into task feature table
    '''
    prefix = "task"
    task_feat = build_feature_table(
        task_df,
        key_col=schema.key_cols,
        category_cols=schema.category_feature,
        numeric_cols=schema.numeric_feature,
        time_cols=schema.time_feature,
        prefix=prefix,
        top_k_per_cat=30,
    )

    task_feat = task_feat.with_columns([
        pl.col(f"{prefix}_DUEDATE").dt.year().alias(f"{prefix}_due_year"),
        pl.col(f"{prefix}_DUEDATE").dt.month().alias(f"{prefix}_due_month"),
        pl.col(f"{prefix}_DUEDATE").dt.day().alias(f"{prefix}_due_day"),
        pl.col(f"{prefix}_DUEDATE").dt.hour().alias(f"{prefix}_due_hour"),
        pl.col(f"{prefix}_DUEDATE").dt.weekday().alias(f"{prefix}_due_weekday"),
        # Cyclic encoding
        (pl.col(f"{prefix}_DUEDATE").dt.month() * 2 * np.pi / 12).sin().alias(f"{prefix}_due_month_sin"),
        (pl.col(f"{prefix}_DUEDATE").dt.month() * 2 * np.pi / 12).cos().alias(f"{prefix}_due_month_cos"),
        (pl.col(f"{prefix}_DUEDATE").dt.hour() * 2 * np.pi / 24).sin().alias(f"{prefix}_due_hour_sin"),
        (pl.col(f"{prefix}_DUEDATE").dt.hour() * 2 * np.pi / 24).cos().alias(f"{prefix}_due_hour_cos"),
        # Durations
        ((pl.col(f"{prefix}_DUEDATE") - pl.col(f"{prefix}_SCHEDULEDSTART")).dt.total_hours() / 3600).alias(f"{prefix}_lead_time_hours"),
        ((pl.col(f"{prefix}_DUEDATE") - pl.col(f"{prefix}_SCHEDULEDFINISH")).dt.total_hours() / 3600).alias(f"{prefix}_slack_time_hours"),
        ((pl.col(f"{prefix}_SCHEDULEDFINISH") - pl.col(f"{prefix}_SCHEDULEDSTART")).dt.total_hours() / 3600).alias(f"{prefix}_scheduled_duration_hours"),
    ])

    task_feat = task_feat.drop([f"{prefix}_DUEDATE", f"{prefix}_SCHEDULEDFINISH", f"{prefix}_SCHEDULEDSTART"])

    # Align to training-time schema (pad missing one-hot cols, drop extras)
    task_feat = align_to_schema(task_feat, node_type="tasks", key_cols=[f"{prefix}_{k}" for k in schema.key_cols])

    return task_feat


def process_assignment_feature(
        assignment_df: pl.DataFrame,
        schema: FeatureSchema
    ) -> pl.DataFrame:
    '''
    Process raw assignment dataframe into assignment feature table
    '''
    prefix="assign"

    assignment_feat = build_feature_table(
        assignment_df,
        key_col=schema.key_cols,
        category_cols=schema.category_feature,
        numeric_cols=schema.numeric_feature,
        time_cols=schema.time_feature,
        prefix=prefix,
        top_k_per_cat=30,
    )

    assignment_feat = assignment_feat.with_columns([
        pl.col(f"{prefix}_STARTTIME").dt.year().alias(f"{prefix}_start_time_year"),
        pl.col(f"{prefix}_STARTTIME").dt.month().alias(f"{prefix}_start_time_month"),
        pl.col(f"{prefix}_STARTTIME").dt.day().alias(f"{prefix}_start_time_day"),
        pl.col(f"{prefix}_STARTTIME").dt.hour().alias(f"{prefix}_start_time_hour"),
        pl.col(f"{prefix}_STARTTIME").dt.weekday().alias(f"{prefix}_start_time_weekday"),
        # Cyclic encoding
        (pl.col(f"{prefix}_STARTTIME").dt.month() * 2 * np.pi / 12).sin().alias(f"{prefix}_start_time_month_sin"),
        (pl.col(f"{prefix}_STARTTIME").dt.month() * 2 * np.pi / 12).cos().alias(f"{prefix}_start_time_month_cos"),
        (pl.col(f"{prefix}_STARTTIME").dt.hour() * 2 * np.pi / 24).sin().alias(f"{prefix}_start_time_hour_sin"),
        (pl.col(f"{prefix}_STARTTIME").dt.hour() * 2 * np.pi / 24).cos().alias(f"{prefix}_start_time_hour_cos"),
    ])
    assignment_feat = assignment_feat.with_columns(
    (pl.col(f"{prefix}_COMPLETIONTIME").dt.total_seconds() / 3600)
    .alias(f"{prefix}_COMPLETIONTIME")
    )
    
    assignment_feat = assignment_feat.drop([f"{prefix}_STARTTIME"])

    # Align to training-time schema
    assignment_feat = align_to_schema(assignment_feat, node_type="assignments", key_cols=[f"{prefix}_{k}" for k in schema.key_cols])

    return assignment_feat


def process_engineer_feature(
        engineer_df: pl.DataFrame,
        schema: FeatureSchema
    ) -> pl.DataFrame:
    '''
    Process raw engineers dataframe into engineers feature table
    '''
    prefix="eng"
    engineer_df = engineer_df.clone()
    
    # convert name column to id 
    engineer_df = engineer_df.with_columns(
        pl.col("NAME").cast(pl.Categorical).to_physical().alias("NAME")
    )
    #engineer_df = engineer_df.drop("NAME")
    
    # filter availability factor to be 0.0 or 1.0
    # engineer_df = engineer_df.filter(
    #     pl.col("AVAILABILITYFACTOR").is_in([0.0, 1.0])
    # )

    engineer_feat = build_feature_table(
        engineer_df,
        key_col=schema.key_cols,
        category_cols=schema.category_feature,
        numeric_cols=schema.numeric_feature,
        time_cols=schema.time_feature,
        prefix=prefix,
        top_k_per_cat=30,
    )

    # Align to training-time schema
    engineer_feat = align_to_schema(engineer_feat, node_type="engineers", key_cols=[f"{prefix}_{k}" for k in schema.key_cols])

    return engineer_feat


def process_districts_feature(
        districts_df: pl.DataFrame,
        schema: FeatureSchema
    ) -> pl.DataFrame:
    '''
    Process raw districts dataframe into districts feature table
    '''
    prefix="district"

    districts_df = districts_df.clone()
    
    # convert city and name column to id 
    districts_df = districts_df.with_columns(
        pl.col("CITY").cast(pl.Categorical).to_physical().alias("CITY_id"), 
        pl.col("NAME").cast(pl.Categorical).to_physical().alias("NAME_id"), 
    )
    districts_df = districts_df.drop(["CITY", "NAME"])

    districts_feat = build_feature_table(
        districts_df,
        key_col=schema.key_cols,
        category_cols=schema.category_feature,
        numeric_cols=schema.numeric_feature,
        time_cols=schema.time_feature,
        prefix=prefix,
        top_k_per_cat=30,
    )

    # Align to training-time schema
    districts_feat = align_to_schema(districts_feat, node_type="districts", key_cols=[f"{prefix}_{k}" for k in schema.key_cols])

    return districts_feat

def process_departments_feature(
    departments_df: pl.DataFrame,
    schema: FeatureSchema,
) -> pl.DataFrame:
    '''
    Process raw district dataframe into districts feature table
    '''
    perfix = 'departments'