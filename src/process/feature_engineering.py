from __future__ import annotations

from typing import List
import math
import polars as pl


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
    key_col: str,
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

    # 1) numeric columns: cast + rename with prefix
    numeric_cols_exist = [c for c in numeric_cols if c in df.columns]
    num_renames = {c: f"{prefix}_{c}" for c in numeric_cols_exist}

    num_exprs = []
    for c in numeric_cols_exist:
        num_exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(num_renames[c]))
    if num_exprs:
        df = df.with_columns(num_exprs)

    prefixed_numeric_cols = list(num_renames.values())

    # 2) time features
    time_cols_exist = [c for c in time_cols if c in df.columns]
    df = _add_time_features(df, time_cols_exist, prefix=prefix)

    # collect derived time cols by prefix pattern
    time_derived_cols: List[str] = []
    for t in time_cols_exist:
        base_prefix = f"{prefix}_{t}_"
        time_derived_cols.extend([c for c in df.columns if c.startswith(base_prefix)])
    time_derived_cols = sorted(set(time_derived_cols))

    # 3) categorical: top-k compression + one-hot
    cat_cols_exist = [c for c in category_cols if c in df.columns]
    for c in cat_cols_exist:
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
    if cat_cols_exist:
        # Prefix each categorical column name in dummy output
        # Polars to_dummies uses col name + "_" + value; we first rename cols to include prefix.
        rename_for_dummy = {c: f"{prefix}_{c}" for c in cat_cols_exist}
        df_for_dummy = df.select([pl.col(c).alias(rename_for_dummy[c]) for c in cat_cols_exist])
        df_cat = df_for_dummy.to_dummies()  # UInt8 columns
    else:
        df_cat = pl.DataFrame()

    # 4) assemble base + dummy
    base_cols = [key_col] + prefixed_numeric_cols + time_derived_cols
    base_cols = [c for c in base_cols if c in df.columns]  # be tolerant

    base_df = df.select(base_cols)

    feature_df = base_df.hstack(df_cat) if df_cat.width > 0 else base_df

    # Drop original categorical/time raw columns (keep only engineered)
    drop_cols = [c for c in (category_cols + time_cols) if c in feature_df.columns]
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