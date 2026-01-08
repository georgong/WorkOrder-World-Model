from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np

def _add_time_features(
    df: pd.DataFrame,
    time_cols: List[str],
    prefix: str,
) -> pd.DataFrame:
    """
    对每个时间列:
      - 若为 datetime-like:
          解析为 datetime
          提取 year/month/day/hour
          对 month/day/hour 做 sin/cos 周期编码
        列名形如:
          {prefix}_{col}_year
          {prefix}_{col}_month / _month_sin / _month_cos
          {prefix}_{col}_day   / _day_sin   / _day_cos
          {prefix}_{col}_hour  / _hour_sin  / _hour_cos

      - 若为 timedelta-like:
          提取 sec / hours / days 数值特征
        列名形如:
          {prefix}_{col}_sec
          {prefix}_{col}_hours
          {prefix}_{col}_days
    """
    df = df.copy()

    for col in time_cols:
        if col not in df.columns:
            continue

        s = df[col]
        dtype = s.dtype

        # 1) Timedelta 分支：用 total_seconds / hours / days
        if np.issubdtype(dtype, np.timedelta64):
            base = f"{prefix}_{col}"
            td = pd.to_timedelta(s, errors="coerce")

            sec = td.dt.total_seconds().astype("Float64")
            df[f"{base}_sec"]   = sec
            df[f"{base}_hours"] = sec / 3600.0
            df[f"{base}_days"]  = sec / 86400.0
            continue  # 不走 datetime 分支

        # 2) 其他情况，当作 datetime 解析
        dt = pd.to_datetime(s, errors="coerce")

        base = f"{prefix}_{col}"

        year_col = f"{base}_year"
        month_col = f"{base}_month"
        day_col = f"{base}_day"
        hour_col = f"{base}_hour"

        df[year_col] = dt.dt.year.astype("Float64")
        df[month_col] = dt.dt.month.astype("Float64")
        df[day_col] = dt.dt.day.astype("Float64")
        df[hour_col] = dt.dt.hour.astype("Float64")

        # 周期特征：month / day / hour
        for base_col, period in [
            (month_col, 12),
            (day_col, 31),
            (hour_col, 24),
        ]:
            vals = df[base_col].astype("Float64")
            radians = 2 * np.pi * vals / period
            df[f"{base_col}_sin"] = np.sin(radians)
            df[f"{base_col}_cos"] = np.cos(radians)

    return df


def build_feature_table(
    df: pd.DataFrame,
    *,
    key_col: str,
    category_cols: List[str],
    numeric_cols: List[str],
    time_cols: List[str],
    prefix: str,
    top_k_per_cat: int = 30,
) -> pd.DataFrame:
    """
    通用特征构造函数（task / engineer / 其它表都能用）。

    参数:
      - key_col: 主键列名（如 "W6KEY" / "NAME"）
      - category_cols: 需要做 one-hot 的类别列
      - numeric_cols: 直接当数值的列（转 numeric）
      - time_cols: 时间列，做 year/month/day/hour + sin/cos
      - prefix: 这一类实体的前缀 ("task", "eng" ...)
      - top_k_per_cat: 每个 categorical 列保留的最多类别数（按频率排序）
                       其余压成 "__OTHER__"
    """
    df = df.copy()


    # 1) 数值列：存在的就转 numeric，然后统一加前缀
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_renames = {c: f"{prefix}_{c}" for c in numeric_cols}
    df = df.rename(columns=numeric_renames)
    prefixed_numeric_cols = list(numeric_renames.values())

    # 2) 时间派生列
    time_cols_exist = [c for c in time_cols if c in df.columns]
    df = _add_time_features(df, time_cols_exist, prefix=prefix)

    time_derived_cols: List[str] = []
    for t in time_cols_exist:
        base = f"{prefix}_{t}_"
        time_derived_cols.extend([c for c in df.columns if c.startswith(base)])
    time_derived_cols = sorted(set(time_derived_cols))

    # 3) 类别列：压缩到 top-K + "__OTHER__"，再 one-hot
    cat_cols_exist = [c for c in category_cols if c in df.columns]

    for c in cat_cols_exist:
        # 先统一成 string
        s = df[c].astype("string")

        # 统计频率，取 top-K
        vc = s.value_counts(dropna=True)
        top_vals = set(vc.head(top_k_per_cat).index)

        def compress(v):
            if pd.isna(v):
                return pd.NA
            return v if v in top_vals else "__OTHER__"

        df[c] = s.map(compress)

    if cat_cols_exist:
        df_cat = pd.get_dummies(
            df[cat_cols_exist],
            prefix=[f"{prefix}_{c}" for c in cat_cols_exist],
            dummy_na=False,  # NaN 不单独出列，留在全 0
        ).astype("Int64")
    else:
        df_cat = pd.DataFrame(index=df.index)

    # 4) 组装：基表 + one-hot 表
    base_cols = key_col + prefixed_numeric_cols + time_derived_cols
    base_df = df[base_cols].copy()

    feature_df = base_df.join(df_cat)
    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
    feature_df = feature_df.drop(category_cols + time_cols, axis=1, errors='ignore')

    return feature_df



def _non_empty_mask(s: pd.Series) -> pd.Series:
    """
    Helper function for clean_feat_by_keys.

    判断一列哪些是“有内容”的：
      - 数值列: notna()
      - 其它: 转 string，非 NaN 且长度 > 0
    """
    if pd.api.types.is_numeric_dtype(s.dtype):
        return ~s.isna()
    else:
        s2 = s.astype("string")
        return s2.notna() & (s2.str.len() > 0)

def clean_feat_by_keys(
    df: pd.DataFrame,
    *,
    key_cols: List[str],
    primary_key: str,
) -> pd.DataFrame:
    """
    通用清洗规则（3 个 feat 都用这个）:

    保留的行满足：
      1) primary_key 非空
      2) 在其它 key_cols 里面，至少有一个非空

    丢弃：
      - primary_key 为空
      - primary_key 非空，但其它 key 全空 → 完全独立点，扔掉
    """
    df = df.copy()
    if len(key_cols) == 1:
        return df.copy()

    if primary_key not in key_cols:
        raise ValueError(f"primary_key {primary_key!r} 不在 key_cols 里")

    for c in key_cols:
        if c not in df.columns:
            raise KeyError(f"df 缺少 key 列 {c!r}")

    # 1) 主键非空
    mask_primary = _non_empty_mask(df[primary_key])

    # 2) 其它 key 至少有一个非空
    other_keys = [c for c in key_cols if c != primary_key]

    if other_keys:
        other_masks = [_non_empty_mask(df[c]) for c in other_keys]
        mask_any_other = other_masks[0]
        for m in other_masks[1:]:
            mask_any_other = mask_any_other | m
    else:
            mask_any_other = False   # 等价于“没有一行满足第二条”
        # 这种情况下一般你也不会来用这个函数
    keep = mask_primary & mask_any_other
    return df[keep].copy()