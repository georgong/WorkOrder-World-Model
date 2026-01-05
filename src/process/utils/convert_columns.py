from __future__ import annotations
from typing import Dict
import pandas as pd

def safe_to_datetime(
    series: pd.Series,
    year_min: int | None = 1900,
    year_max: int | None = 2100,
    errors: str = "coerce",
) -> pd.Series:
    """
    安全地把一列转成 datetime64[ns]：
      - 非法格式 / 溢出 → NaT（errors="coerce"）
      - year_min/year_max 外的年份也当脏数据 → NaT
    """
    s = pd.to_datetime(series, errors=errors)

    if year_min is not None or year_max is not None:
        mask = s.notna()
        years = s.dt.year
        bad = pd.Series(False, index=s.index)
        if year_min is not None:
            bad |= (years < year_min)
        if year_max is not None:
            bad |= (years > year_max)
        s.loc[bad & mask] = pd.NaT

    return s

def convert_with_schema(
    df: pd.DataFrame,
    schema: Dict[str, str],
    *,
    year_min: int | None = 1900,
    year_max: int | None = 2100,
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    按 schema 转换各列 dtype，带容错：
      - datetime64[ns] → safe_to_datetime（带年份裁剪）
      - Int64 / Float64 → to_numeric(errors="coerce") 再 astype
      - string → astype("string")
      - 其它 → 普通 astype(target)

    schema: {列名: "Int64"/"Float64"/"string"/"datetime64[ns]"/...}
    """
    out = df if inplace else df.copy()

    for col, target in schema.items():
        if col not in out.columns:
            if verbose:
                print(f"[convert_with_schema] skip '{col}': not in df")
            continue

        s = out[col]

        try:
            if target.startswith("datetime64"):
                out[col] = safe_to_datetime(s, year_min=year_min, year_max=year_max)

            elif target == "Int64":
                out[col] = pd.to_numeric(s, errors="coerce").astype("Int64")

            elif target == "Float64":
                out[col] = pd.to_numeric(s, errors="coerce").astype("Float64")

            elif target == "boolean":
                # 简易 bool 转换：0/1 + 常见字符串
                tmp = s.astype("string").str.strip().str.lower()
                tmp = tmp.replace(
                    {
                        "": pd.NA,
                        "nan": pd.NA,
                        "none": pd.NA,
                        "null": pd.NA,
                        "false": False,
                        "f": False,
                        "no": False,
                        "n": False,
                        "true": True,
                        "t": True,
                        "yes": True,
                        "y": True,
                    }
                )
                # 再把显式的 0/1 当成布尔
                tmp_num = pd.to_numeric(tmp, errors="ignore")
                if isinstance(tmp_num, pd.Series):
                    tmp = tmp_num
                tmp = tmp.map(
                    lambda v: True
                    if v in [1, True]
                    else (False if v in [0, False] else (pd.NA if pd.isna(v) else v))
                )
                out[col] = tmp.astype("boolean")

            elif target == "string":
                out[col] = s.astype("string")

            else:
                # fallback：你自己指定的奇怪 dtype
                out[col] = s.astype(target)

        except Exception as e:
            if verbose:
                print(f"[WARN] column '{col}' → {target} failed: {e}")

    return out


from typing import Optional, Tuple, Union, Dict, Any, Literal
import numpy as np
import pandas as pd

Num = Union[int, float]
Bounds = Tuple[Optional[Num], Optional[Num]]

def remove_outliers_column(
    df: pd.DataFrame,
    col: str,
    value_bounds: Bounds = (None, None),
    quantile_bounds: Bounds = (None, None),
    inclusive: bool = True,
    keep_na: bool = True,
    return_mask: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in df.")

    s = df[col]
    keep = pd.Series(True, index=df.index)

    is_na = s.isna()
    if not keep_na:
        keep &= ~is_na

    # Rule 1: value bounds
    v_low, v_high = value_bounds
    if not (v_low is None and v_high is None):
        not_na = ~is_na
        if v_low is not None:
            keep &= (~not_na) | ((s >= v_low) if inclusive else (s > v_low))
        if v_high is not None:
            keep &= (~not_na) | ((s <= v_high) if inclusive else (s < v_high))

    # Rule 2: quantile bounds
    q_low, q_high = quantile_bounds
    if not (q_low is None and q_high is None):
        for q, name in [(q_low, "q_low"), (q_high, "q_high")]:
            if q is not None and not (0.0 <= float(q) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {q}.")

        non_na_vals = s[~is_na]
        if len(non_na_vals) > 0:
            qv_low = non_na_vals.quantile(q_low) if q_low is not None else -np.inf
            qv_high = non_na_vals.quantile(q_high) if q_high is not None else np.inf
            not_na = ~is_na
            if inclusive:
                keep &= (~not_na) | ((s >= qv_low) & (s <= qv_high))
            else:
                keep &= (~not_na) | ((s > qv_low) & (s < qv_high))

    out = df.loc[keep].copy()
    return (out, keep) if return_mask else out


def remove_outliers_by_spec(
    df: pd.DataFrame,
    spec: Dict[str, Dict[str, Any]],
    combine: Literal["and", "or"] = "and",
    default_inclusive: bool = True,
    default_keep_na: bool = True,
    return_report: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Apply outlier removal rules for multiple columns based on a spec dict.

    spec example:
    {
      "colA": {"value_bounds": (0, 100), "quantile_bounds": (None, None)},
      "colB": {"quantile_bounds": (0.01, 0.99), "keep_na": False},
      "colC": {"value_bounds": (None, 10), "inclusive": False},
    }

    combine:
      - "and": keep rows that satisfy ALL column rules
      - "or":  keep rows that satisfy ANY column rule (rarely what you want)

    Returns:
      df_filtered (and optional report)
    """
    if combine not in ("and", "or"):
        raise ValueError("combine must be 'and' or 'or'")

    masks: Dict[str, pd.Series] = {}
    per_col_stats: Dict[str, Any] = {}

    for col, rules in spec.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df.")

        value_bounds: Bounds = rules.get("value_bounds", (None, None))
        quantile_bounds: Bounds = rules.get("quantile_bounds", (None, None))
        inclusive: bool = rules.get("inclusive", default_inclusive)
        keep_na: bool = rules.get("keep_na", default_keep_na)

        _, mask = remove_outliers_column(
            df,
            col,
            value_bounds=value_bounds,
            quantile_bounds=quantile_bounds,
            inclusive=inclusive,
            keep_na=keep_na,
            return_mask=True,
        )
        masks[col] = mask

        # Stats per column
        removed = int((~mask).sum())
        per_col_stats[col] = {
            "rows_removed": removed,
            "rows_kept": int(mask.sum()),
            "removed_frac": float(removed / len(df)) if len(df) else 0.0,
            "rules": {
                "value_bounds": value_bounds,
                "quantile_bounds": quantile_bounds,
                "inclusive": inclusive,
                "keep_na": keep_na,
            },
        }

    if not masks:
        # no rules given
        out = df.copy()
        report = {"rows_in": len(df), "rows_out": len(df), "rows_removed": 0, "per_column": {}}
        return (out, report) if return_report else out

    # Combine masks
    if combine == "and":
        final_mask = pd.Series(True, index=df.index)
        for m in masks.values():
            final_mask &= m
    else:  # "or"
        final_mask = pd.Series(False, index=df.index)
        for m in masks.values():
            final_mask |= m

    out = df.loc[final_mask].copy()

    if not return_report:
        return out

    report = {
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "rows_removed": int(len(df) - len(out)),
        "removed_frac": float((len(df) - len(out)) / len(df)) if len(df) else 0.0,
        "combine": combine,
        "per_column": per_col_stats,
    }
    return out, report
