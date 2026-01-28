from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml


# repo 根目录
BASE_DIR = Path(__file__).resolve().parents[1]

#  click 2 
CLICK2_DIR_NAME = "click 2"

# 输出位置（都在 repo 内）
OUT_RAW_DIR = BASE_DIR / "data" / "raw"
OUT_PROFILE_DIR = BASE_DIR / "data" / "profiles"
OUT_YAML = BASE_DIR / "configs" / "data.yaml"


# =====================================================
# Profiling params
# =====================================================
IQR_K = 1.5
NUMERIC_RATIO_THRESHOLD = 0.8  # 可转 numeric 的比例 >= 0.8 才视为 numeric-like
MIN_NUMERIC_COUNT = 50         # numeric 有效值至少 50 才算


# =====================================================
# Helpers
# =====================================================

def find_click2_dir() -> Path:
    """
    在 repo 内搜索名为 'click 2' 的目录。
    - 若找到 1 个：返回
    - 若找到 0 个：报错
    - 若找到多个：报错并列出候选，避免拿错数据
    """
    candidates = [p for p in BASE_DIR.rglob(CLICK2_DIR_NAME) if p.is_dir()]

    if len(candidates) == 0:
        raise RuntimeError(
            f'Cannot find folder named "{CLICK2_DIR_NAME}" under repo root:\n  {BASE_DIR}\n'
            f'Please confirm the folder exists in the repo.'
        )

    if len(candidates) > 1:
        msg = "\n".join([f"  - {c}" for c in candidates])
        raise RuntimeError(
            f'Found MULTIPLE folders named "{CLICK2_DIR_NAME}".\n'
            f'Please rename one, or hardcode the correct one.\n'
            f'Candidates:\n{msg}'
        )

    return candidates[0]


def base_name_from_file(p: Path) -> str:
    """
    W6TASKS-14.csv -> W6TASKS
    W6TASK_TYPES-0.csv -> W6TASK_TYPES
    """
    stem = p.stem
    m = re.match(r"^(.*?)-\d+$", stem)
    return m.group(1) if m else stem


def read_csv_robust(p: Path) -> pd.DataFrame:
    """
    读取 CSV 并做最小清洗：
    - 编码兜底
    - 去掉列名首尾空格
    """
    try:
        df = pd.read_csv(p, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(p, low_memory=False, encoding="latin1")

    df.columns = [c.strip() for c in df.columns]
    return df


def concat_parts(files: List[Path]) -> pd.DataFrame:
    """
    concat 同组分片：
    - sort=True: 允许不同分片列不一致时按列名对齐
    """
    dfs = []
    for f in sorted(files, key=lambda x: x.name):
        dfs.append(read_csv_robust(f))
    return pd.concat(dfs, ignore_index=True, sort=True)


def infer_type_and_yaml_dtype(s: pd.Series) -> Tuple[str, str]:
    """
    返回 (inferred_type, yaml_dtype)
    inferred_type ∈ {numeric, datetime, categorical}
    """
    pd_dtype = str(s.dtype).lower()

    # numeric-like?
    num = pd.to_numeric(s, errors="coerce")
    non_na = int(s.notna().sum())
    num_non_na = int(num.notna().sum())
    num_ratio = (num_non_na / non_na) if non_na else 0.0

    if num_ratio >= NUMERIC_RATIO_THRESHOLD and num_non_na >= MIN_NUMERIC_COUNT:
        if "int" in pd_dtype:
            return "numeric", "Int64"
        return "numeric", "Float64"

    # datetime-like?（只对 object 尝试）
    if s.dtype == "object":
        dt = pd.to_datetime(s, errors="coerce")
        if float(dt.notna().mean()) >= 0.8:
            return "datetime", "datetime64[ns]"

    return "categorical", "string"


def iqr_bounds(num: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    数值列 outlier 初筛：IQR 方法
    返回 (low, high, outlier_ratio)
    """
    x = num.dropna()
    if len(x) < MIN_NUMERIC_COUNT:
        return None, None, None

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return None, None, 0.0

    low = float(q1 - IQR_K * iqr)
    high = float(q3 + IQR_K * iqr)
    ratio = float(((num < low) | (num > high)).mean())
    return low, high, ratio


def profile_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    输出每列体检表：
    - inferred_type, yaml_dtype
    - missing_ratio, n_unique, unique_ratio, top5
    - numeric: IQR outlier bounds & ratio
    """
    n = len(df)
    rows = []

    for col in df.columns:
        s = df[col]

        inferred, yaml_dtype = infer_type_and_yaml_dtype(s)
        missing_ratio = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        unique_ratio = (nunique / n) if n else np.nan

        top_vals = s.value_counts(dropna=True).head(5)
        top5 = "; ".join([f"{k}:{int(v)}" for k, v in top_vals.items()]) if len(top_vals) else ""

        low = high = out_ratio = None
        if inferred == "numeric":
            num = pd.to_numeric(s, errors="coerce")
            low, high, out_ratio = iqr_bounds(num)

        rows.append({
            "dataset": name,
            "column": col,
            "inferred_type": inferred,
            "yaml_dtype": yaml_dtype,
            "missing_ratio": missing_ratio,
            "n_unique": nunique,
            "unique_ratio": unique_ratio,
            "top5": top5,
            "iqr_low": low,
            "iqr_high": high,
            "iqr_outlier_ratio": out_ratio,
        })

    return pd.DataFrame(rows)


# =====================================================
# Main
# =====================================================

def main():
    # 1) 找到 click 2
    input_dir = find_click2_dir()

    print("=== Sanity Check ===")
    print("Repo BASE_DIR:", BASE_DIR)
    print("Auto INPUT_DIR:", input_dir)
    print("INPUT_DIR exists:", input_dir.exists())

    csv_files = list(input_dir.glob("*.csv"))
    print("CSV count:", len(csv_files))
    if len(csv_files) == 0:
        # 顺便列一下目录内容帮助 debug
        sample = [p.name for p in list(input_dir.iterdir())[:50]]
        raise RuntimeError(
            f"No CSV files found in {input_dir}\n"
            f"First files/dirs under INPUT_DIR:\n  {sample}"
        )

    # 2) 输出目录
    OUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)

    # 3) 按 base name 分组
    groups: Dict[str, List[Path]] = {}
    for f in csv_files:
        groups.setdefault(base_name_from_file(f), []).append(f)

    print("\n=== Group Summary ===")
    for k, v in sorted(groups.items()):
        print(f"  - {k}: {len(v)} parts")

    # 4) 处理每组：concat + profile + yaml vars
    datasets_cfg = {}

    for name, files in sorted(groups.items()):
        print(f"\n=== Processing {name} ({len(files)} parts) ===")
        df = concat_parts(files)

        out_csv = OUT_RAW_DIR / f"{name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved merged CSV: {out_csv}  shape={df.shape}")

        prof = profile_dataframe(df, name)
        out_prof = OUT_PROFILE_DIR / f"{name}_profile.csv"
        prof.to_csv(out_prof, index=False)
        print(f"Saved profile: {out_prof}")

        variables = {}
        for _, r in prof.iterrows():
            outlier = [None, None]
            if r["inferred_type"] == "numeric" and pd.notna(r["iqr_low"]) and pd.notna(r["iqr_high"]):
                outlier = [float(r["iqr_low"]), float(r["iqr_high"])]

            variables[str(r["column"])] = {
                "dtype": str(r["yaml_dtype"]),
                "key": False,
                "mask": False,
                "outlier": outlier,
            }

        datasets_cfg[name.lower()] = {
            "file": f"data/raw/{name}.csv",
            "variables": variables,
        }

    # 5) write data.yaml
    with open(OUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump({"datasets": datasets_cfg}, f, sort_keys=False, allow_unicode=True)

    print(f"\n✅ Generated: {OUT_YAML}")
    print("Next: edit configs/data.yaml to set key/mask + adjust outlier if needed.")


if __name__ == "__main__":
    main()
