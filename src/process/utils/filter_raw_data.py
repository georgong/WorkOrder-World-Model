import pandas as pd
from typing import Dict, Tuple

def drop_sparse_columns(
    df: pd.DataFrame,
    min_non_na_ratio: float = 0.5,
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    按非 NaN 占比清理列：
      - 计算每一列的 non-NA 比例: 1 - (num_na / len(df))
      - 若 non-NA 比例 < min_non_na_ratio，则丢掉该列

    参数
    ----
    df : DataFrame
    min_non_na_ratio : float
        最小允许的非 NaN 比例，比如 0.7 表示至少 70% 非 NaN，否则删列。
    inplace : bool
        是否在原 df 上修改。
    verbose : bool
        打印被删的列和比例。

    返回
    ----
    cleaned_df : DataFrame
    """
    _df = df if inplace else df.copy()

    n_rows = len(_df)
    if n_rows == 0:
        if verbose:
            print("[drop_sparse_columns] empty df, nothing to do.")
        return _df

    # 每列 NaN 占比 / 非 NaN 占比
    na_ratio = _df.isna().sum() / n_rows
    non_na_ratio = 1.0 - na_ratio

    # 要丢的列：非 NaN 占比 < 阈值
    to_drop = non_na_ratio[non_na_ratio < min_non_na_ratio].index.tolist()

    if verbose:
        if to_drop:
            print(f"[drop_sparse_columns] dropping {len(to_drop)} columns "
                  f"(min_non_na_ratio={min_non_na_ratio}):")
            for col in to_drop:
                print(f"  - {col}: non_na_ratio={non_na_ratio[col]:.3f}")
        else:
            print(f"[drop_sparse_columns] no columns dropped "
                  f"(min_non_na_ratio={min_non_na_ratio}).")

    if to_drop:
        _df = _df.drop(columns=to_drop)

    return _df


def drop_sparse_columns_for_all(
    dfs: Dict[str, pd.DataFrame],
    min_non_na_ratio: float = 0.5,
    inplace: bool = False,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    对多个表一起按非 NaN 占比清理列。
    返回新的 dict（除非 inplace=True）。
    """
    out = {} if not inplace else dfs
    for name, df in dfs.items():
        if verbose:
            print(f"\n[drop_sparse_columns_for_all] table: {name}")
        cleaned = drop_sparse_columns(
            df,
            min_non_na_ratio=min_non_na_ratio,
            inplace=inplace,
            verbose=verbose,
        )
        if not inplace:
            out[name] = cleaned
    return out
