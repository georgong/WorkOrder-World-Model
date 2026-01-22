from logging import Logger
from typing import Any, Dict, List, Optional, Tuple
import polars as pl
from torch_geometric.data import HeteroData
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
import yaml
from .feature_engineering import process_assignment_feature,process_districts_feature,process_engineer_feature,process_task_feature
from .feature_schema import assignment_schema,district_schema,engineer_schema,task_schema
DTYPE_MAP = {
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "string": pl.Utf8,
    "datetime64[ns]": pl.Datetime("ns"),
}

NULL_VALUES = ["", " ", "NA", "N/A", "NULL", "null", "None"]



TYPE_DIR_MAP = {
    "tasks": "W6TASKS",
    "assignments": "W6ASSIGNMENTS",
    "engineers": "W6ENGINEERS",
    "districts": "W6DISTRICTS",
    "departments": "W6DEPARTMENT",
    "task_statuses": "W6TASK_STATUSES",
    "task_types": "W6TASK_TYPES",
    "equipment": "W6EQUIPMENT",
    "regions": "W6REGIONS",
}

preprocessing_dict = {
    "tasks":{"func":process_task_feature,"schema":task_schema},
    "assignments":{"func":process_assignment_feature,"schema":assignment_schema},
    "districts":{"func":process_districts_feature,"schema":district_schema},
    "engineers":{"func":process_engineer_feature,"schema":engineer_schema},
}
def _resolve_files(data_dir: Path, base: str) -> List[Path]:
    """
    Prefer sharded files: BASE-*.csv
    Fallback: BASE.csv
    """
    shards = sorted(data_dir.glob(f"{base}-*.csv"))
    if shards:
        return shards
    single = data_dir / f"{base}.csv"
    if single.exists():
        return [single]
    # last fallback: maybe user wrote lowercase or weird
    return []

def load_table(schema: Dict[str, Any], df_type: str, data_dir: str = "data/raw") -> pl.DataFrame:
    assert df_type in TYPE_DIR_MAP, f"df_type must be one of {sorted(TYPE_DIR_MAP.keys())} but got {df_type!r}"

    data_dir_p = Path(data_dir)
    base = TYPE_DIR_MAP[df_type]
    files = _resolve_files(data_dir_p, base)
    if not files:
        raise FileNotFoundError(f"No files found for {df_type=} under {data_dir_p} (expected {base}.csv or {base}-*.csv)")

    # schema path: schema["datasets"][df_type]["variables"]  (your earlier layout)
    vars_meta = schema["datasets"][df_type]["variables"]
    cols = list(vars_meta.keys())

    pl_schema = {
        col: DTYPE_MAP.get(meta.get("dtype", "string"), pl.Utf8)
        for col, meta in vars_meta.items()
    }

    dfs: List[pl.DataFrame] = []
    for f in files:
        df = pl.read_csv(
            f,
            columns=cols,                     # only keep known columns
            schema_overrides=pl_schema,        # enforce dtypes
            null_values=NULL_VALUES,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )

        # ensure all requested columns exist (some shards may miss a column)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None).cast(pl_schema[c]).alias(c) for c in missing]).select(cols)
        else:
            df = df.select(cols)

        dfs.append(df)

    df_concat = pl.concat(dfs, how="vertical", rechunk=True)
    return df_concat


def _copy_cols_as_df(df: pl.DataFrame, cols: List[str], prefix: str) -> pl.DataFrame:
    """
    Explicitly copy columns into a new DataFrame to avoid name collisions.
    Missing columns are ignored.
    """
    cols_exist = [c for c in cols if c in df.columns]
    data = {f"{prefix}{c}": df[c].to_list() for c in cols_exist}
    return pl.DataFrame(data)


def build_edge_index(
    *,
    left_df: pl.DataFrame,
    left_join_col: str,
    right_df: pl.DataFrame,
    right_join_col: str,
    left_entity_key: str,
    right_entity_key: str,
    keep_edge_order_from: str = "left",  # "left" or "right"
    left_edge_attr: Optional[List[str]] = None,
    right_edge_attr: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pl.DataFrame]:
    """
    Returns:
      edge_index: [2, E] long
      edge_attr:  [E, F] float (F can be 0 -> returns [E,1] placeholder)
      src_idx:    [E] long
      dst_idx:    [E] long
      edges_df:   Polars DF with src_key,dst_key,src_idx,dst_idx (+ optional edge attr columns for debug)
    """

    left_edge_attr = left_edge_attr or []
    right_edge_attr = right_edge_attr or []

    # --- 1) node idx space: key -> idx (stable via sort on key) ---
    left_k2i = (
        left_df.select(left_entity_key)
               .drop_nulls()
               .unique()
               .sort(left_entity_key)
               .with_row_index("src_idx")
               .rename({left_entity_key: "src_key"})
    )

    right_k2i = (
        right_df.select(right_entity_key)
                .drop_nulls()
                .unique()
                .sort(right_entity_key)
                .with_row_index("dst_idx")
                .rename({right_entity_key: "dst_key"})
    )

    # --- 2) build minimal copies for join (avoid name collisions) + attach attr by _rid ---
    if keep_edge_order_from == "left":
        # base left edge rows: (rid, joined, src_key)
        L_base = pl.DataFrame({
            "_rid": list(range(left_df.height)),
            "joined": left_df[left_join_col].to_list(),
            "src_key": left_df[left_entity_key].to_list(),
        }).drop_nulls(["joined", "src_key"])

        # left edge attrs copied by row order, then joined on _rid to match the filtered L_base
        if left_edge_attr:
            L_attr = _copy_cols_as_df(left_df, left_edge_attr, prefix="l__").with_row_index("_rid")
            L = L_base.join(L_attr, on="_rid", how="left")
        else:
            L = L_base

        # right side for matching destination keys
        R = pl.DataFrame({
            "joined": right_df[right_join_col].to_list(),
            "dst_key": right_df[right_entity_key].to_list(),
        }).drop_nulls(["joined", "dst_key"])

        # join and restore order by left _rid
        J = L.join(R, on="joined", how="inner").sort("_rid")

        # right attrs (node-level-ish) if you insist: map by dst_key later (NOT by rid)
        # If you truly want right attrs per edge instance, you need them to come from the left/event table.
        # We'll support right_edge_attr as "dst-node attrs" via dst_key join after k2i mapping.
        attach_right_by = "dst_key"

    else:
        # symmetric: keep order from right table
        R_base = pl.DataFrame({
            "_rid": list(range(right_df.height)),
            "joined": right_df[right_join_col].to_list(),
            "dst_key": right_df[right_entity_key].to_list(),
        }).drop_nulls(["joined", "dst_key"])

        if right_edge_attr:
            R_attr = _copy_cols_as_df(right_df, right_edge_attr, prefix="r__").with_row_index("_rid")
            R = R_base.join(R_attr, on="_rid", how="left")
        else:
            R = R_base

        L = pl.DataFrame({
            "joined": left_df[left_join_col].to_list(),
            "src_key": left_df[left_entity_key].to_list(),
        }).drop_nulls(["joined", "src_key"])

        J = R.join(L, on="joined", how="inner").sort("_rid")
        attach_right_by = None  # already in J if we kept order from right

    if J.height == 0:
        raise ValueError("Join produced no edges. Check join columns overlap.")

    # --- 3) map keys -> idx ---
    edges_df = (
        J.join(left_k2i, on="src_key", how="inner")
         .join(right_k2i, on="dst_key", how="inner")
    )

    if edges_df.height == 0:
        raise ValueError("After mapping keys->idx, edges are empty. Check key domains.")

    # --- 4) build edge_index ---
    src_idx = torch.tensor(edges_df["src_idx"].to_list(), dtype=torch.long)
    dst_idx = torch.tensor(edges_df["dst_idx"].to_list(), dtype=torch.long)
    edge_index = torch.stack([src_idx, dst_idx], dim=0)

    # --- 5) build edge_attr ---
    # left-side attrs are already in edges_df (prefixed l__)
    attr_cols = [c for c in edges_df.columns if c.startswith("l__") or c.startswith("r__")]

    # Optional: if keep_edge_order_from=="left" and you want right_edge_attr as dst-node attrs,
    # join them by dst_key (one dst row per key)
    if keep_edge_order_from == "left" and right_edge_attr:
        # Build dst_key -> attrs mapping table (unique by dst_key)
        Rn = pl.DataFrame({
            "dst_key": right_df[right_entity_key].to_list(),
        })
        Rattrs = _copy_cols_as_df(right_df, right_edge_attr, prefix="r__")
        Rmap = pl.concat([Rn, Rattrs], how="horizontal").drop_nulls(["dst_key"]).unique(subset=["dst_key"])
        edges_df = edges_df.join(Rmap, on="dst_key", how="left")
        attr_cols = [c for c in edges_df.columns if c.startswith("l__") or c.startswith("r__")]

    if len(attr_cols) == 0:
        edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
    else:
        # Convert to numeric; non-numeric columns must be preprocessed before here.
        attr_np = edges_df.select(attr_cols).to_numpy()
        edge_attr = torch.tensor(attr_np, dtype=torch.float)

    # keep a clean debug view
    debug_cols = ["src_key", "dst_key", "src_idx", "dst_idx"] + attr_cols
    edges_df = edges_df.select([c for c in debug_cols if c in edges_df.columns])

    return edge_index, edge_attr, src_idx, dst_idx, edges_df

def split_tables_by_trait(
    schema: Dict[str, Any],
    table_dict: Dict[str, pl.DataFrame],
    *,
    include_key_cols_in_both: bool = True,
    strict: bool = False,
    treat_null_trait_as: Optional[str] = None,  # None | "node" | "edge"
) -> Dict[str, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Split each table into (node_df, edge_df) based on schema trait_type.

    Args:
        schema: dict loaded from YAML. Expected schema["datasets"][table_name]["variables"][col] -> info dict
        table_dict: {table_name: pl.DataFrame}
        include_key_cols_in_both: if True, columns with key:true are included in both node_df and edge_df.
        strict: if True, raise if schema refers to a column missing in the DataFrame.
        treat_null_trait_as: if trait_type is null/None/missing, optionally treat as "node" or "edge".
                             If None, those cols are ignored (unless key:true and include_key_cols_in_both).

    Returns:
        {table_name: (node_df, edge_df)}
    """

    if "datasets" not in schema:
        raise KeyError("schema missing top-level 'datasets'")

    out: Dict[str, Tuple[pl.DataFrame, pl.DataFrame]] = {}

    for table_name, df in table_dict.items():
        if table_name not in schema["datasets"]:
            if strict:
                raise KeyError(f"schema missing dataset entry for table {table_name!r}")
            # still return empty splits with original df columns ignored
            out[table_name] = (df.head(0), df.head(0))
            continue

        vars_cfg: Dict[str, Dict[str, Any]] = schema["datasets"][table_name].get("variables", {})
        if not isinstance(vars_cfg, dict):
            raise TypeError(f"schema['datasets'][{table_name!r}]['variables'] must be a dict")

        node_cols: List[str] = []
        edge_cols: List[str] = []

        for col, info in vars_cfg.items():
            if col not in df.columns:
                if strict:
                    raise KeyError(f"Table {table_name!r} missing column {col!r} required by schema")
                continue

            is_key = bool(info.get("key", False))
            trait = info.get("trait_type", None)

            if trait is None and treat_null_trait_as is not None:
                trait = treat_null_trait_as

            # Key columns: usually keep everywhere for join/traceability
            if is_key and include_key_cols_in_both:
                if col not in node_cols:
                    node_cols.append(col)
                if col not in edge_cols:
                    edge_cols.append(col)
                # do not "continue": a key col could still be explicitly node/edge, but we've already included it
                continue

            if trait == "node":
                node_cols.append(col)
            elif trait == "edge":
                edge_cols.append(col)
            else:
                # trait_type is null or unknown: ignore by default
                pass

        # Make sure we don't duplicate columns
        node_cols = list(dict.fromkeys(node_cols))
        edge_cols = list(dict.fromkeys(edge_cols))

        out[table_name] = (node_cols, edge_cols)

    return out


class GraphBuilder:
    def __init__(self,yaml):
        self.yaml = yaml
        self.build_graph()
        self.logger = self.setup_logger()

    def build_graph(self):
        #----
        #load tables first
        #----   
        tasks = load_table(self.yaml, "tasks")
        assignments = load_table(self.yaml, "assignments")
        engineers = load_table(self.yaml, "engineers")
        districts = load_table(self.yaml, "districts")
        departments = load_table(self.yaml, "departments")
        #----
        #construct each entity attribute by group by thourgh entity key
        #expect more smart aggregate function instead of first() in the future
        #----   
        # ---
        # split the table into two parts: node features and edge features
        # ---
        TABLE_DICT = {
        "tasks": tasks,
        "assignments": assignments,
        "engineers": engineers,
        "districts": districts,
        "departments": departments,
        #"task_statuses": task_statuses,
        }
        splits = split_tables_by_trait(self.yaml, TABLE_DICT, include_key_cols_in_both=False)
        tasks_node, tasks_edge = splits["tasks"]
        assign_node, assign_edge = splits["assignments"]
        engineer_node, engineer_edge = splits["engineers"]
        district_node, district_edge = splits["districts"]
        department_node, department_edge = splits["departments"]
        ## 这里是特征字典
        #TODO: change the placement tensor into real feature tensor after feature engineering
        feature_dicts = {
            "tasks": (
                list(tasks_node),      # node feature columns
                list(tasks_edge),      # edge feature columns
            ),
            "assignments": (
                list(assign_node),
                list(assign_edge),
            ),
            "engineers": (
                list(engineer_node),
                list(engineer_edge),
            ),
            "districts": (
                list(district_node),
                list(district_edge),
            ),
            "departments": (
                list(department_node),
                list(department_edge),
            ),
        }

        # ---
        # feature engineering for each node and edge type
        # TODO: implement feature engineering functions
        # ---

        #---
        # construct hetero graph
        #---
        self.graph = HeteroData()

        # nodes
        for table_name, df in TABLE_DICT.items():
            feature = preprocessing_dict[table_name]["func"](TABLE_DICT[table_name],preprocessing_dict[table_name]["schema"])
            self.graph[table_name].x = feature

        # edges
        for src_table, metadata in self.yaml["mappings"].items():
            if "links" not in metadata or metadata["links"] is None:
                continue

            for dst_table, link_info in metadata["links"].items():
                left_on = link_info["left_on"]
                right_on = link_info["right_on"]
                edge_type = link_info.get("edge_type", "relates_to")

                edge_index, edge_attr, src_idx, dst_idx, edges_df = build_edge_index(
                    left_df=TABLE_DICT[src_table],
                    left_join_col=left_on,
                    right_df=TABLE_DICT[dst_table],
                    right_join_col=right_on,
                    left_entity_key=self.yaml["mappings"][src_table]["entity_key"],
                    right_entity_key=self.yaml["mappings"][dst_table]["entity_key"],
                    left_edge_attr=feature_dicts[src_table][1],
                    right_edge_attr=feature_dicts[dst_table][1],
                )

                rel = (src_table, edge_type, dst_table)
                self.graph[rel].edge_index = edge_index

                # ✅ edge_attr: per-edge, gathered by src_idx/dst_idx
                src_edge_feat = feature_dicts[src_table][1].float()  # [N_src, F_src_edge]
                dst_edge_feat = feature_dicts[dst_table][1].float()  # [N_dst, F_dst_edge]

                # 若某一侧没有 edge 特征（shape[1]==0），就只用另一侧
                parts = []
                if src_edge_feat.numel() > 0 and src_edge_feat.shape[1] > 0:
                    parts.append(src_edge_feat[src_idx])  # [E, F_src_edge]
                if dst_edge_feat.numel() > 0 and dst_edge_feat.shape[1] > 0:
                    parts.append(dst_edge_feat[dst_idx])  # [E, F_dst_edge]

                if len(parts) == 0:
                    # 占位，至少保证 edge_attr 存在且行数对齐
                    self.graph[rel].edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
                else:
                    self.graph[rel].edge_attr = torch.cat(parts, dim=1)  # [E, F_total]

                # reverse
                rev_rel = (dst_table, f"rev_{edge_type}", src_table)
                self.graph[rev_rel].edge_index = edge_index.flip(0)
                self.graph[rev_rel].edge_attr = self.graph[rel].edge_attr  # 同序，直接复用

        print(self.graph)




        pass

    def visualize_node(self,node_type,node_id):
        pass

    def setup_logger(self):
        logger = Logger(__name__)
        return logger
    
    def visualize_graph(self):
        pass


if __name__ == "__main__":
    with open("configs/graph.yaml", "r") as f:
        schema = yaml.safe_load(f)
    graph = GraphBuilder(yaml=schema)

    
