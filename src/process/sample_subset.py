"""
Sample a small, connected subset of the raw data for use as an example dataset.
The sampled data will preserve relationships defined in graph.yaml and be stored in separate CSVs.

By default, sampling is restricted to the TEST SPLIT of the trained model so that
dashboard demos never leak training data.

The sampled subset will be used as example passed in the dashboard.

Usage:
    python -m src.process.sample_subset \
        --raw_dir data/raw \
        --yaml configs/graph.yaml \
        --out_dir data/sample \
        --n_work_orders 5000 \
        --seed 42

    # To also use the graph .pt for test-set-only sampling:
    python -m src.process.sample_subset \
        --raw_dir data/raw \
        --yaml configs/graph.yaml \
        --pt data/graph/sdge.pt \
        --out_dir data/sample \
        --n_work_orders 500 \
        --seed 42
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import polars as pl
import yaml
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData

# Constants
DTYPE_MAP = {
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "string": pl.Utf8,
    "datetime64[ns]": pl.Datetime("ns"),
}
NULL_VALUES = ["", " ", "NA", "N/A", "NULL", "null", "None"]

# All edges that MUST exist in the sampled subset
REQUIRED_EDGES: List[Tuple[str, str, str]] = [
    ("tasks", "relates_to", "assignments"),
    ("tasks", "relates_to", "task_statuses"),
    ("tasks", "relates_to", "task_types"),
    ("tasks", "relates_to", "districts"),
    ("tasks", "relates_to", "departments"),
    ("assignments", "relates_to", "tasks"),
    ("assignments", "relates_to", "engineers"),
    ("engineers", "relates_to", "assignments"),
    ("engineers", "relates_to", "districts"),
    ("engineers", "relates_to", "departments"),
    ("districts", "relates_to", "tasks"),
    ("districts", "relates_to", "engineers"),
    ("departments", "relates_to", "tasks"),
    ("departments", "relates_to", "engineers"),
    ("task_statuses", "relates_to", "tasks"),
    ("task_types", "relates_to", "tasks"),
]


# ---------------------------------------------------------------------------
# Reproduce the train.py split logic
# ---------------------------------------------------------------------------

def compute_target_degree(data: HeteroData, target: str, *, degree_mode: str = "in") -> torch.Tensor:
    """Identical to train.py's compute_target_degree."""
    N = data[target].num_nodes
    deg = torch.zeros(N, dtype=torch.long)

    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if degree_mode in ("in", "inout") and dst == target:
            deg += torch.bincount(ei[1], minlength=N)
        if degree_mode in ("out", "inout") and src == target:
            deg += torch.bincount(ei[0], minlength=N)

    return deg


def split_indices(
    idx: torch.Tensor, *, seed: int, train_ratio: float, val_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identical to train.py's split_indices."""
    assert idx.ndim == 1
    N = idx.numel()
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(N, generator=g)]

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def get_test_set_indices(
    pt_path: Path,
    target: str = "assignments",
    min_degree: int = 1,
    degree_mode: str = "in",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> torch.Tensor:
    """
    Load the .pt graph and reproduce the exact split from train.py.
    Returns the test-set node indices (in graph index space).
    """
    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData), f"Expected HeteroData, got {type(data)}"
    assert target in data.node_types, f"Target '{target}' not in node types: {data.node_types}"

    deg = compute_target_degree(data, target, degree_mode=degree_mode)
    kept = (deg >= min_degree).nonzero(as_tuple=False).view(-1)

    if kept.numel() == 0:
        raise ValueError(
            f"No target nodes left after degree filter (min_degree={min_degree}, mode={degree_mode})"
        )

    _, _, test_idx = split_indices(
        kept, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio
    )

    print(
        f"[split] target={target} total={data[target].num_nodes} "
        f"kept={kept.numel()} test={test_idx.numel()}"
    )
    return test_idx, data


def graph_indices_to_entity_ids(
    data: HeteroData,
    indices: torch.Tensor,
    target: str,
    schema: Dict,
) -> Set:
    """
    Map graph node indices back to the original entity IDs (e.g. assignment IDs).

    The graph builder typically stores a mapping from graph index → original ID.
    Common attribute names: 'node_id', 'ids', or the entity_key column name.
    If no mapping is stored, we fall back to using the integer indices themselves.
    """
    store = data[target]
    indices_list = indices.tolist()

    # Try common attribute names for stored ID mappings
    for attr in ("node_id", "ids", "raw_id", "original_id"):
        if hasattr(store, attr):
            id_map = getattr(store, attr)
            if isinstance(id_map, torch.Tensor):
                return set(id_map[indices].tolist())
            elif isinstance(id_map, (list, tuple)):
                return {id_map[i] for i in indices_list}
            elif isinstance(id_map, dict):
                # dict: graph_idx -> original_id
                return {id_map[i] for i in indices_list if i in id_map}

    # Try entity_key from schema as attribute name
    entity_key = schema["mappings"].get(target, {}).get("entity_key")
    if entity_key and hasattr(store, entity_key):
        id_map = getattr(store, entity_key)
        if isinstance(id_map, torch.Tensor):
            return set(id_map[indices].tolist())
        elif isinstance(id_map, (list, tuple)):
            return {id_map[i] for i in indices_list}

    # Fallback: use integer indices directly
    print(
        f"[warn] No ID mapping found on graph node store '{target}'. "
        f"Falling back to integer graph indices as IDs."
    )
    return set(indices_list)


# ---------------------------------------------------------------------------
# File / table helpers
# ---------------------------------------------------------------------------

def resolve_files(data_dir: Path, base: str) -> List[Path]:
    """Resolve CSV files for a given dataset."""
    shards = sorted(data_dir.glob(f"{base}-*.csv"))
    if shards:
        return shards
    single = data_dir / f"{base}.csv"
    if single.exists():
        return [single]
    return []


def load_raw_table(schema: Dict[str, any], table_name: str, raw_dir: Path) -> pl.DataFrame:
    """Load a raw CSV table using schema dtypes."""
    base = schema["datasets"][table_name]["file"].split("/")[-1].replace(".csv", "")
    files = resolve_files(raw_dir, base)
    if not files:
        print(f"[warn] No files found for {table_name} ({base}.csv) — skipping")
        return pl.DataFrame()

    vars_meta = schema["datasets"][table_name]["variables"]
    cols = list(vars_meta.keys())
    pl_schema = {
        col: DTYPE_MAP.get(meta.get("dtype", "string"), pl.Utf8)
        for col, meta in vars_meta.items()
    }

    dfs = []
    for f in files:
        df = pl.read_csv(
            f,
            columns=cols,
            schema_overrides=pl_schema,
            null_values=NULL_VALUES,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        missing = [c for c in cols if c not in df.columns]
        if missing:
            df = df.with_columns(
                [pl.lit(None).cast(pl_schema[c]).alias(c) for c in missing]
            ).select(cols)
        else:
            df = df.select(cols)
        dfs.append(df)

    return pl.concat(dfs, how="vertical", rechunk=True)


def _collect_unique_ids(df: pl.DataFrame, col: str) -> Set:
    """Extract unique non-null values from a column."""
    if col not in df.columns:
        return set()
    return set(df[col].drop_nulls().unique().to_list())


def _get_link_columns(schema: Dict, src_table: str, dst_table: str) -> Tuple[str, str]:
    """
    From the mappings section, find the (src_col, dst_col) that links
    src_table → dst_table.

    Returns (column_in_src, column_in_dst) or raises KeyError.
    """
    mapping = schema["mappings"].get(src_table)
    if mapping:
        links = mapping.get("links", {})
        if dst_table in links:
            link = links[dst_table]
            # left_on is the column in src_table, right_on is the column in dst_table
            return link["left_on"], link["right_on"]

    # Try the reverse direction: dst_table's mapping may link to src_table
    mapping = schema["mappings"].get(dst_table)
    if mapping:
        links = mapping.get("links", {})
        if src_table in links:
            link = links[src_table]
            # In the reverse lookup, left_on is in dst_table, right_on is in src_table
            return link["right_on"], link["left_on"]

    raise KeyError(
        f"No link found between '{src_table}' and '{dst_table}' in mappings"
    )


def sample_connected_subset(
    schema: Dict[str, any],
    raw_data: Dict[str, pl.DataFrame],
    n_work_orders: int,
    seed: int,
    test_only_ids: Optional[Set] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Sample a connected subset of the raw data ensuring that ALL edges in
    REQUIRED_EDGES have at least one instance in the final data.

    If test_only_ids is provided, sampling is restricted to assignments
    whose entity key is in that set (i.e. the test split from training).

    Strategy:
      1. Sample N assignments (from test set if provided).
      2. Pull all tasks linked to those assignments.
      3. From tasks, pull linked task_statuses, task_types, districts, departments.
      4. From assignments, pull linked engineers.
      5. From engineers, pull linked districts and departments (union with step 3).
      6. Validate every required edge has coverage.
    """
    import random
    random.seed(seed)

    filtered: Dict[str, pl.DataFrame] = {}

    # --- Helper to get entity key for a table ---
    def entity_key(table: str) -> str:
        return schema["mappings"][table]["entity_key"]

    # ------------------------------------------------------------------
    # Step 1: Sample assignments (restricted to test set if available)
    # ------------------------------------------------------------------
    assignments_all = raw_data["assignments"]

    if test_only_ids is not None:
        asgn_ek = entity_key("assignments")
        # Restrict to test-set assignments only
        assignments_pool = assignments_all.filter(pl.col(asgn_ek).is_in(list(test_only_ids)))
        print(
            f"  [test-set filter] {assignments_all.height} total assignments → "
            f"{assignments_pool.height} in test set"
        )
        if assignments_pool.height == 0:
            raise ValueError(
                "No assignments matched the test-set IDs. Check that the .pt graph "
                "uses the same entity key as the raw CSVs, or inspect the ID mapping."
            )
    else:
        assignments_pool = assignments_all

    if assignments_pool.height <= n_work_orders:
        filtered["assignments"] = assignments_pool
    else:
        filtered["assignments"] = assignments_pool.sample(n=n_work_orders, seed=seed)

    print(f"  assignments: sampled {filtered['assignments'].height} rows"
          f"{' (test-set only)' if test_only_ids is not None else ''}")

    # ------------------------------------------------------------------
    # Step 2: Pull tasks linked to sampled assignments
    #   Edge: assignments → tasks  AND  tasks → assignments
    # ------------------------------------------------------------------
    asgn_to_task_src, asgn_to_task_dst = _get_link_columns(schema, "assignments", "tasks")
    task_ids_from_assignments = _collect_unique_ids(filtered["assignments"], asgn_to_task_src)

    tasks_all = raw_data["tasks"]
    tasks_ek = entity_key("tasks")
    filtered["tasks"] = tasks_all.filter(pl.col(tasks_ek).is_in(task_ids_from_assignments))
    print(f"  tasks: {filtered['tasks'].height} rows linked to assignments")

    # ------------------------------------------------------------------
    # Step 3: Pull engineers linked to sampled assignments
    #   Edge: assignments → engineers  AND  engineers → assignments
    # ------------------------------------------------------------------
    asgn_to_eng_src, asgn_to_eng_dst = _get_link_columns(schema, "assignments", "engineers")
    eng_ids_from_assignments = _collect_unique_ids(filtered["assignments"], asgn_to_eng_src)

    engineers_all = raw_data["engineers"]
    engineers_ek = entity_key("engineers")
    filtered["engineers"] = engineers_all.filter(pl.col(engineers_ek).is_in(eng_ids_from_assignments))
    print(f"  engineers: {filtered['engineers'].height} rows linked to assignments")

    # ------------------------------------------------------------------
    # Step 4: Pull task_statuses linked to sampled tasks
    #   Edge: tasks → task_statuses  AND  task_statuses → tasks
    # ------------------------------------------------------------------
    task_to_status_src, task_to_status_dst = _get_link_columns(schema, "tasks", "task_statuses")
    status_ids = _collect_unique_ids(filtered["tasks"], task_to_status_src)

    task_statuses_all = raw_data["task_statuses"]
    ts_ek = entity_key("task_statuses")
    filtered["task_statuses"] = task_statuses_all.filter(pl.col(ts_ek).is_in(status_ids))
    print(f"  task_statuses: {filtered['task_statuses'].height} rows linked to tasks")

    # ------------------------------------------------------------------
    # Step 5: Pull task_types linked to sampled tasks
    #   Edge: tasks → task_types  AND  task_types → tasks
    # ------------------------------------------------------------------
    task_to_type_src, task_to_type_dst = _get_link_columns(schema, "tasks", "task_types")
    type_ids = _collect_unique_ids(filtered["tasks"], task_to_type_src)

    task_types_all = raw_data["task_types"]
    tt_ek = entity_key("task_types")
    filtered["task_types"] = task_types_all.filter(pl.col(tt_ek).is_in(type_ids))
    print(f"  task_types: {filtered['task_types'].height} rows linked to tasks")

    # ------------------------------------------------------------------
    # Step 6: Pull districts linked to tasks AND engineers
    #   Edges: tasks → districts, districts → tasks,
    #          engineers → districts, districts → engineers
    # ------------------------------------------------------------------
    task_to_dist_src, task_to_dist_dst = _get_link_columns(schema, "tasks", "districts")
    dist_ids_from_tasks = _collect_unique_ids(filtered["tasks"], task_to_dist_src)

    eng_to_dist_src, eng_to_dist_dst = _get_link_columns(schema, "engineers", "districts")
    dist_ids_from_engineers = _collect_unique_ids(filtered["engineers"], eng_to_dist_src)

    all_district_ids = dist_ids_from_tasks | dist_ids_from_engineers
    districts_all = raw_data["districts"]
    dist_ek = entity_key("districts")
    filtered["districts"] = districts_all.filter(pl.col(dist_ek).is_in(all_district_ids))
    print(f"  districts: {filtered['districts'].height} rows "
          f"({len(dist_ids_from_tasks)} from tasks, {len(dist_ids_from_engineers)} from engineers)")

    # ------------------------------------------------------------------
    # Step 7: Pull departments linked to tasks AND engineers
    #   Edges: tasks → departments, departments → tasks,
    #          engineers → departments, departments → engineers
    # ------------------------------------------------------------------
    task_to_dept_src, task_to_dept_dst = _get_link_columns(schema, "tasks", "departments")
    dept_ids_from_tasks = _collect_unique_ids(filtered["tasks"], task_to_dept_src)

    eng_to_dept_src, eng_to_dept_dst = _get_link_columns(schema, "engineers", "departments")
    dept_ids_from_engineers = _collect_unique_ids(filtered["engineers"], eng_to_dept_src)

    all_dept_ids = dept_ids_from_tasks | dept_ids_from_engineers
    departments_all = raw_data["departments"]
    dept_ek = entity_key("departments")
    filtered["departments"] = departments_all.filter(pl.col(dept_ek).is_in(all_dept_ids))
    print(f"  departments: {filtered['departments'].height} rows "
          f"({len(dept_ids_from_tasks)} from tasks, {len(dept_ids_from_engineers)} from engineers)")

    # ------------------------------------------------------------------
    # Validation: ensure every required edge can be realised
    # ------------------------------------------------------------------
    _validate_required_edges(schema, filtered)

    return filtered


def _validate_required_edges(
    schema: Dict[str, any],
    filtered: Dict[str, pl.DataFrame],
) -> None:
    """
    For every triplet in REQUIRED_EDGES, verify that the join between
    the source and destination tables produces at least one row.
    """
    print("\n  Validating required edges...")
    all_ok = True
    for src, rel, dst in REQUIRED_EDGES:
        if src not in filtered or dst not in filtered:
            print(f"  ✗ ({src}, {rel}, {dst}) — table missing "
                  f"(src={'present' if src in filtered else 'MISSING'}, "
                  f"dst={'present' if dst in filtered else 'MISSING'})")
            all_ok = False
            continue

        try:
            src_col, dst_col = _get_link_columns(schema, src, dst)
        except KeyError:
            print(f"  ✗ ({src}, {rel}, {dst}) — no link definition in schema")
            all_ok = False
            continue

        src_ids = _collect_unique_ids(filtered[src], src_col)
        dst_ids = _collect_unique_ids(filtered[dst], dst_col)
        overlap = src_ids & dst_ids

        if overlap:
            print(f"  ✓ ({src}, {rel}, {dst}) — {len(overlap)} shared IDs via "
                  f"{src}.{src_col} ↔ {dst}.{dst_col}")
        else:
            print(f"  ✗ ({src}, {rel}, {dst}) — 0 shared IDs via "
                  f"{src}.{src_col} ↔ {dst}.{dst_col}")
            all_ok = False

    if not all_ok:
        raise ValueError(
            "Some required edges have no coverage in the sampled data. "
            "Try increasing --n_work_orders or check the schema link definitions."
        )
    print("  All required edges validated ✓\n")


def save_filtered_data(filtered_data: Dict[str, pl.DataFrame], out_dir: Path):
    """Save the filtered data to separate CSV files."""
    for table, df in filtered_data.items():
        # Prepend "W6" to the table name
        name = table
        if name in ("departments",):
            name = "department"
        out_path = out_dir / f"W6{name.upper()}.csv"
        df.write_csv(out_path)
        print(f"Saved {table}: {df.height} rows → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Sample a small, connected subset of raw data for dashboard upload"
    )
    ap.add_argument("--raw_dir", type=str, default="data/raw", help="Directory with raw CSVs")
    ap.add_argument("--yaml", type=str, default="configs/graph.yaml", help="Path to graph.yaml")
    ap.add_argument("--out_dir", type=str, default="data/sample", help="Output directory for sampled CSVs")
    ap.add_argument("--n_work_orders", type=int, default=500, help="Number of work orders to sample")
    ap.add_argument("--seed", type=int, default=42)

    # Graph .pt file for test-set splitting
    ap.add_argument("--pt", type=str, default=None,
                     help="Path to .pt HeteroData graph file (enables test-set-only sampling)")
    ap.add_argument("--target", type=str, default="assignments",
                     help="Target node type in the graph (default: assignments)")
    ap.add_argument("--min_degree", type=int, default=1,
                     help="Minimum in-degree for target nodes (must match train.py)")
    ap.add_argument("--degree_mode", type=str, default="in", choices=["in", "out", "inout"],
                     help="Degree mode for filtering (must match train.py)")
    ap.add_argument("--train_ratio", type=float, default=0.8,
                     help="Train ratio used during training (must match train.py)")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                     help="Val ratio used during training (must match train.py)")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"],
                     help="Which split to sample from (default: test)")

    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load schema
    with open(args.yaml, "r") as f:
        schema = yaml.safe_load(f)

    # ---- Determine test-set IDs if .pt is provided ----
    test_only_ids: Optional[Set] = None

    if args.pt is not None:
        pt_path = Path(args.pt)
        if not pt_path.exists():
            raise FileNotFoundError(f"Graph file not found: {pt_path}")

        print(f"[0/3] Reproducing train.py split from {pt_path} ...")
        test_idx, graph_data = get_test_set_indices(
            pt_path,
            target=args.target,
            min_degree=args.min_degree,
            degree_mode=args.degree_mode,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        # Allow choosing which split to sample from
        if args.split != "test":
            deg = compute_target_degree(graph_data, args.target, degree_mode=args.degree_mode)
            kept = (deg >= args.min_degree).nonzero(as_tuple=False).view(-1)
            train_idx, val_idx, split_test_idx = split_indices(
                kept, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
            )
            if args.split == "train":
                chosen_idx = train_idx
            elif args.split == "val":
                chosen_idx = val_idx
            elif args.split == "all":
                chosen_idx = kept
            else:
                chosen_idx = split_test_idx
            print(f"  Using '{args.split}' split: {chosen_idx.numel()} nodes")
        else:
            chosen_idx = test_idx

        test_only_ids = graph_indices_to_entity_ids(
            graph_data, chosen_idx, args.target, schema
        )
        print(f"  Mapped {chosen_idx.numel()} graph indices → {len(test_only_ids)} entity IDs")
    else:
        print("[info] No --pt provided; sampling from ALL assignments (no split filtering)")

    # Load raw data
    print("[1/3] Loading raw data...")
    raw_data = {
        table: load_raw_table(schema, table, raw_dir)
        for table in schema["datasets"]
    }

    # Sample connected subset
    split_label = args.split if args.pt else "all"
    print(f"[2/3] Sampling {args.n_work_orders} connected work orders (split={split_label})...")
    filtered_data = sample_connected_subset(
        schema, raw_data, args.n_work_orders, args.seed,
        test_only_ids=test_only_ids,
    )

    # Save filtered data
    print("[3/3] Saving filtered data...")
    save_filtered_data(filtered_data, out_dir)

    print(f"\n✅ Done! Example dataset ({split_label} split) saved to:", out_dir)


if __name__ == "__main__":
    main()