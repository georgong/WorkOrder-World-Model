"""
Run this ONCE after training to save the feature column schemas.

Usage:
    python -m src.process.save_training_schemas --data-dir data/raw

This reads the full training CSVs, builds feature tables, and saves
the column schemas to api/assets/feature_schemas/*.json so that
inference-time alignment can pad/trim to match.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


from .feature_engineering import (
    process_task_feature,
    process_assignment_feature,
    process_engineer_feature,
    process_districts_feature,
)
# Import your FeatureSchema definitions for each node type
# Adjust these imports to match your actual schema definitions
from .feature_schema import (
    task_schema,
    assignment_schema, 
    engineer_schema,
    district_schema
)
import polars as pl
from .structure_graph_builder import GraphBuilder
import json
SCHEMA_DIR_DEFAULT = Path(__file__).parent.parent / "assets" / "feature_schemas"

def save_feature_schema(
    df: pl.DataFrame,
    node_type: str,
    schema_dir: Path = SCHEMA_DIR_DEFAULT,
) -> Path:
    """
    Persist the ordered list of column names (and dtypes) for a feature table.
    Call this once after building features on the *training* data.
    """
    schema_dir.mkdir(parents=True, exist_ok=True)
    out = schema_dir / f"{node_type}.json"
    payload = {
        "node_type": node_type,
        "columns": df.columns,
        "dtypes": {c: str(df.schema[c]) for c in df.columns},
    }
    out.write_text(json.dumps(payload, indent=2))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to raw training CSVs")
    parser.add_argument("--schema-dir", type=str, default=str(SCHEMA_DIR_DEFAULT))
    args = parser.parse_args()
    
    data = Path(args.data_dir)
    schema_dir = Path(args.schema_dir)
    
    with open('configs/graph.yaml', "r") as f:
        schema = yaml.safe_load(f)
    
    gb = GraphBuilder(yaml=schema, data_dir="data/sample")
    

    print(f"Reading training data from {data}")
    print(f"Saving schemas to {schema_dir}")

    # Tasks
    tasks_df = gb._load_table(schema=schema, df_type='tasks')
    tasks_feat = process_task_feature(tasks_df, task_schema)
    save_feature_schema(tasks_feat, "tasks", schema_dir)
    print(f"  tasks: {tasks_feat.shape[1]} columns")

    # Assignments
    assign_df = gb._load_table(schema=schema, df_type='assignments')
    assign_feat = process_assignment_feature(assign_df, assignment_schema)
    save_feature_schema(assign_feat, "assignments", schema_dir)
    print(f"  assignments: {assign_feat.shape[1]} columns")

    # Engineers
    eng_df = gb._load_table(schema=schema, df_type='engineers')
    eng_feat = process_engineer_feature(eng_df, engineer_schema)
    save_feature_schema(eng_feat, "engineers", schema_dir)
    print(f"  engineers: {eng_feat.shape[1]} columns")

    # Districts
    dist_df = gb._load_table(schema=schema, df_type='districts')
    dist_feat = process_districts_feature(dist_df, district_schema)
    save_feature_schema(dist_feat, "districts", schema_dir)
    print(f"  districts: {dist_feat.shape[1]} columns")

    print("Done! Schema files saved.")


if __name__ == "__main__":
    main()
