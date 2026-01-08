from src.process.graph_builder import ( NodeBuilder,MappingEdgeBuilder,GraphBuilder,build_node_schema_from_yaml,build_edge_builders_from_yaml)
import polars as pl
import yaml
from pathlib import Path

tasks_df = pl.read_parquet(Path("data/processed/tasks_processed.parquet"))
assignments_df = pl.read_parquet(Path("data/processed/assignments_processed.parquet"))
engineers_df = pl.read_parquet(Path("data/processed/engineers_processed.parquet"))
districts_df = pl.read_parquet(Path("data/processed/districts_processed.parquet"))

print(tasks_df)