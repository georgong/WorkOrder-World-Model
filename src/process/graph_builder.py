from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import polars as pl

# Optional torch / pyg
try:
    import torch
    from torch_geometric.data import HeteroData
except Exception:
    torch = None
    HeteroData = None


# ----------------------------
# Small utilities (strict!)
# ----------------------------

def _assert_unique_columns(df: pl.DataFrame, *, where: str) -> None:
    cnt = Counter(df.columns)
    dup = {k: v for k, v in cnt.items() if v > 1}
    assert not dup, f"[{where}] duplicate column names in DataFrame: {dup}. cols={df.columns}"


def _dedup_keep_order(cols: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _trait_type(meta: Dict[str, Any]) -> Optional[str]:
    # unify 'trait' and 'trait_type'
    t = meta.get("trait_type", None)
    if t is None:
        t = meta.get("trait", None)
    return t


def _dtype_map(dtype_str: str) -> Any:
    # You can extend this mapping if your YAML has more types
    DTYPE_MAP = {
        "Int64": pl.Int64,
        "Float64": pl.Float64,
        "string": pl.Utf8,
        "datetime64[ns]": pl.Datetime("ns"),
    }
    return DTYPE_MAP.get(dtype_str, pl.Utf8)


def _is_numeric_polars_dtype(dt: pl.DataType) -> bool:
    return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)


# ----------------------------
# Returned artifact
# ----------------------------

@dataclass
class GraphArtifact:
    # Polars-side
    nodes: Dict[str, pl.DataFrame]                       # columns: [entity_key, (name), node_feats...]
    edges: Dict[Tuple[str, str, str], pl.DataFrame]       # columns: ["src_id","dst_id", edge_feats...]
    node_entities: Dict[str, List[Any]]                   # node_id -> entity_key (list order)
    node_id_maps: Dict[str, Dict[Any, int]]               # entity_key -> node_id
    node_feature_names: Dict[str, List[str]]              # features in x order
    node_mask_names: Dict[str, List[str]]                 # mask=True feature names
    edge_feature_names: Dict[Tuple[str, str, str], List[str]]
    edge_mask_names: Dict[Tuple[str, str, str], List[str]]
    mappings: Dict[str, Any]
    schema_datasets: Dict[str, Any]

    # Torch-side
    pyg: Optional["HeteroData"] = None


# ----------------------------
# GraphBuilder
# ----------------------------

class GraphBuilder:
    """
    Build a heterogeneous graph from:
      - tables: dict[node_type, pl.DataFrame]
      - schema_datasets: schema["datasets"][node_type]["variables"][col] includes:
            dtype, mask, trait/trait_type, (optional) agg
      - mappings: mapping config includes per node_type:
            entity_key, name, links{dst_type: {left_on, right_on}}
    """

    def __init__(
        self,
        *,
        tables: Dict[str, pl.DataFrame],
        schema_datasets: Dict[str, Any],
        mappings: Dict[str, Any],
        default_numeric_agg: str = "mean",
    ) -> None:
        self.tables = tables
        self.schema_datasets = schema_datasets
        self.mappings = mappings
        self.default_numeric_agg = default_numeric_agg

        assert isinstance(self.tables, dict) and self.tables, "tables must be a non-empty dict"
        assert isinstance(self.schema_datasets, dict) and self.schema_datasets, "schema_datasets must be a non-empty dict"
        assert isinstance(self.mappings, dict) and self.mappings, "mappings must be a dict"
        for nt, df in self.tables.items():
            assert isinstance(df, pl.DataFrame), f"tables[{nt}] must be a polars DataFrame, got {type(df)}"
            _assert_unique_columns(df, where=f"init/tables[{nt}]")

    # ----------------------------
    # Schema access
    # ----------------------------

    def _vars_meta(self, node_type: str) -> Dict[str, Dict[str, Any]]:
        assert node_type in self.schema_datasets, f"node_type={node_type!r} missing in schema_datasets keys={list(self.schema_datasets.keys())}"
        ds = self.schema_datasets[node_type]
        assert "variables" in ds, f"schema_datasets[{node_type}] missing 'variables'"
        return ds["variables"]

    def _entity_key(self, node_type: str) -> str:
        assert node_type in self.mappings, f"node_type={node_type!r} missing in mappings keys={list(self.mappings.keys())}"
        ek = self.mappings[node_type].get("entity_key", None)
        assert isinstance(ek, str) and ek, f"mappings[{node_type}].entity_key must be a non-empty string"
        return ek

    def _name_col(self, node_type: str) -> Optional[str]:
        # optional column to display (you used "name" in YAML)
        if node_type not in self.mappings:
            return None
        nc = self.mappings[node_type].get("name", None)
        if nc is None:
            return None
        assert isinstance(nc, str) and nc, f"mappings[{node_type}].name must be a non-empty string if present"
        return nc

    def _links(self, node_type: str) -> Dict[str, Dict[str, str]]:
        cfg = self.mappings.get(node_type, {})
        links = cfg.get("links", {})
        assert isinstance(links, dict), f"mappings[{node_type}].links must be a dict"
        # validate structure
        for dst_type, link in links.items():
            assert isinstance(link, dict), f"mappings[{node_type}].links[{dst_type}] must be a dict"
            assert "left_on" in link and "right_on" in link, f"mappings[{node_type}].links[{dst_type}] must have left_on/right_on"
            assert isinstance(link["left_on"], str) and link["left_on"], f"left_on invalid for {node_type}->{dst_type}"
            assert isinstance(link["right_on"], str) and link["right_on"], f"right_on invalid for {node_type}->{dst_type}"
        return links

    # ----------------------------
    # Node construction
    # ----------------------------

    def _build_nodes(self, node_type: str) -> Tuple[pl.DataFrame, List[Any], Dict[Any, int], List[str], List[str]]:
        """
        Return:
          node_df: [entity_key, (name), feature...]
          entities_list: node_id -> entity_key
          id_map: entity_key -> node_id
          feat_names: x columns (float only)
          mask_names: subset of feat_names where mask=True
        """
        df = self.tables[node_type]
        _assert_unique_columns(df, where=f"build_nodes/input:{node_type}")

        entity_key = self._entity_key(node_type)
        assert entity_key in df.columns, f"[{node_type}] entity_key={entity_key!r} not in df.columns={df.columns}"

        name_col = self._name_col(node_type)
        if name_col is not None:
            assert name_col in df.columns, f"[{node_type}] name_col={name_col!r} not in df.columns={df.columns}"

        vars_meta = self._vars_meta(node_type)

        # pick node traits (trait_type == "node")
        node_trait_cols = []
        node_mask_cols = []
        for col, meta in vars_meta.items():
            t = _trait_type(meta)
            if t == "node":
                if col in df.columns:
                    node_trait_cols.append(col)
                    if bool(meta.get("mask", False)):
                        node_mask_cols.append(col)

        # stable, unique select list: entity_key + optional name + node traits
        raw_select = [entity_key]
        if name_col and name_col != entity_key:
            raw_select.append(name_col)
        raw_select.extend(node_trait_cols)

        select_cols = _dedup_keep_order([c for c in raw_select if c in df.columns])

        assert select_cols.count(entity_key) == 1, f"[{node_type}] entity_key duplicated in select_cols={select_cols}"
        if name_col:
            assert select_cols.count(name_col) <= 1, f"[{node_type}] name_col duplicated in select_cols={select_cols}"

        base = df.select(select_cols).filter(pl.col(entity_key).is_not_null())
        _assert_unique_columns(base, where=f"build_nodes/base:{node_type}")

        # Build aggregations for traits if multiple rows per entity_key
        # - numeric: default mean (or meta['agg'])
        # - non-numeric: first non-null
        agg_exprs = []
        for col in node_trait_cols:
            if col == entity_key:
                continue
            if name_col and col == name_col:
                continue
            if col not in base.columns:
                continue

            meta = vars_meta.get(col, {})
            agg = meta.get("agg", None)

            # infer by dtype in base
            dt = base.schema[col]
            if _is_numeric_polars_dtype(dt):
                use_agg = agg or self.default_numeric_agg
                if use_agg == "mean":
                    agg_exprs.append(pl.col(col).mean().alias(col))
                elif use_agg == "sum":
                    agg_exprs.append(pl.col(col).sum().alias(col))
                elif use_agg == "max":
                    agg_exprs.append(pl.col(col).max().alias(col))
                elif use_agg == "min":
                    agg_exprs.append(pl.col(col).min().alias(col))
                elif use_agg == "first":
                    agg_exprs.append(pl.col(col).drop_nulls().first().alias(col))
                else:
                    raise ValueError(f"[{node_type}] unsupported agg={use_agg!r} for numeric col={col}")
            else:
                # for strings/datetimes/etc: stable default: first non-null
                agg_exprs.append(pl.col(col).drop_nulls().first().alias(col))

        # name column aggregation (if present)
        if name_col and name_col != entity_key:
            # keep first non-null name per entity
            agg_exprs.append(pl.col(name_col).drop_nulls().first().alias(name_col))

        if agg_exprs:
            grouped = base.group_by(entity_key).agg(agg_exprs)
        else:
            grouped = base.unique(subset=[entity_key])

        # ensure entity_key exists
        assert entity_key in grouped.columns, f"[{node_type}] grouped lost entity_key={entity_key}. cols={grouped.columns}"
        _assert_unique_columns(grouped, where=f"build_nodes/grouped:{node_type}")

        # Create node id order and id_map
        # Keep stable ordering by sorting entity_key if sortable; otherwise keep Polars order.
        # Polars entity_key may be numeric/string mixed; safest: keep order as produced.
        entities = grouped[entity_key].to_list()
        assert len(entities) > 0, f"[{node_type}] no entities after grouping"
        id_map = {}
        for i, ek in enumerate(entities):
            # disallow duplicates
            assert ek not in id_map, f"[{node_type}] duplicate entity_key value after grouping: {ek!r}"
            id_map[ek] = i

        # Determine numeric feature columns for x:
        # Only include numeric node traits (float/int) excluding entity_key/name
        feat_names = []
        mask_names = []
        for col in node_trait_cols:
            if col == entity_key:
                continue
            if name_col and col == name_col:
                continue
            if col not in grouped.columns:
                continue
            dt = grouped.schema[col]
            if _is_numeric_polars_dtype(dt):
                feat_names.append(col)
                if col in node_mask_cols:
                    mask_names.append(col)

        node_df = grouped  # includes entity_key + maybe name + all aggregated traits
        return node_df, entities, id_map, feat_names, mask_names

    # ----------------------------
    # Edge construction (mapping-driven, robust)
    # ----------------------------

    def _build_edges(
        self,
        *,
        src_type: str,
        dst_type: str,
        left_on: str,
        right_on: str,
        src_id_map: Dict[Any, int],
        dst_id_map: Dict[Any, int],
    ) -> Tuple[pl.DataFrame, List[str], List[str]]:
        """
        Build edges src_type -> dst_type based on mapping join columns.

        We build a "pair table" of (src_entity, dst_entity) via joining only needed columns.
        Then we map entities to node ids using id_map dicts.

        Important: join columns may equal entity_key on either side.
        We do NOT attempt to keep duplicate physical column names. We rename columns
        into internal names: __src_ent, __dst_ent, __src_key, __dst_key.

        Returns:
          edge_df columns: ["src_id","dst_id", edge_feat...]
          edge_feat_names numeric edge features (currently empty by default)
          edge_mask_names mask=True edge feature names (currently empty by default)
        """
        src_df = self.tables[src_type]
        dst_df = self.tables[dst_type]
        _assert_unique_columns(src_df, where=f"build_edges/src_input:{src_type}")
        _assert_unique_columns(dst_df, where=f"build_edges/dst_input:{dst_type}")

        src_entity_key = self._entity_key(src_type)
        dst_entity_key = self._entity_key(dst_type)

        assert left_on in src_df.columns, f"[edge {src_type}->{dst_type}] left_on={left_on!r} not in src columns={src_df.columns}"
        assert right_on in dst_df.columns, f"[edge {src_type}->{dst_type}] right_on={right_on!r} not in dst columns={dst_df.columns}"
        assert src_entity_key in src_df.columns, f"[edge {src_type}->{dst_type}] src_entity_key={src_entity_key!r} not in src columns={src_df.columns}"
        assert dst_entity_key in dst_df.columns, f"[edge {src_type}->{dst_type}] dst_entity_key={dst_entity_key!r} not in dst columns={dst_df.columns}"

        # Build minimal projection for each side, carefully:
        # We need: (src_entity_key, left_on) from src table
        #          (dst_entity_key, right_on) from dst table
        # But if src_entity_key == left_on, we must keep both semantics without duplicate names.
        src_select = []
        if src_entity_key == left_on:
            # same column supplies both entity and join key
            src_select = [
                pl.col(src_entity_key).alias("__src_ent"),
                pl.col(left_on).alias("__src_key"),
            ]
        else:
            src_select = [
                pl.col(src_entity_key).alias("__src_ent"),
                pl.col(left_on).alias("__src_key"),
            ]

        dst_select = []
        if dst_entity_key == right_on:
            dst_select = [
                pl.col(dst_entity_key).alias("__dst_ent"),
                pl.col(right_on).alias("__dst_key"),
            ]
        else:
            dst_select = [
                pl.col(dst_entity_key).alias("__dst_ent"),
                pl.col(right_on).alias("__dst_key"),
            ]

        src_proj = src_df.select(src_select).filter(
            pl.col("__src_ent").is_not_null() & pl.col("__src_key").is_not_null()
        )
        dst_proj = dst_df.select(dst_select).filter(
            pl.col("__dst_ent").is_not_null() & pl.col("__dst_key").is_not_null()
        )

        _assert_unique_columns(src_proj, where=f"build_edges/src_proj:{src_type}")
        _assert_unique_columns(dst_proj, where=f"build_edges/dst_proj:{dst_type}")

        # Now the join is ALWAYS on __src_key == __dst_key
        joined = src_proj.join(dst_proj, left_on="__src_key", right_on="__dst_key", how="inner")
        _assert_unique_columns(joined, where=f"build_edges/joined:{src_type}->{dst_type}")

        # Convert entities to node ids using mapping dicts.
        # We keep only pairs that exist in node maps (should, but guard anyway).
        src_ents = joined["__src_ent"].to_list()
        dst_ents = joined["__dst_ent"].to_list()
        assert len(src_ents) == len(dst_ents), "internal error: join produced mismatched lengths"

        src_ids: List[int] = []
        dst_ids: List[int] = []
        dropped = 0
        for se, de in zip(src_ents, dst_ents):
            si = src_id_map.get(se, None)
            di = dst_id_map.get(de, None)
            if si is None or di is None:
                dropped += 1
                continue
            src_ids.append(si)
            dst_ids.append(di)

        # if everything dropped, it's usually a mapping/key mismatch
        assert len(src_ids) > 0, (
            f"[edge {src_type}->{dst_type}] no edges produced after id mapping. "
            f"dropped={dropped}. check entity_key values/types and mappings left_on/right_on."
        )

        edge_df = pl.DataFrame({"src_id": src_ids, "dst_id": dst_ids})

        # Edge traits:
        # Your YAML idea: "trait_type == edge" columns belong to edges.
        # BUT: those columns live in *tables*, not in links config. Attaching them is ambiguous.
        # Here we do the minimal correct thing: build topology first.
        # You can extend by defining per-link edge_attr_source in YAML later.
        edge_feat_names: List[str] = []
        edge_mask_names: List[str] = []
        return edge_df, edge_feat_names, edge_mask_names

    # ----------------------------
    # Build full artifact
    # ----------------------------

    def build(self) -> GraphArtifact:
        # 1) Build nodes for every table that has mapping info
        nodes: Dict[str, pl.DataFrame] = {}
        node_entities: Dict[str, List[Any]] = {}
        node_id_maps: Dict[str, Dict[Any, int]] = {}
        node_feat_names: Dict[str, List[str]] = {}
        node_mask_names: Dict[str, List[str]] = {}

        for node_type in self.tables.keys():
            # Only build nodes if it exists in mappings (otherwise it's "unused")
            if node_type not in self.mappings:
                continue

            node_df, entities, id_map, feat_names, mask_names = self._build_nodes(node_type)

            nodes[node_type] = node_df
            node_entities[node_type] = entities
            node_id_maps[node_type] = id_map
            node_feat_names[node_type] = feat_names
            node_mask_names[node_type] = mask_names

            # asserts for sanity
            assert len(entities) == node_df.height, f"[{node_type}] entities length != node_df height"
            assert all(isinstance(i, int) for i in id_map.values()), f"[{node_type}] id_map values must be ints"
            # no duplicate entity keys
            assert len(set(entities)) == len(entities), f"[{node_type}] duplicate entity keys after build_nodes"

        assert nodes, "no nodes built (check mappings keys vs tables keys)"

        # 2) Build edges according to mappings.links
        edges: Dict[Tuple[str, str, str], pl.DataFrame] = {}
        edge_feat_names: Dict[Tuple[str, str, str], List[str]] = {}
        edge_mask_names: Dict[Tuple[str, str, str], List[str]] = {}

        for src_type in nodes.keys():
            links = self._links(src_type)
            for dst_type, link_cfg in links.items():
                # We only build if dst nodes exist
                if dst_type not in nodes:
                    continue

                left_on = link_cfg["left_on"]
                right_on = link_cfg["right_on"]

                # Relation name: deterministic
                rel = f"{src_type}__to__{dst_type}"
                etype = (src_type, rel, dst_type)

                edge_df, ef, em = self._build_edges(
                    src_type=src_type,
                    dst_type=dst_type,
                    left_on=left_on,
                    right_on=right_on,
                    src_id_map=node_id_maps[src_type],
                    dst_id_map=node_id_maps[dst_type],
                )

                edges[etype] = edge_df
                edge_feat_names[etype] = ef
                edge_mask_names[etype] = em

                # more asserts
                assert "src_id" in edge_df.columns and "dst_id" in edge_df.columns, f"[{etype}] edge_df missing src_id/dst_id"
                assert edge_df.height > 0, f"[{etype}] empty edge_df (should have been caught earlier)"

        art = GraphArtifact(
            nodes=nodes,
            edges=edges,
            node_entities=node_entities,
            node_id_maps=node_id_maps,
            node_feature_names=node_feat_names,
            node_mask_names=node_mask_names,
            edge_feature_names=edge_feat_names,
            edge_mask_names=edge_mask_names,
            mappings=self.mappings,
            schema_datasets=self.schema_datasets,
        )
        return art

    # ----------------------------
    # Torch / PyG conversion (reversible at entity_key level)
    # ----------------------------

    def to_pyg(self, art: GraphArtifact) -> GraphArtifact:
        assert torch is not None and HeteroData is not None, "torch_geometric not available in this environment"
        data = HeteroData()

        # nodes
        for node_type, node_df in art.nodes.items():
            feat_names = art.node_feature_names.get(node_type, [])
            # build x (N, F) float32
            if feat_names:
                x_df = node_df.select(feat_names)
                # fill nulls with 0.0 for numeric
                x_df = x_df.with_columns([pl.col(c).fill_null(0.0) for c in feat_names])
                x = torch.tensor(x_df.to_numpy(), dtype=torch.float32)
            else:
                x = torch.empty((node_df.height, 0), dtype=torch.float32)

            data[node_type].x = x
            data[node_type].x_names = feat_names
            data[node_type].mask_names = art.node_mask_names.get(node_type, [])

            # store entity_key list for reversibility
            entity_key = self._entity_key(node_type)
            data[node_type].entity_key_name = entity_key
            data[node_type].entity_keys = art.node_entities[node_type]  # python list

            # optional name col
            name_col = self._name_col(node_type)
            if name_col and name_col in node_df.columns:
                # create parallel python list (not tensor)
                data[node_type].name_col = name_col
                data[node_type].names = node_df[name_col].to_list()

        # edges
        for etype, edge_df in art.edges.items():
            src_type, rel, dst_type = etype
            src = torch.tensor(edge_df["src_id"].to_list(), dtype=torch.long)
            dst = torch.tensor(edge_df["dst_id"].to_list(), dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
            data[etype].edge_index = edge_index
            data[etype].edge_attr_names = art.edge_feature_names.get(etype, [])
            data[etype].mask_names = art.edge_mask_names.get(etype, [])
            # edge_attr currently empty by default
            data[etype].edge_attr = torch.empty((edge_df.height, 0), dtype=torch.float32)

        art.pyg = data
        return art

    def from_pyg_nodes(self, art: GraphArtifact, *, node_type: str) -> pl.DataFrame:
        assert art.pyg is not None, "artifact has no pyg graph; call to_pyg first"
        data = art.pyg
        assert node_type in data.node_types, f"node_type={node_type} not in pyg node_types={data.node_types}"

        entity_keys = data[node_type].entity_keys
        x = data[node_type].x
        x_names = getattr(data[node_type], "x_names", [])

        # x might be empty
        if x.numel() == 0 or len(x_names) == 0:
            return pl.DataFrame({data[node_type].entity_key_name: entity_keys})

        # convert to python lists
        mat = x.detach().cpu().numpy()
        out = {data[node_type].entity_key_name: entity_keys}
        for j, name in enumerate(x_names):
            out[name] = mat[:, j].tolist()
        return pl.DataFrame(out)

    def from_pyg_edges(self, art: GraphArtifact, *, etype: Tuple[str, str, str]) -> pl.DataFrame:
        assert art.pyg is not None, "artifact has no pyg graph; call to_pyg first"
        data = art.pyg
        assert etype in data.edge_types, f"etype={etype} not in pyg edge_types={data.edge_types}"

        src_type, _, dst_type = etype
        edge_index = data[etype].edge_index.detach().cpu().numpy()

        src_ids = edge_index[0].tolist()
        dst_ids = edge_index[1].tolist()

        src_keys = data[src_type].entity_keys
        dst_keys = data[dst_type].entity_keys

        # map node_id -> entity_key using stored lists
        src_ek = [src_keys[i] for i in src_ids]
        dst_ek = [dst_keys[i] for i in dst_ids]

        return pl.DataFrame({"src_entity": src_ek, "dst_entity": dst_ek})


# ----------------------------
# Demo test (synthetic)
# ----------------------------

def _demo_test() -> None:
    # Minimal synthetic tables that mimic your mapping patterns.

    tasks = pl.DataFrame({
        "W6KEY": [1, 1, 2, 3],
        "STATUS": [10, 10, 11, 10],
        "DURATION": [5.0, 7.0, 3.0, None],  # node trait? (you put DURATION as edge in tasks, but we only use node traits here)
        "PRIORITY": [2, 2, 1, 3],
    })

    task_statuses = pl.DataFrame({
        "W6KEY": [10, 11],
        "NAME": ["Open", "Closed"],
    })

    engineers = pl.DataFrame({
        "NAME": ["alice", "bob", "carl"],
        "EFFICIENCY": [0.9, 0.8, 0.7],
        "ACTIVE": [1, 1, 0],
    })

    assignments = pl.DataFrame({
        "W6KEY": [100, 101, 102],
        "TASK": [1, 2, 3],
        "ASSIGNEDENGINEERS": ["alice", "bob", "alice"],
        "DURATION": [1.5, 2.0, 3.0],
    })

    schema_datasets = {
        "tasks": {
            "variables": {
                "W6KEY": {"dtype": "Int64", "mask": False, "trait_type": "node"},
                "STATUS": {"dtype": "Int64", "mask": False, "trait_type": "node"},
                "PRIORITY": {"dtype": "Int64", "mask": False, "trait_type": "node"},
                # DURATION not used as node trait in this demo
            }
        },
        "task_statuses": {
            "variables": {
                "W6KEY": {"dtype": "Int64", "mask": False, "trait_type": "node"},
                "NAME": {"dtype": "string", "mask": False, "trait_type": "node"},
            }
        },
        "engineers": {
            "variables": {
                "NAME": {"dtype": "string", "mask": False, "trait_type": "node"},
                "EFFICIENCY": {"dtype": "Float64", "mask": False, "trait_type": "node"},
                "ACTIVE": {"dtype": "Int64", "mask": False, "trait_type": "node"},
            }
        },
        "assignments": {
            "variables": {
                "W6KEY": {"dtype": "Int64", "mask": False, "trait": "node"},
                "TASK": {"dtype": "Int64", "mask": False, "trait": None},
                "ASSIGNEDENGINEERS": {"dtype": "string", "mask": False, "trait": None},
                "DURATION": {"dtype": "Float64", "mask": False, "trait": "node"},
            }
        },
    }

    mappings = {
        "tasks": {
            "entity_key": "W6KEY",
            "name": "W6KEY",
            "links": {
                "task_statuses": {"left_on": "STATUS", "right_on": "W6KEY"},
                "assignments": {"left_on": "W6KEY", "right_on": "TASK"},
            },
        },
        "task_statuses": {
            "entity_key": "W6KEY",
            "name": "NAME",
            "links": {
                "tasks": {"left_on": "W6KEY", "right_on": "STATUS"},
            },
        },
        "engineers": {
            "entity_key": "NAME",
            "name": "NAME",  # entity_key == name (this is where you blew up before)
            "links": {
                "assignments": {"left_on": "NAME", "right_on": "ASSIGNEDENGINEERS"},
            },
        },
        "assignments": {
            "entity_key": "W6KEY",
            "name": "W6KEY",
            "links": {
                "tasks": {"left_on": "TASK", "right_on": "W6KEY"},
                "engineers": {"left_on": "ASSIGNEDENGINEERS", "right_on": "NAME"},
            },
        },
    }

    tables = {
        "tasks": tasks,
        "task_statuses": task_statuses,
        "engineers": engineers,
        "assignments": assignments,
    }

    gb = GraphBuilder(
        tables=tables,
        schema_datasets=schema_datasets,
        mappings=mappings,
        default_numeric_agg="mean",
    )

    art = gb.build()

    # Assertions to show what we built
    assert "tasks" in art.nodes and art.nodes["tasks"].height == 3, "tasks should have 3 unique W6KEY nodes"
    assert "engineers" in art.nodes and art.nodes["engineers"].height == 3, "engineers should have 3 NAME nodes"
    assert any(et[0] == "tasks" and et[2] == "assignments" for et in art.edges.keys()), "tasks->assignments edges missing"
    assert any(et[0] == "assignments" and et[2] == "engineers" for et in art.edges.keys()), "assignments->engineers edges missing"

    print("Built nodes:", {k: v.shape for k, v in art.nodes.items()})
    print("Built edges:", {k: v.shape for k, v in art.edges.items()})

    # Optional torch conversion
    if torch is not None and HeteroData is not None:
        art = gb.to_pyg(art)
        # reverse back example
        df_back = gb.from_pyg_nodes(art, node_type="engineers")
        print("Back from pyg (engineers nodes):")
        print(df_back)

        # pick an edge type and reverse
        some_etype = next(iter(art.edges.keys()))
        eback = gb.from_pyg_edges(art, etype=some_etype)
        print("Back from pyg (one edge type):", some_etype)
        print(eback)


if __name__ == "__main__":
    _demo_test()
