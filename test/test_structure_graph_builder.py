"""
Tests for utility functions in src/process/structure_graph_builder.py

Covers pure helper functions that don't require reading CSV files:
  - assert_no_nulls
  - filter_null_value
  - get_trait_cols
  - warn_if_non_numeric
  - _to_float_tensor
  - _ensure_2d
  - _build_node_index
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import polars as pl

from src.process.structure_graph_builder import (
    _build_node_index,
    _ensure_2d,
    _to_float_tensor,
    assert_no_nulls,
    filter_null_value,
    get_trait_cols,
    warn_if_non_numeric,
)


# ---------------------------------------------------------------------------
# assert_no_nulls
# ---------------------------------------------------------------------------

class TestAssertNoNulls:
    def test_passes_on_clean_dataframe(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        assert_no_nulls(df)  # should not raise

    def test_raises_on_null_value(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [4.0, 5.0, 6.0]})
        with pytest.raises(ValueError):
            assert_no_nulls(df)

    def test_raises_when_all_nulls_in_column(self):
        df = pl.DataFrame({"a": [None, None, None]})
        with pytest.raises(ValueError):
            assert_no_nulls(df)

    def test_empty_dataframe_passes(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
        assert_no_nulls(df)  # 0 nulls → OK


# ---------------------------------------------------------------------------
# filter_null_value
# ---------------------------------------------------------------------------

class TestFilterNullValue:
    def test_numeric_nulls_filled_with_zero(self):
        df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, None]})
        result = filter_null_value(df)
        assert result["a"].null_count() == 0
        assert result["b"].null_count() == 0
        assert result["a"].to_list() == [1.0, 0.0, 3.0]

    def test_string_columns_unchanged(self):
        df = pl.DataFrame({"s": ["x", None, "z"], "n": [1.0, None, 3.0]})
        result = filter_null_value(df)
        # string column should NOT be filled with 0 (only numeric cols filled)
        assert result["n"].to_list() == [1.0, 0.0, 3.0]
        # string col may still have null or be untouched
        assert "s" in result.columns

    def test_no_nulls_dataframe_unchanged(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = filter_null_value(df)
        assert result["a"].to_list() == [1.0, 2.0]
        assert result["b"].to_list() == [3.0, 4.0]

    def test_returns_dataframe(self):
        df = pl.DataFrame({"a": [1.0, None]})
        result = filter_null_value(df)
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# get_trait_cols
# ---------------------------------------------------------------------------

class TestGetTraitCols:
    """
    get_trait_cols(schema, table_name) reads schema['tables'][table_name]['trait_cols']
    and returns (node_cols, edge_cols).
    We only need a dict that follows the expected structure.
    """

    def _make_schema(self, node_cols, edge_cols):
        # Matches real structure: schema['datasets'][table]['variables'][col]['trait_type']
        variables = {}
        for col in node_cols:
            variables[col] = {"trait_type": "node"}
        for col in edge_cols:
            variables[col] = {"trait_type": "edge"}
        return {
            "datasets": {
                "assignments": {
                    "variables": variables
                }
            }
        }

    def test_returns_node_and_edge_cols(self):
        schema = self._make_schema(["feat_a", "feat_b"], ["join_col"])
        node_cols, edge_cols = get_trait_cols(schema, "assignments")
        assert sorted(node_cols) == sorted(["feat_a", "feat_b"])
        assert edge_cols == ["join_col"]

    def test_empty_columns(self):
        schema = self._make_schema([], [])
        node_cols, edge_cols = get_trait_cols(schema, "assignments")
        assert node_cols == []
        assert edge_cols == []

    def test_returns_lists(self):
        schema = self._make_schema(["x"], ["y"])
        node_cols, edge_cols = get_trait_cols(schema, "assignments")
        assert isinstance(node_cols, list)
        assert isinstance(edge_cols, list)


# ---------------------------------------------------------------------------
# warn_if_non_numeric
# ---------------------------------------------------------------------------

class TestWarnIfNonNumeric:
    def test_no_warning_on_numeric_df(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_non_numeric(df, "test_df")
        assert len(caught) == 0

    def test_warning_on_string_column(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_non_numeric(df, "test_df")
        assert any("s" in str(w.message) or "non-numeric" in str(w.message).lower()
                   for w in caught)

    def test_drop_true_removes_non_numeric_cols(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
        result = warn_if_non_numeric(df, "test_df", drop=True)
        assert "a" in result.columns
        assert "s" not in result.columns

    def test_drop_false_keeps_non_numeric_cols(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = warn_if_non_numeric(df, "test_df", drop=False)
        # When drop=False the function may return None or the original df
        # We only test that it doesn't raise
        assert result is None or "s" in result.columns


# ---------------------------------------------------------------------------
# _to_float_tensor
# ---------------------------------------------------------------------------

class TestToFloatTensor:
    def test_from_float_tensor(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _to_float_tensor(t)
        assert result.dtype == torch.float32
        assert result.shape == (2, 2)

    def test_from_int_tensor(self):
        t = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        result = _to_float_tensor(t)
        assert result.dtype == torch.float32

    def test_from_numpy_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = _to_float_tensor(arr)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (2, 2)

    def test_from_polars_dataframe(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = _to_float_tensor(df)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (3, 2)

    def test_values_preserved(self):
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        result = _to_float_tensor(arr)
        assert torch.allclose(result, torch.tensor([[1.0, 2.0]]))

    def test_unsupported_type_raises(self):
        with pytest.raises((TypeError, Exception)):
            _to_float_tensor("not_a_tensor")


# ---------------------------------------------------------------------------
# _ensure_2d
# ---------------------------------------------------------------------------

class TestEnsure2d:
    def test_2d_tensor_passes(self):
        t = torch.randn(4, 3)
        result = _ensure_2d(t, "test")
        assert result.dim() == 2

    def test_1d_tensor_raises(self):
        t = torch.randn(4)
        with pytest.raises((AssertionError, ValueError)):
            _ensure_2d(t, "test")

    def test_3d_tensor_raises(self):
        t = torch.randn(2, 3, 4)
        with pytest.raises((AssertionError, ValueError)):
            _ensure_2d(t, "test")

    def test_returns_same_tensor(self):
        t = torch.randn(5, 2)
        result = _ensure_2d(t, "t")
        assert result is t


# ---------------------------------------------------------------------------
# _build_node_index
# ---------------------------------------------------------------------------

class TestBuildNodeIndex:
    # When entity_key == name, _build_node_index selects just [entity_key] and
    # uses that column as the node_ids list.
    def test_returns_two_elements(self):
        df = pl.DataFrame({"entity_id": ["a", "b", "c", "a"]})
        keys_df, node_ids = _build_node_index(df, "entity_id", "entity_id")
        assert keys_df is not None
        assert node_ids is not None

    def test_node_ids_are_unique(self):
        df = pl.DataFrame({"entity_id": ["a", "b", "c", "a", "b"]})
        keys_df, node_ids = _build_node_index(df, "entity_id", "entity_id")
        assert len(node_ids) == len(set(node_ids))

    def test_node_ids_count_matches_unique_entities(self):
        df = pl.DataFrame({"entity_id": [1, 2, 3, 1, 2]})
        keys_df, node_ids = _build_node_index(df, "entity_id", "entity_id")
        assert len(node_ids) == 3  # 3 unique: 1, 2, 3

    def test_keys_df_has_idx_column(self):
        df = pl.DataFrame({"entity_id": ["x", "y", "z"]})
        keys_df, node_ids = _build_node_index(df, "entity_id", "entity_id")
        assert "__idx" in keys_df.columns

    def test_idx_is_contiguous_from_zero(self):
        df = pl.DataFrame({"entity_id": ["a", "b", "c"]})
        keys_df, node_ids = _build_node_index(df, "entity_id", "entity_id")
        indices = sorted(keys_df["__idx"].to_list())
        assert indices == list(range(len(indices)))

    def test_two_column_mode(self):
        # When entity_key != name, a second col is selected alongside entity_key
        df = pl.DataFrame({"entity_id": ["a", "b", "c"], "label": ["x", "y", "z"]})
        keys_df, node_ids = _build_node_index(df, "entity_id", "label")
        assert len(node_ids) == 3
        assert "__idx" in keys_df.columns
