"""
Tests for utility functions in src/runner/interpret_subgraph.py

Covers:
  - extract_metadata_maps: correct extraction of attr_name, node_ids, mask_cols
  - ensure_all_node_types_have_x: fills missing x with zeros
  - sanitize_for_neighbor_loader: non-tensor fields removed
  - load_ckpt: valid checkpoint returned; bad format raises ValueError
  - summarize_attrib: correct structure, topk respected, sorted by abs mass
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from torch_geometric.data import HeteroData

from src.runner.interpret_subgraph import (
    ensure_all_node_types_have_x,
    extract_metadata_maps,
    load_ckpt,
    sanitize_for_neighbor_loader,
    summarize_attrib,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data_with_metadata() -> HeteroData:
    """
    HeteroData with 'a' and 'b' nodes.
    'a' has attr_name, node_ids, mask_cols.
    'b' has only attr_name.
    """
    data = HeteroData()
    data["a"].x = torch.randn(5, 3)
    data["a"].attr_name = ["f0", "f1", "f2"]
    data["a"].node_ids = ["id_0", "id_1", "id_2", "id_3", "id_4"]
    data["a"].mask_cols = ["f1"]

    data["b"].x = torch.randn(4, 2)
    data["b"].attr_name = ["feat_x", "feat_y"]

    data[("a", "connects", "b")].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    return data


def _make_attrib(n_nodes: int = 5, n_feats: int = 3) -> Dict[str, Any]:
    """Synthetic attribution dict matching the structure returned by gradxinput_attrib."""
    node_signed = torch.randn(n_nodes).tolist()
    node_abs = torch.randn(n_nodes).abs().tolist()
    feat_signed = torch.randn(n_feats).tolist()
    feat_abs = torch.randn(n_feats).abs().tolist()
    return {
        "pred": 2.5,
        "per_type": {
            "a": {
                "num_nodes": n_nodes,
                "num_feats": n_feats,
                "node_signed": node_signed,
                "node_abs": node_abs,
                "feat_signed": feat_signed,
                "feat_abs": feat_abs,
            }
        },
    }


# ---------------------------------------------------------------------------
# extract_metadata_maps
# ---------------------------------------------------------------------------

class TestExtractMetadataMaps:
    def test_returns_three_dicts(self):
        data = _make_data_with_metadata()
        result = extract_metadata_maps(data)
        assert isinstance(result, tuple) and len(result) == 3

    def test_attr_name_extracted(self):
        data = _make_data_with_metadata()
        attr_name_map, _, _ = extract_metadata_maps(data)
        assert "a" in attr_name_map
        assert attr_name_map["a"] == ["f0", "f1", "f2"]

    def test_node_ids_extracted(self):
        data = _make_data_with_metadata()
        _, node_ids_map, _ = extract_metadata_maps(data)
        assert "a" in node_ids_map
        assert len(node_ids_map["a"]) == 5

    def test_mask_cols_extracted(self):
        data = _make_data_with_metadata()
        _, _, mask_cols_map = extract_metadata_maps(data)
        assert "a" in mask_cols_map
        assert mask_cols_map["a"] == ["f1"]

    def test_missing_optional_fields_not_in_maps(self):
        data = _make_data_with_metadata()
        _, node_ids_map, mask_cols_map = extract_metadata_maps(data)
        # 'b' has no node_ids or mask_cols
        assert "b" not in node_ids_map
        assert "b" not in mask_cols_map

    def test_node_ids_as_tensor_converted_to_list(self):
        data = HeteroData()
        data["a"].x = torch.randn(4, 2)
        data["a"].attr_name = ["x", "y"]
        data["a"].node_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)
        attr_map, node_ids_map, _ = extract_metadata_maps(data)
        assert isinstance(node_ids_map["a"], list)
        assert node_ids_map["a"] == [10, 20, 30, 40]

    def test_attr_name_length_mismatch_raises(self):
        data = HeteroData()
        data["a"].x = torch.randn(5, 3)
        data["a"].attr_name = ["f0", "f1"]  # should be 3, not 2
        with pytest.raises(AssertionError, match="attr_name mismatch"):
            extract_metadata_maps(data)

    def test_no_metadata_returns_empty_dicts(self):
        data = HeteroData()
        data["a"].x = torch.randn(5, 3)
        attr_map, node_ids_map, mask_cols_map = extract_metadata_maps(data)
        assert attr_map == {}
        assert node_ids_map == {}
        assert mask_cols_map == {}


# ---------------------------------------------------------------------------
# ensure_all_node_types_have_x
# ---------------------------------------------------------------------------

class TestEnsureAllNodeTypesHaveX:
    def test_returns_dict_of_in_dims(self):
        data = _make_data_with_metadata()
        in_dims = ensure_all_node_types_have_x(data)
        assert isinstance(in_dims, dict)

    def test_existing_x_has_correct_dim(self):
        data = _make_data_with_metadata()
        in_dims = ensure_all_node_types_have_x(data)
        assert in_dims["a"] == 3
        assert in_dims["b"] == 2

    def test_missing_x_filled_with_zeros(self):
        data = HeteroData()
        data["a"].x = torch.randn(5, 3)
        data["b"].num_nodes = 4  # no x
        in_dims = ensure_all_node_types_have_x(data)
        assert "b" in in_dims
        assert in_dims["b"] == 1
        assert hasattr(data["b"], "x")
        assert data["b"].x.shape == (4, 1)
        assert data["b"].x.sum().item() == 0.0

    def test_all_node_types_covered(self):
        data = _make_data_with_metadata()
        in_dims = ensure_all_node_types_have_x(data)
        for nt in data.node_types:
            assert nt in in_dims


# ---------------------------------------------------------------------------
# sanitize_for_neighbor_loader (from interpret_subgraph)
# ---------------------------------------------------------------------------

class TestSanitizeForNeighborLoaderInterpret:
    def test_non_tensor_attrs_removed(self):
        data = _make_data_with_metadata()
        sanitize_for_neighbor_loader(data)
        # These should be gone
        assert not hasattr(data["a"], "attr_name")
        assert not hasattr(data["a"], "node_ids")
        assert not hasattr(data["a"], "mask_cols")

    def test_tensor_x_kept(self):
        data = _make_data_with_metadata()
        sanitize_for_neighbor_loader(data)
        assert hasattr(data["a"], "x")
        assert isinstance(data["a"].x, torch.Tensor)

    def test_returns_hetero_data(self):
        data = _make_data_with_metadata()
        result = sanitize_for_neighbor_loader(data)
        assert isinstance(result, HeteroData)


# ---------------------------------------------------------------------------
# load_ckpt
# ---------------------------------------------------------------------------

class TestLoadCkpt:
    def test_valid_checkpoint_loaded(self, tmp_path):
        ckpt = tmp_path / "model_epoch001.pt"
        payload = {
            "model_state": {"some_param": torch.tensor(1.0)},
            "args": {"hidden": 64, "layers": 2},
            "epoch": 1,
        }
        torch.save(payload, ckpt)
        result = load_ckpt(ckpt)
        assert "model_state" in result
        assert "args" in result

    def test_missing_model_state_raises(self, tmp_path):
        ckpt = tmp_path / "bad.pt"
        torch.save({"args": {"hidden": 64}}, ckpt)
        with pytest.raises(ValueError, match="Bad checkpoint format"):
            load_ckpt(ckpt)

    def test_missing_args_raises(self, tmp_path):
        ckpt = tmp_path / "bad2.pt"
        torch.save({"model_state": {}}, ckpt)
        with pytest.raises(ValueError, match="Bad checkpoint format"):
            load_ckpt(ckpt)

    def test_both_keys_present_no_error(self, tmp_path):
        ckpt = tmp_path / "ok.pt"
        torch.save({"model_state": {}, "args": {}, "epoch": 5}, ckpt)
        payload = load_ckpt(ckpt)
        assert isinstance(payload, dict)


# ---------------------------------------------------------------------------
# summarize_attrib
# ---------------------------------------------------------------------------

class TestSummarizeAttrib:
    def test_returns_dict_with_expected_keys(self):
        data = _make_data_with_metadata()
        attrib = _make_attrib()
        result = summarize_attrib(data, attrib)
        assert "pred" in result
        assert "type_mass" in result
        assert "top_nodes" in result
        assert "top_feats" in result

    def test_pred_value_preserved(self):
        data = _make_data_with_metadata()
        attrib = _make_attrib()
        result = summarize_attrib(data, attrib)
        assert result["pred"] == 2.5

    def test_type_mass_sorted_by_abs(self):
        data = _make_data_with_metadata()
        # two node types with different mass
        attrib = {
            "pred": 1.0,
            "per_type": {
                "a": {
                    "num_nodes": 5, "num_feats": 3,
                    "node_signed": [1.0] * 5,
                    "node_abs": [10.0, 5.0, 2.0, 1.0, 0.5],
                    "feat_signed": [3.0, 2.0, 1.0],
                    "feat_abs": [3.0, 2.0, 1.0],
                },
                "b": {
                    "num_nodes": 4, "num_feats": 2,
                    "node_signed": [0.1] * 4,
                    "node_abs": [0.1, 0.2, 0.1, 0.1],
                    "feat_signed": [0.1, 0.2],
                    "feat_abs": [0.1, 0.2],
                },
            },
        }
        result = summarize_attrib(data, attrib)
        masses = [d["mass_abs"] for d in result["type_mass"]]
        assert masses == sorted(masses, reverse=True)

    def test_topk_nodes_respected(self):
        data = _make_data_with_metadata()
        attrib = _make_attrib(n_nodes=20, n_feats=3)
        result = summarize_attrib(data, attrib, topk_nodes=3)
        for nodes in result["top_nodes"].values():
            assert len(nodes) <= 3

    def test_topk_feats_respected(self):
        data = _make_data_with_metadata()
        attrib = _make_attrib(n_nodes=5, n_feats=10)
        result = summarize_attrib(data, attrib, topk_feats=2)
        for feats in result["top_feats"].values():
            assert len(feats) <= 2

    def test_feat_names_from_attr_name_map(self):
        data = _make_data_with_metadata()
        attrib = _make_attrib(n_nodes=5, n_feats=3)
        attr_name_map = {"a": ["f0", "f1", "f2"]}
        result = summarize_attrib(data, attrib, attr_name_map=attr_name_map)
        feat_names_in_result = [d["feat"] for d in result["top_feats"]["a"]]
        for name in feat_names_in_result:
            assert name in ["f0", "f1", "f2"]

    def test_type_mass_has_ratio_field(self):
        data = _make_data_with_metadata()
        attrib = _make_attrib()
        result = summarize_attrib(data, attrib)
        for entry in result["type_mass"]:
            assert "ratio" in entry
            assert 0.0 <= entry["ratio"] <= 1.0

    def test_empty_per_type_returns_empty_lists(self):
        data = _make_data_with_metadata()
        attrib = {"pred": 0.0, "per_type": {}}
        result = summarize_attrib(data, attrib)
        assert result["type_mass"] == []
        assert result["top_nodes"] == {}
        assert result["top_feats"] == {}
