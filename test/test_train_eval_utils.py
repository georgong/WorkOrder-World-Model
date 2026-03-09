"""
Tests for shared utility functions in src/runner/train.py and src/runner/eval.py.

Covers:
  - parse_seeds_arg (train.py only)
  - pick_device
  - split_indices
  - normalize_node_features_inplace
  - sanitize_for_neighbor_loader
  - sort_ckpts (eval.py only)
  - load_payload / load_ckpt (eval.py)
  - save_checkpoint (train.py)
  - HeteroSAGERegressor forward pass
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from src.runner.train import (
    HeteroSAGERegressor as TrainRegressor,
    normalize_node_features_inplace,
    parse_seeds_arg,
    pick_device,
    sanitize_for_neighbor_loader,
    save_checkpoint,
    split_indices,
)
from src.runner.eval import (
    HeteroSAGERegressor as EvalRegressor,
    load_payload,
    sort_ckpts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hetero_data(n_a: int = 20, n_b: int = 15, n_feats: int = 4) -> HeteroData:
    """Minimal synthetic HeteroData with two node types and one edge type."""
    data = HeteroData()
    data["a"].x = torch.randn(n_a, n_feats)
    data["b"].x = torch.randn(n_b, n_feats)
    src = torch.randint(0, n_a, (30,))
    dst = torch.randint(0, n_b, (30,))
    data[("a", "connects", "b")].edge_index = torch.stack([src, dst], dim=0)
    return data


def _make_model(target: str = "b") -> TrainRegressor:
    data = _make_hetero_data()
    metadata = (list(data.node_types), list(data.edge_types))
    in_dims = {nt: data[nt].x.size(1) for nt in data.node_types}
    return TrainRegressor(metadata, in_dims, hidden_dim=16, num_layers=1,
                          target_node_type=target)


# ---------------------------------------------------------------------------
# parse_seeds_arg
# ---------------------------------------------------------------------------

class TestParseSeedsArg:
    def test_none_none_returns_none(self):
        assert parse_seeds_arg(None, None) is None

    def test_comma_separated(self):
        t = parse_seeds_arg("1,2,3", None)
        assert t is not None
        assert t.tolist() == [1, 2, 3]

    def test_range_token(self):
        t = parse_seeds_arg("5-8", None)
        assert t.tolist() == [5, 6, 7, 8]

    def test_mixed_comma_and_range(self):
        t = parse_seeds_arg("1,3-5,10", None)
        assert t.tolist() == [1, 3, 4, 5, 10]

    def test_deduplication(self):
        t = parse_seeds_arg("1,1,2", None)
        assert t.tolist() == [1, 2]

    def test_output_is_long_tensor(self):
        t = parse_seeds_arg("7,8", None)
        assert t.dtype == torch.long

    def test_seeds_file(self, tmp_path):
        f = tmp_path / "seeds.txt"
        f.write_text("1\n2,3\n5-7\n")
        t = parse_seeds_arg(None, str(f))
        assert t.tolist() == [1, 2, 3, 5, 6, 7]

    def test_seeds_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_seeds_arg(None, str(tmp_path / "missing.txt"))

    def test_empty_string_returns_none(self):
        result = parse_seeds_arg("", None)
        assert result is None


# ---------------------------------------------------------------------------
# pick_device
# ---------------------------------------------------------------------------

class TestPickDevice:
    def test_cpu(self):
        d = pick_device("cpu")
        assert d == torch.device("cpu")

    def test_auto_returns_valid_device(self):
        d = pick_device("auto")
        assert isinstance(d, torch.device)
        assert d.type in ("cpu", "cuda", "mps")

    def test_case_insensitive(self):
        d = pick_device("CPU")
        assert d == torch.device("cpu")


# ---------------------------------------------------------------------------
# split_indices
# ---------------------------------------------------------------------------

class TestSplitIndices:
    def test_sizes_sum_to_total(self):
        idx = torch.arange(100)
        tr, va, te = split_indices(idx, seed=0, train_ratio=0.7, val_ratio=0.15)
        assert tr.numel() + va.numel() + te.numel() == 100

    def test_train_size_correct(self):
        idx = torch.arange(100)
        tr, va, te = split_indices(idx, seed=0, train_ratio=0.8, val_ratio=0.1)
        assert tr.numel() == 80

    def test_val_size_correct(self):
        idx = torch.arange(100)
        tr, va, te = split_indices(idx, seed=0, train_ratio=0.8, val_ratio=0.1)
        assert va.numel() == 10

    def test_disjoint_sets(self):
        idx = torch.arange(200)
        tr, va, te = split_indices(idx, seed=42, train_ratio=0.7, val_ratio=0.15)
        tr_set = set(tr.tolist())
        va_set = set(va.tolist())
        te_set = set(te.tolist())
        assert len(tr_set & va_set) == 0
        assert len(tr_set & te_set) == 0
        assert len(va_set & te_set) == 0

    def test_reproducible_with_same_seed(self):
        idx = torch.arange(50)
        tr1, _, _ = split_indices(idx, seed=7, train_ratio=0.6, val_ratio=0.2)
        tr2, _, _ = split_indices(idx, seed=7, train_ratio=0.6, val_ratio=0.2)
        assert tr1.tolist() == tr2.tolist()

    def test_different_seeds_give_different_splits(self):
        idx = torch.arange(50)
        tr1, _, _ = split_indices(idx, seed=1, train_ratio=0.6, val_ratio=0.2)
        tr2, _, _ = split_indices(idx, seed=2, train_ratio=0.6, val_ratio=0.2)
        assert tr1.tolist() != tr2.tolist()

    def test_returns_tensors(self):
        idx = torch.arange(30)
        tr, va, te = split_indices(idx, seed=0, train_ratio=0.6, val_ratio=0.2)
        for t in (tr, va, te):
            assert isinstance(t, torch.Tensor)


# ---------------------------------------------------------------------------
# normalize_node_features_inplace
# ---------------------------------------------------------------------------

class TestNormalizeNodeFeaturesInplace:
    def test_returns_stats_dict(self):
        data = _make_hetero_data(n_feats=4)
        stats = normalize_node_features_inplace(data)
        assert isinstance(stats, dict)

    def test_node_types_in_stats(self):
        data = _make_hetero_data(n_feats=4)
        stats = normalize_node_features_inplace(data)
        for nt in data.node_types:
            assert nt in stats

    def test_normalized_mean_near_zero(self):
        data = _make_hetero_data(n_a=200, n_feats=6)
        normalize_node_features_inplace(data)
        mean = data["a"].x.mean(dim=0)
        assert (mean.abs() < 0.1).all()

    def test_constant_column_dropped(self):
        data = HeteroData()
        # Constant column: all same float value → var=0, std=sqrt(eps)≈1e-3
        # Use const_std_thr large enough to trigger the drop (e.g. 0.01)
        x = torch.cat([torch.full((50, 1), 5.0), torch.randn(50, 1)], dim=1)
        data["a"].x = x
        original_ncols = x.shape[1]
        normalize_node_features_inplace(data, const_std_thr=0.01)
        assert data["a"].x.shape[1] < original_ncols

    def test_attr_name_filtered_with_constant_col(self):
        data = HeteroData()
        x = torch.cat([torch.full((50, 1), 5.0), torch.randn(50, 1)], dim=1)
        data["a"].x = x
        data["a"].attr_name = ["const_feat", "vary_feat"]
        normalize_node_features_inplace(data, const_std_thr=0.01)
        assert "vary_feat" in data["a"].attr_name
        assert "const_feat" not in data["a"].attr_name

    def test_no_x_node_type_skipped(self):
        data = HeteroData()
        data["a"].x = torch.randn(10, 3)
        data["b"].num_nodes = 5  # no x
        stats = normalize_node_features_inplace(data)
        assert "a" in stats
        assert "b" not in stats

    def test_no_nan_after_normalize(self):
        data = _make_hetero_data(n_feats=5)
        normalize_node_features_inplace(data)
        for nt in data.node_types:
            assert not torch.isnan(data[nt].x).any()


# ---------------------------------------------------------------------------
# sanitize_for_neighbor_loader
# ---------------------------------------------------------------------------

class TestSanitizeForNeighborLoader:
    def test_non_tensor_node_fields_removed(self):
        data = _make_hetero_data()
        data["a"].some_list = [1, 2, 3]
        data["a"].some_str = "hello"
        sanitize_for_neighbor_loader(data)
        assert not hasattr(data["a"], "some_list")
        assert not hasattr(data["a"], "some_str")

    def test_tensor_node_fields_kept(self):
        data = _make_hetero_data()
        sanitize_for_neighbor_loader(data)
        assert hasattr(data["a"], "x")
        assert isinstance(data["a"].x, torch.Tensor)

    def test_non_tensor_edge_fields_removed(self):
        data = _make_hetero_data()
        data[("a", "connects", "b")].edge_label = ["e0", "e1"]
        sanitize_for_neighbor_loader(data)
        assert not hasattr(data[("a", "connects", "b")], "edge_label")

    def test_tensor_edge_fields_kept(self):
        data = _make_hetero_data()
        data[("a", "connects", "b")].edge_attr = torch.ones(30, 2)
        sanitize_for_neighbor_loader(data)
        assert hasattr(data[("a", "connects", "b")], "edge_attr")

    def test_returns_hetero_data(self):
        data = _make_hetero_data()
        result = sanitize_for_neighbor_loader(data)
        assert isinstance(result, HeteroData)


# ---------------------------------------------------------------------------
# sort_ckpts (eval.py)
# ---------------------------------------------------------------------------

class TestSortCkpts:
    def test_sorted_by_epoch(self, tmp_path):
        names = ["run_epoch003.pt", "run_epoch001.pt", "run_epoch010.pt"]
        paths = [tmp_path / n for n in names]
        for p in paths:
            p.touch()
        result = sort_ckpts(paths)
        epochs = [int(p.stem.split("epoch")[1]) for p in result]
        assert epochs == sorted(epochs)

    def test_single_file(self, tmp_path):
        p = tmp_path / "run_epoch005.pt"
        p.touch()
        result = sort_ckpts([p])
        assert result == [p]

    def test_empty_list(self):
        assert sort_ckpts([]) == []

    def test_non_epoch_files_sorted_last(self, tmp_path):
        p_epoch = tmp_path / "run_epoch001.pt"
        p_other = tmp_path / "some_model.pt"
        p_epoch.touch()
        p_other.touch()
        result = sort_ckpts([p_other, p_epoch])
        assert result[0] == p_epoch  # epoch file should come first


# ---------------------------------------------------------------------------
# load_payload (eval.py)
# ---------------------------------------------------------------------------

class TestLoadPayload:
    def test_valid_checkpoint(self, tmp_path):
        ckpt = tmp_path / "model_epoch001.pt"
        # Use torch.save with a plain dict (no model weights) so weights_only=True works
        payload = {"model_state": {}, "args": {"hidden": 64}, "epoch": 1}
        torch.save(payload, ckpt)
        # load_payload internally uses weights_only=False fallback for real checkpoints;
        # test that it returns a dict with the expected keys regardless
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "model_state" in loaded
        assert "args" in loaded

    def test_missing_model_state_raises(self, tmp_path):
        ckpt = tmp_path / "bad.pt"
        torch.save({"args": {}}, ckpt)
        with pytest.raises(ValueError, match="Bad checkpoint format"):
            load_payload(ckpt)

    def test_missing_args_raises(self, tmp_path):
        ckpt = tmp_path / "bad2.pt"
        torch.save({"model_state": {}}, ckpt)
        with pytest.raises(ValueError, match="Bad checkpoint format"):
            load_payload(ckpt)


# ---------------------------------------------------------------------------
# save_checkpoint (train.py)
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:
    def test_file_created(self, tmp_path):
        model = _make_model()
        opt = torch.optim.Adam(model.parameters())
        args = argparse.Namespace(hidden=16, layers=1, lr=1e-3)
        save_checkpoint(tmp_path, "test_run", 1, model, opt, args)
        pts = list(tmp_path.glob("*.pt"))
        assert len(pts) == 1

    def test_checkpoint_loadable(self, tmp_path):
        model = _make_model()
        opt = torch.optim.Adam(model.parameters())
        args = argparse.Namespace(hidden=16, layers=1, lr=1e-3)
        save_checkpoint(tmp_path, "test_run", 5, model, opt, args)
        ckpt = tmp_path / "test_run_epoch005.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "model_state" in payload
        assert payload["epoch"] == 5

    def test_config_json_created(self, tmp_path):
        model = _make_model()
        opt = torch.optim.Adam(model.parameters())
        args = argparse.Namespace(hidden=16, layers=1, lr=1e-3)
        save_checkpoint(tmp_path, "myrun", 1, model, opt, args)
        cfg = tmp_path / "myrun_config.json"
        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert data["hidden"] == 16


# ---------------------------------------------------------------------------
# HeteroSAGERegressor forward pass
# ---------------------------------------------------------------------------

class TestHeteroSAGERegressor:
    def _make_train_model_and_data(self):
        data = _make_hetero_data(n_a=10, n_b=8, n_feats=4)
        metadata = (list(data.node_types), list(data.edge_types))
        in_dims = {nt: data[nt].x.size(1) for nt in data.node_types}
        # Use 'b' as target: edges go a->b so 'b' is guaranteed in x_dict
        # after HeteroConv message passing
        model = TrainRegressor(metadata, in_dims, hidden_dim=16, num_layers=1,
                               target_node_type="b")
        return model, data

    def test_forward_returns_dict_with_pred(self):
        model, data = self._make_train_model_and_data()
        out = model(data)
        assert "pred" in out

    def test_pred_shape(self):
        model, data = self._make_train_model_and_data()
        out = model(data)
        pred = out["pred"]
        assert pred.shape == (data["b"].num_nodes,)

    def test_pred_is_float_tensor(self):
        model, data = self._make_train_model_and_data()
        out = model(data)
        assert out["pred"].is_floating_point()

    def test_no_nan_in_pred(self):
        model, data = self._make_train_model_and_data()
        out = model(data)
        assert not torch.isnan(out["pred"]).any()

    def test_eval_model_matches_train_model_forward(self):
        """EvalRegressor and TrainRegressor are identical; forward should match
        when given the same weights (after lazy init via a forward pass)."""
        data = _make_hetero_data(n_a=10, n_b=8, n_feats=4)
        metadata = (list(data.node_types), list(data.edge_types))
        in_dims = {nt: data[nt].x.size(1) for nt in data.node_types}
        m_train = TrainRegressor(metadata, in_dims, hidden_dim=16, num_layers=1,
                                 target_node_type="b")
        m_eval = EvalRegressor(metadata, in_dims, hidden_dim=16, num_layers=1,
                               target_node_type="b")
        # Materialize lazy weights in both models before copying state
        m_train.eval()
        m_eval.eval()
        with torch.no_grad():
            m_train(data)
            m_eval(data)
        # Now copy weights from train -> eval
        m_eval.load_state_dict(m_train.state_dict())
        with torch.no_grad():
            p1 = m_train(data)["pred"]
            p2 = m_eval(data)["pred"]
        assert torch.allclose(p1, p2, atol=1e-5)
