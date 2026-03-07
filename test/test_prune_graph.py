"""
Tests for src/process/prune_graph.py

Covers:
  - compute_node_degree_all: correct degree counting
  - prune_isolated_nodes: isolated nodes removed, connected nodes kept,
    edge re-indexing, non-tensor field re-attachment, min_degree kwarg
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from src.process.prune_graph import compute_node_degree_all, prune_isolated_nodes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_graph() -> HeteroData:
    """
    Two node types: 'a' (4 nodes), 'b' (3 nodes)
    Edges a->b: 0->0, 1->1, 2->2   (node a[3] and no b[?] are isolated)
    """
    data = HeteroData()
    data["a"].x = torch.randn(4, 2)
    data["b"].x = torch.randn(3, 2)
    # edges: a[0]->b[0], a[1]->b[1], a[2]->b[2]   a[3] has degree 0
    data[("a", "connects", "b")].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    return data


def _make_graph_with_non_tensor() -> HeteroData:
    """Same as above but adds list / string metadata on nodes."""
    data = _make_simple_graph()
    data["a"].node_ids = ["a0", "a1", "a2", "a3"]  # list field
    data["b"].attr_name = ["f0", "f1"]              # list field
    return data


def _make_disconnected_graph() -> HeteroData:
    """All nodes are isolated (no edges at all)."""
    data = HeteroData()
    data["a"].x = torch.randn(5, 2)
    return data


# ---------------------------------------------------------------------------
# compute_node_degree_all
# ---------------------------------------------------------------------------

class TestComputeNodeDegreeAll:
    def test_returns_dict_for_all_node_types(self):
        data = _make_simple_graph()
        deg = compute_node_degree_all(data)
        assert set(deg.keys()) == {"a", "b"}

    def test_connected_nodes_have_positive_degree(self):
        data = _make_simple_graph()
        deg = compute_node_degree_all(data)
        # a[0..2] are sources -> degree >= 1
        assert deg["a"][0] >= 1
        assert deg["a"][1] >= 1
        assert deg["a"][2] >= 1
        # b[0..2] are destinations -> degree >= 1
        assert deg["b"][0] >= 1
        assert deg["b"][1] >= 1
        assert deg["b"][2] >= 1

    def test_isolated_node_has_zero_degree(self):
        data = _make_simple_graph()
        deg = compute_node_degree_all(data)
        # a[3] has no edges at all
        assert deg["a"][3].item() == 0

    def test_degree_tensor_shape(self):
        data = _make_simple_graph()
        deg = compute_node_degree_all(data)
        assert deg["a"].shape == (4,)
        assert deg["b"].shape == (3,)

    def test_no_edges_graph(self):
        data = _make_disconnected_graph()
        deg = compute_node_degree_all(data)
        assert deg["a"].sum().item() == 0

    def test_dtype_is_long(self):
        data = _make_simple_graph()
        deg = compute_node_degree_all(data)
        for v in deg.values():
            assert v.dtype == torch.long


# ---------------------------------------------------------------------------
# prune_isolated_nodes
# ---------------------------------------------------------------------------

class TestPruneIsolatedNodes:
    def test_isolated_node_removed(self):
        data = _make_simple_graph()
        pruned = prune_isolated_nodes(data, min_degree=1)
        # a[3] was isolated; only 3 'a' nodes remain
        assert pruned["a"].num_nodes == 3

    def test_connected_nodes_kept(self):
        data = _make_simple_graph()
        pruned = prune_isolated_nodes(data, min_degree=1)
        assert pruned["b"].num_nodes == 3

    def test_edge_index_valid_after_pruning(self):
        data = _make_simple_graph()
        pruned = prune_isolated_nodes(data, min_degree=1)
        ei = pruned[("a", "connects", "b")].edge_index
        assert ei.shape[0] == 2
        # all indices must be within bounds
        assert ei[0].max().item() < pruned["a"].num_nodes
        assert ei[1].max().item() < pruned["b"].num_nodes

    def test_edge_count_preserved(self):
        data = _make_simple_graph()
        pruned = prune_isolated_nodes(data, min_degree=1)
        ei = pruned[("a", "connects", "b")].edge_index
        assert ei.shape[1] == 3

    def test_non_tensor_fields_reattached(self):
        data = _make_graph_with_non_tensor()
        pruned = prune_isolated_nodes(data, min_degree=1)
        # node_ids list should be sliced to 3 elements (a[0..2] kept)
        assert hasattr(pruned["a"], "node_ids")
        assert len(pruned["a"].node_ids) == 3
        # b attr_name should survive (length==num_features, not num_nodes, so kept as-is)
        assert hasattr(pruned["b"], "attr_name")

    def test_min_degree_zero_keeps_all(self):
        data = _make_simple_graph()
        pruned = prune_isolated_nodes(data, min_degree=0)
        assert pruned["a"].num_nodes == 4
        assert pruned["b"].num_nodes == 3

    def test_returns_hetero_data(self):
        data = _make_simple_graph()
        pruned = prune_isolated_nodes(data, min_degree=1)
        assert isinstance(pruned, HeteroData)

    def test_feature_tensor_shape_preserved(self):
        data = _make_simple_graph()
        pruned = prune_isolated_nodes(data, min_degree=1)
        # 3 surviving 'a' nodes, 2 features
        assert pruned["a"].x.shape == (3, 2)
        assert pruned["b"].x.shape == (3, 2)
