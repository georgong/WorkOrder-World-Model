"""
Tests for src/process/graph_connectivity.py

Covers:
  - compute_type_connectivity_heatmap: ratio/count maps correct
  - compute_second_order_connectivity: full / partial / missing edge scenarios
  - MetaPathResult fields

NOTE: graph_connectivity.py has a module-level call to analyze_graph_connectivity()
that tries to load data/graph/sdge.pt. We patch torch.load at import time to
prevent the FileNotFoundError during collection.
"""
from __future__ import annotations
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import HeteroData

# Patch torch.load before importing so the module-level analyze_graph_connectivity()
# call doesn't blow up when sdge.pt is absent.
_dummy_data = HeteroData()
_dummy_data["a"].x = torch.zeros(2, 1)

with patch("torch.load", return_value=_dummy_data), \
     patch("builtins.print"):
    from src.process.graph_connectivity import (
        MetaPathResult,
        compute_type_connectivity_heatmap,
        compute_second_order_connectivity,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_two_type_graph() -> HeteroData:
    """
    node types: 'a' (4 nodes), 'b' (3 nodes)
    edges a->b: 0->0, 1->1, 2->2   (a[3] has no outgoing edges)
    """
    data = HeteroData()
    data["a"].x = torch.zeros(4, 1)
    data["b"].x = torch.zeros(3, 1)
    data[("a", "connects", "b")].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    return data


def _make_three_type_graph() -> HeteroData:
    """
    node types: 'a' (4), 'b' (3), 'c' (2)
    edges a->b: 0->0, 1->1, 2->2
    edges b->c: 0->0, 1->1, 2->1  (all b nodes reach c)
    """
    data = HeteroData()
    data["a"].x = torch.zeros(4, 1)
    data["b"].x = torch.zeros(3, 1)
    data["c"].x = torch.zeros(2, 1)
    data[("a", "to_b", "b")].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data[("b", "to_c", "c")].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 1]], dtype=torch.long
    )
    return data


def _make_partial_metapath_graph() -> HeteroData:
    """
    node types: 'a' (4), 'b' (3), 'c' (2)
    edges a->b: only a[0]->b[0], a[1]->b[1]
    edges b->c: only b[0]->c[0]  (b[1] and b[2] do NOT connect to c)
    """
    data = HeteroData()
    data["a"].x = torch.zeros(4, 1)
    data["b"].x = torch.zeros(3, 1)
    data["c"].x = torch.zeros(2, 1)
    data[("a", "to_b", "b")].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[("b", "to_c", "c")].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    return data


# ---------------------------------------------------------------------------
# compute_type_connectivity_heatmap
# ---------------------------------------------------------------------------

class TestComputeTypeConnectivityHeatmap:
    def test_returns_two_dicts(self):
        data = _make_two_type_graph()
        result = compute_type_connectivity_heatmap(data)
        assert isinstance(result, tuple) and len(result) == 2

    def test_ratio_for_connected_pair(self):
        data = _make_two_type_graph()
        ratio_map, count_map = compute_type_connectivity_heatmap(data)
        # 3 out of 4 'a' nodes connect to 'b'
        assert abs(ratio_map["a"]["b"] - 3 / 4) < 1e-6

    def test_count_for_connected_pair(self):
        data = _make_two_type_graph()
        ratio_map, count_map = compute_type_connectivity_heatmap(data)
        assert count_map["a"]["b"] == 3

    def test_zero_ratio_for_non_existent_direction(self):
        data = _make_two_type_graph()
        ratio_map, _ = compute_type_connectivity_heatmap(data)
        # no b->a edges
        assert ratio_map["b"]["a"] == 0.0

    def test_ratio_between_zero_and_one(self):
        data = _make_three_type_graph()
        ratio_map, _ = compute_type_connectivity_heatmap(data)
        for src in ratio_map:
            for dst in ratio_map[src]:
                assert 0.0 <= ratio_map[src][dst] <= 1.0

    def test_all_node_types_in_map(self):
        data = _make_three_type_graph()
        ratio_map, count_map = compute_type_connectivity_heatmap(data)
        for nt in ["a", "b", "c"]:
            assert nt in ratio_map
            assert nt in count_map

    def test_self_loop_pair_zero_when_no_self_loops(self):
        data = _make_two_type_graph()
        ratio_map, _ = compute_type_connectivity_heatmap(data)
        # no a->a edges
        assert ratio_map["a"]["a"] == 0.0

    def test_full_connectivity(self):
        """All 3 a-nodes connect to b => ratio == 3/4"""
        data = HeteroData()
        data["a"].x = torch.zeros(3, 1)
        data["b"].x = torch.zeros(3, 1)
        data[("a", "e", "b")].edge_index = torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.long
        )
        ratio_map, _ = compute_type_connectivity_heatmap(data)
        assert abs(ratio_map["a"]["b"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_second_order_connectivity
# ---------------------------------------------------------------------------

class TestComputeSecondOrderConnectivity:
    def test_returns_list_of_metapath_results(self):
        data = _make_three_type_graph()
        results = compute_second_order_connectivity(data, [("a", "b", "c")])
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], MetaPathResult)

    def test_full_reachability(self):
        """All 3 a[0..2] connect to b[0..2], all b[0..2] connect to c => frac=3/4"""
        data = _make_three_type_graph()
        results = compute_second_order_connectivity(data, [("a", "b", "c")])
        r = results[0]
        assert r.missing_edges is False
        assert r.missing_legs == []
        # a[0..2] -> b[0..2] -> c; a[3] has no outgoing edge => 3/4
        assert abs(r.frac_A_reach_C_via_B - 3 / 4) < 1e-6
        assert r.num_A_reachable == 3
        assert r.num_A_total == 4

    def test_partial_reachability(self):
        """a[0]->b[0]->c[0]; a[1]->b[1] but b[1] doesn't reach c => 1/4"""
        data = _make_partial_metapath_graph()
        results = compute_second_order_connectivity(data, [("a", "b", "c")])
        r = results[0]
        assert r.missing_edges is False
        assert r.num_A_reachable == 1
        assert abs(r.frac_A_reach_C_via_B - 1 / 4) < 1e-6

    def test_missing_node_type(self):
        """Metapath with non-existent node type -> missing_edges True"""
        data = _make_two_type_graph()  # only 'a' and 'b'
        results = compute_second_order_connectivity(data, [("a", "b", "c")])
        r = results[0]
        assert r.missing_edges is True
        assert r.frac_A_reach_C_via_B == 0.0

    def test_missing_edge_leg(self):
        """Node types exist but no b->c edges => missing_edges True"""
        data = HeteroData()
        data["a"].x = torch.zeros(3, 1)
        data["b"].x = torch.zeros(3, 1)
        data["c"].x = torch.zeros(2, 1)
        # only a->b, no b->c
        data[("a", "to_b", "b")].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        results = compute_second_order_connectivity(data, [("a", "b", "c")])
        r = results[0]
        assert r.missing_edges is True
        assert ("b", "c") in r.missing_legs

    def test_multiple_metapaths(self):
        data = _make_three_type_graph()
        metapaths = [("a", "b", "c"), ("b", "c", "a")]
        results = compute_second_order_connectivity(data, metapaths)
        assert len(results) == 2

    def test_metapath_result_fields(self):
        data = _make_three_type_graph()
        results = compute_second_order_connectivity(data, [("a", "b", "c")])
        r = results[0]
        assert r.metapath == ("a", "b", "c")
        assert isinstance(r.num_A_total, int)
        assert isinstance(r.num_A_reachable, int)
        assert isinstance(r.frac_A_reach_C_via_B, float)
        assert isinstance(r.missing_legs, list)

    def test_empty_metapaths_list(self):
        data = _make_three_type_graph()
        results = compute_second_order_connectivity(data, [])
        assert results == []
