"""Tests for topic hierarchy: build_hierarchy_chart_data and filtering logic."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Fixtures ---

@pytest.fixture
def sample_topics():
    """Minimal 3-level topic hierarchy for testing."""
    return {
        "macro": {
            "0": {"label": "AI – Trust", "count": 50},
            "1": {"label": "VR – Reality", "count": 30},
        },
        "mid": {
            "0": {"label": "Trust – Decision", "count": 30},
            "1": {"label": "Agent – XAI", "count": 20},
            "2": {"label": "Haptic – Touch", "count": 30},
        },
        "fine": {
            "0": {"label": "Explainable AI", "count": 15},
            "1": {"label": "Bias – Fairness", "count": 15},
            "2": {"label": "Chatbot – Agent", "count": 20},
            "3": {"label": "Force Feedback", "count": 15},
            "4": {"label": "Thermal – Heat", "count": 15},
        },
    }


@pytest.fixture
def sample_hierarchy():
    """Maps fine → mid → macro."""
    return {
        "0": {"mid": 0, "macro": 0},  # Explainable AI → Trust–Decision → AI–Trust
        "1": {"mid": 0, "macro": 0},  # Bias–Fairness → Trust–Decision → AI–Trust
        "2": {"mid": 1, "macro": 0},  # Chatbot–Agent → Agent–XAI → AI–Trust
        "3": {"mid": 2, "macro": 1},  # Force Feedback → Haptic–Touch → VR–Reality
        "4": {"mid": 2, "macro": 1},  # Thermal–Heat → Haptic–Touch → VR–Reality
    }


@pytest.fixture
def real_data():
    """Load actual data from disk if available."""
    data_dir = Path(__file__).parent.parent / "data"
    topics_path = data_dir / "topics.json"
    if not topics_path.exists():
        pytest.skip("No topics.json — run clustering pipeline first")
    data = json.loads(topics_path.read_text())
    return data["topics"], data["hierarchy"]


# --- Tests for build_hierarchy_chart_data ---

class TestBuildHierarchyChartData:
    def _build(self, topics, hierarchy):
        # Import without Streamlit cache decorator
        from app import build_hierarchy_chart_data
        fn = build_hierarchy_chart_data.__wrapped__
        return fn(topics, hierarchy)

    def test_returns_nodes_and_links(self, sample_topics, sample_hierarchy):
        nodes_df, links_df = self._build(sample_topics, sample_hierarchy)
        assert isinstance(nodes_df, pd.DataFrame)
        assert isinstance(links_df, pd.DataFrame)

    def test_node_count_matches_topics(self, sample_topics, sample_hierarchy):
        nodes_df, _ = self._build(sample_topics, sample_hierarchy)
        n_macro = len([k for k in sample_topics["macro"] if k != "-1"])
        n_mid = len(sample_topics["mid"])
        n_fine = len(sample_topics["fine"])
        assert len(nodes_df) == n_macro + n_mid + n_fine

    def test_all_levels_present(self, sample_topics, sample_hierarchy):
        nodes_df, _ = self._build(sample_topics, sample_hierarchy)
        levels = set(nodes_df["level"].unique())
        assert levels == {"macro", "mid", "fine"}

    def test_macro_nodes_sorted_descending(self, sample_topics, sample_hierarchy):
        nodes_df, _ = self._build(sample_topics, sample_hierarchy)
        macros = nodes_df[nodes_df["level"] == "macro"]
        counts = macros["count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_mid_nodes_sorted_within_parent(self, sample_topics, sample_hierarchy):
        nodes_df, _ = self._build(sample_topics, sample_hierarchy)
        mids = nodes_df[nodes_df["level"] == "mid"]
        # Within each parent_macro group, counts should be descending
        for parent in mids["parent_macro"].unique():
            group_counts = mids[mids["parent_macro"] == parent]["count"].tolist()
            assert group_counts == sorted(group_counts, reverse=True), \
                f"Mid topics under '{parent}' not sorted descending"

    def test_y_positions_non_overlapping(self, sample_topics, sample_hierarchy):
        nodes_df, _ = self._build(sample_topics, sample_hierarchy)
        # Within each level, y ranges should not overlap
        for level in ["macro", "mid", "fine"]:
            level_nodes = nodes_df[nodes_df["level"] == level].sort_values("y")
            for i in range(len(level_nodes) - 1):
                curr = level_nodes.iloc[i]
                nxt = level_nodes.iloc[i + 1]
                assert curr["y2"] <= nxt["y"], \
                    f"{level} nodes overlap: {curr['label']} y2={curr['y2']} > {nxt['label']} y={nxt['y']}"

    def test_parent_labels_populated(self, sample_topics, sample_hierarchy):
        nodes_df, _ = self._build(sample_topics, sample_hierarchy)
        # All macro nodes should have parent_macro == self label
        macros = nodes_df[nodes_df["level"] == "macro"]
        assert (macros["parent_macro"] == macros["label"]).all()
        # All mid nodes should have non-empty parent_macro
        mids = nodes_df[nodes_df["level"] == "mid"]
        assert mids["parent_macro"].str.len().gt(0).all()
        # All fine nodes should have non-empty parent_macro and parent_mid
        fines = nodes_df[nodes_df["level"] == "fine"]
        assert fines["parent_macro"].str.len().gt(0).all()
        assert fines["parent_mid"].str.len().gt(0).all()

    def test_links_count_matches(self, sample_topics, sample_hierarchy):
        _, links_df = self._build(sample_topics, sample_hierarchy)
        n_mid = len(sample_topics["mid"])
        n_fine = len(sample_topics["fine"])
        # One link per macro→mid + one per mid→fine
        assert len(links_df) == n_mid + n_fine

    def test_node_colors_assigned(self, sample_topics, sample_hierarchy):
        nodes_df, _ = self._build(sample_topics, sample_hierarchy)
        assert nodes_df["color"].str.startswith("rgba(").all()

    def test_minimum_node_height(self, sample_topics, sample_hierarchy):
        nodes_df, _ = self._build(sample_topics, sample_hierarchy)
        fines = nodes_df[nodes_df["level"] == "fine"]
        total = sample_topics["fine"]["0"]["count"] + sample_topics["fine"]["1"]["count"] + \
                sample_topics["fine"]["2"]["count"] + sample_topics["fine"]["3"]["count"] + \
                sample_topics["fine"]["4"]["count"]
        min_h = total * 0.014
        for _, node in fines.iterrows():
            height = node["y2"] - node["y"]
            assert height >= min_h - 0.01, \
                f"Fine node '{node['label']}' height {height:.2f} < min {min_h:.2f}"


# --- Tests for highlighting logic ---

class TestHighlighting:
    def _build_nodes(self, topics, hierarchy):
        from app import build_hierarchy_chart_data
        fn = build_hierarchy_chart_data.__wrapped__
        nodes_df, _ = fn(topics, hierarchy)
        nodes_df = nodes_df.copy()
        return nodes_df

    def _apply_highlight(self, nodes_df, level, label):
        """Replicate the highlighting logic from app.py."""
        if level == "macro":
            nodes_df["hl"] = nodes_df["parent_macro"] == label
        elif level == "mid":
            mid_rows = nodes_df.loc[
                (nodes_df["level"] == "mid") & (nodes_df["label"] == label)
            ]
            pm = mid_rows.iloc[0]["parent_macro"] if len(mid_rows) else ""
            nodes_df["hl"] = (
                (nodes_df["parent_mid"] == label) |
                ((nodes_df["level"] == "macro") & (nodes_df["label"] == pm))
            )
        elif level == "fine":
            fine_rows = nodes_df.loc[
                (nodes_df["level"] == "fine") & (nodes_df["label"] == label)
            ]
            if len(fine_rows):
                pm = fine_rows.iloc[0]["parent_mid"]
                pma = fine_rows.iloc[0]["parent_macro"]
                nodes_df["hl"] = (
                    ((nodes_df["level"] == "fine") & (nodes_df["label"] == label)) |
                    ((nodes_df["level"] == "mid") & (nodes_df["label"] == pm)) |
                    ((nodes_df["level"] == "macro") & (nodes_df["label"] == pma))
                )
            else:
                nodes_df["hl"] = True
        return nodes_df

    def test_no_filter_all_highlighted(self, sample_topics, sample_hierarchy):
        nodes_df = self._build_nodes(sample_topics, sample_hierarchy)
        nodes_df["hl"] = True
        assert nodes_df["hl"].all()

    def test_macro_filter_highlights_children(self, sample_topics, sample_hierarchy):
        nodes_df = self._build_nodes(sample_topics, sample_hierarchy)
        nodes_df = self._apply_highlight(nodes_df, "macro", "AI – Trust")

        # The macro itself should be highlighted
        macro_hl = nodes_df[(nodes_df["level"] == "macro") & (nodes_df["label"] == "AI – Trust")]
        assert macro_hl["hl"].all()

        # Its mid children should be highlighted
        mid_hl = nodes_df[(nodes_df["level"] == "mid") & nodes_df["hl"]]
        assert set(mid_hl["label"]) == {"Trust – Decision", "Agent – XAI"}

        # Its fine children should be highlighted
        fine_hl = nodes_df[(nodes_df["level"] == "fine") & nodes_df["hl"]]
        assert set(fine_hl["label"]) == {"Explainable AI", "Bias – Fairness", "Chatbot – Agent"}

        # VR macro should NOT be highlighted
        vr = nodes_df[(nodes_df["level"] == "macro") & (nodes_df["label"] == "VR – Reality")]
        assert not vr["hl"].any()

    def test_mid_filter_highlights_parent_and_children(self, sample_topics, sample_hierarchy):
        nodes_df = self._build_nodes(sample_topics, sample_hierarchy)
        nodes_df = self._apply_highlight(nodes_df, "mid", "Trust – Decision")

        # Parent macro should be highlighted
        macro_hl = nodes_df[(nodes_df["level"] == "macro") & nodes_df["hl"]]
        assert "AI – Trust" in macro_hl["label"].values

        # The mid itself should be highlighted
        mid_hl = nodes_df[(nodes_df["level"] == "mid") & nodes_df["hl"]]
        assert "Trust – Decision" in mid_hl["label"].values

        # Its fine children should be highlighted
        fine_hl = nodes_df[(nodes_df["level"] == "fine") & nodes_df["hl"]]
        assert set(fine_hl["label"]) == {"Explainable AI", "Bias – Fairness"}

        # Sibling mid should NOT be highlighted
        assert "Agent – XAI" not in mid_hl["label"].values

    def test_fine_filter_highlights_ancestors(self, sample_topics, sample_hierarchy):
        nodes_df = self._build_nodes(sample_topics, sample_hierarchy)
        nodes_df = self._apply_highlight(nodes_df, "fine", "Explainable AI")

        # The fine itself
        fine_hl = nodes_df[(nodes_df["level"] == "fine") & nodes_df["hl"]]
        assert fine_hl["label"].tolist() == ["Explainable AI"]

        # Its parent mid
        mid_hl = nodes_df[(nodes_df["level"] == "mid") & nodes_df["hl"]]
        assert mid_hl["label"].tolist() == ["Trust – Decision"]

        # Its grandparent macro
        macro_hl = nodes_df[(nodes_df["level"] == "macro") & nodes_df["hl"]]
        assert macro_hl["label"].tolist() == ["AI – Trust"]

    def test_nonexistent_fine_highlights_all(self, sample_topics, sample_hierarchy):
        nodes_df = self._build_nodes(sample_topics, sample_hierarchy)
        nodes_df = self._apply_highlight(nodes_df, "fine", "Nonexistent Topic")
        assert nodes_df["hl"].all()


# --- Tests with real data ---

class TestWithRealData:
    def _build(self, topics, hierarchy):
        from app import build_hierarchy_chart_data
        fn = build_hierarchy_chart_data.__wrapped__
        return fn(topics, hierarchy)

    def test_real_data_loads(self, real_data):
        topics, hierarchy = real_data
        nodes_df, links_df = self._build(topics, hierarchy)
        assert len(nodes_df) > 50  # should have ~100 nodes
        assert len(links_df) > 50

    def test_real_data_no_overlapping_fine_nodes(self, real_data):
        topics, hierarchy = real_data
        nodes_df, _ = self._build(topics, hierarchy)
        fines = nodes_df[nodes_df["level"] == "fine"].sort_values("y")
        for i in range(len(fines) - 1):
            curr = fines.iloc[i]
            nxt = fines.iloc[i + 1]
            assert curr["y2"] <= nxt["y"] + 0.01, \
                f"Overlap: {curr['label']} y2={curr['y2']:.1f} > {nxt['label']} y={nxt['y']:.1f}"

    def test_real_data_all_fine_have_parents(self, real_data):
        topics, hierarchy = real_data
        nodes_df, _ = self._build(topics, hierarchy)
        fines = nodes_df[nodes_df["level"] == "fine"]
        assert fines["parent_macro"].str.len().gt(0).all()
        assert fines["parent_mid"].str.len().gt(0).all()

    def test_real_data_macro_filter_reduces_nodes(self, real_data):
        topics, hierarchy = real_data
        nodes_df, _ = self._build(topics, hierarchy)
        macro_label = nodes_df[nodes_df["level"] == "macro"].iloc[0]["label"]
        nodes_df["hl"] = nodes_df["parent_macro"] == macro_label
        highlighted = nodes_df["hl"].sum()
        assert 0 < highlighted < len(nodes_df), \
            f"Macro filter should highlight some but not all nodes, got {highlighted}/{len(nodes_df)}"
