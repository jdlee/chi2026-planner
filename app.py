"""CHI 2026 Paper Viewer & Agenda Planner — Streamlit App"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from export import generate_ics, generate_markdown

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

MACRO_COLORS = [
    "rgba(232,213,183,0.8)", "rgba(183,213,232,0.8)", "rgba(213,232,183,0.8)",
    "rgba(232,183,213,0.8)", "rgba(183,232,213,0.8)", "rgba(232,201,183,0.8)",
    "rgba(183,201,232,0.8)", "rgba(201,232,183,0.8)", "rgba(232,183,201,0.8)",
    "rgba(201,183,232,0.8)", "rgba(213,183,232,0.8)", "rgba(183,232,201,0.8)",
    "rgba(232,213,201,0.8)", "rgba(201,213,232,0.8)", "rgba(213,201,183,0.8)",
]

st.set_page_config(page_title="CHI 2026 Planner", layout="wide")


@st.cache_data
def load_data():
    """Load clustered papers and topics."""
    clustered_path = DATA_DIR / "chi2026_clustered.json"
    topics_path = DATA_DIR / "topics.json"

    if not clustered_path.exists():
        st.error(
            "No clustered data found. Run the clustering pipeline first:\n"
            "```\npython chi_pipeline.py cluster\n```"
        )
        st.stop()

    papers = json.loads(clustered_path.read_text())
    topics_data = json.loads(topics_path.read_text()) if topics_path.exists() else {}
    topic_names = topics_data.get("topic_names", [])
    topics = topics_data.get("topics", {})
    hierarchy = topics_data.get("hierarchy", {})
    return papers, topic_names, topics, hierarchy



def init_selection():
    """Initialize or load paper selection state."""
    if "selected" not in st.session_state:
        saved = DATA_DIR / "selection.json"
        if saved.exists():
            st.session_state.selected = set(json.loads(saved.read_text()))
        else:
            st.session_state.selected = set()


def save_selection():
    """Persist selection to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "selection.json"
    path.write_text(json.dumps(sorted(st.session_state.selected)))


@st.cache_data
def build_dataframe(papers, topic_names):
    """Build a DataFrame from papers with topic score columns."""
    df = pd.DataFrame(papers)
    df["_index"] = range(len(papers))

    if "topic_scores" in df.columns:
        scores_df = pd.json_normalize(df["topic_scores"])
        for col in topic_names:
            if col not in scores_df.columns:
                scores_df[col] = 0.0
        df = pd.concat([df.drop(columns=["topic_scores"]), scores_df[topic_names]], axis=1)

    return df


def _has_sentence_transformers():
    """Check if sentence-transformers is available."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_resource
def _load_embedding_model():
    """Load sentence-transformers model (cached across reruns)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def compute_paper_embeddings(papers) -> np.ndarray:
    """Embed paper title + abstract. Returns (n_papers, 384) array."""
    embeddings_path = DATA_DIR / "paper_embeddings.npy"
    # Use cached file if it matches the current paper count
    if embeddings_path.exists():
        emb = np.load(str(embeddings_path))
        if emb.shape[0] == len(papers):
            return emb

    if not _has_sentence_transformers():
        return None

    model = _load_embedding_model()
    texts = [
        f"{p.get('title', '')}. {p.get('abstract', '')}" for p in papers
    ]
    emb = model.encode(texts, show_progress_bar=True, batch_size=64)
    np.save(str(embeddings_path), emb)
    return emb


@st.cache_data
def _build_vocabulary_embeddings(papers) -> tuple[list[str], np.ndarray, dict]:
    """Pre-compute embeddings for all unique words across all papers.

    Returns (word_list, word_embeddings, paper_word_indices).
    paper_word_indices maps paper index -> list of indices into word_list.
    """
    model = _load_embedding_model()

    vocab_path = DATA_DIR / "vocab_embeddings.npy"
    vocab_words_path = DATA_DIR / "vocab_words.json"

    # Collect unique words per paper
    all_words = set()
    paper_words = {}
    for i, p in enumerate(papers):
        text = f"{p.get('title', '')} {p.get('abstract', '')}"
        words = set()
        for w in text.lower().split():
            w = w.strip(".,;:!?()\"'[]{}:")
            if len(w) > 3:
                words.add(w)
        paper_words[i] = words
        all_words |= words

    word_list = sorted(all_words)

    # Check cache
    if vocab_path.exists() and vocab_words_path.exists():
        cached_words = json.loads(vocab_words_path.read_text())
        if cached_words == word_list:
            word_embs = np.load(str(vocab_path))
            return word_list, word_embs, paper_words

    word_embs = model.encode(word_list, batch_size=512, show_progress_bar=True)
    np.save(str(vocab_path), word_embs)
    vocab_words_path.write_text(json.dumps(word_list))

    return word_list, word_embs, paper_words


def semantic_search(query: str, paper_embeddings: np.ndarray, papers: list[dict],
                    threshold: float = 0.33) -> pd.DataFrame:
    """Search papers by semantic similarity to query.

    Returns DataFrame with columns: _index, search_score, search_keywords.
    Papers below threshold get score=NaN, keywords="".
    """
    model = _load_embedding_model()
    query_emb = model.encode([query])  # (1, 384)

    # Cosine similarity (vectorized)
    norms_p = np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
    norms_q = np.linalg.norm(query_emb, axis=1, keepdims=True)
    similarities = (paper_embeddings @ query_emb.T) / (norms_p * norms_q.T + 1e-10)
    scores = similarities.flatten()

    # Get pre-computed vocabulary embeddings
    word_list, word_embs, paper_words = _build_vocabulary_embeddings(papers)

    # Compute similarity of all vocab words to query (vectorized, instant)
    word_sims = (word_embs @ query_emb.T).flatten() / (
        np.linalg.norm(word_embs, axis=1) * np.linalg.norm(query_emb) + 1e-10
    )
    word_sim_map = dict(zip(word_list, word_sims))

    query_terms = set(query.lower().split())
    matching_indices = [i for i, s in enumerate(scores) if s >= threshold]

    # For each matching paper, pick top 3 unique-stem keywords
    keywords_list = [""] * len(papers)
    for i in matching_indices:
        words = paper_words.get(i, set())
        ranked = sorted(
            [(w, word_sim_map.get(w, 0)) for w in words if w not in query_terms],
            key=lambda x: -x[1],
        )
        top_words = []
        seen_stems = set()
        for w, sim in ranked:
            if sim < 0.2:
                break
            stem = w[:5]
            if stem not in seen_stems:
                seen_stems.add(stem)
                top_words.append(w)
            if len(top_words) == 3:
                break
        keywords_list[i] = ", ".join(top_words)

    result = pd.DataFrame({
        "_index": range(len(papers)),
        "search_score": scores,
        "search_keywords": keywords_list,
    })

    result.loc[result["search_score"] < threshold, "search_score"] = np.nan
    result.loc[result["search_score"].isna(), "search_keywords"] = ""

    return result


@st.cache_data
def build_hierarchy_chart_data(topics: dict, hierarchy: dict):
    """Build node and link DataFrames for the Altair strip chart.

    Returns (nodes_df, links_df) with layout positions computed so that
    children are grouped under their parent to minimise crossing.
    """
    macro_topics = topics.get("macro", {})
    mid_topics = topics.get("mid", {})
    fine_topics = topics.get("fine", {})

    # --- parent maps ---
    mid_parent = {}   # mid_id -> macro_id
    fine_parent = {}  # fine_id -> mid_id
    for fine_id, links in hierarchy.items():
        fine_parent[fine_id] = str(links["mid"])
        mid_parent[str(links["mid"])] = str(links["macro"])

    # --- ordering (grouped, descending size) ---
    macro_order = sorted(
        [k for k in macro_topics if k != "-1"],
        key=lambda k: macro_topics[k]["count"], reverse=True,
    )
    mid_order = []
    for macro_id in macro_order:
        mid_order.extend(sorted(
            [m for m, p in mid_parent.items() if p == macro_id],
            key=lambda m: mid_topics.get(m, {}).get("count", 0), reverse=True,
        ))
    fine_order = []
    for mid_id in mid_order:
        fine_order.extend(sorted(
            [f for f, p in fine_parent.items() if p == mid_id],
            key=lambda f: fine_topics.get(f, {}).get("count", 0), reverse=True,
        ))

    total = sum(fine_topics.get(f, {}).get("count", 0) for f in fine_order)
    if total == 0:
        total = 1
    gap = total * 0.008  # gap between mid-level groups
    # Minimum height per fine node so text labels don't overlap
    min_node_height = total * 0.014

    # --- compute y extents per node ---
    # Fine level: stack sequentially with minimum height
    fine_y0 = {}
    fine_y1 = {}
    cum = 0.0
    prev_mid = None
    for fid in fine_order:
        mid_id = fine_parent[fid]
        if prev_mid is not None and mid_id != prev_mid:
            cum += gap
        prev_mid = mid_id
        cnt = fine_topics.get(fid, {}).get("count", 0)
        height = max(cnt, min_node_height)
        fine_y0[fid] = cum
        cum += height
        fine_y1[fid] = cum

    # Mid level: span of its children
    mid_y0 = {}
    mid_y1 = {}
    for mid_id in mid_order:
        children = [f for f in fine_order if fine_parent[f] == mid_id]
        if children:
            mid_y0[mid_id] = fine_y0[children[0]]
            mid_y1[mid_id] = fine_y1[children[-1]]
        else:
            mid_y0[mid_id] = 0
            mid_y1[mid_id] = 0

    # Macro level: span of its mid children
    macro_y0 = {}
    macro_y1 = {}
    for macro_id in macro_order:
        children = [m for m in mid_order if mid_parent.get(m) == macro_id]
        if children:
            macro_y0[macro_id] = mid_y0[children[0]]
            macro_y1[macro_id] = mid_y1[children[-1]]
        else:
            macro_y0[macro_id] = 0
            macro_y1[macro_id] = 0

    # X positions for the three columns
    x_positions = {"macro": 0, "mid": 1, "fine": 2}
    node_width = 0.15

    # --- build nodes_df with parent labels for hierarchical highlighting ---
    # Pre-compute label lookups
    macro_id_to_label = {k: v["label"] for k, v in macro_topics.items() if k != "-1"}
    mid_id_to_label = {k: v.get("label", "") for k, v in mid_topics.items()}

    nodes = []
    for macro_id in macro_order:
        info = macro_topics[macro_id]
        color = MACRO_COLORS[int(macro_id) % len(MACRO_COLORS)]
        nodes.append({
            "level": "macro", "id": macro_id, "label": info["label"],
            "count": info["count"], "color": color,
            "parent_macro": info["label"],  # self
            "parent_mid": "",
            "x": x_positions["macro"], "x2": x_positions["macro"] + node_width,
            "y": macro_y0[macro_id], "y2": macro_y1[macro_id],
        })
    for mid_id in mid_order:
        info = mid_topics.get(mid_id, {})
        macro_id = mid_parent.get(mid_id, "0")
        color = MACRO_COLORS[int(macro_id) % len(MACRO_COLORS)]
        nodes.append({
            "level": "mid", "id": mid_id, "label": info.get("label", ""),
            "count": info.get("count", 0), "color": color,
            "parent_macro": macro_id_to_label.get(macro_id, ""),
            "parent_mid": info.get("label", ""),  # self
            "x": x_positions["mid"], "x2": x_positions["mid"] + node_width,
            "y": mid_y0.get(mid_id, 0), "y2": mid_y1.get(mid_id, 0),
        })
    for fine_id in fine_order:
        info = fine_topics.get(fine_id, {})
        macro_id = str(hierarchy.get(fine_id, {}).get("macro", 0))
        mid_id = fine_parent[fine_id]
        color = MACRO_COLORS[int(macro_id) % len(MACRO_COLORS)]
        nodes.append({
            "level": "fine", "id": fine_id, "label": info.get("label", ""),
            "count": info.get("count", 0), "color": color,
            "parent_macro": macro_id_to_label.get(macro_id, ""),
            "parent_mid": mid_id_to_label.get(mid_id, ""),
            "x": x_positions["fine"], "x2": x_positions["fine"] + node_width,
            "y": fine_y0.get(fine_id, 0), "y2": fine_y1.get(fine_id, 0),
        })

    nodes_df = pd.DataFrame(nodes)

    # --- build links_df (one row per link with source/target y extents) ---
    # For macro→mid links: the source side spans the portion of the macro
    # that corresponds to this mid's paper count, stacked in order.
    links = []
    # Track cumulative offset within each macro for the source side
    macro_src_offset = {mid: macro_y0.get(mid, 0) for mid in macro_order}

    for mid_id in mid_order:
        macro_id = mid_parent.get(mid_id, "0")
        cnt = mid_topics.get(mid_id, {}).get("count", 0)
        color = MACRO_COLORS[int(macro_id) % len(MACRO_COLORS)]
        src_y = macro_src_offset[macro_id]
        src_y2 = src_y + cnt
        macro_src_offset[macro_id] = src_y2
        links.append({
            "src_x": x_positions["macro"] + node_width,
            "tgt_x": x_positions["mid"],
            "src_y": src_y, "src_y2": src_y2,
            "tgt_y": mid_y0.get(mid_id, 0), "tgt_y2": mid_y1.get(mid_id, 0),
            "color": color,
        })

    # Mid→fine links
    mid_src_offset = {mid_id: mid_y0.get(mid_id, 0) for mid_id in mid_order}
    for fine_id in fine_order:
        mid_id = fine_parent[fine_id]
        macro_id = str(hierarchy.get(fine_id, {}).get("macro", 0))
        cnt = fine_topics.get(fine_id, {}).get("count", 0)
        color = MACRO_COLORS[int(macro_id) % len(MACRO_COLORS)]
        src_y = mid_src_offset.get(mid_id, 0)
        src_y2 = src_y + cnt
        mid_src_offset[mid_id] = src_y2
        links.append({
            "src_x": x_positions["mid"] + node_width,
            "tgt_x": x_positions["fine"],
            "src_y": src_y, "src_y2": src_y2,
            "tgt_y": fine_y0.get(fine_id, 0), "tgt_y2": fine_y1.get(fine_id, 0),
            "color": color,
        })

    links_df = pd.DataFrame(links)

    return nodes_df, links_df


def main():
    # Global styling — SF Pro / system font, consistent type scale, HIG spacing
    st.markdown("""
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: -apple-system, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 1rem; }

    /* ── Title ── */
    h1 {
        font-family: -apple-system, "SF Pro Display", "Helvetica Neue", sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 0.25rem !important;
        color: #1d1d1f !important;
    }
    h2, h3, [data-testid="stSubheader"] p {
        font-family: -apple-system, "SF Pro Display", "Helvetica Neue", sans-serif !important;
        font-weight: 600 !important;
        color: #1d1d1f !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { min-width: 300px !important; width: 300px !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p {
        font-size: 0.8125rem;
        color: #3a3a3c;
    }

    /* Section labels in sidebar */
    .sidebar-label {
        font-size: 0.6875rem;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }

    /* Schedule grid */
    .day-col {
        text-align: center;
        line-height: 1.2;
    }
    .day-abbr {
        font-size: 0.6875rem;
        font-weight: 500;
        color: #86868b;
    }
    .day-num {
        font-size: 1rem;
        font-weight: 700;
        color: #1d1d1f;
    }
    .slot-label {
        text-align: center;
        font-size: 0.625rem;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: -14px;
        margin-bottom: -4px;
    }
    .sel-count {
        text-align: center;
        font-size: 0.8125rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }
    .sel-count.has-items { color: #1d1d1f; }
    .sel-count.empty { color: #d1d1d6; }
    .sel-slot-label {
        text-align: center;
        font-size: 0.8125rem;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: -8px;
        margin-bottom: -4px;
    }

    /* Checkboxes — compact and centered */
    [data-testid="stSidebar"] .stCheckbox > label {
        padding: 0; min-height: 0; gap: 0;
        justify-content: center;
    }
    [data-testid="stSidebar"] .stCheckbox > label > span:last-child {
        font-size: 0.6875rem;
    }
    [data-testid="stSidebar"] .stCheckbox > label > span:first-child {
        transform: scale(0.8);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(.stCheckbox) {
        margin-top: -6px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    [data-testid="stSidebar"] .stCheckbox {
        display: flex;
        justify-content: center;
    }

    /* Type toggles */
    [data-testid="stSidebar"] .stCheckbox > label > span:last-child p {
        font-size: 0.8125rem !important;
    }

    /* Buttons — minimal, rounded */
    [data-testid="stSidebar"] button[kind="secondary"] {
        font-size: 0.8125rem;
        border-radius: 8px;
        padding: 0.3rem 0.75rem;
    }

    /* Expander — clean header */
    [data-testid="stExpander"] summary span p {
        font-size: 0.9375rem;
        font-weight: 600;
        color: #1d1d1f;
    }

    /* Detail cards */
    .detail-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 2px;
    }

    /* Filter info bar */
    [data-testid="stAlert"] {
        border-radius: 8px;
        font-size: 0.8125rem;
    }

    /* Caption styling */
    [data-testid="stCaptionContainer"] p {
        font-size: 0.75rem;
        color: #86868b;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("CHI 2026 Planner")

    # Active topic filter indicator (below title, matches expander header size)
    _active = st.session_state.get("sankey_filter")
    if _active:
        col_f, col_c = st.columns([4, 1])
        with col_f:
            st.markdown(
                f'<p style="font-size:0.9375rem;font-weight:600;color:#1d1d1f;'
                f'margin:0;padding:0">'
                f'Filtered by {_active[0]}: {_active[1]}</p>',
                unsafe_allow_html=True,
            )
        with col_c:
            if st.button("Clear", key="clear_filter_top", type="tertiary"):
                st.session_state.pop("sankey_filter", None)

    papers, topic_names, topics, hierarchy = load_data()
    init_selection()
    themes = [{"name": t} for t in topic_names]
    min_relevance = 0.2

    # --- Topic Hierarchy (clickable text chart) ---
    # Keep expander open if user has interacted with topic selection
    _hierarchy_open = "sankey_filter" in st.session_state
    with st.expander("Topic Hierarchy", expanded=_hierarchy_open):
        if topics and hierarchy:
            import streamlit.components.v1 as components

            nodes_df, _links_df = build_hierarchy_chart_data(topics, hierarchy)
            nodes_df = nodes_df.copy()
            nodes_df["mid_y"] = (nodes_df["y"] + nodes_df["y2"]) / 2
            nodes_df["display_label"] = (
                nodes_df["label"] + " (" + nodes_df["count"].astype(str) + ")"
            )
            y_max = nodes_df["y2"].max()

            active_filter = st.session_state.get("sankey_filter")

            # Compute highlighting
            if active_filter:
                al, ab = active_filter
                if al == "macro":
                    nodes_df["hl"] = nodes_df["parent_macro"] == ab
                elif al == "mid":
                    mr = nodes_df.loc[(nodes_df["level"] == "mid") & (nodes_df["label"] == ab)]
                    pm = mr.iloc[0]["parent_macro"] if len(mr) else ""
                    nodes_df["hl"] = (
                        (nodes_df["parent_mid"] == ab) |
                        ((nodes_df["level"] == "macro") & (nodes_df["label"] == pm))
                    )
                elif al == "fine":
                    fr = nodes_df.loc[(nodes_df["level"] == "fine") & (nodes_df["label"] == ab)]
                    if len(fr):
                        nodes_df["hl"] = (
                            ((nodes_df["level"] == "fine") & (nodes_df["label"] == ab)) |
                            ((nodes_df["level"] == "mid") & (nodes_df["label"] == fr.iloc[0]["parent_mid"])) |
                            ((nodes_df["level"] == "macro") & (nodes_df["label"] == fr.iloc[0]["parent_macro"]))
                        )
                    else:
                        nodes_df["hl"] = True
                else:
                    nodes_df["hl"] = True
            else:
                nodes_df["hl"] = True

            # Darken colors
            _dc = {}
            def _dk(s):
                if s not in _dc:
                    p = s.replace("rgba(", "").replace(")", "").split(",")
                    _dc[s] = f"rgb({max(0,int(p[0])-100)},{max(0,int(p[1])-100)},{max(0,int(p[2])-100)})"
                return _dc[s]
            nodes_df["dark"] = nodes_df["color"].map(_dk)

            # Build HTML for 3-column clickable text layout
            col_x = {"macro": 0, "mid": 1, "fine": 2}
            col_headers = {"macro": "Macro", "mid": "Mid", "fine": "Micro"}

            # Static HTML chart (no click handlers, no JS)
            items_html = []
            for _, r in nodes_df.iterrows():
                color = r["dark"] if r["hl"] else "rgb(190,190,190)"
                weight = "600" if r["hl"] else "400"
                size = "12px" if r["hl"] else "10px"
                y_pct = (r["mid_y"] / y_max) * 100 if y_max > 0 else 0
                col = col_x[r["level"]]
                align = "right" if col == 0 else ("left" if col == 2 else "center")
                transform = "-100%" if col == 0 else ("0%" if col == 2 else "-50%")
                pad_r = "8px" if col == 0 else "0"
                pad_l = "8px" if col == 2 else "0"
                col_pct = [16.67, 50, 83.33][col]
                display = r["display_label"].replace("&", "&amp;").replace("<", "&lt;")
                items_html.append(
                    f'<div style="position:absolute;top:{y_pct:.2f}%;left:{col_pct}%;'
                    f'color:{color};font-weight:{weight};font-size:{size};'
                    f'transform:translate({transform},-50%);white-space:nowrap;'
                    f'padding-right:{pad_r};padding-left:{pad_l}">{display}</div>'
                )

            html = f"""
            <div style="position:relative;width:100%;height:1100px;
                        font-family:-apple-system,'Helvetica Neue',sans-serif">
              <div style="display:flex;font-size:13px;font-weight:600;
                          color:#86868b;padding-bottom:8px">
                <span style="flex:1;text-align:center">Macro</span>
                <span style="flex:1;text-align:center">Mid</span>
                <span style="flex:1;text-align:center">Micro</span>
              </div>
              <div style="position:relative;width:100%;height:1060px">
                {''.join(items_html)}
              </div>
            </div>
            """

            import streamlit.components.v1 as components
            components.html(html, height=1120, scrolling=False)

            # --- Topic filter via pills (native widget, no rerun loops) ---
            macro_topics_data = topics.get("macro", {})
            mid_topics_data = topics.get("mid", {})
            fine_topics_data = topics.get("fine", {})

            macro_labels = sorted(
                [info["label"] for k, info in macro_topics_data.items() if k != "-1"],
                key=lambda l: next(
                    (v["count"] for v in macro_topics_data.values() if v.get("label") == l), 0
                ), reverse=True,
            )

            sel_macro = st.pills(
                "Macro", macro_labels, key="pills_macro",
                default=active_label if active_level == "macro" else None,
            )

            if sel_macro:
                st.session_state["sankey_filter"] = ("macro", sel_macro)

                # Show mid pills for selected macro
                macro_id = next(
                    (k for k, v in macro_topics_data.items() if v.get("label") == sel_macro), None
                )
                if macro_id:
                    mid_ids = sorted(set(
                        str(links["mid"]) for fid, links in hierarchy.items()
                        if str(links["macro"]) == macro_id
                    ))
                    mid_labels = sorted(
                        [mid_topics_data[m]["label"] for m in mid_ids if m in mid_topics_data],
                        key=lambda l: next(
                            (v["count"] for v in mid_topics_data.values() if v.get("label") == l), 0
                        ), reverse=True,
                    )
                    sel_mid = st.pills(
                        "Mid", mid_labels, key="pills_mid",
                        default=active_label if active_level in ("mid", "fine") else None,
                    )
                    if sel_mid:
                        st.session_state["sankey_filter"] = ("mid", sel_mid)

                        # Show fine pills for selected mid
                        mid_id = next(
                            (k for k, v in mid_topics_data.items() if v.get("label") == sel_mid), None
                        )
                        if mid_id:
                            fine_ids = sorted(
                                [fid for fid, links in hierarchy.items() if str(links["mid"]) == mid_id],
                                key=lambda f: fine_topics_data.get(f, {}).get("count", 0),
                                reverse=True,
                            )
                            fine_labels = [fine_topics_data[f]["label"] for f in fine_ids if f in fine_topics_data]
                            sel_fine = st.pills(
                                "Micro", fine_labels, key="pills_fine",
                                default=active_label if active_level == "fine" else None,
                            )
                            if sel_fine:
                                st.session_state["sankey_filter"] = ("fine", sel_fine)
            else:
                st.session_state.pop("sankey_filter", None)
        else:
            st.info("No hierarchy data. Run `python chi_pipeline.py cluster` first.")

    # --- Sidebar ---
    with st.sidebar:
        # --- Schedule grid ---
        from datetime import date as dt_date
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        dates = sorted(set(p.get("date", "") for p in papers if p.get("date")))
        day_info = []
        for d in dates:
            try:
                parts = d.split("-")
                dow = dt_date(int(parts[0]), int(parts[1]), int(parts[2])).weekday()
                day_abbr = day_names[dow]
                day_num = parts[2]
            except Exception:
                day_abbr = "?"
                day_num = d[-2:]
            day_info.append({"date": d, "abbr": day_abbr, "num": day_num})

        date_slots = []
        selected_slots = []

        if day_info:
            st.markdown('<p class="sidebar-label">Schedule</p>', unsafe_allow_html=True)

            cols = st.columns(len(day_info))
            for i, d in enumerate(day_info):
                with cols[i]:
                    st.markdown(
                        f'<div class="day-col">'
                        f'<span class="day-abbr">{d["abbr"]}</span><br>'
                        f'<span class="day-num">{d["num"]}</span></div>',
                        unsafe_allow_html=True,
                    )

            cols_am = st.columns(len(day_info))
            for i, d in enumerate(day_info):
                slot_key = f"{d['abbr']} {d['date'][5:]} AM"
                date_slots.append(slot_key)
                with cols_am[i]:
                    if st.checkbox("am", value=True, key=f"sched_am_{d['date']}", label_visibility="collapsed"):
                        selected_slots.append(slot_key)
                    st.markdown('<div class="slot-label">AM</div>', unsafe_allow_html=True)

            cols_pm = st.columns(len(day_info))
            for i, d in enumerate(day_info):
                slot_key = f"{d['abbr']} {d['date'][5:]} PM"
                date_slots.append(slot_key)
                with cols_pm[i]:
                    if st.checkbox("pm", value=True, key=f"sched_pm_{d['date']}", label_visibility="collapsed"):
                        selected_slots.append(slot_key)
                    st.markdown('<div class="slot-label">PM</div>', unsafe_allow_html=True)

        # --- Type toggles ---
        type_display_names = {
            "Interactive Demos": "Demos",
            "Plenary And Keynote": "Plenary",
            "Student Mentoring Program": "Student Mentoring",
            "SIGCHI Awards": "Awards",
        }
        session_types = sorted(set(p.get("session_type", "") for p in papers if p.get("session_type")))
        selected_types = []
        if session_types:
            st.markdown('<p class="sidebar-label">Type</p>', unsafe_allow_html=True)
            n_cols = 2
            type_cols = st.columns(n_cols)
            for i, stype in enumerate(session_types):
                display_name = type_display_names.get(stype, stype)
                with type_cols[i % n_cols]:
                    if st.checkbox(display_name, value=True, key=f"type_{stype}"):
                        selected_types.append(stype)

        st.divider()
        n_selected = len(st.session_state.selected)
        st.markdown(
            f'<p class="sidebar-label">Selected &nbsp;·&nbsp; {n_selected}</p>',
            unsafe_allow_html=True,
        )

        # Show selected paper counts per schedule slot
        if day_info:
            sel_counts = {}
            for idx in st.session_state.selected:
                if idx < len(papers):
                    p = papers[idx]
                    sched = p.get("schedule", [])
                    entries = sched if sched else [p]
                    for s in entries:
                        d = s.get("date", p.get("date", ""))
                        st_time = s.get("start_time", p.get("start_time", "12:00"))
                        period = "AM" if st_time < "12:00" else "PM"
                        sel_counts[(d, period)] = sel_counts.get((d, period), 0) + 1

            sel_cols_am = st.columns(len(day_info))
            for i, d in enumerate(day_info):
                with sel_cols_am[i]:
                    count = sel_counts.get((d["date"], "AM"), 0)
                    cls = "has-items" if count else "empty"
                    st.markdown(
                        f'<div class="sel-count {cls}">{count}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="sel-slot-label">AM</div>', unsafe_allow_html=True)

            sel_cols_pm = st.columns(len(day_info))
            for i, d in enumerate(day_info):
                with sel_cols_pm[i]:
                    count = sel_counts.get((d["date"], "PM"), 0)
                    cls = "has-items" if count else "empty"
                    st.markdown(
                        f'<div class="sel-count {cls}">{count}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="sel-slot-label">PM</div>', unsafe_allow_html=True)

        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("Save", use_container_width=True):
                save_selection()
                st.success("Saved!")
        with col_clear:
            if st.button("Clear", use_container_width=True, disabled=n_selected == 0):
                st.session_state.selected = set()
                save_selection()
                st.rerun()

        st.divider()
        st.markdown('<p class="sidebar-label">Export</p>', unsafe_allow_html=True)

        selected_papers = [p for i, p in enumerate(papers) if i in st.session_state.selected]

        st.download_button(
            "Markdown agenda",
            data=generate_markdown(selected_papers, themes, min_relevance),
            file_name="chi2026_agenda.md",
            mime="text/markdown",
            disabled=len(selected_papers) == 0,
            use_container_width=True,
        )
        st.download_button(
            "Calendar (.ics)",
            data=generate_ics(selected_papers, themes, min_relevance),
            file_name="chi2026_agenda.ics",
            mime="text/calendar",
            disabled=len(selected_papers) == 0,
            use_container_width=True,
        )

    # --- Build and filter DataFrame ---
    df = build_dataframe(papers, topic_names)
    has_hierarchy = "macro_label" in df.columns

    # Filter by date + AM/PM slot
    if selected_slots and date_slots and set(selected_slots) != set(date_slots):
        allowed_dates_am = set()
        allowed_dates_pm = set()
        for slot in selected_slots:
            parts = slot.rsplit(" ", 1)
            period = parts[-1]
            date_part = parts[0].split(" ", 1)[-1]
            full_date = f"2026-{date_part}"
            if period == "AM":
                allowed_dates_am.add(full_date)
            else:
                allowed_dates_pm.add(full_date)

        # Pre-compute valid paper indices for slot filtering (avoids row-by-row apply)
        valid_indices = set()
        for idx, p in enumerate(papers):
            sched = p.get("schedule", [])
            entries = sched if sched else [p]
            for s in entries:
                d = s.get("date", p.get("date", ""))
                st_time = s.get("start_time", p.get("start_time", "12:00"))
                if (st_time < "12:00" and d in allowed_dates_am) or \
                   (st_time >= "12:00" and d in allowed_dates_pm):
                    valid_indices.add(idx)
                    break
        df = df[df["_index"].isin(valid_indices)]

    if selected_types:
        df = df[df["session_type"].isin(selected_types)]

    # Apply topic filter from hierarchy chart click
    if "sankey_filter" in st.session_state:
        s_level, s_label = st.session_state["sankey_filter"]
        if s_level == "macro" and "macro_label" in df.columns:
            df = df[df["macro_label"] == s_label]
        elif s_level == "mid" and "mid_label" in df.columns:
            df = df[df["mid_label"] == s_label]
        elif s_level == "fine" and "cluster_label" in df.columns:
            df = df[df["cluster_label"] == s_label]

    # --- Semantic Search ---
    # Check if embeddings are available (pre-computed .npy or sentence-transformers installed)
    embeddings_path = DATA_DIR / "paper_embeddings.npy"
    search_available = embeddings_path.exists() or _has_sentence_transformers()

    search_query = ""
    if search_available:
        search_query = st.text_input(
            "Search papers",
            placeholder="e.g. accessibility assistive technology blind",
            key="semantic_search",
            label_visibility="collapsed",
        )

    # --- Papers Table ---
    paper_count_placeholder = st.empty()

    if has_hierarchy:
        display_cols = ["title", "authors", "award", "macro_label", "mid_label", "cluster_label", "date", "time", "location"]
    else:
        display_cols = ["title", "authors", "award", "cluster_label", "date", "time", "location"]
    display_df = df[["_index"] + [c for c in display_cols if c in df.columns]].copy()

    # Apply semantic search if query is provided
    if search_query and len(search_query.strip()) >= 2:
        paper_embeddings = compute_paper_embeddings(papers)
        if paper_embeddings is None:
            st.warning("Semantic search unavailable — install sentence-transformers for this feature.")
            search_query = ""
        else:
            search_results = semantic_search(search_query.strip(), paper_embeddings, papers)
            display_df = display_df.merge(
                search_results[["_index", "search_score", "search_keywords"]],
                on="_index", how="left",
            )
            display_df["search_match"] = display_df.apply(
                lambda r: f"{r['search_keywords']} ({r['search_score']:.2f})"
                if pd.notna(r["search_score"]) and r["search_keywords"]
                else ("" if pd.isna(r["search_score"]) else f"({r['search_score']:.2f})"),
                axis=1,
            )
            display_df = display_df.sort_values("search_score", ascending=False, na_position="last")
    else:
        display_df["search_match"] = ""
        display_df["search_score"] = np.nan

    # Build display table
    has_search = search_query and len(search_query.strip()) >= 2

    # Select visible columns
    visible_cols = ["title"]
    if has_search:
        visible_cols.append("search_match")
    visible_cols.append("authors")
    if "award" in display_df.columns and display_df["award"].any():
        visible_cols.append("award")
    if "macro_label" in display_df.columns:
        visible_cols.extend(["macro_label", "mid_label", "cluster_label"])
    visible_cols.extend(["date", "time", "location"])

    # Keep _index mapping separate, exclude from displayed dataframe
    index_map = display_df["_index"].tolist()
    grid_df = display_df[[c for c in visible_cols if c in display_df.columns]].copy()
    grid_df = grid_df.reset_index(drop=True)

    # Column display config
    col_config = {
        "title": st.column_config.TextColumn("Title", width="large"),
        "search_match": st.column_config.TextColumn("Match", width="medium"),
        "authors": st.column_config.TextColumn("Authors", width="medium"),
        "award": st.column_config.TextColumn("Award", width="small"),
        "macro_label": st.column_config.TextColumn("Macro", width="small"),
        "mid_label": st.column_config.TextColumn("Mid", width="small"),
        "cluster_label": st.column_config.TextColumn("Micro", width="small"),
        "date": st.column_config.TextColumn("Date", width="small"),
        "time": st.column_config.TextColumn("Time", width="small"),
        "location": st.column_config.TextColumn("Room", width="small"),
    }

    paper_count_placeholder.markdown(
        f"**Papers** &nbsp;&middot;&nbsp; {len(grid_df)}"
    )

    # Build a stable key that changes when any filter changes (prevents stale selection reruns)
    table_key = f"paper_table_{len(grid_df)}"

    event = st.dataframe(
        grid_df,
        column_config=col_config,
        use_container_width=True,
        height=500,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=table_key,
    )

    # Sync selections back to session state (merge, don't replace)
    if event and event.selection and event.selection.rows:
        selected_indices = event.selection.rows
        grid_selected = set()
        for row_idx in selected_indices:
            if row_idx < len(index_map):
                grid_selected.add(int(index_map[row_idx]))
        if grid_selected:
            updated = st.session_state.selected | grid_selected
            if updated != st.session_state.selected:
                st.session_state.selected = updated
                save_selection()

    # --- Paper Details ---
    n_sel = len(st.session_state.selected)
    st.markdown(f"**Selected Papers** &nbsp;&middot;&nbsp; {n_sel}")
    detail_rows = df[df["_index"].isin(st.session_state.selected)].copy()
    if detail_rows.empty:
        st.caption("Select papers from the table above.")

    # Sort by date then start_time
    detail_rows = detail_rows.sort_values(
        ["date", "start_time"], ascending=True, na_position="last"
    )

    # Track current slot for section dividers
    prev_slot = None
    removed_any = False
    for _, row in detail_rows.iterrows():
        # Insert divider when day or AM/PM changes
        date = row.get("date", "")
        start_time = row.get("start_time", "12:00")
        period = "Morning" if start_time < "12:00" else "Afternoon"
        current_slot = (date, period)
        if current_slot != prev_slot:
            if prev_slot is not None:
                st.markdown(
                    "<hr style='border:none;border-top:1px solid #e5e5ea;margin:1.2rem 0 0.8rem'>",
                    unsafe_allow_html=True,
                )
            # Section label
            from datetime import date as dt_date
            try:
                parts = date.split("-")
                dow = dt_date(int(parts[0]), int(parts[1]), int(parts[2])).strftime("%A")
                label = f"{dow}, {date} &nbsp;&middot;&nbsp; {period}"
            except Exception:
                label = f"{date} &nbsp;&middot;&nbsp; {period}"
            st.markdown(
                f"<p style='font-size:0.6875rem;font-weight:600;color:#86868b;"
                f"text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.3rem'>"
                f"{label}</p>",
                unsafe_allow_html=True,
            )
            prev_slot = current_slot
        idx = int(row["_index"])
        award_badge = f" &nbsp;**[{row['award']}]**" if row.get("award") else ""

        col_exp, col_remove = st.columns([20, 1])
        with col_exp:
            expander = st.expander(f"{row['title']}{award_badge}", expanded=True)
        with col_remove:
            if st.button("✕", key=f"remove_{idx}", help="Remove from selection",
                         type="tertiary"):
                st.session_state.selected.discard(idx)
                save_selection()
                removed_any = True

        with expander:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(row.get("authors", ""))
                schedule = papers[idx].get("schedule") if idx < len(papers) else None
                if schedule and len(schedule) > 1:
                    for s in schedule:
                        st.caption(
                            f"{s['date']} &nbsp; {s['time']} &nbsp; {s.get('location', '')} "
                            f"&nbsp;&middot;&nbsp; {s['session']}"
                        )
                else:
                    st.caption(
                        f"{row.get('date', '')} &nbsp; {row.get('time', '')} "
                        f"&nbsp; {row.get('location', '')} "
                        f"&nbsp;&middot;&nbsp; {row.get('session', '')}"
                    )
            with col_b:
                if row.get("macro_label"):
                    st.caption(
                        f"{row['macro_label']}  &rarr;  {row.get('mid_label', '')}  "
                        f"&rarr;  {row.get('cluster_label', '')}"
                    )
            if row.get("abstract"):
                st.markdown(
                    f"<p style='font-size:0.8125rem;color:#3a3a3c;line-height:1.5'>"
                    f"{row['abstract']}</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No abstract available.")
    if removed_any:
        st.rerun()


if __name__ == "__main__":
    main()
