"""CHI 2026 Paper Viewer & Agenda Planner — Streamlit App"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from export import generate_ics, generate_markdown

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

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
    return papers, topic_names, topics


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


def main():
    st.title("CHI 2026 Paper Viewer & Agenda Planner")

    papers, topic_names, topics = load_data()
    init_selection()

    themes = [{"name": t} for t in topic_names]

    # --- Sidebar: Filters & Export ---
    with st.sidebar:
        st.header("Filters")

        keyword = st.text_input("Search titles & abstracts")

        selected_topics = st.multiselect("Filter by topics", topic_names, default=[])
        min_relevance = st.slider("Min topic relevance", 0.0, 1.0, 0.2, 0.05)

        dates = sorted(set(p.get("date", "") for p in papers if p.get("date")))
        if dates:
            selected_dates = st.multiselect("Filter by date", dates, default=dates)
        else:
            selected_dates = []

        session_types = sorted(set(p.get("session_type", "") for p in papers if p.get("session_type")))
        if session_types:
            selected_types = st.multiselect("Filter by type", session_types, default=session_types)
        else:
            selected_types = []

        st.divider()
        st.header("Selection")
        st.metric("Selected papers", len(st.session_state.selected))

        if st.button("Save selection"):
            save_selection()
            st.success("Saved!")

        uploaded = st.file_uploader("Load selection (.json)", type="json")
        if uploaded:
            st.session_state.selected = set(json.loads(uploaded.read()))
            st.rerun()

        st.divider()
        st.header("Export")

        selected_papers = [p for i, p in enumerate(papers) if i in st.session_state.selected]

        st.download_button(
            "Download Markdown agenda",
            data=generate_markdown(selected_papers, themes, min_relevance),
            file_name="chi2026_agenda.md",
            mime="text/markdown",
            disabled=len(selected_papers) == 0,
        )
        st.download_button(
            "Download .ics calendar",
            data=generate_ics(selected_papers, themes, min_relevance),
            file_name="chi2026_agenda.ics",
            mime="text/calendar",
            disabled=len(selected_papers) == 0,
        )

    # --- Build and filter DataFrame ---
    df = build_dataframe(papers, topic_names)

    if keyword:
        mask = df["title"].str.contains(keyword, case=False, na=False) | df[
            "abstract"
        ].str.contains(keyword, case=False, na=False)
        df = df[mask]

    if selected_dates:
        df = df[df["date"].isin(selected_dates)]

    if selected_types:
        df = df[df["session_type"].isin(selected_types)]

    if selected_topics:
        topic_mask = df[selected_topics].max(axis=1) >= min_relevance
        df = df[topic_mask]

    # --- UMAP Scatter Plot ---
    st.subheader(f"Paper Landscape ({len(df)} papers)")

    if "umap_x" in df.columns and len(df) > 0:
        scatter = (
            alt.Chart(df)
            .mark_circle(size=40, opacity=0.7)
            .encode(
                x=alt.X("umap_x:Q", axis=alt.Axis(title="UMAP 1", labels=False, ticks=False)),
                y=alt.Y("umap_y:Q", axis=alt.Axis(title="UMAP 2", labels=False, ticks=False)),
                color=alt.Color(
                    "cluster_label:N",
                    legend=alt.Legend(title="Topic Cluster", columns=2),
                ),
                tooltip=["title", "cluster_label", "session", "date", "time", "location"],
            )
            .properties(height=450)
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

    # --- Sortable Topic Table ---
    st.subheader("Papers")

    sort_col = st.selectbox(
        "Sort by",
        ["start_time", "cluster_label"] + topic_names,
        index=0,
    )
    ascending = sort_col == "start_time"
    df_sorted = df.sort_values(sort_col, ascending=ascending)

    for _, row in df_sorted.iterrows():
        idx = int(row["_index"])
        is_selected = idx in st.session_state.selected

        has_conflict = False
        if is_selected:
            for other_idx in st.session_state.selected:
                if other_idx != idx and other_idx < len(papers):
                    other = papers[other_idx]
                    if (
                        row.get("date") == other.get("date")
                        and row.get("start_time")
                        and other.get("end_time")
                        and row["start_time"] < other["end_time"]
                        and other.get("start_time", "") < row.get("end_time", "99:99")
                    ):
                        has_conflict = True
                        break

        conflict_icon = " ⚠️" if has_conflict else ""
        cluster_tag = f" [{row.get('cluster_label', '')}]" if row.get("cluster_label") else ""

        col1, col2 = st.columns([0.05, 0.95])

        with col1:
            checked = st.checkbox(
                "sel",
                value=is_selected,
                key=f"cb_{idx}",
                label_visibility="collapsed",
            )
            if checked and idx not in st.session_state.selected:
                st.session_state.selected.add(idx)
                st.rerun()
            elif not checked and idx in st.session_state.selected:
                st.session_state.selected.discard(idx)
                st.rerun()

        with col2:
            with st.expander(
                f"{conflict_icon} **{row['title']}** — {row.get('time', '')} @ {row.get('location', '')}{cluster_tag}"
            ):
                st.write(f"**Authors:** {row.get('authors', '')}")
                st.write(f"**Session:** {row.get('session', '')} ({row.get('session_type', '')})")
                if row.get("abstract"):
                    st.write(f"**Abstract:** {row['abstract']}")

                scores = {
                    t: row[t] for t in topic_names
                    if t in row.index and row[t] >= min_relevance
                }
                if scores:
                    st.write(
                        "**Topics:** "
                        + ", ".join(
                            f"`{t}` ({s:.2f})"
                            for t, s in sorted(scores.items(), key=lambda x: -x[1])
                        )
                    )


if __name__ == "__main__":
    main()
