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

    # --- Cluster filter ---
    cluster_labels = sorted(df["cluster_label"].dropna().unique()) if "cluster_label" in df.columns else []
    if cluster_labels:
        selected_clusters = st.multiselect("Filter by cluster", cluster_labels, default=[])
        if selected_clusters:
            df = df[df["cluster_label"].isin(selected_clusters)]

    # --- Papers Table ---
    st.subheader(f"Papers ({len(df)})")

    sort_col = st.selectbox(
        "Sort by",
        ["start_time", "cluster_label"] + topic_names,
        index=0,
    )
    ascending = sort_col in ("start_time", "cluster_label")
    df_sorted = df.sort_values(sort_col, ascending=ascending, na_position="last")

    # Build display table columns
    display_cols = ["title", "authors", "cluster_label", "session", "session_type", "date", "time", "location"]
    if "award" in df_sorted.columns:
        display_cols.insert(1, "award")
    for t in topic_names:
        if t in df_sorted.columns:
            display_cols.append(t)

    display_df = df_sorted[["_index"] + [c for c in display_cols if c in df_sorted.columns]].copy()
    for t in topic_names:
        if t in display_df.columns:
            display_df[t] = display_df[t].round(2)

    # Selection checkboxes
    display_df.insert(0, "Select", display_df["_index"].apply(lambda x: x in st.session_state.selected))

    edited = st.data_editor(
        display_df.drop(columns=["_index"]),
        column_config={
            "Select": st.column_config.CheckboxColumn("Sel", width="small"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "authors": st.column_config.TextColumn("Authors", width="medium"),
            "session": st.column_config.TextColumn("Session", width="medium"),
            "session_type": st.column_config.TextColumn("Type", width="small"),
            "date": st.column_config.TextColumn("Date", width="small"),
            "time": st.column_config.TextColumn("Time", width="small"),
            "location": st.column_config.TextColumn("Room", width="small"),
            "cluster_label": st.column_config.TextColumn("Cluster", width="medium"),
            "award": st.column_config.TextColumn("Award", width="small"),
            **{
                t: st.column_config.ProgressColumn(t, min_value=0, max_value=1, width="small")
                for t in topic_names if t in display_df.columns
            },
        },
        hide_index=True,
        use_container_width=True,
        key="paper_table",
    )

    # Sync selections back from edited table
    if edited is not None:
        new_selected = set()
        for i, selected in enumerate(edited["Select"]):
            if selected:
                idx = int(display_df.iloc[i]["_index"])
                new_selected.add(idx)
        if new_selected != st.session_state.selected:
            st.session_state.selected = new_selected
            save_selection()
            st.rerun()

    # --- Paper Details: click title to expand ---
    st.subheader("Paper Details")
    for _, row in df_sorted.iterrows():
        idx = int(row["_index"])
        is_selected = idx in st.session_state.selected
        award_badge = f" **[{row['award']}]**" if row.get("award") else ""
        sel_marker = " ✅" if is_selected else ""

        with st.expander(f"{row['title']}{award_badge}{sel_marker}"):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"**Authors:** {row.get('authors', '')}")
                st.markdown(f"**Session:** {row.get('session', '')} ({row.get('session_type', '')})")
                st.markdown(f"**When:** {row.get('date', '')} {row.get('time', '')} @ {row.get('location', '')}")
                if row.get("cluster_label"):
                    st.markdown(f"**Cluster:** {row['cluster_label']}")
            with col_b:
                scores = {
                    t: row[t] for t in topic_names
                    if t in row.index and row[t] >= min_relevance
                }
                if scores:
                    st.markdown(
                        "**Topics:** "
                        + ", ".join(
                            f"`{t}` ({s:.2f})"
                            for t, s in sorted(scores.items(), key=lambda x: -x[1])
                        )
                    )
            if row.get("abstract"):
                st.markdown(f"**Abstract:** {row['abstract']}")
            else:
                st.caption("No abstract available — run `python chi_pipeline.py abstracts` to fetch.")


if __name__ == "__main__":
    main()
