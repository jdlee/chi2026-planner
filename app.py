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
    """Load classified papers and themes."""
    papers_path = DATA_DIR / "chi2026_classified.json"
    themes_path = DATA_DIR / "themes.json"

    if not papers_path.exists():
        st.error(
            "No data found. Run the pipeline first:\n"
            "```\npython chi_pipeline.py\n```"
        )
        st.stop()

    papers = json.loads(papers_path.read_text())
    themes = json.loads(themes_path.read_text()) if themes_path.exists() else []
    return papers, themes


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


def main():
    st.title("CHI 2026 Paper Viewer & Agenda Planner")

    papers, themes = load_data()
    init_selection()
    theme_names = [t["name"] for t in themes]

    # --- Sidebar: Filters & Export ---
    with st.sidebar:
        st.header("Filters")

        keyword = st.text_input("Search titles & abstracts")

        selected_themes = st.multiselect(
            "Filter by themes", theme_names, default=[]
        )
        min_relevance = st.slider(
            "Min theme relevance", 0.0, 1.0, 0.3, 0.05
        )

        dates = sorted(set(p.get("date", "") for p in papers if p.get("date")))
        if dates:
            selected_dates = st.multiselect("Filter by date", dates, default=dates)
        else:
            selected_dates = []

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

    # --- Apply filters ---
    df = pd.DataFrame(papers)
    df["_index"] = range(len(papers))

    theme_score_df = pd.json_normalize(df["theme_scores"])
    for col in theme_names:
        if col not in theme_score_df.columns:
            theme_score_df[col] = 0.0
    df = pd.concat([df.drop(columns=["theme_scores"]), theme_score_df[theme_names]], axis=1)

    if keyword:
        mask = df["title"].str.contains(keyword, case=False, na=False) | df[
            "abstract"
        ].str.contains(keyword, case=False, na=False)
        df = df[mask]

    if selected_dates:
        df = df[df["date"].isin(selected_dates)]

    if selected_themes:
        theme_mask = df[selected_themes].max(axis=1) >= min_relevance
        df = df[theme_mask]

    # --- Main view: Paper x Theme matrix ---
    st.subheader(f"Papers ({len(df)} shown)")

    sort_col = st.selectbox(
        "Sort by",
        ["start_time"] + theme_names,
        index=0,
    )
    df = df.sort_values(sort_col, ascending=sort_col == "start_time")

    # Heatmap visualization
    if theme_names and len(df) > 0:
        heatmap_data = df.melt(
            id_vars=["_index", "title"],
            value_vars=theme_names,
            var_name="Theme",
            value_name="Relevance",
        )

        heatmap = (
            alt.Chart(heatmap_data)
            .mark_rect()
            .encode(
                x=alt.X("Theme:N", sort=theme_names, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("title:N", sort=list(df["title"]), axis=alt.Axis(labelLimit=300)),
                color=alt.Color(
                    "Relevance:Q",
                    scale=alt.Scale(scheme="blues", domain=[0, 1]),
                ),
                tooltip=["title", "Theme", alt.Tooltip("Relevance:Q", format=".2f")],
            )
            .properties(height=max(len(df) * 25, 200))
        )
        st.altair_chart(heatmap, use_container_width=True)

    # Paper list with selection
    for _, row in df.iterrows():
        idx = int(row["_index"])
        is_selected = idx in st.session_state.selected

        has_conflict = False
        if is_selected:
            for other_idx in st.session_state.selected:
                if other_idx != idx and other_idx < len(papers):
                    other = papers[other_idx]
                    if (
                        row["date"] == other.get("date")
                        and row["start_time"]
                        and other.get("end_time")
                        and row["start_time"] < other["end_time"]
                        and other.get("start_time", "") < row.get("end_time", "99:99")
                    ):
                        has_conflict = True
                        break

        conflict_icon = " ⚠️" if has_conflict else ""
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
            with st.expander(f"{conflict_icon} **{row['title']}** — {row['time']} @ {row['location']}"):
                st.write(f"**Authors:** {row['authors']}")
                st.write(f"**Session:** {row['session']}")
                st.write(f"**Abstract:** {row['abstract']}")

                scores = {t: row[t] for t in theme_names if row[t] >= min_relevance}
                if scores:
                    st.write("**Themes:** " + ", ".join(
                        f"`{t}` ({s:.2f})" for t, s in sorted(scores.items(), key=lambda x: -x[1])
                    ))


if __name__ == "__main__":
    main()
