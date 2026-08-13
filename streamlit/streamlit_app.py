"""
RouteHunter — Streamlit app

Sidebar-navigated demo covering Review, Search, Monitor, and Predict.
Nothing here reimplements any logic — every section only calls into
the existing routehunter package, same as RouteHunterApp.ipynb does.

Run locally with:
    streamlit run streamlit/streamlit_app.py
"""

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from routehunter import RouteHunterApp
from routehunter.app import DEFAULT_CSV_SUBPATH
from routehunter.core import InvalidSMILESError

# --- Example molecules, taken straight from RouteHunterApp.ipynb ---
EXAMPLE_FOUND = "C(O)(C(O)=O)C(C1C=CC=CC=1)NC(C1C=CC=CC=1)=O"
EXAMPLE_NOT_FOUND = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

# Change this if your data directory lives somewhere else relative to
# wherever you run `streamlit run` from (see RouteHunterApp docs).
DATA_DIR = "rh-data"

SECTIONS = ["Review", "Search", "Monitor", "Predict"]

# - Bumps the sidebar nav text up a size and bolds it (st.radio's own
#   formatting only supports inline markdown, not font-size, hence the
#   CSS override).
# - Hides the sidebar's collapse control so it can't be closed. Two
#   selectors are targeted because Streamlit renamed this element in
#   1.38 (old: collapsedControl, new: stSidebarCollapseButton) --
#   hiding both keeps this working across versions.
# All of this is tied to Streamlit's current DOM structure -- if a
# future version changes it, these selectors may need adjusting.
SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 1.15rem;
}
div[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapseButton"] {
    display: none;
}
</style>
"""


@st.cache_resource(show_spinner="Loading RouteHunter dataset...")
def load_app(data_dir: str) -> RouteHunterApp:
    """
    Build the RouteHunterApp exactly once per app process, not once
    per rerun. This is the same call the notebook makes:
        RouteHunterApp.from_data_dir("rh-data")
    """
    return RouteHunterApp.from_data_dir(data_dir)


@st.cache_data(show_spinner="Loading seed CSV...")
def load_seed_csv(data_dir: str) -> pd.DataFrame:
    """Raw contents of the seed CSV, unprocessed -- straight pd.read_csv,
    no pass through RouteHunterStore/Target parsing."""
    return pd.read_csv(Path(data_dir) / DEFAULT_CSV_SUBPATH)


def render_stat_list(counts: dict, empty_message: str) -> None:
    """
    A compact "name -- count" list: count sized down from st.metric's
    (quite large) default, but still visually bigger than body text,
    since this may render many rows.
    """
    if not counts:
        st.info(empty_message)
        return

    rows = "".join(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:baseline;
                    padding:0.45rem 0; border-bottom:1px solid rgba(128,128,128,0.25);">
            <span style="font-size:0.95rem;">{html.escape(str(name))}</span>
            <span style="font-size:1.3rem; font-weight:600;">{count}</span>
        </div>
        """
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    st.markdown(rows, unsafe_allow_html=True)


def smiles_input(key: str) -> str:
    return st.text_input(
        "SMILES",
        key=key,
        placeholder=EXAMPLE_FOUND,
        help=f"Try a known molecule, e.g. `{EXAMPLE_FOUND}`, "
        f"or an absent one, e.g. `{EXAMPLE_NOT_FOUND}`.",
    )


# --- Sections --------------------------------------------------------

def render_review(app: RouteHunterApp) -> None:
    st.title("📊 Review")
    st.caption(
        "Dataset overview — the same numbers as RouteHunterApp.ipynb's "
        "Review section, plus a preview of the raw seed CSV."
    )

    stats = app.store.stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Targets", stats["n_targets"])
    c2.metric("Papers", stats["n_papers"])
    c3.metric("Targets w/ >1 route", stats["n_multi_paper_targets"])
    c4.metric(
        "Cached CASP routes",
        stats["n_cached_casp_routes"],
        help="Session-only — never written back to the CSV.",
    )

    st.write("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Papers by journal")
        render_stat_list(stats["papers_by_journal"], "No journal data available.")

    with col_b:
        st.subheader("Papers by contributor")
        render_stat_list(stats["papers_by_contributor"], "No contributor data available.")

    st.divider()
    st.subheader("Seed dataset preview")
    st.caption(f"Raw, unprocessed contents of `{DATA_DIR}/{DEFAULT_CSV_SUBPATH}`.")
    try:
        seed_df = load_seed_csv(DATA_DIR)
    except FileNotFoundError:
        st.error(f"Could not find the seed CSV at `{DATA_DIR}/{DEFAULT_CSV_SUBPATH}`.")
    else:
        st.caption(f"{len(seed_df):,} rows × {len(seed_df.columns)} columns")
        st.dataframe(seed_df, use_container_width=True, hide_index=True)


def render_search(app: RouteHunterApp) -> None:
    st.title("🔎 Search")
    st.caption(
        "Give a SMILES, get literature papers and/or CASP-predicted routes for it"
    )

    smiles = smiles_input("search_smiles")

    if st.button("Search", type="primary"):
        if not smiles.strip():
            st.warning("Enter a SMILES string first.")
            return
        try:
            result = app.search(smiles)
        except InvalidSMILESError as e:
            st.error(f"Couldn't parse that SMILES: {e}")
            return

        if result.papers:
            st.subheader(f"📄 {len(result.papers)} paper(s) found")
            st.dataframe(
                [
                    {
                        "Title": p.title,
                        "Journal": p.journal,
                        "Year": p.year,
                        "DOI": p.doi,
                    }
                    for p in result.papers
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("🤖 Cached CASP-predicted routes")
        if result.casp_solved:
            st.dataframe(
                [
                    {"Tool": e.tool_display, "Result": "Solved", "Link": e.link}
                    for e in result.casp_solved
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Cached predicted routes are not available yet.")

        if not result.found:
            st.info("This molecule was not found in the dataset. Predicted solvability:")
            cols = st.columns(len(result.properties) or 1)
            for col, (name, value) in zip(cols, result.properties.items()):
                col.metric(
                    f"Chance to be solved by {name}",
                    f"{value:.0%}" if value is not None else "n/a",
                )

        with st.expander("Raw report() text (same as the notebook)"):
            st.code(result.report(), language="text")


def render_monitor(app: RouteHunterApp) -> None:
    st.title("🗞️ Monitor")
    st.caption(
        "Browse recent papers, pre-scored by predicted route probability. "
        "Display-only — nothing here is written into the dataset."
    )

    year = st.number_input("Year", min_value=1900, max_value=2100, value=2025, step=1)

    if st.button("Load", type="primary"):
        result = app.monitor(int(year))

        if not result.available:
            st.info(result.message)
            return

        st.write(result.message)
        st.dataframe(
            result.to_dataframe(),
            use_container_width=True,
            hide_index=True,
            column_config={
                "route_prob": st.column_config.TextColumn("Route probability"),
                "journal": st.column_config.TextColumn("Journal"),
                "title": st.column_config.TextColumn("Title", width="large"),
                "doi": st.column_config.TextColumn("DOI"),
            },
        )


def render_predict(app: RouteHunterApp) -> None:
    st.title("🧪 Predict")
    st.caption(
        "Predict solvability for a target with no known literature "
        "synthesis, using the same models Search uses at level 1."
    )

    smiles = smiles_input("predict_smiles")

    if st.button("Predict", type="primary"):
        if not smiles.strip():
            st.warning("Enter a SMILES string first.")
            return
        try:
            result = app.predict(smiles)
        except InvalidSMILESError as e:
            st.error(f"Couldn't parse that SMILES: {e}")
            return

        st.dataframe(
            result.to_dataframe(),
            use_container_width=True,
            hide_index=True,
            column_config={
                "tool": st.column_config.TextColumn("Tool"),
                "probability": st.column_config.TextColumn("Probability"),
                "url": st.column_config.TextColumn("Link"),
            },
        )


# --- Page setup + sidebar navigation ---------------------------------

st.set_page_config(
    page_title="RouteHunter",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

app = load_app(DATA_DIR)

st.sidebar.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
st.sidebar.title("RouteHunter")
section = st.sidebar.radio(
    "Section",
    SECTIONS,
    format_func=lambda s: f"**{s}**",
    label_visibility="collapsed",
)

if section == "Review":
    render_review(app)
elif section == "Search":
    render_search(app)
elif section == "Monitor":
    render_monitor(app)
elif section == "Predict":
    render_predict(app)