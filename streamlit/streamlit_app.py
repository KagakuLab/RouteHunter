import html
import os.path
from pathlib import Path

import pandas as pd
import streamlit as st

from routehunter import RouteHunterApp
from routehunter.app import DEFAULT_CSV_SUBPATH
from routehunter.core import InvalidSMILESError


# Global settings
DATA_DIR = "rh-data"
SECTIONS = ["📊 Review", "🔎 Search", "📄️ Monitor", "💻 Predict", "💾 Download", "⬇️ Contribute"]

# Sidebar settings
SIDEBAR_WIDTH_PX = 220
SIDEBAR_CSS = f"""
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    font-size: 1.15rem;
}}
div[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapseButton"] {{
    display: none;
}}
section[data-testid="stSidebar"] {{
    min-width: {SIDEBAR_WIDTH_PX}px;
    max-width: {SIDEBAR_WIDTH_PX}px;
}}
</style>
"""

# Metrics font size
st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] p {
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header space size
st.markdown(
    """
    <style>
    .block-container,
    .stAppViewBlockContainer,
    .stMainBlockContainer,
    section.stMain .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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


def caption(text: str, size: str = "1.1rem", color: str = "#000000") -> None:
    st.caption(
        f'<span style="font-size:{size}; color:{color};">{text}</span>',
        unsafe_allow_html=True,
    )

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
    return st.text_input("SMILES", key=key)


# --- Sections --------------------------------------------------------

def render_review(app: RouteHunterApp) -> None:
    st.title("📊 Review")
    st.write("RouteHunter dataset introduction")
    st.divider()

    # render database stats
    stats = app.store.stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Targets", stats["n_targets"])
    c2.metric("Papers", stats["n_papers"])
    c3.metric("Targets with >1 paper", stats["n_multi_paper_targets"])
    c4.metric("Targets with predicted routes", stats["n_cached_casp_routes"])
    c5.metric("Targets predicted for digitalization", stats["n_cached_casp_routes"])

    # render journals and contributors
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Targets by journal")
        render_stat_list(stats["targets_by_journal"], "No journal data available")
    with col_b:
        st.subheader("Targets by contributor")
        render_stat_list(stats["targets_by_contributor"], "No contributor data available")

    # dataset preview
    st.write("")
    st.subheader("Dataset preview")
    st.write("RouteHunter dataset file preview")

    # load dataset
    seed_df = load_seed_csv(DATA_DIR)
    seed_df = seed_df[["title", "doi", "smiles"]]
    seed_df.index = seed_df.index + 1
    st.dataframe(seed_df, use_container_width=True, hide_index=False)


def render_search(app: RouteHunterApp) -> None:
    st.title("🔎 Search")
    st.write("Give a SMILES, get literature papers reporting the route for this molecule")

    # process smiles
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

        # positive search results
        if result.found:
            if result.papers:
                st.subheader(f"📄 Found {len(result.papers)} paper(s) with route for this molecule")
                st.dataframe(
                    [
                        {
                            "Journal": p.journal,
                            "Title": p.title,
                            "Year": p.year,
                            "DOI": p.doi,
                        }
                        for p in result.papers
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            st.subheader(f"💻 Found {len(result.casp_solved)} tool(s) with prediction for this molecule")
            if result.casp_solved:
                st.dataframe(
                    [
                        {"Tool": e.tool_display, "Result": f"Solved by {e.tool_display}", "Link": e.link}
                        for e in result.casp_solved
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No predictions were obtained by CASP tools for this molecule")

        # negative search result
        if not result.found:
            st.subheader("📄 Found 0 paper(s) with route for this molecule")
            st.info("No papers with route were found for this molecule. You can try CASP tools for prediction:")
            cols = st.columns(len(result.properties) or 1)
            for col, (name, value) in zip(cols, result.properties.items()):
                col.metric(
                    f"Chance to be solved by {name}",
                    f"{value:.0%}" if value is not None else "n/a",
                )


def render_monitor(app: RouteHunterApp) -> None:
    st.title("📄️ Monitor")
    st.write("Browse recent papers, pre-scored by predicted route probability. ")

    # search by year
    year = st.number_input("Year", min_value=1832, max_value=2100, value=2025, step=1)
    if st.button("Predict", type="primary"):
        result = app.monitor(int(year))
        if not result.available:
            st.info(result.message)
            return

        # display search result
        st.write(result.message)
        monitor_df = result.to_dataframe()
        monitor_df.index = monitor_df.index + 1
        st.dataframe(
            monitor_df,
            use_container_width=True,
            hide_index=False,
            column_config={
                "route_prob": st.column_config.TextColumn("Route report probability"),
                "journal": st.column_config.TextColumn("Journal"),
                "title": st.column_config.TextColumn("Title", width="large"),
                "publication_date": st.column_config.TextColumn("Publication date"),
                "doi": st.column_config.TextColumn("DOI"),
            },
        )


def render_predict(app: RouteHunterApp) -> None:
    st.title("💻 Predict")
    st.write("Predict solvability for a target with no known literature")

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
                "probability": st.column_config.TextColumn("Predicted chance to be solved"),
                "url": st.column_config.LinkColumn("Link"),
            },
        )


def render_download(app: RouteHunterApp) -> None:
    st.title("💾 Download")
    st.write("Export RouteHunter's underlying data files.")

    # target dataset
    st.subheader("Target collection")
    st.write("Digitalized target collection")
    file_path = Path(DATA_DIR) / "core/routehunter_seed.csv"
    csv_data = file_path.read_bytes()
    pdf_data = file_path.read_bytes()

    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        st.download_button(label="Download CSV", data=csv_data, file_name="target_collection.csv", mime="text/csv")
    with col2:
        st.download_button(label="Download PDF", data=pdf_data, file_name="target_collection.pdf", mime="application/pdf")


def render_contribute(app: RouteHunterApp) -> None:
    st.title("⬇️ Contribute")
    st.write(
        "RouteHunter's dataset is static — there is no in-app way to add or "
        "edit records. Contributions are reviewed and validated by an "
        "administrator before being added to the seed CSV."
    )

    st.write(
        "To submit a new record or correct an existing one, send your "
        "request to [dvzankov@gmail.com](mailto:dvzankov@gmail.com). "
        "Please include the name you'd like registered as the contributor "
        "on your records — that's the name that will appear in the dataset "
        "(see Review → \"Papers by contributor\")."
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

if section == "📊 Review":
    render_review(app)
elif section == "🔎 Search":
    render_search(app)
elif section == "📄️ Monitor":
    render_monitor(app)
elif section == "💻 Predict":
    render_predict(app)
elif section == "💾 Download":
    render_download(app)
elif section == "⬇️ Contribute":
    render_contribute(app)
