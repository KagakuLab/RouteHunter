import html
from pathlib import Path

import pandas as pd
import streamlit as st

from routehunter import RouteHunterApp
from routehunter.app import TargetStaticData, CandidateStaticData, AbstractTrainingData
from routehunter.utils import load_config
from routehunter.core import InvalidSMILESError


# Global settings
DATA_DIR = "rh_data"
SECTIONS = ["📊 Review", "🔎 Search", "💻 Predict", "📄️ Monitor", "💾 Download", "⬇️ Contribute"]

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
        font-size: 1.0rem;
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

@st.cache_resource(show_spinner="Loading RouteHunter ...")
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
    no pass through TargetStore/Target parsing. Path comes from
    config.csv's TargetStaticData entry -- there's no default path to
    fall back to anymore."""
    seed_path = load_config(data_dir)[TargetStaticData]
    return pd.read_csv(seed_path)


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
                    padding:0.1rem 0; border-bottom:1px solid rgba(128,128,128,0.25);">
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
    st.write("**RouteHunter** is a system for collecting and sharing reference data on multi-step synthesis routes. "
             "Its current database contains **1,000+ digitized target molecules** (stored as SMILES), "
             "each linked to the paper reporting a multi-step route to it. This dataset supports benchmarking "
             "**computer-aided synthesis planning (CASP)** tools by how many targets they can solve, "
             "and - for solved targets - comparing a tool's predicted route against the one published in the original paper. "
             "Beyond the dataset itself, **RouteHunter** lets you **search for a molecule** to find whether a route to it has "
             "already been **published or solved by a CASP tool**, **predict the molecule solvability** by open-source CASP tools, "
             "and **monitor newly published papers** likely to report new routes.")
    st.divider()

    # render database stats
    stats = app.review_engine.stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Targets", stats["n_targets"])
    c2.metric("Papers", stats["n_papers"])
    c3.metric("Targets with >1 paper", stats["n_multi_paper_targets"])
    c4.metric("Targets with predicted routes", stats["n_cached_casp_routes"])
    c5.metric("Papers predicted for digitalization", stats["n_predicted_candidate_papers"])
    st.divider()

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
    st.subheader("Target collection preview")
    st.write("Each paper was checked to confirm it reports a detailed multi-step synthesis route; if so, the target "
             "molecule was extracted and stored as its SMILES. A preview of the resulting dataset is shown below:")

    # load dataset
    seed_df = load_seed_csv(DATA_DIR)
    seed_df = seed_df[["title", "doi", "target"]]
    seed_df.index = seed_df.index + 1
    st.dataframe(seed_df, use_container_width=True, hide_index=False)


def render_search(app: RouteHunterApp) -> None:
    st.title("🔎 Search")
    st.write(
        "**Search service** lets you check whether a given target molecule already has a known route - "
        "either a paper reporting it, or a CASP tool that has already predicted a route for it")
    st.write(
        "**Try examples for positive search results:**\n"
        "- ``C#CCOC1=C(C=C(C(=C1)N2C(=O)N3CCCCC3=N2)Cl)Cl``\n"
        "- ``C(O)(C(O)=O)C(C1C=CC=CC=1)NC(C1C=CC=CC=1)=O``\n"
        "- ``C1CCC(=C(C1)CC(=O)O)N2C(=O)C=CC(=N2)C3=C4C=CC=CN4N=C3C5=CC=CC=C5``\n\n"
        "**Try examples for a negative search result:**\n"
        "- ``CC(C)Cc1ccc(cc1)C(C)C(=O)O``"
    )

    # process smiles
    smiles = smiles_input("search_smiles")
    if st.button("Search", type="primary"):
        if not smiles.strip():
            st.warning("Enter a SMILES string first")
            return
        try:
            result = app.search(smiles)
        except InvalidSMILESError as e:
            st.error(f"Couldn't parse that SMILES: {e}")
            return

        # positive search results
        if result.found:
            if not result.paper_report.empty:
                st.subheader(f"📄 {result.paper_message}")
                st.dataframe(
                    result.paper_report,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "journal": st.column_config.TextColumn("Journal"),
                        "title": st.column_config.TextColumn("Title"),
                        "year": st.column_config.NumberColumn("Year"),
                        "doi": st.column_config.TextColumn("DOI"),
                    },
                )
            if not result.tool_report.empty:
                st.subheader(f"💻 {result.tool_message}")
                st.dataframe(
                    result.tool_report,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "tool": st.column_config.TextColumn("Tool"),
                        "result": st.column_config.TextColumn("Result"),
                        "route": st.column_config.TextColumn("Route"),
                    },
                )
            else:
                st.info(result.tool_message)

        # negative search result
        if not result.found:
            st.subheader(f"📄 {result.paper_message}")
            st.info("No papers with route were found for this molecule. You can try CASP tools for prediction:")

            try:
                result = app.predict(smiles)
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
            except InvalidSMILESError as e:
                st.error(f"Couldn't parse that SMILES: {e}")
                return


def render_predict(app: RouteHunterApp) -> None:
    st.title("💻 Predict")
    st.write(
        "**Predict service** can predict the solvability of a molecule - the chance that it can be solved by open-source CASP tools")
    st.write(
        "**Try these examples:**\n"
        "- ``C#CCOC1=C(C=C(C(=C1)N2C(=O)N3CCCCC3=N2)Cl)Cl``\n"
        "- ``C(O)(C(O)=O)C(C1C=CC=CC=1)NC(C1C=CC=CC=1)=O``\n"
        "- ``CC(C)Cc1ccc(cc1)C(C)C(=O)O``")

    smiles = smiles_input("predict_smiles")
    if st.button("Predict", type="primary"):
        if not smiles.strip():
            st.warning("Enter a SMILES string first")
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


def render_monitor(app: RouteHunterApp) -> None:
    st.title("📄️ Monitor")
    st.write(
        "**Monitor service** ranks papers by the predicted probability that they describe a multi-step synthesis route, "
        "based only on the paper's title and abstract.")

    # search by year range
    col1, col2 = st.columns(2)
    with col1:
        year_min = st.number_input("From year", min_value=1832, max_value=2026, value=2020, step=1)
    with col2:
        year_max = st.number_input("To year", min_value=1832, max_value=2026, value=2025, step=1)

    if st.button("Predict", type="primary"):
        if year_min > year_max:
            st.error("'From year' can't be later than 'To year'.")
            return

        result = app.monitor(year_min=int(year_min), year_max=int(year_max))
        if result.empty:
            st.info("No papers found for that year range.")
            return

        st.write(f"{len(result)} paper(s) found, sorted by predicted route probability.")
        monitor_df = result.copy()
        monitor_df = monitor_df.drop("abstract", axis=1)
        monitor_df["route_prob"] = monitor_df["route_prob"].map(lambda p: f"{p:.0%}")
        monitor_df["publication_date"] = monitor_df["publication_date"].dt.strftime("%d/%m/%Y")
        monitor_df.index = range(1, len(monitor_df) + 1)
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


def render_download(app: RouteHunterApp) -> None:
    st.title("💾 Download")
    st.write("Here one can download RouteHunter underlying data files, each useful for a different purpose.")

    # 1. Target dataset
    st.subheader("Digitalized target collection")
    st.write("The full collection of digitized target molecules (stored as SMILES), each linked to its source paper. "
             "Can be used for looking up literature routes for a given target, or for benchmarking CASP tools.")
    file_path = Path(load_config(DATA_DIR)[TargetStaticData])
    st.download_button(label="Download CSV", data=file_path.read_bytes(), file_name="target_collection.csv", mime="text/csv")

    # 2. Candidate dataset
    st.subheader("Candidate paper collection")
    st.write("Papers predicted, but not yet manually confirmed, to report a synthesis route. "
             "Can be used as a starting point for digitizing new targets.")
    file_path = Path(load_config(DATA_DIR)[CandidateStaticData])
    st.download_button(label="Download CSV", data=file_path.read_bytes(), file_name="candidate_collection.csv", mime="text/csv")

    # 3. Paper-with-route dataset
    st.subheader("Paper-with-route training data")
    st.write("Paper titles and abstracts, each labeled with whether that paper reports a synthesis route (1 - Yes / 0 - No). "
             "Can be used for training a route report probability model from open metadata (paper title and abstract) alone.")
    file_path = Path(load_config(DATA_DIR)[AbstractTrainingData])
    st.download_button(label="Download CSV", data=file_path.read_bytes(), file_name="abstract_training_data.csv", mime="text/csv")


def render_contribute(app: RouteHunterApp) -> None:
    st.title("⬇️ Contribute")
    st.write(
        "**RouteHunter** is a static application - for now, there's no way to add or edit any of its data files directly. "
        "All data is updated manually by an administrator, after the submitted data has been validated. "
        "Contributions of any kind are welcome - send them to dvzankov@gmail.com, and please include the name you'd like "
        "registered as the contributor on your records.")
    st.write(
        "**For scientists:** propose new papers and their target molecules for the dataset, or flag targets with "
        "no known route that you'd like tested against open-source CASP .")
    st.write(
        "**For developers:** propose having your CASP tool integrated into **RouteHunter**, for use in route search and solvability prediction.")


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
elif section == "💻 Predict":
    render_predict(app)
elif section == "📄️ Monitor":
    render_monitor(app)
elif section == "💾 Download":
    render_download(app)
elif section == "⬇️ Contribute":
    render_contribute(app)