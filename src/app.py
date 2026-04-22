import math
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_mart import build_data_mart

ROOT_DIR = SRC_DIR.parent
load_dotenv(ROOT_DIR / ".env")

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)
SHORTLIST_SIZE = 10


@st.cache_data
def load_data() -> pd.DataFrame:
    engine = create_engine(DB_DSN)
    query = "SELECT * FROM mart.player_ranking"

    with engine.connect() as conn:
        table_exists = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'mart'
                      AND table_name = 'player_ranking'
                )
                """
            )
        ).scalar()

    if not table_exists:
        build_data_mart()

    return pd.read_sql(query, engine)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "smart_value_index" not in normalized.columns and "final_dss_score" in normalized.columns:
        normalized["smart_value_index"] = normalized["final_dss_score"]

    numeric_columns = [
        "market_value_eur",
        "total_minutes",
        "goal_contributions",
        "attacking_contribution_per_90",
        "smart_value_index",
        "production_score",
        "value_score",
        "reliability_score",
        "discipline_score",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    required_columns = ["market_value_eur", "total_minutes", "smart_value_index"]
    normalized = normalized.dropna(subset=[column for column in required_columns if column in normalized.columns])
    return normalized


def build_filter_defaults(df: pd.DataFrame) -> dict[str, int]:
    budget_min_m = max(1, int(math.floor(df["market_value_eur"].min() / 1_000_000)))
    budget_max_m = max(budget_min_m + 1, int(math.ceil(df["market_value_eur"].max() / 1_000_000)))
    budget_default_m = min(max(50, budget_min_m), budget_max_m)

    minutes_min = int(df["total_minutes"].min())
    minutes_max = int(df["total_minutes"].max())
    minutes_default = max(minutes_min, min(1500, minutes_max))

    return {
        "budget_min_m": budget_min_m,
        "budget_max_m": budget_max_m,
        "budget_default_m": budget_default_m,
        "minutes_min": minutes_min,
        "minutes_max": minutes_max,
        "minutes_default": minutes_default,
    }


def apply_filters(df: pd.DataFrame, max_budget_m: int, min_minutes: int, positions: list[str]) -> pd.DataFrame:
    filtered = df[
        (df["market_value_eur"] <= max_budget_m * 1_000_000)
        & (df["total_minutes"] >= min_minutes)
    ].copy()

    if positions:
        filtered = filtered[filtered["position"].isin(positions)]

    return filtered


def format_eur_millions(value: float) -> str:
    return f"€{value / 1_000_000:.1f}M"


def render_summary(filtered_df: pd.DataFrame, total_players: int) -> None:
    matched_players = len(filtered_df)
    avg_score = filtered_df["smart_value_index"].mean() if matched_players else 0.0
    avg_value = filtered_df["market_value_eur"].mean() if matched_players else 0.0
    avg_contribution_rate = (
        filtered_df["attacking_contribution_per_90"].mean()
        if matched_players and "attacking_contribution_per_90" in filtered_df.columns
        else 0.0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched candidates", f"{matched_players}", f"of {total_players} ranked players")
    col2.metric("Average Smart Value Index", f"{avg_score:.1f}")
    col3.metric("Average market value", format_eur_millions(avg_value) if matched_players else "€0.0M")
    col4.metric("Average contributions / 90", f"{avg_contribution_rate:.2f}")


def render_score_guide() -> None:
    st.subheader("How the score works")
    st.caption("Smart Value Index is the final ranking score shown in this dashboard.")

    formula_col, component_col = st.columns([1, 1])

    with formula_col:
        st.markdown("**Final formula**")
        st.code(
            "Smart Value Index = 40% Production + 35% Value + 20% Reliability + 5% Discipline",
            language="text",
        )

    with component_col:
        st.markdown("**Component formulas**")
        st.markdown(
            "- **Production** = attacking contribution per 90 + goal contributions\n"
            "- **Value** = value efficiency + value retention\n"
            "- **Reliability** = total minutes + appearances\n"
            "- **Discipline** = discipline risk + yellow/red cards"
        )


def render_visual_analysis(filtered_df: pd.DataFrame, shortlist_df: pd.DataFrame) -> None:
    st.subheader("Visual analysis")
    st.caption("Use these charts to compare cost, score, and the players leading the current scenario.")

    scatter_col, bar_col = st.columns(2)

    with scatter_col:
        st.markdown("**Market value vs Smart Value Index**")
        scatter_data = filtered_df[["market_value_eur", "smart_value_index", "position"]].dropna()
        if not scatter_data.empty:
            st.scatter_chart(
                scatter_data,
                x="market_value_eur",
                y="smart_value_index",
                color="position",
                width="stretch",
            )
        else:
            st.info("Not enough filtered data to draw the value vs score chart.")

    with bar_col:
        st.markdown("**Top shortlist candidates by Smart Value Index**")
        bar_data = shortlist_df[["name", "smart_value_index"]].set_index("name")
        if not bar_data.empty:
            st.bar_chart(bar_data, width="stretch")
        else:
            st.info("No shortlist candidates available for the score chart.")


def render_shortlist(shortlist_df: pd.DataFrame) -> None:
    st.subheader(f"Top {SHORTLIST_SIZE} shortlist")
    st.caption("This is the main recommendation list. Players are ordered by Smart Value Index, highest first.")

    if shortlist_df.empty:
        st.info("No players match the current filter settings. Try increasing budget or lowering the reliability threshold.")
        return

    display_columns = [
        "name",
        "position",
        "market_value_eur",
        "total_minutes",
        "goal_contributions",
        "attacking_contribution_per_90",
        "smart_value_index",
    ]
    optional_columns = ["production_score", "value_score", "reliability_score", "discipline_score"]
    for column in optional_columns:
        if column in shortlist_df.columns:
            display_columns.append(column)

    st.dataframe(
        shortlist_df[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "market_value_eur": st.column_config.NumberColumn("Market value (€)", format="€%d"),
            "total_minutes": st.column_config.NumberColumn("Minutes", format="%d"),
            "goal_contributions": st.column_config.NumberColumn("Goal contributions", format="%.0f"),
            "attacking_contribution_per_90": st.column_config.NumberColumn("Contrib / 90", format="%.2f"),
            "smart_value_index": st.column_config.NumberColumn("Smart Value Index", format="%.2f"),
            "production_score": st.column_config.NumberColumn("Production", format="%.2f"),
            "value_score": st.column_config.NumberColumn("Value", format="%.2f"),
            "reliability_score": st.column_config.NumberColumn("Reliability", format="%.2f"),
            "discipline_score": st.column_config.NumberColumn("Discipline", format="%.2f"),
        },
    )


st.set_page_config(page_title="FC Barcelona DSS", layout="wide")
st.title("FC Barcelona: Recruitment Decision Support System")
st.markdown(
    "Support recruitment decisions with transparent filtering, shortlist indicators, and simple tradeoff visuals built from the warehouse mart."
)

try:
    df = normalize_data(load_data())
except SQLAlchemyError as exc:
    st.error(
        "Unable to load mart data. Please run ETL and mart build first: "
        "`python src/etl.py` then `python src/build_mart.py`."
    )
    st.exception(exc)
    st.stop()

if df.empty:
    st.warning("No valid rows found in `mart.player_ranking` after data preparation.")
    st.stop()

filter_defaults = build_filter_defaults(df)

st.sidebar.header("Decision criteria")
st.sidebar.caption("Filter the ranked player pool by affordability, reliability, and role fit.")
max_budget = st.sidebar.slider(
    "Max Budget (€ Millions)",
    filter_defaults["budget_min_m"],
    filter_defaults["budget_max_m"],
    filter_defaults["budget_default_m"],
)
min_minutes = st.sidebar.slider(
    "Reliability filter (Min. Minutes Played)",
    filter_defaults["minutes_min"],
    filter_defaults["minutes_max"],
    filter_defaults["minutes_default"],
)
positions = st.sidebar.multiselect(
    "Position",
    options=sorted(df["position"].dropna().unique()),
)

filtered_df = apply_filters(df, max_budget, min_minutes, positions)
shortlist_df = filtered_df.sort_values(by="smart_value_index", ascending=False).head(SHORTLIST_SIZE)

st.subheader("Dashboard overview")
st.caption(
    f"Current scenario: players valued at or below {format_eur_millions(max_budget * 1_000_000)} "
    f"with at least {min_minutes:,} minutes played."
)
render_summary(filtered_df, len(df))

with st.expander("How to read this dashboard", expanded=True):
    st.markdown(
        "- **Budget** narrows the shortlist to financially realistic targets.\n"
        "- **Reliability** uses minutes played as a simple confidence threshold.\n"
        "- **Smart Value Index** is the final ranking score shown in this dashboard.\n"
        "- The index combines Production, Value, Reliability, and Discipline."
    )

if filtered_df.empty:
    st.warning("No players match the current recruitment scenario. Try increasing the budget or lowering the reliability threshold.")
    st.stop()

render_shortlist(shortlist_df)
render_score_guide()
render_visual_analysis(filtered_df, shortlist_df)
