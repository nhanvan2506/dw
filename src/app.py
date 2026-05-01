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
from evidence_windows import (
    DEFAULT_EVIDENCE_WINDOW,
    DEFAULT_RELIABILITY_LEVEL,
    get_evidence_window_label,
    get_evidence_window_options,
    get_recent_minutes_column,
    get_reliability_level_options,
    get_reliability_score_column,
    get_reliability_threshold,
    get_smart_value_index_column,
    get_window_config,
)
from similar_player_recommender import recommend_similar_cheaper_players

ROOT_DIR = SRC_DIR.parent
load_dotenv(ROOT_DIR / ".env")

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)
SHORTLIST_SIZE = 10
REQUIRED_RANKING_COLUMNS = [
    "recent_minutes_last_season",
    "recent_minutes_last_3_seasons",
    "recent_minutes_last_5_seasons",
    "reliability_score_last_season",
    "reliability_score_last_3_seasons",
    "reliability_score_last_5_seasons",
    "smart_value_index_last_season",
    "smart_value_index_last_3_seasons",
    "smart_value_index_last_5_seasons",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    engine = create_engine(DB_DSN)
    query = """
    WITH latest_club AS (
        SELECT player_id, club_id
        FROM (
            SELECT
                player_id,
                club_id,
                date_id,
                ROW_NUMBER() OVER (
                    PARTITION BY player_id
                    ORDER BY date_id DESC
                ) AS rn
            FROM warehouse.fact_player_valuations
            WHERE market_value_eur IS NOT NULL
        ) x
        WHERE rn = 1
    )
    SELECT
        pr.player_id,
        pr.name,
        pr.position,
        EXTRACT(YEAR FROM AGE(pr.latest_value_date, dp.birth_date)) AS age,
        dp.nationality,
        dc.club_name,
        dc.country AS club_country,
        pr.appearances_count,
        pr.total_minutes,
        pr.minutes_per_appearance,
        pr.total_goals,
        pr.total_assists,
        pr.goal_contributions,
        pr.attacking_contribution_per_90,
        pr.yellow_cards,
        pr.red_cards,
        pr.discipline_risk_per_90,
        pr.recent_minutes_last_season,
        pr.recent_appearances_last_season,
        pr.recent_minutes_last_3_seasons,
        pr.recent_appearances_last_3_seasons,
        pr.recent_minutes_last_5_seasons,
        pr.recent_appearances_last_5_seasons,
        pr.market_value_eur,
        pr.peak_market_value_eur,
        pr.latest_value_date,
        pr.value_efficiency_index,
        pr.value_retention_ratio,
        pr.production_score,
        pr.value_score,
        pr.discipline_score,
        pr.reliability_score_last_season,
        pr.reliability_score_last_3_seasons,
        pr.reliability_score_last_5_seasons,
        pr.smart_value_index_last_season,
        pr.smart_value_index_last_3_seasons,
        pr.smart_value_index_last_5_seasons,
        pr.final_dss_score,
        pr.smart_value_index
    FROM mart.player_ranking pr
    LEFT JOIN warehouse.dim_players dp
        ON pr.player_id = dp.player_id
    LEFT JOIN latest_club lc
        ON pr.player_id = lc.player_id
    LEFT JOIN warehouse.dim_clubs dc
        ON lc.club_id = dc.club_id
    """

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

    df = pd.read_sql(query, engine)
    missing_columns = [column for column in REQUIRED_RANKING_COLUMNS if column not in df.columns]
    if missing_columns:
        raise RuntimeError(
            "mart.player_ranking is out of date. Rebuild the mart with `python src/build_mart.py` after the warehouse ETL. "
            f"Missing columns: {', '.join(missing_columns)}"
        )

    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    numeric_columns = [
        "age",
        "market_value_eur",
        "total_minutes",
        "goal_contributions",
        "attacking_contribution_per_90",
        "production_score",
        "value_score",
        "discipline_score",
        "reliability_score",
        "smart_value_index",
        "final_dss_score",
        "recent_minutes_last_season",
        "recent_appearances_last_season",
        "recent_minutes_last_3_seasons",
        "recent_appearances_last_3_seasons",
        "recent_minutes_last_5_seasons",
        "recent_appearances_last_5_seasons",
        "reliability_score_last_season",
        "reliability_score_last_3_seasons",
        "reliability_score_last_5_seasons",
        "smart_value_index_last_season",
        "smart_value_index_last_3_seasons",
        "smart_value_index_last_5_seasons",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    required_columns = ["market_value_eur", "total_minutes", "smart_value_index"]
    normalized = normalized.dropna(subset=[column for column in required_columns if column in normalized.columns])
    return normalized


def prepare_evidence_view(df: pd.DataFrame, evidence_window: str) -> pd.DataFrame:
    window_config = get_window_config(evidence_window)
    view = df.copy()

    recent_minutes_col = get_recent_minutes_column(evidence_window)
    recent_appearances_col = window_config["recent_appearances_column"]
    reliability_col = get_reliability_score_column(evidence_window)
    score_col = get_smart_value_index_column(evidence_window)

    view["recent_minutes"] = pd.to_numeric(view[recent_minutes_col], errors="coerce")
    view["recent_appearances"] = pd.to_numeric(view[recent_appearances_col], errors="coerce")
    view["reliability_score"] = pd.to_numeric(view[reliability_col], errors="coerce")
    view["smart_value_index"] = pd.to_numeric(view[score_col], errors="coerce")
    view["final_dss_score"] = view["smart_value_index"]

    return view


def build_budget_defaults(df: pd.DataFrame) -> dict[str, int]:
    budget_min_m = max(1, int(math.floor(df["market_value_eur"].min() / 1_000_000)))
    budget_max_m = max(budget_min_m + 1, int(math.ceil(df["market_value_eur"].max() / 1_000_000)))
    budget_default_m = min(max(50, budget_min_m), budget_max_m)

    return {
        "budget_min_m": budget_min_m,
        "budget_max_m": budget_max_m,
        "budget_default_m": budget_default_m,
    }


def apply_filters(df: pd.DataFrame, max_budget_m: int, min_recent_minutes: int, positions: list[str]) -> pd.DataFrame:
    filtered = df[
        (df["market_value_eur"] <= max_budget_m * 1_000_000)
        & (df["recent_minutes"] >= min_recent_minutes)
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
    avg_recent_minutes = filtered_df["recent_minutes"].mean() if matched_players else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched candidates", f"{matched_players}", f"of {total_players} ranked players")
    col2.metric("Average Smart Value Index", f"{avg_score:.1f}")
    col3.metric("Average market value", format_eur_millions(avg_value) if matched_players else "€0.0M")
    col4.metric("Average recent minutes", f"{avg_recent_minutes:.0f}")


def render_visual_analysis(filtered_df: pd.DataFrame) -> None:
    st.subheader("Value vs Score")
    st.caption("The chart compares market value against the active Smart Value Index for the current evidence window.")

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


def render_shortlist(shortlist_df: pd.DataFrame, evidence_window: str) -> None:
    window_label = get_evidence_window_label(evidence_window)
    st.subheader("Top Potential Players")
    st.caption(f"Players are ordered by the active Smart Value Index using {window_label.lower()} evidence.")

    if shortlist_df.empty:
        st.info("No players match the current filter settings. Try increasing budget or lowering the reliability level.")
        return

    display_columns = [
        "smart_value_index",
        "name",
        "position",
        "age",
        "nationality",
        "club_name",
        "market_value_eur",
        "recent_minutes",
        "goal_contributions",
        "attacking_contribution_per_90",
        "production_score",
        "value_score",
        "reliability_score",
        "discipline_score",
    ]
    display_columns = [column for column in display_columns if column in shortlist_df.columns]

    st.dataframe(
        shortlist_df[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "smart_value_index": st.column_config.NumberColumn("Smart Value Index", format="%.2f"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
            "nationality": st.column_config.TextColumn("Nationality"),
            "club_name": st.column_config.TextColumn("Current club"),
            "market_value_eur": st.column_config.NumberColumn("Market value (€)", format="€%,d"),
            "recent_minutes": st.column_config.NumberColumn(f"Recent minutes ({window_label})", format="%d"),
            "goal_contributions": st.column_config.NumberColumn("Goal contributions", format="%.0f"),
            "attacking_contribution_per_90": st.column_config.NumberColumn("Contribution / 90", format="%.2f"),
            "production_score": st.column_config.NumberColumn("Production", format="%.2f"),
            "value_score": st.column_config.NumberColumn("Value", format="%.2f"),
            "reliability_score": st.column_config.NumberColumn("Reliability", format="%.2f"),
            "discipline_score": st.column_config.NumberColumn("Discipline", format="%.2f"),
        },
    )


def render_similar_alternatives(df: pd.DataFrame, evidence_window: str, reliability_level: str, age_range: tuple[int, int]) -> None:
    window_label = get_evidence_window_label(evidence_window)

    if df.empty:
        st.info("No player data available for similarity recommendation.")
        return

    candidate_names = sorted(df["name"].dropna().unique())

    control_col1, control_col2 = st.columns([2, 1])

    with control_col1:
        target_player = st.selectbox(
            "Target player",
            options=candidate_names,
            index=None,
            placeholder="Choose a target player",
        )

    with control_col2:
        same_position = st.checkbox(
            "Same position only",
            value=True,
        )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        max_value_ratio = st.slider(
            "Max value ratio",
            min_value=0.30,
            max_value=1.00,
            value=0.75,
            step=0.05,
            help="Candidate value must be below this ratio of the target player's market value.",
        )

    with filter_col2:
        min_similarity = st.slider(
            "Min similarity",
            min_value=0,
            max_value=100,
            value=70,
            step=5,
            help="Minimum similarity score for a recommendation to stay on the list. Higher values return players whose profile is closer to the target.",
        )

    if target_player is None:
        st.info("Choose a target player to see similar alternatives.")
        return

    try:
        target, alternatives = recommend_similar_cheaper_players(
            player_name=target_player,
            top_n=10,
            evidence_window=evidence_window,
            reliability_level=reliability_level,
            max_value_ratio=max_value_ratio,
            min_age=age_range[0],
            max_age=age_range[1],
            min_similarity=min_similarity,
            same_position=same_position,
        )
    except Exception as exc:
        st.error("Unable to generate similar player recommendations.")
        st.exception(exc)
        return

    st.markdown("**Target player**")

    target_display = pd.DataFrame(
        [
            {
                "name": target["name"],
                "position": target["position"],
                "age": target.get("age", None),
                "club_name": target.get("club_name", "Unknown"),
                "market_value_eur": target["market_value_eur"],
                "recent_minutes": target["recent_minutes"],
                "smart_value_index": target["smart_value_index"],
            }
        ]
    )

    st.dataframe(
        target_display,
        hide_index=True,
        width="stretch",
        column_config={
            "club_name": st.column_config.TextColumn("Current club"),
            "market_value_eur": st.column_config.NumberColumn("Market value (€)", format="€%,d"),
            "recent_minutes": st.column_config.NumberColumn(f"Recent minutes ({window_label})", format="%d"),
            "smart_value_index": st.column_config.NumberColumn("Smart Value Index", format="%.2f"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
        },
    )

    st.markdown("**Recommended alternatives**")

    if alternatives.empty:
        threshold = get_reliability_threshold(evidence_window, reliability_level)
        st.info(
            f"No similar alternatives found with {window_label.lower()} evidence and {reliability_level.lower()} reliability ({threshold:,}+ recent minutes). "
            "Try increasing Max value ratio, lowering Min similarity, or allowing different positions."
        )
        return

    display_columns = [
        "smart_value_index",
        "name",
        "position",
        "age",
        "club_name",
        "market_value_eur",
        "recent_minutes",
        "similarity_score",
        "alternative_score",
        "affordability_score",
    ]
    display_columns = [column for column in display_columns if column in alternatives.columns]

    st.dataframe(
        alternatives[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "smart_value_index": st.column_config.NumberColumn("Smart Value Index", format="%.2f"),
            "club_name": st.column_config.TextColumn("Current club"),
            "market_value_eur": st.column_config.NumberColumn("Market value (€)", format="€%,d"),
            "recent_minutes": st.column_config.NumberColumn(f"Recent minutes ({window_label})", format="%d"),
            "similarity_score": st.column_config.NumberColumn("Similarity", format="%.2f"),
            "alternative_score": st.column_config.NumberColumn("Alternative Score", format="%.2f"),
            "affordability_score": st.column_config.NumberColumn("Affordability", format="%.2f"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
        },
    )


st.set_page_config(page_title="FC Barcelona DSS", layout="wide")
st.title("FC Barcelona: Recruitment Decision Support System")
st.markdown(
    "Support recruitment decisions with transparent filtering, shortlist indicators, and simple tradeoff visuals."
)

try:
    df = normalize_data(load_data())
except (SQLAlchemyError, RuntimeError) as exc:
    st.error(
        "Unable to load mart data. Please run ETL and mart build first: "
        "`python src/etl.py` then `python src/build_mart.py`."
    )
    st.exception(exc)
    st.stop()

if df.empty:
    st.warning("No valid rows found in `mart.player_ranking` after data preparation.")
    st.stop()

navigation = st.sidebar.radio(
    "Go to",
    ["Potential Shortlist", "Similar Alternatives"],
)

st.sidebar.header("Decision context")
evidence_window = st.sidebar.radio(
    "Evidence window",
    options=get_evidence_window_options(),
    index=get_evidence_window_options().index(DEFAULT_EVIDENCE_WINDOW),
    format_func=get_evidence_window_label,
)
reliability_level = st.sidebar.radio(
    "Reliability level",
    options=get_reliability_level_options(),
    index=get_reliability_level_options().index(DEFAULT_RELIABILITY_LEVEL),
)
age_range = st.sidebar.slider(
    "Age range",
    min_value=16,
    max_value=40,
    value=(18, 30),
    step=1,
)

window_label = get_evidence_window_label(evidence_window)
min_recent_minutes = get_reliability_threshold(evidence_window, reliability_level)

evidence_df = prepare_evidence_view(df, evidence_window)

if navigation == "Potential Shortlist":
    budget_defaults = build_budget_defaults(evidence_df)

    max_budget = st.sidebar.slider(
        "Max Budget (€ Millions)",
        budget_defaults["budget_min_m"],
        budget_defaults["budget_max_m"],
        budget_defaults["budget_default_m"],
    )

    positions = st.sidebar.multiselect(
        "Position",
        options=sorted(evidence_df["position"].dropna().unique()),
    )

    filtered_df = apply_filters(evidence_df, max_budget, min_recent_minutes, positions)
    filtered_df = filtered_df[filtered_df["age"].between(age_range[0], age_range[1])].copy()
    shortlist_df = filtered_df.sort_values(by="smart_value_index", ascending=False).head(SHORTLIST_SIZE)

    st.subheader("Dashboard Overview")
    st.caption(
        f"Current scenario: players valued at or below {format_eur_millions(max_budget * 1_000_000)}, "
        f"using {window_label.lower()} evidence, {reliability_level.lower()} reliability, and age {age_range[0]}–{age_range[1]}."
    )

    render_summary(filtered_df, len(evidence_df))

    with st.expander("How to read this dashboard", expanded=False):
        st.markdown(
            "- **Budget** narrows the shortlist to financially realistic targets.\n"
            f"- **Evidence window** chooses which recent seasons count: `{get_evidence_window_label('last_season')}`, `{get_evidence_window_label('last_3_seasons')}`, or `{get_evidence_window_label('last_5_seasons')}`.\n"
            f"- **Reliability level** maps to a minimum recent-minute threshold for the selected window: `{reliability_level}` = `{min_recent_minutes:,}+` minutes.\n"
            f"- **Age range** limits the shortlist to players between `{age_range[0]}` and `{age_range[1]}` years old.\n"
            "- **Smart Value Index** is the final ranking score shown in this dashboard.\n"
            "- **Formula**: `Smart Value Index = 40% Production + 35% Value + 20% Reliability + 5% Discipline`.\n"
            "- **Production** = attacking contribution per 90 + goal contributions.\n"
            "- **Value** = value efficiency + value retention ratio.\n"
            "- **Reliability** = recent minutes + recent appearances in the selected evidence window.\n"
            "- **Discipline** = discipline risk per 90 + yellow/red cards.\n"
            "- The chart compares market value against the active Smart Value Index."
        )

    if filtered_df.empty:
        st.warning(
            "No players match the current recruitment scenario. "
            "Try increasing the budget or lowering the reliability level."
        )
        st.stop()

    render_shortlist(shortlist_df, evidence_window)
    render_visual_analysis(filtered_df)

else:
    st.subheader("Similar Alternative Recommendation")
    st.caption(
        f"Choose a target player and the system recommends players with similar performance profiles using {window_label.lower()} evidence."
    )

    with st.expander("How to read this feature", expanded=False):
        st.markdown(
            "- **Target player** is the player you want to replace or benchmark.\n"
            "- **Similarity** measures how close another player's profile is to the target.\n"
            "- **Affordability** rewards players that are much cheaper than the target.\n"
            f"- **Recent minutes** come from the selected {window_label.lower()} evidence window.\n"
            "- **Smart Value Index** brings in the overall ranking score from the dashboard.\n"
            "- **Alternative Score** combines Similarity, Affordability, and Smart Value Index."
        )

    render_similar_alternatives(evidence_df, evidence_window, reliability_level, age_range)
