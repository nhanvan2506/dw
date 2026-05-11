import html
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


def render_brand_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
        }
        .dashboard-title {
            background: linear-gradient(90deg, #a50044 0%, #004d98 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            color: transparent;
            margin: 0;
        }
        .dashboard-title-accent {
            width: 1000px;
            height: 4px;
            border-radius: 999px;
            margin: 0.45rem 0 1rem;
            background: linear-gradient(90deg, #a50044 0%, #004d98 100%);
        }
        .dashboard-subtitle {
            color: #6b7280;
            margin: 0 0 1.1rem;
        }
        .accent-card {
            background: #f7f9fc;
            border: 1px solid #e5e7eb;
            border-left: 6px solid #004d98;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            margin: 0.25rem 0 0.9rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .accent-title {
            color: #004d98;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .accent-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.24rem 0;
            border-top: 1px solid rgba(0, 77, 152, 0.08);
        }
        .accent-row:first-of-type { border-top: 0; padding-top: 0; }
        .accent-label { color: #6b7280; font-weight: 600; }
        .accent-value { color: #1f2937; font-weight: 800; text-align: right; }
        .compact-card-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem 0.55rem;
        }
        .compact-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.38rem 0.62rem;
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid #dbe4f0;
            color: #1f2937;
            font-size: 0.94rem;
            line-height: 1.2;
            white-space: nowrap;
        }
        .compact-badge strong {
            color: #004d98;
            font-weight: 800;
        }
        @media (max-width: 768px) {
            .accent-row { flex-direction: column; gap: 0.15rem; }
            .accent-value { text-align: left; }
            .compact-badge { white-space: normal; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header() -> None:
    st.markdown(
        """
        <h1 class="dashboard-title">FC Barcelona: Recruitment Decision Support System</h1>
        <div class="dashboard-title-accent"></div>
        <p class="dashboard-subtitle">
            Support FC Barcelona recruitment decisions with transparent filtering, scenario summaries,
            shortlist rankings, and simple tradeoff visuals.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_accent_card(title: str, rows: list[tuple[str, str]]) -> None:
    row_html = "".join(
        f'<div class="accent-row"><span class="accent-label">{html.escape(label)}</span>'
        f'<span class="accent-value">{html.escape(value)}</span></div>'
        for label, value in rows
    )
    st.markdown(
        f"""
        <div class="accent-card">
            <div class="accent-title">{html.escape(title)}</div>
            {row_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_card(title: str, badges: list[str]) -> None:
    badge_html = "".join(
        f'<span class="compact-badge">{badge}</span>'
        for badge in badges
    )
    st.markdown(
        f"""
        <div class="accent-card">
            <div class="accent-title">{html.escape(title)}</div>
            <div class="compact-card-grid">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sync_widget_state(persistent_key: str, widget_key: str) -> None:
    st.session_state[persistent_key] = st.session_state.get(widget_key)


def render_summary(filtered_df: pd.DataFrame, total_players: int) -> None:
    matched_players = len(filtered_df)
    avg_score = filtered_df["smart_value_index"].mean() if matched_players else 0.0
    avg_value = filtered_df["market_value_eur"].mean() if matched_players else 0.0
    avg_recent_minutes = filtered_df["recent_minutes"].mean() if matched_players else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched candidates", f"{matched_players}", f"out of {total_players} ranked players")
    col2.metric("Average DSS Score", f"{avg_score:.1f}")
    col3.metric("Average market value", format_eur_millions(avg_value) if matched_players else "€0.0M")
    col4.metric("Average recent minutes", f"{avg_recent_minutes:.0f}")


def render_visual_analysis(filtered_df: pd.DataFrame) -> None:
    st.subheader("Market Value vs DSS Score")
    st.caption("The chart compares market value in €M against the active DSS Score for the current evidence period.")

    scatter_data = filtered_df[["market_value_eur", "smart_value_index", "position"]].dropna().copy()
    if not scatter_data.empty:
        scatter_data["Market value (€M)"] = scatter_data["market_value_eur"] / 1_000_000
        scatter_data["DSS Score"] = scatter_data["smart_value_index"]
        st.scatter_chart(
            scatter_data[["Market value (€M)", "DSS Score", "position"]],
            x="Market value (€M)",
            y="DSS Score",
            color="position",
            width="stretch",
        )
    else:
        st.info("Not enough filtered data to draw the Market Value vs DSS Score chart.")


def render_shortlist_summary(shortlist_df: pd.DataFrame) -> None:
    if shortlist_df.empty:
        return

    top_pick = shortlist_df.iloc[0]
    best_value_pick = shortlist_df.sort_values(by=["value_score", "smart_value_index"], ascending=False).iloc[0]
    most_reliable_pick = shortlist_df.sort_values(by=["reliability_score", "smart_value_index"], ascending=False).iloc[0]
    render_compact_card(
        "Recommended shortlist summary",
        [
            f"<strong>Top overall:</strong> {html.escape(top_pick['name'])} · DSS Score {top_pick['smart_value_index']:.1f}",
            f"<strong>Best value:</strong> {html.escape(best_value_pick['name'])} · Value Score {best_value_pick['value_score']:.1f}",
            f"<strong>Most reliable:</strong> {html.escape(most_reliable_pick['name'])} · Reliability {most_reliable_pick['reliability_score']:.1f}",
        ],
    )


def render_shortlist(shortlist_df: pd.DataFrame, evidence_window: str) -> None:
    window_label = get_evidence_window_label(evidence_window)
    st.subheader("Recommended Shortlist")
    st.caption(f"Players are ordered by the active DSS Score using {window_label.lower()} evidence.")

    if shortlist_df.empty:
        st.info("No players match the current filter settings. Try increasing budget or lowering the minimum reliability.")
        return

    primary_columns = [
        "smart_value_index",
        "name",
        "position",
        "age",
        "club_name",
        "market_value_eur",
        "recent_minutes",
        "production_score",
        "value_score",
        "reliability_score",
    ]
    primary_columns = [column for column in primary_columns if column in shortlist_df.columns]

    st.dataframe(
        shortlist_df[primary_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "smart_value_index": st.column_config.NumberColumn("DSS Score", format="%.2f"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
            "club_name": st.column_config.TextColumn("Current club"),
            "market_value_eur": st.column_config.NumberColumn("Market value (€)", format="€%,d"),
            "recent_minutes": st.column_config.NumberColumn(f"Recent playing time ({window_label})", format="%d"),
            "production_score": st.column_config.NumberColumn("Production", format="%.2f"),
            "value_score": st.column_config.NumberColumn("Value", format="%.2f"),
            "reliability_score": st.column_config.NumberColumn("Reliability", format="%.2f"),
        },
    )

    with st.expander("Show detailed player metrics", expanded=False):
        detail_columns = [
            "name",
            "nationality",
            "goal_contributions",
            "attacking_contribution_per_90",
            "yellow_cards",
            "red_cards",
            "discipline_risk_per_90",
            "discipline_score",
        ]
        detail_columns = [column for column in detail_columns if column in shortlist_df.columns]
        st.dataframe(
            shortlist_df[detail_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "nationality": st.column_config.TextColumn("Nationality"),
                "goal_contributions": st.column_config.NumberColumn("Goal contributions", format="%.0f"),
                "attacking_contribution_per_90": st.column_config.NumberColumn("Attacking contribution / 90", format="%.2f"),
                "yellow_cards": st.column_config.NumberColumn("Yellow cards", format="%.0f"),
                "red_cards": st.column_config.NumberColumn("Red cards", format="%.0f"),
                "discipline_risk_per_90": st.column_config.NumberColumn("Discipline risk / 90", format="%.2f"),
                "discipline_score": st.column_config.NumberColumn("Discipline", format="%.2f"),
            },
        )


def render_similar_alternatives(df: pd.DataFrame, evidence_window: str, reliability_level: str, age_range: tuple[int, int]) -> None:
    window_label = get_evidence_window_label(evidence_window)

    if df.empty:
        st.info("No player data available for similarity recommendation.")
        return

    candidate_names = sorted(df["name"].dropna().unique())

    st.session_state.setdefault("similar_target_player", None)
    st.session_state.setdefault("similar_same_position", True)
    st.session_state.setdefault("similar_max_value_ratio", 0.75)
    st.session_state.setdefault("similar_min_similarity", 70)

    if st.session_state["similar_target_player"] not in candidate_names:
        st.session_state["similar_target_player"] = None
        st.session_state.pop("similar_target_player_widget", None)

    if "similar_target_player_widget" not in st.session_state:
        st.session_state["similar_target_player_widget"] = st.session_state["similar_target_player"]
    if "similar_same_position_widget" not in st.session_state:
        st.session_state["similar_same_position_widget"] = st.session_state["similar_same_position"]
    if "similar_max_value_ratio_widget" not in st.session_state:
        st.session_state["similar_max_value_ratio_widget"] = st.session_state["similar_max_value_ratio"]
    if "similar_min_similarity_widget" not in st.session_state:
        st.session_state["similar_min_similarity_widget"] = st.session_state["similar_min_similarity"]

    control_col1, control_col2 = st.columns([2, 1])

    with control_col1:
        target_player = st.selectbox(
            "Target player",
            options=candidate_names,
            index=None,
            placeholder="Choose a target player",
            key="similar_target_player_widget",
            on_change=sync_widget_state,
            args=("similar_target_player", "similar_target_player_widget"),
        )

    with control_col2:
        same_position = st.checkbox(
            "Same position only",
            key="similar_same_position_widget",
            on_change=sync_widget_state,
            args=("similar_same_position", "similar_same_position_widget"),
        )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        max_value_ratio = st.slider(
            "Maximum price compared to target",
            min_value=0.30,
            max_value=1.00,
            step=0.05,
            help="For example, 0.75 means alternatives must cost at most 75% of the target player's market value.",
            key="similar_max_value_ratio_widget",
            on_change=sync_widget_state,
            args=("similar_max_value_ratio", "similar_max_value_ratio_widget"),
        )

    with filter_col2:
        min_similarity = st.slider(
            "Minimum similarity",
            min_value=0,
            max_value=100,
            step=5,
            help="Higher values keep players whose profiles are closer to the target.",
            key="similar_min_similarity_widget",
            on_change=sync_widget_state,
            args=("similar_min_similarity", "similar_min_similarity_widget"),
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

    st.markdown("**Target player profile**")

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
            "recent_minutes": st.column_config.NumberColumn(f"Recent playing time ({window_label})", format="%d"),
            "smart_value_index": st.column_config.NumberColumn("DSS Score", format="%.2f"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
        },
    )

    st.markdown("**Recommended alternatives**")

    if alternatives.empty:
        threshold = get_reliability_threshold(evidence_window, reliability_level)
        st.info(
            f"No cheaper similar alternatives found with {window_label.lower()} evidence and {reliability_level.lower()} minimum reliability ({threshold:,}+ recent minutes). "
            "Try increasing Maximum price compared to target, lowering Minimum similarity, or allowing different positions."
        )
        return

    top_alternative = alternatives.iloc[0]
    render_accent_card(
        "Recommendation summary",
        [
            ("Found alternatives", f"{len(alternatives)} cheaper options for {target['name']}"),
            (
                "Top match",
                f"{top_alternative['name']} — Similarity {top_alternative['similarity_score']:.2f} | "
                f"Affordability {top_alternative['affordability_score']:.2f} | Alternative Score {top_alternative['alternative_score']:.2f}",
            ),
        ],
    )

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
            "smart_value_index": st.column_config.NumberColumn("DSS Score", format="%.2f"),
            "club_name": st.column_config.TextColumn("Current club"),
            "market_value_eur": st.column_config.NumberColumn("Market value (€)", format="€%,d"),
            "recent_minutes": st.column_config.NumberColumn(f"Recent playing time ({window_label})", format="%d"),
            "similarity_score": st.column_config.NumberColumn("Similarity", format="%.2f"),
            "alternative_score": st.column_config.NumberColumn("Alternative Score", format="%.2f"),
            "affordability_score": st.column_config.NumberColumn("Affordability", format="%.2f"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
        },
    )


st.set_page_config(page_title="FC Barcelona DSS", layout="wide")
render_brand_styles()
render_page_header()

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

st.sidebar.header("Navigation")
navigation = st.sidebar.radio(
    "Go to",
    ["Potential Shortlist", "Similar Alternatives"],
    label_visibility="collapsed",
)

st.sidebar.header("Scenario settings")
evidence_window = st.sidebar.radio(
    "Performance evidence period",
    options=get_evidence_window_options(),
    index=get_evidence_window_options().index(DEFAULT_EVIDENCE_WINDOW),
    format_func=get_evidence_window_label,
)
reliability_level = st.sidebar.radio(
    "Minimum reliability",
    options=get_reliability_level_options(),
    index=get_reliability_level_options().index(DEFAULT_RELIABILITY_LEVEL),
)

window_label = get_evidence_window_label(evidence_window)
min_recent_minutes = get_reliability_threshold(evidence_window, reliability_level)

evidence_df = prepare_evidence_view(df, evidence_window)

max_budget = None
positions: list[str] = []
if navigation == "Potential Shortlist":
    budget_defaults = build_budget_defaults(evidence_df)
    max_budget = st.sidebar.slider(
        "Maximum budget (€M)",
        budget_defaults["budget_min_m"],
        budget_defaults["budget_max_m"],
        budget_defaults["budget_default_m"],
    )

st.sidebar.header("Player filters")
age_range = st.sidebar.slider(
    "Age range",
    min_value=16,
    max_value=40,
    value=(18, 30),
    step=1,
)

if navigation == "Potential Shortlist":
    positions = st.sidebar.multiselect(
        "Position",
        options=sorted(evidence_df["position"].dropna().unique()),
    )

    filtered_df = apply_filters(evidence_df, max_budget, min_recent_minutes, positions)
    filtered_df = filtered_df[filtered_df["age"].between(age_range[0], age_range[1])].copy()
    shortlist_df = filtered_df.sort_values(by="smart_value_index", ascending=False).head(SHORTLIST_SIZE)

    st.subheader("Dashboard Overview")
    with st.expander("How to read this dashboard", expanded=False):
        st.markdown(
            "- **Maximum budget** limits the shortlist to financially realistic targets.\n"
            f"- **Evidence period** chooses which recent seasons count: `{get_evidence_window_label('last_season')}`, `{get_evidence_window_label('last_3_seasons')}`, or `{get_evidence_window_label('last_5_seasons')}`.\n"
            f"- **Minimum reliability** maps to a minimum recent-minute threshold for the selected period: `{reliability_level.title()}` = `{min_recent_minutes:,}+` minutes.\n"
            f"- **Age range** limits the shortlist to players between `{age_range[0]}` and `{age_range[1]}` years old.\n"
            "- **DSS Score** is the final ranking score shown in this dashboard.\n"
            "- **Formula**: `DSS Score = 40% Production + 35% Value + 20% Reliability + 5% Discipline`.\n"
            "- **Production** = attacking contribution per 90 + goal contributions.\n"
            "- **Value** = value efficiency + value retention ratio.\n"
            "- **Reliability** = recent minutes + recent appearances in the selected evidence period.\n"
            "- **Discipline** = discipline risk per 90 + yellow/red cards.\n"
            "- The chart compares market value against the active DSS Score."
        )

    render_summary(filtered_df, len(evidence_df))

    render_compact_card(
        "Current scenario",
        [
            f"<strong>Position:</strong> {html.escape(', '.join(positions) if positions else 'All positions')}",
            f"<strong>Age range:</strong> {age_range[0]}–{age_range[1]}",
            f"<strong>Budget:</strong> ≤ {html.escape(format_eur_millions(max_budget * 1_000_000))}",
            f"<strong>Evidence period:</strong> {html.escape(window_label)}",
            f"<strong>Reliability:</strong> {html.escape(reliability_level.title())} ({min_recent_minutes:,}+ mins)",
        ],
    )

    render_shortlist_summary(shortlist_df)

    if filtered_df.empty:
        st.warning(
            "No players match the current recruitment scenario. "
            "Try increasing the budget or lowering the minimum reliability."
        )
        st.stop()

    render_shortlist(shortlist_df, evidence_window)
    render_visual_analysis(filtered_df)

else:
    st.subheader("Cheaper Similar Alternatives")
    st.caption(
        f"Select a target player to find cheaper similar alternatives using {window_label.lower()} evidence and the current reliability setting."
    )

    with st.expander("How to read this feature", expanded=False):
        st.markdown(
            "- **Target player** is the player you want to replace or benchmark.\n"
            "- **Similarity** measures how close another player's profile is to the target.\n"
            "- **Affordability** rewards players that are much cheaper than the target.\n"
            f"- **Recent playing time** comes from the selected {window_label.lower()} evidence period.\n"
            "- **DSS Score** brings in the overall ranking score from the dashboard.\n"
            "- **Alternative Score** = 55% Similarity + 25% Affordability + 20% DSS Score."
        )

    render_similar_alternatives(evidence_df, evidence_window, reliability_level, age_range)
