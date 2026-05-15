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
SHORTLIST_DISPLAY_OPTIONS = {
    "Top 10": 10,
    "Top 20": 20,
    "All matches": None,
}


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


def render_app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.5rem;
            padding-bottom: 2rem;
        }
        .app-eyebrow {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .app-title {
            color: #0f172a;
            margin: 0;
            line-height: 1.05;
        }
        .app-subtitle {
            color: #475569;
            margin: 0.7rem 0 0;
            max-width: 72rem;
        }
        .panel-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .panel-copy {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }
        .summary-panel {
            background: #f8fafc;
            border: 1px solid #dbe4f0;
            border-radius: 12px;
            padding: 1rem 1rem 0.95rem;
            margin: 0.75rem 0 1.1rem;
        }
        .summary-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }
        .summary-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            border: 1px solid #d1d5db;
            background: #ffffff;
            color: #1f2937;
            font-size: 0.94rem;
            line-height: 1.2;
        }
        .summary-chip strong {
            color: #0f172a;
        }
        .recommendation-card {
            border-radius: 14px;
            padding: 1rem 1rem 1.05rem;
            border: 1px solid #dbe4f0;
            border-top-width: 5px;
            background: #ffffff;
            min-height: 210px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .recommendation-card--overall {
            border-top-color: #2563eb;
        }
        .recommendation-card--value {
            border-top-color: #16a34a;
        }
        .recommendation-card--reliability {
            border-top-color: #ea580c;
        }
        .recommendation-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }
        .recommendation-name {
            color: #0f172a;
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.95rem;
        }
        .recommendation-metric {
            display: inline-flex;
            align-items: center;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.95rem;
        }
        .recommendation-metric--overall {
            background: #dbeafe;
            color: #1d4ed8;
        }
        .recommendation-metric--value {
            background: #dcfce7;
            color: #15803d;
        }
        .recommendation-metric--reliability {
            background: #ffedd5;
            color: #c2410c;
        }
        .recommendation-copy {
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.55;
        }
        .workflow-panel {
            background: #ffffff;
            border: 1px solid #dbe4f0;
            border-radius: 14px;
            padding: 1rem 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            margin: 0.75rem 0 1.1rem;
        }
        .workflow-panel--quiet {
            background: #f8fafc;
        }
        .workflow-panel--featured {
            border-left: 6px solid #1e3a8a;
        }
        .workflow-panel--empty {
            background: #eff6ff;
            border-color: #bfdbfe;
        }
        .workflow-title {
            color: #0f172a;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .workflow-copy {
            color: #475569;
            font-size: 0.96rem;
            line-height: 1.5;
        }
        .featured-card {
            display: grid;
            gap: 0.55rem;
        }
        .featured-kicker {
            color: #64748b;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .featured-name {
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .featured-metric-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .featured-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.34rem 0.72rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            color: #1f2937;
            background: #eef2ff;
        }
        .featured-pill--strong {
            background: #dbeafe;
            color: #1d4ed8;
        }
        .featured-pill--price {
            background: #dcfce7;
            color: #15803d;
        }
        .featured-pill--fit {
            background: #ffedd5;
            color: #c2410c;
        }
        .workflow-steps {
            margin: 0.65rem 0 0;
            padding-left: 1.1rem;
            color: #334155;
        }
        .workflow-steps li {
            margin-bottom: 0.35rem;
        }
        .table-note {
            color: #64748b;
            font-size: 0.92rem;
            margin-top: -0.15rem;
            margin-bottom: 0.55rem;
        }
        .section-spacer {
            margin-top: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-eyebrow">{html.escape(eyebrow)}</div>
        <h1 class="app-title">{html.escape(title)}</h1>
        <p class="app-subtitle">{html.escape(subtitle)}</p>
        """,
        unsafe_allow_html=True,
    )


def render_scenario_summary(chips: list[str]) -> None:
    chip_html = "".join(f'<span class="summary-chip">{chip}</span>' for chip in chips)
    st.markdown(
        f"""
        <div class="summary-panel">
            <div class="panel-title">Active scenario</div>
            <div class="panel-copy">These controls define the shortlist currently being evaluated.</div>
            <div class="summary-grid">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def render_methodology(window_label: str, reliability_level: str, min_recent_minutes: int) -> None:
    with st.expander("Methodology and score interpretation", expanded=False):
        st.markdown(
            "- **DSS Score** is the final ranking score shown in this dashboard.\n"
            "- **Formula**: `DSS Score = 40% Production + 35% Value + 20% Reliability + 5% Discipline`.\n"
            "- **Evidence period** controls which seasons feed recent playing-time and reliability calculations.\n"
            f"- **Active evidence period**: `{window_label}`.\n"
            f"- **Reliability floor**: `{reliability_level.title()}` = `{min_recent_minutes:,}+` recent minutes for the selected evidence period.\n"
            "- **Value Score** rewards strong output relative to market price.\n"
            "- **Reliability Score** reflects recent minutes and appearances in the selected period.\n"
        )


def sync_widget_state(persistent_key: str, widget_key: str) -> None:
    st.session_state[persistent_key] = st.session_state.get(widget_key)


def build_recommendation_cards(shortlist_df: pd.DataFrame) -> list[dict[str, str | float]]:
    top_pick = shortlist_df.iloc[0]
    best_value_pick = shortlist_df.sort_values(by=["value_score", "smart_value_index"], ascending=False).iloc[0]
    most_reliable_pick = shortlist_df.sort_values(by=["reliability_score", "smart_value_index"], ascending=False).iloc[0]

    return [
        {
            "label": "Top overall",
            "name": top_pick["name"],
            "metric": f"DSS Score {top_pick['smart_value_index']:.1f}",
            "copy": "Highest composite ranking in the current shortlist.",
        },
        {
            "label": "Best value",
            "name": best_value_pick["name"],
            "metric": f"Value Score {best_value_pick['value_score']:.1f}",
            "copy": "Strongest value-for-price signal among the shortlisted options.",
        },
        {
            "label": "Most reliable",
            "name": most_reliable_pick["name"],
            "metric": f"Reliability {most_reliable_pick['reliability_score']:.1f}",
            "copy": "Safest recent-minutes profile in the current shortlist.",
        },
    ]


def render_recommendation_cards(shortlist_df: pd.DataFrame) -> str | None:
    if shortlist_df.empty:
        return None

    cards = build_recommendation_cards(shortlist_df)

    st.subheader("Recommendation summary")

    variant_map = {
        "Top overall": "overall",
        "Best value": "value",
        "Most reliable": "reliability",
    }

    columns = st.columns(3)
    for column, card in zip(columns, cards, strict=False):
        variant = variant_map[card["label"]]
        with column:
            st.markdown(
                f"""
                <div class="recommendation-card recommendation-card--{variant}">
                    <div class="recommendation-label">{html.escape(str(card['label']))}</div>
                    <div class="recommendation-name">{html.escape(str(card['name']))}</div>
                    <div class="recommendation-metric recommendation-metric--{variant}">{html.escape(str(card['metric']))}</div>
                    <div class="recommendation-copy">{html.escape(str(card['copy']))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    return None


def render_shortlist(ranked_df: pd.DataFrame, evidence_window: str, selected_player: str | None) -> None:
    window_label = get_evidence_window_label(evidence_window)
    st.subheader("Priority targets")

    if ranked_df.empty:
        st.info("No players match the current filter settings. Try increasing budget or lowering the minimum reliability.")
        return

    row_count_label = st.selectbox(
        "Rows shown",
        options=list(SHORTLIST_DISPLAY_OPTIONS.keys()),
        index=0,
        key="shortlist_rows_shown",
    )
    row_limit = SHORTLIST_DISPLAY_OPTIONS[row_count_label]

    display_df = ranked_df.copy()
    if row_limit is not None:
        display_df = display_df.head(row_limit)

    display_df = display_df.reset_index(drop=True)
    display_df.insert(0, "Rank", display_df.index + 1)
    display_df["Player"] = display_df["name"]

    primary_columns = [
        "Rank",
        "Player",
        "position",
        "club_name",
        "age",
        "market_value_eur",
        "smart_value_index",
        "value_score",
        "reliability_score",
    ]

    st.dataframe(
        display_df[primary_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Player": st.column_config.TextColumn("Player"),
            "position": st.column_config.TextColumn("Position"),
            "club_name": st.column_config.TextColumn("Current club"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
            "market_value_eur": st.column_config.NumberColumn("Market value (€)", format="€%,d"),
            "smart_value_index": st.column_config.NumberColumn("DSS Score", format="%.2f"),
            "value_score": st.column_config.NumberColumn("Value", format="%.2f"),
            "reliability_score": st.column_config.NumberColumn("Reliability", format="%.2f"),
        },
    )

    with st.expander("Show supporting player metrics", expanded=False):
        detail_df = display_df.copy()
        detail_df["recent_minutes"] = detail_df["recent_minutes"].astype("Int64")
        detail_columns = [
            "Rank",
            "Player",
            "nationality",
            "recent_minutes",
            "production_score",
            "goal_contributions",
            "attacking_contribution_per_90",
            "discipline_score",
            "yellow_cards",
            "red_cards",
            "discipline_risk_per_90",
        ]
        st.dataframe(
            detail_df[detail_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Player": st.column_config.TextColumn("Player"),
                "nationality": st.column_config.TextColumn("Nationality"),
                "recent_minutes": st.column_config.NumberColumn(
                    f"Recent playing time ({window_label})",
                    format="%d",
                ),
                "production_score": st.column_config.NumberColumn("Production", format="%.2f"),
                "goal_contributions": st.column_config.NumberColumn("Goal contributions", format="%.0f"),
                "attacking_contribution_per_90": st.column_config.NumberColumn("Attacking contribution / 90", format="%.2f"),
                "discipline_score": st.column_config.NumberColumn("Discipline", format="%.2f"),
                "yellow_cards": st.column_config.NumberColumn("Yellow cards", format="%.0f"),
                "red_cards": st.column_config.NumberColumn("Red cards", format="%.0f"),
                "discipline_risk_per_90": st.column_config.NumberColumn("Discipline risk / 90", format="%.2f"),
            },
        )


def render_visual_analysis(filtered_df: pd.DataFrame, shortlist_df: pd.DataFrame, selected_player: str | None) -> None:
    st.subheader("Value opportunity map")
    st.caption("Use the chart to validate whether shortlisted players sit in strong score-versus-price positions relative to the wider filtered market.")

    plot_columns = ["name", "position", "club_name", "market_value_eur", "smart_value_index"]
    background_df = filtered_df[plot_columns].dropna().copy()
    shortlist_plot_df = shortlist_df[plot_columns].dropna().copy()

    if background_df.empty or shortlist_plot_df.empty:
        st.info("Not enough filtered data to draw the Market Value versus DSS Score chart.")
        return

    for frame in (background_df, shortlist_plot_df):
        frame["market_value_m"] = (frame["market_value_eur"] / 1_000_000).round(2)
        frame["dss_score"] = frame["smart_value_index"].round(2)

    selected_df = shortlist_plot_df[shortlist_plot_df["name"] == selected_player].copy() if selected_player else pd.DataFrame()

    encoding = {
        "x": {"field": "market_value_m", "type": "quantitative", "title": "Market value (€M)"},
        "y": {"field": "dss_score", "type": "quantitative", "title": "DSS Score"},
        "tooltip": [
            {"field": "name", "type": "nominal", "title": "Player"},
            {"field": "position", "type": "nominal", "title": "Position"},
            {"field": "club_name", "type": "nominal", "title": "Club"},
            {"field": "market_value_m", "type": "quantitative", "title": "Market value (€M)", "format": ".2f"},
            {"field": "dss_score", "type": "quantitative", "title": "DSS Score", "format": ".2f"},
        ],
    }

    layers = [
        {
            "data": {"values": background_df.to_dict("records")},
            "mark": {"type": "circle", "size": 55, "opacity": 0.18, "color": "#94a3b8"},
            "encoding": encoding,
        },
        {
            "data": {"values": shortlist_plot_df.to_dict("records")},
            "mark": {
                "type": "circle",
                "size": 145,
                "opacity": 0.9,
                "color": "#1e3a8a",
                "stroke": "white",
                "strokeWidth": 1,
            },
            "encoding": encoding,
        },
    ]

    if not selected_df.empty:
        selected_records = selected_df.to_dict("records")
        layers.extend(
            [
                {
                    "data": {"values": selected_records},
                    "mark": {
                        "type": "circle",
                        "size": 260,
                        "opacity": 1,
                        "color": "#b45309",
                        "stroke": "white",
                        "strokeWidth": 2,
                    },
                    "encoding": encoding,
                },
                {
                    "data": {"values": selected_records},
                    "mark": {
                        "type": "text",
                        "align": "left",
                        "dx": 8,
                        "dy": -8,
                        "fontSize": 12,
                        "color": "#92400e",
                    },
                    "encoding": {
                        "x": {"field": "market_value_m", "type": "quantitative", "title": "Market value (€M)"},
                        "y": {"field": "dss_score", "type": "quantitative", "title": "DSS Score"},
                        "text": {"field": "name", "type": "nominal"},
                    },
                },
            ]
        )

    st.vega_lite_chart(
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "height": 420,
            "layer": layers,
        },
        use_container_width=True,
    )


def render_no_shortlist_matches(max_budget: int, reliability_level: str, positions: list[str]) -> None:
    st.warning("No candidates match the current shortlist scenario.")
    st.markdown(
        "Try one of the following to widen the pool:\n"
        f"- Increase the budget cap above `{format_eur_millions(max_budget * 1_000_000)}`.\n"
        f"- Lower the reliability floor from `{reliability_level.title()}`.\n"
        "- Expand the age range.\n"
        f"- {'Add more positions to the filter.' if positions else 'Select one or more positions only if you need a narrower shortlist.'}"
    )


def render_similarity_methodology(window_label: str, reliability_level: str) -> None:
    with st.expander("Methodology and score interpretation", expanded=False):
        st.markdown(
            "- **Target player** is the player you want to replace or benchmark.\n"
            "- **Similarity** measures how close another player's profile is to the target.\n"
            "- **Affordability** rewards players that are much cheaper than the target.\n"
            f"- **Recent playing time** comes from the selected {window_label.lower()} evidence period.\n"
            f"- **Reliability floor**: `{reliability_level.title()}`.\n"
            "- **Alternative Score** = 55% Similarity + 25% Affordability + 20% DSS Score."
        )


def render_similarity_empty_state() -> None:
    st.markdown(
        """
        <div class="workflow-panel workflow-panel--empty">
            <div class="workflow-title">How this page works</div>
            <div class="workflow-copy">Choose a target player and then compare cheaper replacements under the active scenario. The featured recommendation appears first, followed by a ranked table of substitutes.</div>
            <ol class="workflow-steps">
                <li>Set the scenario context.</li>
                <li>Choose the target player and constraints.</li>
                <li>Review the featured cheaper replacement and compare the fallback options.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similarity_no_results(window_label: str, reliability_level: str, threshold: int) -> None:
    st.warning("No cheaper similar alternatives found for the current target and filters.")
    st.markdown(
        f"""
        Try relaxing the search in this order:
        - Raise the maximum price compared to the target.
        - Lower the minimum similarity threshold.
        - Allow different positions.
        - Widen the age range if needed.

        The current `{window_label}` evidence period and `{reliability_level.title()}` reliability floor require at least `{threshold:,}+` recent minutes.
        """
    )


def render_similarity_scenario_block(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, tuple[int, int]]:
    with st.container(border=True):
        st.markdown("#### Scenario context")
        st.caption("Define the search universe before choosing the target player.")

        st.session_state.setdefault("similar_evidence_window", DEFAULT_EVIDENCE_WINDOW)
        st.session_state.setdefault("similar_reliability_level", DEFAULT_RELIABILITY_LEVEL)
        st.session_state.setdefault("similar_age_range", (18, 30))

        evidence_col, reliability_col = st.columns(2)
        with evidence_col:
            evidence_window = st.radio(
                "Evidence period",
                options=get_evidence_window_options(),
                index=get_evidence_window_options().index(st.session_state["similar_evidence_window"]),
                format_func=get_evidence_window_label,
                key="similar_evidence_window",
                horizontal=True,
            )
        with reliability_col:
            reliability_level = st.radio(
                "Minimum reliability",
                options=get_reliability_level_options(),
                index=get_reliability_level_options().index(st.session_state["similar_reliability_level"]),
                key="similar_reliability_level",
                horizontal=True,
            )

        evidence_df = prepare_evidence_view(df, evidence_window)

        age_range = st.slider(
            "Age range",
            min_value=16,
            max_value=40,
            value=st.session_state["similar_age_range"],
            step=1,
            key="similar_age_range",
        )

    return evidence_df, evidence_window, reliability_level, age_range


def render_similarity_target_block(evidence_df: pd.DataFrame) -> tuple[str | None, bool, float, int]:
    with st.container(border=True):
        st.markdown("#### Target player and replacement constraints")
        st.caption("Pick the player you want to replace, then tighten or relax the replacement rules.")

        candidate_names = sorted(evidence_df["name"].dropna().unique())

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

        target_player = st.selectbox(
            "Target player",
            options=candidate_names,
            index=None,
            placeholder="Choose a target player",
            key="similar_target_player_widget",
            on_change=sync_widget_state,
            args=("similar_target_player", "similar_target_player_widget"),
        )

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

    return target_player, same_position, max_value_ratio, min_similarity


def render_similarity_target_profile(target: dict, window_label: str) -> None:
    st.markdown("#### Target profile")
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


def render_similarity_featured_card(target: dict, top_alternative: pd.Series) -> None:
    target_price = target.get("market_value_eur") or 0
    alt_price = top_alternative.get("market_value_eur") or 0
    price_delta_pct = 0.0
    if target_price:
        price_delta_pct = max(0.0, (1 - (alt_price / target_price)) * 100)

    similarity = float(top_alternative.get("similarity_score", 0) or 0)
    affordability = float(top_alternative.get("affordability_score", 0) or 0)
    alt_score = float(top_alternative.get("alternative_score", 0) or 0)

    st.markdown("#### Featured replacement")
    st.markdown(
        f"""
        <div class="workflow-panel workflow-panel--featured featured-card">
            <div class="featured-kicker">Best cheaper alternative</div>
            <div class="featured-name">{html.escape(str(top_alternative['name']))}</div>
            <div class="featured-metric-row">
                <span class="featured-pill featured-pill--strong">Alternative Score {alt_score:.2f}</span>
                <span class="featured-pill featured-pill--price">{price_delta_pct:.0f}% cheaper than target</span>
                <span class="featured-pill featured-pill--fit">Similarity {similarity:.2f}</span>
            </div>
            <div class="workflow-copy">This is the strongest cheaper replacement under the current scenario, balancing fit, lower cost, and overall decision value.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similarity_table(alternatives: pd.DataFrame, window_label: str) -> None:
    st.markdown("#### Alternatives comparison")
    display_df = alternatives.head(10).copy().reset_index(drop=True)
    display_df.insert(0, "Rank", display_df.index + 1)

    columns = [
        "Rank",
        "name",
        "position",
        "club_name",
        "market_value_eur",
        "similarity_score",
        "alternative_score",
        "affordability_score",
        "age",
        "smart_value_index",
        "recent_minutes",
    ]
    columns = [column for column in columns if column in display_df.columns]

    st.dataframe(
        display_df[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "name": st.column_config.TextColumn("Player"),
            "position": st.column_config.TextColumn("Position"),
            "club_name": st.column_config.TextColumn("Current club"),
            "market_value_eur": st.column_config.NumberColumn("Market value (€)", format="€%,d"),
            "similarity_score": st.column_config.NumberColumn("Similarity", format="%.2f"),
            "alternative_score": st.column_config.NumberColumn("Alternative Score", format="%.2f"),
            "affordability_score": st.column_config.NumberColumn("Affordability", format="%.2f"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
            "smart_value_index": st.column_config.NumberColumn("DSS Score", format="%.2f"),
            "recent_minutes": st.column_config.NumberColumn(f"Recent playing time ({window_label})", format="%d"),
        },
    )


def render_similar_alternatives(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No player data available for similarity recommendation.")
        return

    evidence_df, evidence_window, reliability_level, age_range = render_similarity_scenario_block(df)
    target_player, same_position, max_value_ratio, min_similarity = render_similarity_target_block(evidence_df)
    window_label = get_evidence_window_label(evidence_window)
    threshold = get_reliability_threshold(evidence_window, reliability_level)

    render_similarity_methodology(window_label, reliability_level)

    if target_player is None:
        render_similarity_empty_state()
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

    render_similarity_target_profile(target, window_label)

    if alternatives.empty:
        render_similarity_no_results(window_label, reliability_level, threshold)
        return

    top_alternative = alternatives.iloc[0]
    render_similarity_featured_card(target, top_alternative)
    render_similarity_table(alternatives, window_label)


st.set_page_config(page_title="Recruitment Decision Support System", layout="wide")
render_app_styles()

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
    ["Potential Shortlist", "Cost-Efficient Alternatives"],
    label_visibility="collapsed",
)

if navigation == "Potential Shortlist":
    render_page_header(
        "Recruitment intelligence",
        "Potential Shortlist",
        "",
    )

    evidence_df = prepare_evidence_view(df, DEFAULT_EVIDENCE_WINDOW)
    budget_defaults = build_budget_defaults(evidence_df)

    st.session_state.setdefault("shortlist_evidence_window", DEFAULT_EVIDENCE_WINDOW)
    st.session_state.setdefault("shortlist_reliability_level", DEFAULT_RELIABILITY_LEVEL)
    st.session_state.setdefault("shortlist_budget_m", budget_defaults["budget_default_m"])
    st.session_state.setdefault("shortlist_age_range", (18, 30))
    st.session_state.setdefault("shortlist_positions", [])

    with st.container(border=True):
        st.markdown("#### Scenario controls")
        st.caption("Adjust the shortlist definition here. The recommendation cards, table, and chart all update from this active scenario.")

        evidence_col, reliability_col = st.columns(2)
        with evidence_col:
            evidence_window = st.radio(
                "Evidence period",
                options=get_evidence_window_options(),
                index=get_evidence_window_options().index(st.session_state["shortlist_evidence_window"]),
                format_func=get_evidence_window_label,
                key="shortlist_evidence_window",
                horizontal=True,
            )
        with reliability_col:
            reliability_level = st.radio(
                "Minimum reliability",
                options=get_reliability_level_options(),
                index=get_reliability_level_options().index(st.session_state["shortlist_reliability_level"]),
                key="shortlist_reliability_level",
                horizontal=True,
            )

        evidence_df = prepare_evidence_view(df, evidence_window)
        budget_defaults = build_budget_defaults(evidence_df)
        if st.session_state["shortlist_budget_m"] < budget_defaults["budget_min_m"]:
            st.session_state["shortlist_budget_m"] = budget_defaults["budget_min_m"]
        if st.session_state["shortlist_budget_m"] > budget_defaults["budget_max_m"]:
            st.session_state["shortlist_budget_m"] = budget_defaults["budget_max_m"]

        budget_col, age_col = st.columns(2)
        with budget_col:
            max_budget = st.slider(
                "Maximum budget (€M)",
                min_value=budget_defaults["budget_min_m"],
                max_value=budget_defaults["budget_max_m"],
                value=st.session_state["shortlist_budget_m"],
                key="shortlist_budget_m",
                help="Caps the shortlist to financially realistic targets.",
            )
        with age_col:
            age_range = st.slider(
                "Age range",
                min_value=16,
                max_value=40,
                value=st.session_state["shortlist_age_range"],
                step=1,
                key="shortlist_age_range",
            )

        positions = st.multiselect(
            "Position",
            options=sorted(evidence_df["position"].dropna().unique()),
            key="shortlist_positions",
            help="Leave empty to compare all outfield positions in the filtered pool.",
        )

    window_label = get_evidence_window_label(evidence_window)
    min_recent_minutes = get_reliability_threshold(evidence_window, reliability_level)

    render_scenario_summary(
        [
            f"<strong>Evidence period:</strong> {html.escape(window_label)}",
            f"<strong>Reliability floor:</strong> {html.escape(reliability_level.title())} ({min_recent_minutes:,}+ mins)",
            f"<strong>Budget cap:</strong> {html.escape(format_eur_millions(max_budget * 1_000_000))}",
            f"<strong>Age range:</strong> {age_range[0]}–{age_range[1]}",
            f"<strong>Positions:</strong> {html.escape(', '.join(positions) if positions else 'All positions')}",
        ]
    )

    filtered_df = apply_filters(evidence_df, max_budget, min_recent_minutes, positions)
    filtered_df = filtered_df[filtered_df["age"].between(age_range[0], age_range[1])].copy()
    filtered_df = filtered_df.sort_values(by=["smart_value_index", "value_score"], ascending=False)
    shortlist_df = filtered_df.head(SHORTLIST_SIZE).copy()

    render_summary(filtered_df, len(evidence_df))
    render_methodology(window_label, reliability_level, min_recent_minutes)

    if filtered_df.empty:
        render_no_shortlist_matches(max_budget, reliability_level, positions)
        st.stop()

    selected_player = render_recommendation_cards(shortlist_df)
    render_shortlist(filtered_df, evidence_window, selected_player)
    render_visual_analysis(filtered_df, shortlist_df, selected_player)

else:
    render_page_header(
        "Recruitment intelligence",
        "Cost-Efficient Alternatives",
        ".",
    )

    render_similar_alternatives(df)
