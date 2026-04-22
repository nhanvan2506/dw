import os
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)

MIN_RANKED_APPEARANCES = 10
MIN_RANKED_MINUTES = 900
RECENT_VALUATION_DAYS = 730
MIN_MARKET_VALUE_FLOOR = 250_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

MART_SQL = f"""
CREATE SCHEMA IF NOT EXISTS mart;

DROP TABLE IF EXISTS mart.player_ranking;
DROP TABLE IF EXISTS mart.player_features;

CREATE TABLE mart.player_features AS
WITH performance_base AS (
    SELECT
        p.player_id,
        p.name,
        p.position,
        COUNT(*) AS appearances_count,
        SUM(fpp.minutes_played) AS total_minutes,
        SUM(fpp.goals) AS total_goals,
        SUM(fpp.assists) AS total_assists,
        SUM(fpp.yellow_cards) AS yellow_cards,
        SUM(fpp.red_cards) AS red_cards
    FROM warehouse.dim_players p
    JOIN warehouse.fact_player_performance fpp ON p.player_id = fpp.player_id
    GROUP BY p.player_id, p.name, p.position
),
performance_features AS (
    SELECT
        player_id,
        name,
        position,
        appearances_count,
        total_minutes,
        total_goals,
        total_assists,
        total_goals + total_assists AS goal_contributions,
        yellow_cards,
        red_cards,
        ROUND(total_minutes::NUMERIC / NULLIF(appearances_count, 0), 1) AS minutes_per_appearance,
        ROUND(((total_goals + total_assists)::NUMERIC * 90) / NULLIF(total_minutes, 0), 3) AS attacking_contribution_per_90,
        ROUND(((yellow_cards + (red_cards * 3))::NUMERIC * 90) / NULLIF(total_minutes, 0), 3) AS discipline_risk_per_90
    FROM performance_base
),
latest_valuations AS (
    SELECT player_id, market_value_eur, date_id AS latest_value_date
    FROM (
        SELECT
            player_id,
            market_value_eur,
            date_id,
            ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY date_id DESC) AS rn
        FROM warehouse.fact_player_valuations
        WHERE market_value_eur IS NOT NULL
    ) ranked_values
    WHERE rn = 1
),
peak_valuations AS (
    SELECT
        player_id,
        MAX(market_value_eur) AS peak_market_value_eur
    FROM warehouse.fact_player_valuations
    WHERE market_value_eur IS NOT NULL
    GROUP BY player_id
)
SELECT
    pf.player_id,
    pf.name,
    pf.position,
    pf.appearances_count,
    pf.total_minutes,
    pf.minutes_per_appearance,
    pf.total_goals,
    pf.total_assists,
    pf.goal_contributions,
    pf.attacking_contribution_per_90,
    pf.yellow_cards,
    pf.red_cards,
    pf.discipline_risk_per_90,
    lv.market_value_eur,
    pv.peak_market_value_eur,
    lv.latest_value_date,
    ROUND(
        pf.attacking_contribution_per_90 /
        NULLIF(GREATEST(lv.market_value_eur, {MIN_MARKET_VALUE_FLOOR})::NUMERIC / 10000000.0, 0),
        3
    ) AS value_efficiency_index,
    ROUND(lv.market_value_eur / NULLIF(pv.peak_market_value_eur, 0), 3) AS value_retention_ratio
FROM performance_features pf
LEFT JOIN latest_valuations lv ON pf.player_id = lv.player_id
LEFT JOIN peak_valuations pv ON pf.player_id = pv.player_id;

CREATE TABLE mart.player_ranking AS
WITH recent_cutoff AS (
    SELECT MAX(date_id) - INTERVAL '{RECENT_VALUATION_DAYS} days' AS cutoff_date
    FROM warehouse.fact_player_valuations
),
eligible_players AS (
    SELECT *
    FROM mart.player_features, recent_cutoff
    WHERE appearances_count >= {MIN_RANKED_APPEARANCES}
      AND total_minutes >= {MIN_RANKED_MINUTES}
      AND market_value_eur IS NOT NULL
      AND attacking_contribution_per_90 IS NOT NULL
      AND latest_value_date >= cutoff_date
      AND position IS NOT NULL
      AND position <> 'Goalkeeper'
),
scored_players AS (
    SELECT
        player_id,
        name,
        position,
        appearances_count,
        total_minutes,
        minutes_per_appearance,
        total_goals,
        total_assists,
        goal_contributions,
        attacking_contribution_per_90,
        yellow_cards,
        red_cards,
        discipline_risk_per_90,
        market_value_eur,
        peak_market_value_eur,
        latest_value_date,
        value_efficiency_index,
        value_retention_ratio,
        ROUND((PERCENT_RANK() OVER (ORDER BY total_minutes, appearances_count) * 100)::NUMERIC, 2) AS reliability_score,
        ROUND((PERCENT_RANK() OVER (ORDER BY attacking_contribution_per_90, goal_contributions) * 100)::NUMERIC, 2) AS production_score,
        ROUND((PERCENT_RANK() OVER (ORDER BY value_efficiency_index, value_retention_ratio) * 100)::NUMERIC, 2) AS value_score,
        ROUND(((1 - PERCENT_RANK() OVER (ORDER BY discipline_risk_per_90, yellow_cards, red_cards)) * 100)::NUMERIC, 2) AS discipline_score
    FROM eligible_players
)
SELECT
    player_id,
    name,
    position,
    appearances_count,
    total_minutes,
    minutes_per_appearance,
    total_goals,
    total_assists,
    goal_contributions,
    attacking_contribution_per_90,
    yellow_cards,
    red_cards,
    discipline_risk_per_90,
    market_value_eur,
    peak_market_value_eur,
    latest_value_date,
    value_efficiency_index,
    value_retention_ratio,
    reliability_score,
    production_score,
    value_score,
    discipline_score,
    ROUND(
        (production_score * 0.40) +
        (value_score * 0.35) +
        (reliability_score * 0.20) +
        (discipline_score * 0.05),
        2
    ) AS final_dss_score,
    ROUND(
        (production_score * 0.40) +
        (value_score * 0.35) +
        (reliability_score * 0.20) +
        (discipline_score * 0.05),
        2
    ) AS smart_value_index
FROM scored_players
ORDER BY final_dss_score DESC, total_minutes DESC, goal_contributions DESC;
"""


def validate_mart(engine) -> None:
    with engine.connect() as conn:
        feature_rows = conn.execute(text("SELECT COUNT(*) FROM mart.player_features")).scalar_one()
        ranking_rows = conn.execute(text("SELECT COUNT(*) FROM mart.player_ranking")).scalar_one()
        duplicate_ranked_players = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT player_id
                    FROM mart.player_ranking
                    GROUP BY player_id
                    HAVING COUNT(*) > 1
                ) duplicates
                """
            )
        ).scalar_one()
        invalid_ranked_rows = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM mart.player_ranking
                WHERE final_dss_score IS NULL
                   OR smart_value_index IS NULL
                   OR market_value_eur IS NULL
                   OR total_minutes IS NULL
                   OR production_score IS NULL
                   OR value_score IS NULL
                   OR reliability_score IS NULL
                """
            )
        ).scalar_one()

    errors = []
    if feature_rows <= 0:
        errors.append("mart.player_features is empty")
    if ranking_rows <= 0:
        errors.append("mart.player_ranking is empty")
    if duplicate_ranked_players > 0:
        errors.append("mart.player_ranking contains duplicate player_id rows")
    if invalid_ranked_rows > 0:
        errors.append("mart.player_ranking contains invalid key numeric fields")

    if errors:
        raise RuntimeError("Mart validation failed: " + "; ".join(errors))

    log.info(
        "Mart validation passed: %s feature rows, %s ranked rows, 0 duplicate ranked players, 0 invalid ranked rows",
        feature_rows,
        ranking_rows,
    )



def build_data_mart() -> None:
    engine = create_engine(DB_DSN, future=True)
    log.info("Connected to database. Building Data Mart...")

    with engine.begin() as conn:
        conn.execute(text(MART_SQL))

    validate_mart(engine)
    log.info("✅ Data Mart built successfully: mart.player_features and mart.player_ranking created.")


if __name__ == "__main__":
    build_data_mart()
