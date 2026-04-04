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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

def build_data_mart():
    engine = create_engine(DB_DSN)
    log.info("Connected to database. Building Data Mart...")

    mart_sql = """
        CREATE SCHEMA IF NOT EXISTS mart;

        -- STEP 1: Create player_features (Feature Engineering)
        DROP TABLE IF EXISTS mart.player_features;
        CREATE TABLE mart.player_features AS
        SELECT 
            p.player_id,
            p.name,
            p.position,
            SUM(fpp.minutes_played) AS total_minutes,
            -- Performance Features
            CAST(SUM(fpp.goals) + SUM(fpp.assists) AS FLOAT) AS goal_contributions,
            -- Value Features
            MAX(fpv.market_value_eur) AS market_value_eur,
            -- Derived Feature: Contributions per Million Euro
            ROUND(CAST(SUM(fpp.goals) + SUM(fpp.assists) AS NUMERIC) / 
                NULLIF(MAX(fpv.market_value_eur) / 1000000.0, 0), 2) AS value_efficiency
        FROM warehouse.dim_players p
        JOIN warehouse.fact_player_performance fpp ON p.player_id = fpp.player_id
        JOIN warehouse.fact_player_valuations fpv ON p.player_id = fpv.player_id
        GROUP BY p.player_id, p.name, p.position
        HAVING SUM(fpp.minutes_played) > 500;

        -- STEP 2: Create player_ranking (The Scoring Model)
        DROP TABLE IF EXISTS mart.player_ranking;
        CREATE TABLE mart.player_ranking AS
        SELECT 
            *,
            -- Weighted Scoring Model: 70% Efficiency, 30% Raw Output
            ROUND(((value_efficiency * 0.7) + (goal_contributions * 0.3))::NUMERIC, 2) AS final_dss_score
        FROM mart.player_features
        ORDER BY final_dss_score DESC;
        """

    with engine.begin() as conn:
        conn.execute(text(mart_sql))

    log.info("✅ Data Mart built successfully: mart.player_features and mart.player_ranking created.")

if __name__ == "__main__":
    build_data_mart()