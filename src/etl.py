import os
import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

csv_env = os.getenv("CSV_DIR", "data")
CSV_DIR = (ROOT_DIR / csv_env).resolve()

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)

CHUNK_SIZE = 200_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def read_csv(filename: str, **kwargs) -> pd.DataFrame:
    path = CSV_DIR / filename
    log.info(f"Reading {path} …")
    df = pd.read_csv(path, low_memory=False, **kwargs)
    log.info(f"  → {len(df):,} rows, {df.shape[1]} columns")
    return df


def upsert(df: pd.DataFrame, table: str, engine, conflict_cols: list[str]) -> None:
    """
    Simple upsert via INSERT … ON CONFLICT DO NOTHING.
    For Postgres ≥ 9.5.
    """
    df = df.where(df.notna(), other=None)
    schema, tbl = table.split(".")
    log.info(f"Loading {len(df):,} rows → {table}")
    cols = ", ".join(df.columns)
    placeholders = ", ".join([f":{c}" for c in df.columns])
    conflict = ", ".join(conflict_cols)

    stmt = text(
        f"""
        INSERT INTO {table} ({cols})
        VALUES ({placeholders})
        ON CONFLICT ({conflict}) DO NOTHING
        """
    )

    with engine.begin() as conn:
        for start in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[start : start + CHUNK_SIZE]
            conn.execute(stmt, chunk.to_dict(orient="records"))

    log.info(f"  ✓ {table} loaded")


def init_db_schema(engine) -> None:
    """Create warehouse schema and target tables if they do not exist."""
    ddl_statements = [
        "CREATE SCHEMA IF NOT EXISTS warehouse",
        """
        CREATE TABLE IF NOT EXISTS warehouse.dim_date (
            date_id DATE PRIMARY KEY,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            week INTEGER,
            quarter INTEGER,
            day_of_week INTEGER
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS warehouse.dim_competitions (
            competition_id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS warehouse.dim_clubs (
            club_id BIGINT PRIMARY KEY,
            club_name TEXT,
            country TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS warehouse.dim_players (
            player_id BIGINT PRIMARY KEY,
            name TEXT,
            birth_date DATE,
            position TEXT,
            nationality TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS warehouse.fact_matches (
            match_id BIGINT PRIMARY KEY,
            competition_id INTEGER NOT NULL,
            date_id DATE NOT NULL,
            season INTEGER,
            home_club_id BIGINT,
            away_club_id BIGINT,
            home_score INTEGER,
            away_score INTEGER,
            FOREIGN KEY (competition_id) REFERENCES warehouse.dim_competitions (competition_id),
            FOREIGN KEY (date_id) REFERENCES warehouse.dim_date (date_id),
            FOREIGN KEY (home_club_id) REFERENCES warehouse.dim_clubs (club_id),
            FOREIGN KEY (away_club_id) REFERENCES warehouse.dim_clubs (club_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS warehouse.fact_player_performance (
            player_id BIGINT NOT NULL,
            match_id BIGINT NOT NULL,
            date_id DATE NOT NULL,
            minutes_played INTEGER,
            goals INTEGER,
            assists INTEGER,
            yellow_cards INTEGER,
            red_cards INTEGER,
            PRIMARY KEY (player_id, match_id),
            FOREIGN KEY (player_id) REFERENCES warehouse.dim_players (player_id),
            FOREIGN KEY (match_id) REFERENCES warehouse.fact_matches (match_id),
            FOREIGN KEY (date_id) REFERENCES warehouse.dim_date (date_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS warehouse.fact_player_valuations (
            player_id BIGINT NOT NULL,
            date_id DATE NOT NULL,
            market_value_eur NUMERIC,
            club_id BIGINT,
            PRIMARY KEY (player_id, date_id),
            FOREIGN KEY (player_id) REFERENCES warehouse.dim_players (player_id),
            FOREIGN KEY (date_id) REFERENCES warehouse.dim_date (date_id),
            FOREIGN KEY (club_id) REFERENCES warehouse.dim_clubs (club_id)
        )
        """,
    ]

    with engine.begin() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))
        conn.execute(text("ALTER TABLE warehouse.fact_matches ADD COLUMN IF NOT EXISTS season INTEGER"))

    log.info("Warehouse schema and tables are ready")

def extract() -> dict[str, pd.DataFrame]:
    return {
        "players":      read_csv("players.csv"),
        "valuations":   read_csv("player_valuations.csv"),
        "appearances":  read_csv("appearances.csv"),
        "games":        read_csv("games.csv"),
        "clubs":        read_csv("clubs.csv"),
        "competitions": read_csv("competitions.csv"),
    }


def get_extra_clubs(games_raw: pd.DataFrame, existing_club_ids: set) -> pd.DataFrame:
    """Build stub rows for clubs that appear in games but not in clubs.csv."""
    home = games_raw[["home_club_id", "home_club_name"]].rename(
        columns={"home_club_id": "club_id", "home_club_name": "club_name"})
    away = games_raw[["away_club_id", "away_club_name"]].rename(
        columns={"away_club_id": "club_id", "away_club_name": "club_name"})

    extra = pd.concat([home, away], ignore_index=True)
    extra["club_id"] = pd.to_numeric(extra["club_id"], errors="coerce")
    extra = extra.dropna(subset=["club_id"]).drop_duplicates("club_id")
    extra["country"] = None  
    return extra[~extra["club_id"].isin(existing_club_ids)]

def get_extra_players(appearances_raw: pd.DataFrame, existing_player_ids: set) -> pd.DataFrame:
    """Build stub rows for players in appearances but missing from players.csv."""
    df = appearances_raw[["player_id", "player_name"]].copy()
    df.rename(columns={"player_name": "name"}, inplace=True)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df = df.dropna(subset=["player_id"]).drop_duplicates("player_id")

    df["birth_date"]   = None
    df["position"]     = None
    df["nationality"]  = None

    return df[~df["player_id"].isin(existing_player_ids)]

def build_dim_date(date_series_list: list[pd.Series]) -> pd.DataFrame:
    """
    Collect every unique date across all fact tables and build dim_date.
    """
    all_dates = pd.concat(date_series_list, ignore_index=True)
    all_dates = pd.to_datetime(all_dates, errors="coerce").dropna().unique()
    dt = pd.DatetimeIndex(all_dates)
    df = pd.DataFrame(
        {
            "date_id":     pd.to_datetime(all_dates).date,
            "year":        dt.year,
            "month":       dt.month,
            "day":         dt.day,
            "week":        dt.isocalendar().week.values,
            "quarter":     dt.quarter,
            "day_of_week": dt.dayofweek,  
        }
    )
    return df.drop_duplicates("date_id").sort_values("date_id").reset_index(drop=True)

def build_competition_id_map(competitions_raw: pd.DataFrame) -> dict:
    """Maps string competition_id codes → stable integers."""
    codes = competitions_raw["competition_id"].unique()
    return {code: idx + 1 for idx, code in enumerate(sorted(codes))}

def transform_dim_competitions(competitions_raw: pd.DataFrame, id_map: dict) -> pd.DataFrame:
    df = competitions_raw[["competition_id", "name", "country_name"]].copy()
    df.rename(columns={"country_name": "country"}, inplace=True)
    df["competition_id"] = df["competition_id"].map(id_map)
    return df.dropna(subset=["competition_id"]).drop_duplicates("competition_id")


def transform_dim_clubs(clubs_raw: pd.DataFrame, competitions_raw: pd.DataFrame) -> pd.DataFrame:
    comp_country = competitions_raw[["competition_id", "country_name"]].drop_duplicates()
    df = clubs_raw[["club_id", "name", "domestic_competition_id"]].copy()
    df = df.merge(comp_country, left_on="domestic_competition_id",
                  right_on="competition_id", how="left")
    df.rename(columns={"name": "club_name", "country_name": "country"}, inplace=True)
    df["club_id"] = pd.to_numeric(df["club_id"], errors="coerce")
    return (
        df[["club_id", "club_name", "country"]]
        .dropna(subset=["club_id"])
        .drop_duplicates("club_id")
    )


def transform_dim_players(players_raw: pd.DataFrame) -> pd.DataFrame:
    df = players_raw[["player_id", "name", "date_of_birth", "position",
                       "country_of_citizenship"]].copy()
    df.rename(
        columns={
            "date_of_birth":          "birth_date",
            "country_of_citizenship": "nationality",
        },
        inplace=True,
    )
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce").dt.date
    df["birth_date"] = df["birth_date"].where(df["birth_date"].notna(), other=None)  
    df["player_id"]  = pd.to_numeric(df["player_id"], errors="coerce")
    return df.dropna(subset=["player_id"]).drop_duplicates("player_id")


def transform_fact_matches(games_raw: pd.DataFrame, id_map: dict, valid_club_ids: set) -> pd.DataFrame:
    df = games_raw[
        ["game_id", "competition_id", "date", "season", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]
    ].copy()
    df.rename(
        columns={
            "game_id": "match_id",
            "date": "date_id",
            "home_club_goals": "home_score",
            "away_club_goals": "away_score",
        },
        inplace=True,
    )
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["competition_id"] = df["competition_id"].map(id_map)
    df["date_id"] = pd.to_datetime(df["date_id"], errors="coerce").dt.date
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df["home_club_id"] = pd.to_numeric(df["home_club_id"], errors="coerce")
    df["away_club_id"] = pd.to_numeric(df["away_club_id"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)
    df = df[df["home_club_id"].isin(valid_club_ids) & df["away_club_id"].isin(valid_club_ids)]

    return df.dropna(subset=["match_id", "date_id", "competition_id"]).drop_duplicates("match_id")

def transform_fact_player_performance(appearances_raw: pd.DataFrame, valid_player_ids, valid_match_ids) -> pd.DataFrame:
    df = appearances_raw[
        ["player_id", "game_id", "date",
         "minutes_played", "goals", "assists",
         "yellow_cards", "red_cards"]
    ].copy()
    df.rename(columns={"game_id": "match_id", "date": "date_id"}, inplace=True)
    df["date_id"]        = pd.to_datetime(df["date_id"], errors="coerce").dt.date
    df["player_id"]      = pd.to_numeric(df["player_id"],      errors="coerce")
    df["match_id"]       = pd.to_numeric(df["match_id"],       errors="coerce")
    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0).astype(int)
    df["goals"]          = pd.to_numeric(df["goals"],          errors="coerce").fillna(0).astype(int)
    df["assists"]        = pd.to_numeric(df["assists"],        errors="coerce").fillna(0).astype(int)
    df["yellow_cards"]   = pd.to_numeric(df["yellow_cards"],   errors="coerce").fillna(0).astype(int)
    df["red_cards"]      = pd.to_numeric(df["red_cards"],      errors="coerce").fillna(0).astype(int)

    df = df[df["player_id"].isin(valid_player_ids) & df["match_id"].isin(valid_match_ids)]

    return df.dropna(subset=["player_id", "match_id", "date_id"]).drop_duplicates(["player_id", "match_id"])


def transform_fact_player_valuations(valuations_raw: pd.DataFrame, valid_player_ids: set, valid_club_ids: set) -> pd.DataFrame:
    df = valuations_raw[
        ["player_id", "date", "market_value_in_eur", "current_club_id"]
    ].copy()
    df.rename(
        columns={
            "date":               "date_id",
            "market_value_in_eur": "market_value_eur",
            "current_club_id":    "club_id",
        },
        inplace=True,
    )
    df["date_id"]          = pd.to_datetime(df["date_id"], errors="coerce").dt.date
    df["player_id"]        = pd.to_numeric(df["player_id"],        errors="coerce")
    df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")
    df["club_id"]          = pd.to_numeric(df["club_id"],          errors="coerce")
    df = df[df["player_id"].isin(valid_player_ids)]
    df = df[df["club_id"].isin(valid_club_ids) | df["club_id"].isna()]

    return df.dropna(subset=["player_id", "date_id"]).drop_duplicates(["player_id", "date_id"])

def run_etl() -> None:
    engine = create_engine(DB_DSN, future=True)
    log.info(f"Connected to database: {engine.url}")

    init_db_schema(engine)

    raw = extract()

    log.info("Transforming data …")

    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE warehouse.fact_player_valuations, warehouse.fact_player_performance, warehouse.fact_matches CASCADE"
            )
        )

    comp_id_map = build_competition_id_map(raw["competitions"])

    dim_competitions = transform_dim_competitions(raw["competitions"],comp_id_map)

    dim_clubs        = transform_dim_clubs(raw["clubs"], raw["competitions"])
    extra_clubs = get_extra_clubs(raw["games"], set(dim_clubs["club_id"].dropna().astype(int)))
    dim_clubs_full = pd.concat([dim_clubs, extra_clubs], ignore_index=True)

    dim_players      = transform_dim_players(raw["players"])
    extra_players = get_extra_players(raw["appearances"], set(dim_players["player_id"].dropna().astype(int)))
    dim_players_full = pd.concat([dim_players, extra_players], ignore_index=True)

    fact_matches  = transform_fact_matches(raw["games"], comp_id_map, set(dim_clubs_full["club_id"].dropna().astype(int)))
    valid_players = set(dim_players_full["player_id"].dropna().astype(int))
    valid_matches = set(fact_matches["match_id"].dropna().astype(int))

    fact_perf = transform_fact_player_performance(raw["appearances"], valid_players, valid_matches)
    valid_clubs_for_vals = set(dim_clubs_full["club_id"].dropna().astype(int))
    fact_vals = transform_fact_player_valuations(raw["valuations"], valid_players, valid_clubs_for_vals)

    dim_date = build_dim_date(
        [
            pd.to_datetime(raw["games"]["date"],       errors="coerce"),
            pd.to_datetime(raw["appearances"]["date"], errors="coerce"),
            pd.to_datetime(raw["valuations"]["date"],  errors="coerce"),
        ]
    )

    log.info("Loading dimensions …")
    upsert(dim_date,         "warehouse.dim_date",         engine, ["date_id"])
    upsert(dim_competitions, "warehouse.dim_competitions", engine, ["competition_id"])
    upsert(dim_clubs_full,   "warehouse.dim_clubs",        engine, ["club_id"])
    upsert(dim_players_full, "warehouse.dim_players",      engine, ["player_id"])

    log.info("Loading fact tables …")
    upsert(fact_matches, "warehouse.fact_matches",             engine, ["match_id"])
    upsert(fact_perf,    "warehouse.fact_player_performance",  engine, ["player_id", "match_id"])
    upsert(fact_vals,    "warehouse.fact_player_valuations",   engine, ["player_id", "date_id"])

    log.info("✅  ETL pipeline completed successfully.")


if __name__ == "__main__":
    run_etl()