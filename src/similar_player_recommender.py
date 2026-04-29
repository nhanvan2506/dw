import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_FEATURES = [
    "age",
    "appearances_count",
    "total_minutes",
    "minutes_per_appearance",
    "total_goals",
    "total_assists",
    "goal_contributions",
    "attacking_contribution_per_90",
    "discipline_risk_per_90",
    "production_score",
    "reliability_score",
    "discipline_score",
    "smart_value_index",
]

CATEGORICAL_FEATURES = [
    "position",
]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_player_pool() -> pd.DataFrame:
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

        pr.market_value_eur,
        pr.production_score,
        pr.value_score,
        pr.reliability_score,
        pr.discipline_score,
        pr.smart_value_index

    FROM mart.player_ranking pr
    LEFT JOIN warehouse.dim_players dp
        ON pr.player_id = dp.player_id
    LEFT JOIN latest_club lc
        ON pr.player_id = lc.player_id
    LEFT JOIN warehouse.dim_clubs dc
        ON lc.club_id = dc.club_id
    WHERE pr.market_value_eur IS NOT NULL
      AND pr.market_value_eur > 0
      AND pr.position IS NOT NULL
      AND pr.position <> 'Goalkeeper'
      AND pr.total_minutes IS NOT NULL;
    """

    df = pd.read_sql(query, engine)

    for col in NUMERIC_FEATURES + ["market_value_eur"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["position"] = df["position"].fillna("Unknown").astype(str)
    df["club_name"] = df["club_name"].fillna("Unknown")
    df["club_country"] = df["club_country"].fillna("Unknown")
    df["nationality"] = df["nationality"].fillna("Unknown")

    df = df.dropna(subset=["age", "market_value_eur", "smart_value_index"])
    df = df[df["age"].between(16, 40)]

    return df.reset_index(drop=True)


def find_target_player(df: pd.DataFrame, player_name: str) -> pd.Series:
    exact = df[df["name"].str.lower() == player_name.lower()]

    if len(exact) == 1:
        return exact.iloc[0]

    contains = df[df["name"].str.lower().str.contains(player_name.lower(), regex=False)]

    if contains.empty:
        raise ValueError(f"No player found matching: {player_name}")

    if len(contains) > 1:
        print("\nMultiple players matched. Using the first one:")
        print(contains[["name", "position", "age", "market_value_eur"]].head(10).to_string(index=False))

    return contains.iloc[0]


def build_similarity_model(df: pd.DataFrame):
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[feature_cols].copy()

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    knn = NearestNeighbors(
        n_neighbors=min(200, len(df)),
        metric="cosine",
        algorithm="brute",
    )

    knn.fit(X_processed)

    return preprocessor, knn, X_processed


def recommend_similar_cheaper_players(
    player_name: str,
    top_n: int = 10,
    max_value_ratio: float = 0.75,
    max_budget: float | None = None,
    min_minutes: int = 900,
    min_age: int = 18,
    max_age: int = 30,
    min_similarity: float = 70.0,
    same_position: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    df = load_player_pool()
    target = find_target_player(df, player_name)

    preprocessor, knn, X_processed = build_similarity_model(df)

    target_idx = int(target.name)

    # Keep target player as a 2D matrix: shape = (1, n_features)
    target_vector = X_processed[target_idx : target_idx + 1]

    distances, indices = knn.kneighbors(
        target_vector,
        n_neighbors=min(200, len(df))
)

    result = df.iloc[indices[0]].copy()
    result["cosine_distance"] = distances[0]
    result["similarity_score"] = ((1 - result["cosine_distance"]) * 100).clip(0, 100)

    result = result[result["player_id"] != target["player_id"]].copy()

    if same_position:
        result = result[result["position"] == target["position"]].copy()

    target_value = float(target["market_value_eur"])

    # Cheaper alternative condition
    result = result[result["market_value_eur"] < target_value].copy()

    # Stronger affordability condition
    result = result[result["market_value_eur"] <= target_value * max_value_ratio].copy()

    if max_budget is not None:
        result = result[result["market_value_eur"] <= max_budget].copy()

    result = result[result["total_minutes"] >= min_minutes].copy()

    result = result[
        result["age"].between(min_age, max_age)
    ].copy()

    result = result[
        result["similarity_score"] >= min_similarity
    ].copy()

    if result.empty:
        return target, result

    result["affordability_score"] = (
        100 * (1 - result["market_value_eur"] / target_value)
    ).clip(0, 100)

    result["alternative_score"] = (
        0.55 * result["similarity_score"]
        + 0.25 * result["affordability_score"]
        + 0.20 * result["smart_value_index"]
    ).round(2)

    result = result.sort_values(
        by=["alternative_score", "similarity_score", "market_value_eur", "smart_value_index"],
        ascending=[False, False, True, False],
    )

    return target, result.head(top_n)


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "name",
        "position",
        "age",
        "club_name",
        "club_country",
        "market_value_eur",
        "total_minutes",
        "goal_contributions",
        "attacking_contribution_per_90",
        "similarity_score",
        "affordability_score",
        "smart_value_index",
        "alternative_score",
    ]

    display_cols = [c for c in display_cols if c in df.columns]
    out = df[display_cols].copy()

    for col in ["age", "similarity_score", "affordability_score", "smart_value_index", "alternative_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    if "market_value_eur" in out.columns:
        out["market_value_eur"] = out["market_value_eur"].apply(lambda x: f"€{x / 1_000_000:.2f}M")

    if "attacking_contribution_per_90" in out.columns:
        out["attacking_contribution_per_90"] = pd.to_numeric(
            out["attacking_contribution_per_90"], errors="coerce"
        ).round(3)

    return out


def main():
    parser = argparse.ArgumentParser(description="Find cheaper similar alternatives for a target football player.")
    parser.add_argument("--player", required=True, help="Target player name")
    parser.add_argument("--top", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--max-value-ratio", type=float, default=0.75, help="Candidate max value as ratio of target value")
    parser.add_argument("--max-budget", type=float, default=None, help="Optional max budget in EUR")
    parser.add_argument("--min-minutes", type=int, default=900, help="Minimum total minutes")
    parser.add_argument("--allow-different-position", action="store_true", help="Allow recommendations from different positions")

    args = parser.parse_args()

    target, recommendations = recommend_similar_cheaper_players(
        player_name=args.player,
        top_n=args.top,
        max_value_ratio=args.max_value_ratio,
        max_budget=args.max_budget,
        min_minutes=args.min_minutes,
        same_position=not args.allow_different_position,
    )

    print("\nTARGET PLAYER")
    print(
        pd.DataFrame(
            [
                {
                    "name": target["name"],
                    "position": target["position"],
                    "age": round(float(target["age"]), 2),
                    "market_value_eur": f"€{float(target['market_value_eur']) / 1_000_000:.2f}M",
                    "club_name": target.get("club_name", "Unknown"),
                    "club_country": target.get("club_country", "Unknown"),
                    "smart_value_index": round(float(target["smart_value_index"]), 2),
                }
            ]
        ).to_string(index=False)
    )

    if recommendations.empty:
        print("\nNo cheaper similar alternatives found with the current filters.")
        print("Try increasing --max-value-ratio, lowering --min-minutes, or using --allow-different-position.")
        return

    print("\nCHEAPER SIMILAR ALTERNATIVES")
    print(format_output(recommendations).to_string(index=False))

    output_path = OUTPUT_DIR / f"similar_alternatives_{slugify(str(target['name']))}.csv"
    recommendations.to_csv(output_path, index=False)

    print(f"\n[OK] Saved recommendations to: {output_path}")


if __name__ == "__main__":
    main()