import os
import math
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from pathlib import Path
from build_mart import build_data_mart

# Setup environment and connection
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DB_DSN = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)

@st.cache_data
def load_data():
    engine = create_engine(DB_DSN)
    query = "SELECT * FROM mart.player_ranking"

    # If the mart table is missing, try building it once, then load again.
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

# UI Layout
st.set_page_config(page_title="FC Barcelona DSS", layout="wide")
st.title("FC Barcelona: Recruitment Decision Support System")
st.markdown("Identify undervalued talent compliant with La Liga's 1:1 rule.")

try:
    df = load_data()
except SQLAlchemyError as exc:
    st.error(
        "Unable to load mart data. Please run ETL and mart build first: "
        "`python src/etl.py` then `python src/build_mart.py`."
    )
    st.exception(exc)
    st.stop()

if df.empty:
    st.warning("No rows found in mart.player_ranking.")
    st.stop()

# Normalize numeric fields so slider filters always work as expected.
df = df.copy()

# Keep compatibility with older mart schema naming.
if "smart_value_index" not in df.columns and "final_dss_score" in df.columns:
    df["smart_value_index"] = df["final_dss_score"]

df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")
df["total_minutes"] = pd.to_numeric(df["total_minutes"], errors="coerce")
df["smart_value_index"] = pd.to_numeric(df["smart_value_index"], errors="coerce")
df = df.dropna(subset=["market_value_eur", "total_minutes", "smart_value_index"])

if df.empty:
    st.warning("No valid rows available after numeric cleanup.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Decision Criteria")

budget_min_m = max(1, int(math.floor(df["market_value_eur"].min() / 1_000_000)))
budget_max_m = max(budget_min_m + 1, int(math.ceil(df["market_value_eur"].max() / 1_000_000)))
budget_default_m = min(max(50, budget_min_m), budget_max_m)

minutes_min = int(df["total_minutes"].min())
minutes_max = int(df["total_minutes"].max())
minutes_default = max(minutes_min, min(1500, minutes_max))

max_budget = st.sidebar.slider("Max Budget (€ Millions)", budget_min_m, budget_max_m, budget_default_m)
min_minutes = st.sidebar.slider("Reliability Filter (Min. Minutes Played)", minutes_min, minutes_max, minutes_default)
positions = st.sidebar.multiselect("Position", options=df['position'].dropna().unique())

# Apply Logic
filtered_df = df[
    (df['market_value_eur'] <= max_budget * 1000000) &
    (df['total_minutes'] >= min_minutes)
]

if positions:
    filtered_df = filtered_df[filtered_df['position'].isin(positions)]

st.caption(f"Matched {len(filtered_df)} of {len(df)} players")

# Sort by Optimal Buy (Smart-Value Index)
smart_shortlist = filtered_df.sort_values(by="smart_value_index", ascending=False).head(10)

# Display Results
st.subheader("Top 10 Smart Shortlist")
if smart_shortlist.empty:
    st.info("No players match the current filter settings. Try increasing budget or lowering minutes.")
else:
    st.dataframe(
        smart_shortlist[['name', 'position', 'total_minutes', 'goal_contributions', 'market_value_eur', 'smart_value_index']],
        width='stretch',
        hide_index=True
    )