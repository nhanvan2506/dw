# Football DW Project Run Guide

## What this project does
This project loads football CSV data into PostgreSQL (warehouse), builds mart tables for analysis, and serves a Streamlit app for recruitment decision support.

## Prerequisites
- Python 3.12
- Docker and Docker Compose
- Git

## 1) Clone and enter project
- git clone <your-repo-url>
- cd dw

## 2) Create and activate virtual environment
- python3 -m venv .venv
- source .venv/bin/activate

## 3) Install Python dependencies
- pip install -r requirements.txt

## 4) Configure environment variables
Create .env (or copy from .env.example) with:
- DB_USER
- DB_PASSWORD
- DB_NAME
- DB_HOST (usually localhost)
- DB_PORT (usually 5432)
- CSV_DIR (usually data)

## 5) Start PostgreSQL
- docker compose up -d

Optional check:
- docker ps

## 6) Build warehouse tables from CSV
- python3 src/etl.py

## 7) Build mart tables
- python3 src/build_mart.py

This creates:
- mart.player_features
- mart.player_ranking

### Refined mart output

`mart.player_features` now keeps one analytic row per player and exposes recruitment-oriented fields such as:
- appearances_count
- total_minutes
- minutes_per_appearance
- total_goals
- total_assists
- goal_contributions
- attacking_contribution_per_90
- recent minutes and appearances for the last 1 / 3 / 5 seasons
- yellow_cards / red_cards
- discipline_risk_per_90
- market_value_eur (latest known valuation)
- peak_market_value_eur
- latest_value_date
- value_efficiency_index
- value_retention_ratio

`mart.player_ranking` builds on those features and keeps the score transparent with:
- window-specific reliability scores for the last 1 / 3 / 5 seasons
- window-specific Smart Value Index columns
- production_score
- value_score
- discipline_score
- final_dss_score
- smart_value_index

### Ranking logic

The refined ranking keeps the model simple and explainable for the course project:
- only players with enough sample size are ranked (`appearances_count >= 10`, `total_minutes >= 900`)
- only players with recent market valuations are ranked
- goalkeepers are excluded from this ranking because the current model is built around outfield attacking/value metrics
- the final score combines production, value, reliability, and discipline into a weighted DSS score
- the dashboard can switch between exact-season evidence windows without recomputing the mart in Streamlit

### Validation

When `python3 src/build_mart.py` runs successfully, it now also validates that:
- `mart.player_features` is populated
- `mart.player_ranking` is populated
- ranked `player_id` values are unique
- key scoring fields are not null

## 8) Run Streamlit app
- streamlit run src/app.py

App URL:
- http://localhost:8501

## Dashboard flow
The Streamlit dashboard is organized to support a simple recruitment decision flow:
- **Potential Shortlist** — the main page starts with editable scenario controls in the page body, followed by KPI summaries, a methodology explainer, recommendation cards, a ranked shortlist table, and a value-opportunity chart
- **Scenario controls** — evidence period, reliability, budget, age range, and position are configured directly on the shortlist page so the active recruitment scenario is always visible near the decision output
- **Recommendation summary** — three cards highlight the top overall, best value, and most reliable shortlist candidates and can focus the corresponding player in the evidence views
- **Priority targets table** — the default table keeps the comparison shortlist-first with rank, player identity, market value, DSS, value, and reliability prioritized before secondary metrics
- **Value opportunity map** — the chart shows the wider filtered market in the background and emphasizes the shortlisted players so the score-versus-price tradeoff is easier to explain in demos
- **Cost-Efficient Alternatives** — the second workflow now starts with scenario context, then target selection and replacement constraints, followed by a target summary, one featured cheaper replacement, and a ranked comparison table focused on substitute comparisons

This keeps the app easy to demo while still showing clear decision-support logic from the mart. Both pages now use a neutral financial-style theme and a stronger information hierarchy so the DSS logic is easier to explain to a course audience without relying on club-themed styling.

## Daily restart flow
If data/model SQL changes:
1. python3 src/etl.py (only when source data or warehouse logic changed)
2. python3 src/build_mart.py
3. Refresh Streamlit page (or rerun app)

## Quick troubleshooting
- Error: relation does not exist
  - Run python3 src/build_mart.py again.
- DB connection errors
  - Ensure docker compose up -d is running and .env values are correct.
- Empty app table
  - Check ETL completed and mart build completed without errors.
