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

## 8) Run Streamlit app
- streamlit run src/app.py

App URL:
- http://localhost:8501

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
