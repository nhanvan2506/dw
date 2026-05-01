EVIDENCE_WINDOWS = {
    "last_season": {
        "label": "Last season",
        "recent_minutes_column": "recent_minutes_last_season",
        "recent_appearances_column": "recent_appearances_last_season",
        "reliability_score_column": "reliability_score_last_season",
        "smart_value_index_column": "smart_value_index_last_season",
    },
    "last_3_seasons": {
        "label": "Last 3 seasons",
        "recent_minutes_column": "recent_minutes_last_3_seasons",
        "recent_appearances_column": "recent_appearances_last_3_seasons",
        "reliability_score_column": "reliability_score_last_3_seasons",
        "smart_value_index_column": "smart_value_index_last_3_seasons",
    },
    "last_5_seasons": {
        "label": "Last 5 seasons",
        "recent_minutes_column": "recent_minutes_last_5_seasons",
        "recent_appearances_column": "recent_appearances_last_5_seasons",
        "reliability_score_column": "reliability_score_last_5_seasons",
        "smart_value_index_column": "smart_value_index_last_5_seasons",
    },
}

RELIABILITY_LEVELS = ["Low", "Medium", "High"]
DEFAULT_EVIDENCE_WINDOW = "last_3_seasons"
DEFAULT_RELIABILITY_LEVEL = "Medium"

RELIABILITY_THRESHOLDS = {
    "last_season": {
        "Low": 300,
        "Medium": 900,
        "High": 1800,
    },
    "last_3_seasons": {
        "Low": 900,
        "Medium": 1800,
        "High": 3600,
    },
    "last_5_seasons": {
        "Low": 1500,
        "Medium": 3000,
        "High": 6000,
    },
}


def get_window_config(evidence_window: str) -> dict:
    return EVIDENCE_WINDOWS[evidence_window]


def get_evidence_window_options() -> list[str]:
    return list(EVIDENCE_WINDOWS.keys())


def get_reliability_level_options() -> list[str]:
    return RELIABILITY_LEVELS[:]


def get_evidence_window_label(evidence_window: str) -> str:
    return EVIDENCE_WINDOWS[evidence_window]["label"]


def get_recent_minutes_column(evidence_window: str) -> str:
    return EVIDENCE_WINDOWS[evidence_window]["recent_minutes_column"]


def get_recent_appearances_column(evidence_window: str) -> str:
    return EVIDENCE_WINDOWS[evidence_window]["recent_appearances_column"]


def get_reliability_score_column(evidence_window: str) -> str:
    return EVIDENCE_WINDOWS[evidence_window]["reliability_score_column"]


def get_smart_value_index_column(evidence_window: str) -> str:
    return EVIDENCE_WINDOWS[evidence_window]["smart_value_index_column"]


def get_reliability_threshold(evidence_window: str, reliability_level: str) -> int:
    return RELIABILITY_THRESHOLDS[evidence_window][reliability_level]


def evidence_window_caption(evidence_window: str) -> str:
    return f"recent evidence from {get_evidence_window_label(evidence_window).lower()}"
