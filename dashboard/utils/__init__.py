"""Dashboard utilities."""

from dashboard.utils.database import (
    init_database,
    get_db_connection,
    record_prediction,
    record_outcome,
    get_recent_predictions,
    get_performance_stats,
)

__all__ = [
    "init_database",
    "get_db_connection",
    "record_prediction",
    "record_outcome",
    "get_recent_predictions",
    "get_performance_stats",
]
