"""
SQLite database for prediction logging and performance tracking.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Database location
DB_PATH = Path(__file__).parent.parent.parent / "data" / "predictions.db"


def get_db_connection() -> sqlite3.Connection:
    """Get database connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Predictions table - stores daily ML signals
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date DATE NOT NULL,

            -- Direction model outputs
            direction_pred TEXT NOT NULL,  -- 'up', 'down', 'neutral'
            direction_prob REAL,           -- Probability of predicted direction

            -- Volatility model outputs
            volatility_pred TEXT,          -- 'high', 'medium', 'low'
            volatility_prob REAL,

            -- IV metrics at prediction time
            iv_rank REAL,
            iv_percentile REAL,
            current_iv REAL,

            -- Strategy recommendation
            recommended_strategy TEXT,     -- 'long_call', 'bull_put_spread', etc.
            strategy_confidence REAL,

            -- Price at prediction
            underlying_price REAL,

            -- Metadata
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(ticker, prediction_date)
        )
    """)

    # Outcomes table - stores actual results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,

            -- Actual direction after holding period
            actual_direction TEXT,         -- 'up', 'down', 'neutral'
            actual_return REAL,            -- % return of underlying

            -- Trade results (if trade was taken)
            trade_taken INTEGER DEFAULT 0,
            entry_price REAL,
            exit_price REAL,
            pnl_dollars REAL,
            pnl_percent REAL,

            -- Timing
            holding_days INTEGER,
            outcome_date DATE,

            -- Notes
            notes TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)

    # Model metrics table - stores daily model performance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_date DATE NOT NULL,
            ticker TEXT NOT NULL,

            -- Rolling accuracy metrics
            accuracy_7d REAL,
            accuracy_30d REAL,

            -- Drift detection
            psi_score REAL,                -- Population Stability Index
            drift_detected INTEGER DEFAULT 0,

            -- Feature importance snapshot
            top_features TEXT,             -- JSON array of top features

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(metric_date, ticker)
        )
    """)

    # Create indexes for common queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date
        ON predictions(ticker, prediction_date DESC)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_outcomes_prediction
        ON outcomes(prediction_id)
    """)

    conn.commit()
    conn.close()


def record_prediction(
    ticker: str,
    prediction_date: str,
    direction_pred: str,
    direction_prob: float,
    volatility_pred: Optional[str] = None,
    volatility_prob: Optional[float] = None,
    iv_rank: Optional[float] = None,
    iv_percentile: Optional[float] = None,
    current_iv: Optional[float] = None,
    recommended_strategy: Optional[str] = None,
    strategy_confidence: Optional[float] = None,
    underlying_price: Optional[float] = None,
    model_version: str = "v0.1",
) -> int:
    """
    Record a new prediction.

    Returns:
        prediction_id for linking outcomes
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO predictions (
            ticker, prediction_date, direction_pred, direction_prob,
            volatility_pred, volatility_prob, iv_rank, iv_percentile,
            current_iv, recommended_strategy, strategy_confidence,
            underlying_price, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker, prediction_date, direction_pred, direction_prob,
        volatility_pred, volatility_prob, iv_rank, iv_percentile,
        current_iv, recommended_strategy, strategy_confidence,
        underlying_price, model_version
    ))

    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return prediction_id


def record_outcome(
    prediction_id: int,
    actual_direction: str,
    actual_return: float,
    trade_taken: bool = False,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    pnl_dollars: Optional[float] = None,
    pnl_percent: Optional[float] = None,
    holding_days: int = 5,
    outcome_date: Optional[str] = None,
    notes: Optional[str] = None,
):
    """Record the outcome of a prediction."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO outcomes (
            prediction_id, actual_direction, actual_return,
            trade_taken, entry_price, exit_price, pnl_dollars, pnl_percent,
            holding_days, outcome_date, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_id, actual_direction, actual_return,
        1 if trade_taken else 0, entry_price, exit_price,
        pnl_dollars, pnl_percent, holding_days, outcome_date, notes
    ))

    conn.commit()
    conn.close()


def get_recent_predictions(
    ticker: Optional[str] = None,
    days: int = 30,
) -> pd.DataFrame:
    """Get recent predictions with outcomes."""
    conn = get_db_connection()

    query = """
        SELECT
            p.*,
            o.actual_direction,
            o.actual_return,
            o.trade_taken,
            o.pnl_dollars,
            o.pnl_percent,
            CASE
                WHEN o.actual_direction = p.direction_pred THEN 1
                ELSE 0
            END as correct
        FROM predictions p
        LEFT JOIN outcomes o ON p.id = o.prediction_id
        WHERE p.prediction_date >= date('now', ?)
    """
    params = [f"-{days} days"]

    if ticker:
        query += " AND p.ticker = ?"
        params.append(ticker)

    query += " ORDER BY p.prediction_date DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df


def get_performance_stats(
    ticker: Optional[str] = None,
    days: int = 30,
) -> Dict:
    """Calculate performance statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Base query for predictions with outcomes
    where_clause = "WHERE p.prediction_date >= date('now', ?)"
    params = [f"-{days} days"]

    if ticker:
        where_clause += " AND p.ticker = ?"
        params.append(ticker)

    # Total predictions
    cursor.execute(f"""
        SELECT COUNT(*) FROM predictions p {where_clause}
    """, params)
    total_predictions = cursor.fetchone()[0]

    # Predictions with outcomes
    cursor.execute(f"""
        SELECT COUNT(*)
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        {where_clause}
    """, params)
    with_outcomes = cursor.fetchone()[0]

    # Accuracy
    cursor.execute(f"""
        SELECT
            SUM(CASE WHEN o.actual_direction = p.direction_pred THEN 1 ELSE 0 END) as correct,
            COUNT(*) as total
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        {where_clause}
    """, params)
    row = cursor.fetchone()
    correct = row[0] or 0
    total = row[1] or 0
    accuracy = correct / total if total > 0 else 0

    # P&L stats
    cursor.execute(f"""
        SELECT
            SUM(o.pnl_dollars) as total_pnl,
            AVG(o.pnl_percent) as avg_return,
            SUM(CASE WHEN o.pnl_dollars > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN o.pnl_dollars < 0 THEN 1 ELSE 0 END) as losses
        FROM predictions p
        JOIN outcomes o ON p.id = o.prediction_id
        {where_clause} AND o.trade_taken = 1
    """, params)
    pnl_row = cursor.fetchone()

    conn.close()

    return {
        "total_predictions": total_predictions,
        "predictions_with_outcomes": with_outcomes,
        "accuracy": accuracy,
        "correct_predictions": correct,
        "total_with_outcomes": total,
        "total_pnl": pnl_row[0] or 0,
        "avg_return": pnl_row[1] or 0,
        "wins": pnl_row[2] or 0,
        "losses": pnl_row[3] or 0,
        "win_rate": pnl_row[2] / (pnl_row[2] + pnl_row[3]) if (pnl_row[2] or 0) + (pnl_row[3] or 0) > 0 else 0,
    }


def get_predictions_for_date(date_str: str) -> List[Dict]:
    """Get all predictions for a specific date."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        WHERE prediction_date = ?
        ORDER BY ticker
    """, (date_str,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]
