"""
ML Options Trading Dashboard - Main Entry Point

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.utils.database import init_database

# Page config
st.set_page_config(
    page_title="ML Options Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database on startup
init_database()

# Main page content
st.title("ML Options Trading Dashboard")
st.markdown("---")

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Model Accuracy",
        value="57.1%",
        delta="+2.6%",
        help="Direction classifier accuracy on validation set"
    )

with col2:
    st.metric(
        label="Active Signals",
        value="3",
        help="Number of open positions being tracked"
    )

with col3:
    st.metric(
        label="Win Rate (30d)",
        value="--",
        delta=None,
        help="Win rate over last 30 days (needs data)"
    )

with col4:
    st.metric(
        label="Total P&L",
        value="--",
        help="Cumulative P&L from recorded signals"
    )

st.markdown("---")

# Quick navigation
st.subheader("Quick Navigation")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Daily Signals
    View today's ML predictions and recommended strategies.

    - Direction predictions (up/down/neutral)
    - IV-based strategy recommendations
    - Confidence scores
    """)

with col2:
    st.markdown("""
    ### Performance
    Track paper trading results and analyze performance.

    - Win/loss tracking
    - P&L analysis
    - Strategy breakdown
    """)

with col3:
    st.markdown("""
    ### Model Metrics
    Monitor model health and detect drift.

    - Accuracy over time
    - Feature importance
    - Drift detection (PSI)
    """)

st.markdown("---")

# System status
st.subheader("System Status")

from dashboard.utils.database import get_db_connection
import sqlite3

try:
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get prediction count
    cursor.execute("SELECT COUNT(*) FROM predictions")
    pred_count = cursor.fetchone()[0]

    # Get latest prediction date
    cursor.execute("SELECT MAX(created_at) FROM predictions")
    latest_pred = cursor.fetchone()[0] or "No predictions yet"

    conn.close()

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Database**: {pred_count} predictions recorded")
    with col2:
        st.info(f"**Latest Signal**: {latest_pred}")

except Exception as e:
    st.warning(f"Database not initialized: {e}")

# Footer
st.markdown("---")
st.caption("ML Options Trading System v0.1 | Phase 6: Paper Trading & Monitoring")
