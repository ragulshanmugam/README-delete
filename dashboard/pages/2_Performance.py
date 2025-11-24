"""
Performance Page - Track paper trading results and analyze performance.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Performance", page_icon="📈", layout="wide")

st.title("Performance Tracking")
st.markdown("Track paper trading results and analyze prediction accuracy.")

# Time period selector
col1, col2 = st.columns([1, 3])
with col1:
    period = st.selectbox("Time Period", ["7 days", "30 days", "90 days", "All time"])
    days_map = {"7 days": 7, "30 days": 30, "90 days": 90, "All time": 9999}
    days = days_map[period]

st.markdown("---")

# Performance metrics
try:
    from dashboard.utils.database import get_performance_stats, get_recent_predictions

    stats = get_performance_stats(days=days)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        accuracy_pct = stats["accuracy"] * 100
        st.metric(
            label="Prediction Accuracy",
            value=f"{accuracy_pct:.1f}%",
            delta=f"{accuracy_pct - 50:.1f}% vs random" if accuracy_pct > 0 else None,
            help="Percentage of correct direction predictions"
        )

    with col2:
        st.metric(
            label="Win Rate",
            value=f"{stats['win_rate'] * 100:.1f}%" if stats["win_rate"] > 0 else "--",
            help="Percentage of profitable trades"
        )

    with col3:
        st.metric(
            label="Total P&L",
            value=f"${stats['total_pnl']:,.2f}" if stats["total_pnl"] != 0 else "--",
            delta=None,
            help="Cumulative profit/loss from paper trades"
        )

    with col4:
        st.metric(
            label="Predictions",
            value=f"{stats['predictions_with_outcomes']}/{stats['total_predictions']}",
            help="Predictions with recorded outcomes / Total predictions"
        )

    st.markdown("---")

    # Win/Loss breakdown
    if stats["wins"] + stats["losses"] > 0:
        st.subheader("Win/Loss Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig = px.pie(
                values=[stats["wins"], stats["losses"]],
                names=["Wins", "Losses"],
                color_discrete_sequence=["#00CC96", "#EF553B"],
                hole=0.4,
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Winning Trades | {stats['wins']} |
            | Losing Trades | {stats['losses']} |
            | Avg Return | {stats['avg_return']:.2f}% |
            | Total P&L | ${stats['total_pnl']:,.2f} |
            """)

    st.markdown("---")

    # Recent predictions table
    st.subheader("Recent Predictions")

    df = get_recent_predictions(days=days)

    if not df.empty:
        # Add visual indicators
        def style_direction(val):
            if val == "up":
                return "🟢 UP"
            elif val == "down":
                return "🔴 DOWN"
            return "⚪ NEUTRAL"

        def style_correct(val):
            if pd.isna(val):
                return "⏳ Pending"
            return "✅ Correct" if val == 1 else "❌ Wrong"

        display_df = df[["ticker", "prediction_date", "direction_pred", "direction_prob",
                        "recommended_strategy", "correct", "actual_return", "pnl_dollars"]].copy()

        display_df["direction_pred"] = display_df["direction_pred"].apply(style_direction)
        display_df["correct"] = display_df["correct"].apply(style_correct)
        display_df["direction_prob"] = display_df["direction_prob"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        display_df["actual_return"] = display_df["actual_return"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "Pending")
        display_df["pnl_dollars"] = display_df["pnl_dollars"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "--")

        display_df.columns = ["Ticker", "Date", "Direction", "Confidence", "Strategy", "Result", "Actual Return", "P&L"]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Accuracy over time chart
        st.markdown("---")
        st.subheader("Accuracy Over Time")

        df_with_outcomes = df[df["correct"].notna()].copy()
        if len(df_with_outcomes) > 0:
            df_with_outcomes["prediction_date"] = pd.to_datetime(df_with_outcomes["prediction_date"])
            df_with_outcomes = df_with_outcomes.sort_values("prediction_date")

            # Rolling accuracy
            df_with_outcomes["rolling_accuracy"] = df_with_outcomes["correct"].rolling(window=5, min_periods=1).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_with_outcomes["prediction_date"],
                y=df_with_outcomes["rolling_accuracy"] * 100,
                mode="lines+markers",
                name="5-day Rolling Accuracy",
                line=dict(color="#636EFA", width=2),
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Random (50%)")
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0, 100],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions with outcomes yet to chart.")

    else:
        st.info("No predictions recorded yet. Use the Daily Signals page to generate predictions.")

    # Record outcome section
    st.markdown("---")
    st.subheader("Record Outcome")

    with st.expander("Record outcome for a prediction"):
        # Get predictions without outcomes
        predictions_pending = df[df["correct"].isna()] if not df.empty else pd.DataFrame()

        if not predictions_pending.empty:
            pred_options = [
                f"{row['ticker']} - {row['prediction_date']} ({row['direction_pred']})"
                for _, row in predictions_pending.iterrows()
            ]

            selected_pred = st.selectbox("Select Prediction", pred_options)

            if selected_pred:
                pred_idx = pred_options.index(selected_pred)
                pred_row = predictions_pending.iloc[pred_idx]

                col1, col2 = st.columns(2)

                with col1:
                    actual_direction = st.selectbox(
                        "Actual Direction",
                        ["up", "down", "neutral"],
                        index=0 if pred_row["direction_pred"] == "up" else (1 if pred_row["direction_pred"] == "down" else 2)
                    )
                    actual_return = st.number_input("Actual Return (%)", value=0.0, step=0.1)

                with col2:
                    trade_taken = st.checkbox("Trade was taken")
                    if trade_taken:
                        pnl = st.number_input("P&L ($)", value=0.0, step=10.0)
                    else:
                        pnl = 0.0

                notes = st.text_area("Notes (optional)")

                if st.button("Save Outcome"):
                    from dashboard.utils.database import record_outcome

                    record_outcome(
                        prediction_id=pred_row["id"],
                        actual_direction=actual_direction,
                        actual_return=actual_return,
                        trade_taken=trade_taken,
                        pnl_dollars=pnl if trade_taken else None,
                        notes=notes if notes else None,
                    )
                    st.success("Outcome recorded!")
                    st.rerun()
        else:
            st.info("No pending predictions to record outcomes for.")

except Exception as e:
    st.error(f"Error loading performance data: {e}")
    st.info("Make sure the database is initialized and predictions have been recorded.")
