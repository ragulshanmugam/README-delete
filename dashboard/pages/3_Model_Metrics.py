"""
Model Metrics Page - Monitor model health and detect drift.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Model Metrics", page_icon="🔬", layout="wide")

st.title("Model Metrics & Health")
st.markdown("Monitor model performance, feature importance, and detect drift.")

# Ticker selector
ticker = st.selectbox("Select Ticker", ["SPY", "QQQ", "IWM"])

st.markdown("---")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for drift detection.

    PSI < 0.1: No significant change
    PSI 0.1-0.25: Moderate change, monitor
    PSI > 0.25: Significant change, investigate

    Args:
        expected: Reference distribution (training data)
        actual: Current distribution (recent predictions)
        bins: Number of bins for discretization

    Returns:
        PSI score
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Calculate frequencies
    expected_freq = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_freq = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_freq = np.clip(expected_freq, 0.0001, 1)
    actual_freq = np.clip(actual_freq, 0.0001, 1)

    # Calculate PSI
    psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))

    return psi


# Load model and get metrics
try:
    from src.models.inference import ModelInference
    import joblib

    # Model paths
    model_dir = Path(__file__).parent.parent.parent / "models"
    direction_model_path = model_dir / f"{ticker.lower()}_direction_model.joblib"
    vol_model_path = model_dir / f"{ticker.lower()}_volatility_model.joblib"

    col1, col2 = st.columns(2)

    # Direction Model
    with col1:
        st.subheader("Direction Classifier")

        if direction_model_path.exists():
            model_data = joblib.load(direction_model_path)
            model = model_data.get("model")
            metadata = model_data.get("metadata", {})
            feature_names = model_data.get("feature_names", [])

            # Model info
            st.markdown(f"""
            **Model Type**: {type(model).__name__}
            **Features**: {len(feature_names)}
            **Training Date**: {metadata.get('trained_at', 'Unknown')[:10] if metadata.get('trained_at') else 'Unknown'}
            """)

            # Metrics from training
            metrics = metadata.get("metrics", {})
            if metrics:
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                with metric_col2:
                    st.metric("AUC", f"{metrics.get('auc', 0):.3f}")

            # Feature importance
            if hasattr(model, "coef_"):
                st.markdown("**Feature Importance (Top 10)**")

                # Get coefficients (absolute values for importance)
                if len(model.coef_.shape) > 1:
                    importances = np.abs(model.coef_).mean(axis=0)
                else:
                    importances = np.abs(model.coef_)

                importance_df = pd.DataFrame({
                    "Feature": feature_names[:len(importances)],
                    "Importance": importances
                }).sort_values("Importance", ascending=True).tail(10)

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"Direction model not found for {ticker}")
            st.info("Run training with: `docker-compose run app python scripts/train_model.py train`")

    # Volatility Model
    with col2:
        st.subheader("Volatility Forecaster")

        if vol_model_path.exists():
            model_data = joblib.load(vol_model_path)
            model = model_data.get("model")
            metadata = model_data.get("metadata", {})
            feature_names = model_data.get("feature_names", [])

            # Model info
            st.markdown(f"""
            **Model Type**: {type(model).__name__}
            **Features**: {len(feature_names)}
            **Training Date**: {metadata.get('trained_at', 'Unknown')[:10] if metadata.get('trained_at') else 'Unknown'}
            """)

            # Metrics from training
            metrics = metadata.get("metrics", {})
            if metrics:
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                with metric_col2:
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")

            # Feature importance
            if hasattr(model, "coef_"):
                st.markdown("**Feature Importance (Top 10)**")

                if len(model.coef_.shape) > 1:
                    importances = np.abs(model.coef_).mean(axis=0)
                else:
                    importances = np.abs(model.coef_)

                importance_df = pd.DataFrame({
                    "Feature": feature_names[:len(importances)],
                    "Importance": importances
                }).sort_values("Importance", ascending=True).tail(10)

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Oranges",
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"Volatility model not found for {ticker}")

    st.markdown("---")

    # Drift Detection
    st.subheader("Drift Detection")

    st.markdown("""
    **Population Stability Index (PSI)** measures how much the distribution of input features
    has changed compared to the training data.

    - **PSI < 0.1**: No significant change (green)
    - **PSI 0.1-0.25**: Moderate change, monitor (yellow)
    - **PSI > 0.25**: Significant change, investigate (red)
    """)

    if st.button("Run Drift Analysis"):
        with st.spinner("Analyzing feature distributions..."):
            try:
                from src.data.data_fetcher import DataFetcher
                from src.models.feature_pipeline import FeaturePipeline

                data_fetcher = DataFetcher()

                # Fetch data
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

                price_data = data_fetcher.fetch_ticker_data(ticker, start_date, end_date)

                if price_data is not None and not price_data.empty:
                    # Build features
                    pipeline = FeaturePipeline(ticker)
                    features_df, _ = pipeline.prepare_training_data(
                        price_df=price_data,
                        target_col="direction",
                        forward_days=5,
                    )

                    # Split into training period (older) and recent (newer)
                    split_idx = int(len(features_df) * 0.8)
                    training_data = features_df.iloc[:split_idx]
                    recent_data = features_df.iloc[split_idx:]

                    # Calculate PSI for each numeric feature
                    psi_results = []
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns

                    for col in numeric_cols:
                        if col in ["direction", "direction_binary"]:
                            continue

                        train_vals = training_data[col].dropna().values
                        recent_vals = recent_data[col].dropna().values

                        if len(train_vals) > 10 and len(recent_vals) > 10:
                            psi = calculate_psi(train_vals, recent_vals)
                            psi_results.append({
                                "Feature": col,
                                "PSI": psi,
                                "Status": "OK" if psi < 0.1 else ("Monitor" if psi < 0.25 else "DRIFT")
                            })

                    psi_df = pd.DataFrame(psi_results).sort_values("PSI", ascending=False)

                    # Summary
                    drift_count = len(psi_df[psi_df["Status"] == "DRIFT"])
                    monitor_count = len(psi_df[psi_df["Status"] == "Monitor"])

                    if drift_count > 0:
                        st.error(f"**{drift_count} features show significant drift!** Consider retraining.")
                    elif monitor_count > 0:
                        st.warning(f"**{monitor_count} features show moderate drift.** Monitor closely.")
                    else:
                        st.success("**No significant drift detected.** Model inputs are stable.")

                    # Show PSI values
                    st.markdown("**PSI by Feature (Top 15)**")

                    top_psi = psi_df.head(15)

                    # Color code
                    def color_psi(val):
                        if val >= 0.25:
                            return "background-color: #ffcccc"
                        elif val >= 0.1:
                            return "background-color: #fff3cd"
                        return "background-color: #d4edda"

                    styled_df = top_psi.style.map(color_psi, subset=["PSI"])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

                    # PSI chart
                    fig = px.bar(
                        top_psi,
                        x="Feature",
                        y="PSI",
                        color="Status",
                        color_discrete_map={"OK": "#00CC96", "Monitor": "#FFA15A", "DRIFT": "#EF553B"},
                    )
                    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Monitor threshold")
                    fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Drift threshold")
                    fig.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error running drift analysis: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # Model retraining section
    st.subheader("Model Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Retrain Models"):
            st.info("Run in terminal: `docker-compose run app python scripts/train_model.py train --ticker " + ticker + "`")

    with col2:
        if st.button("Export Model Report"):
            st.info("Model report export coming soon...")

except Exception as e:
    st.error(f"Error loading model metrics: {e}")
    import traceback
    st.code(traceback.format_exc())
