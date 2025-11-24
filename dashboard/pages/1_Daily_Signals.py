"""
Daily Signals Page - View today's ML predictions and strategies.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Daily Signals", page_icon="📊", layout="wide")

st.title("Daily Signals")
st.markdown("View ML predictions and recommended strategies for today.")

# Date selector
col1, col2 = st.columns([1, 3])
with col1:
    selected_date = st.date_input("Signal Date", value=date.today())

st.markdown("---")

# Generate signals button
if st.button("Generate New Signals", type="primary"):
    with st.spinner("Running ML models..."):
        try:
            from src.models.inference import ModelInference
            from src.data.data_fetcher import DataFetcher
            from dashboard.utils.database import record_prediction

            inference = ModelInference()
            data_fetcher = DataFetcher()

            tickers = ["SPY", "QQQ", "IWM"]
            results = []

            for ticker in tickers:
                try:
                    # Fetch latest data
                    price_data = data_fetcher.fetch_ticker_data(
                        ticker,
                        start_date=(datetime.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
                        end_date=datetime.now().strftime("%Y-%m-%d")
                    )

                    # Get prediction
                    prediction = inference.predict(ticker, price_data)

                    if prediction:
                        # Record to database
                        pred_id = record_prediction(
                            ticker=ticker,
                            prediction_date=str(selected_date),
                            direction_pred=prediction.get("direction", "neutral"),
                            direction_prob=prediction.get("direction_confidence", 0.5),
                            volatility_pred=prediction.get("volatility_regime", "medium"),
                            iv_rank=prediction.get("iv_rank"),
                            recommended_strategy=prediction.get("strategy"),
                            strategy_confidence=prediction.get("strategy_confidence"),
                            underlying_price=prediction.get("current_price"),
                        )

                        results.append({
                            "Ticker": ticker,
                            "Direction": prediction.get("direction", "neutral").upper(),
                            "Confidence": f"{prediction.get('direction_confidence', 0.5):.1%}",
                            "Strategy": prediction.get("strategy", "N/A"),
                            "IV Rank": f"{prediction.get('iv_rank', 0):.0f}%",
                            "Price": f"${prediction.get('current_price', 0):.2f}",
                        })

                except Exception as e:
                    st.warning(f"Error processing {ticker}: {e}")

            if results:
                st.success(f"Generated {len(results)} signals!")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

        except ImportError as e:
            st.error(f"Model not available: {e}")
            st.info("Run `docker-compose run app python scripts/train_model.py train` first.")

st.markdown("---")

# Display existing signals for selected date
st.subheader(f"Signals for {selected_date}")

try:
    from dashboard.utils.database import get_predictions_for_date

    predictions = get_predictions_for_date(str(selected_date))

    if predictions:
        # Create display dataframe
        display_data = []
        for p in predictions:
            # Direction styling
            direction = p["direction_pred"].upper()
            if direction == "UP":
                direction_display = "🟢 UP"
            elif direction == "DOWN":
                direction_display = "🔴 DOWN"
            else:
                direction_display = "⚪ NEUTRAL"

            display_data.append({
                "Ticker": p["ticker"],
                "Direction": direction_display,
                "Confidence": f"{(p['direction_prob'] or 0.5) * 100:.1f}%",
                "Strategy": p["recommended_strategy"] or "N/A",
                "IV Rank": f"{p['iv_rank'] or 0:.0f}%" if p["iv_rank"] else "N/A",
                "Price": f"${p['underlying_price']:.2f}" if p["underlying_price"] else "N/A",
            })

        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Strategy explanations
        st.markdown("---")
        st.subheader("Strategy Details")

        for p in predictions:
            strategy = p["recommended_strategy"]
            if strategy:
                with st.expander(f"{p['ticker']} - {strategy}"):
                    st.markdown(get_strategy_explanation(strategy, p))

    else:
        st.info("No signals recorded for this date. Click 'Generate New Signals' to create predictions.")

except Exception as e:
    st.warning(f"Could not load predictions: {e}")


def get_strategy_explanation(strategy: str, prediction: dict) -> str:
    """Get detailed explanation for a strategy."""
    explanations = {
        "long_call": """
**Long Call** - Bullish directional play

- **Setup**: Buy ATM or slightly OTM call
- **Max Profit**: Unlimited
- **Max Loss**: Premium paid
- **Best When**: Strong bullish conviction, low IV
        """,
        "long_put": """
**Long Put** - Bearish directional play

- **Setup**: Buy ATM or slightly OTM put
- **Max Profit**: Strike - Premium (if stock goes to 0)
- **Max Loss**: Premium paid
- **Best When**: Strong bearish conviction, low IV
        """,
        "bull_put_spread": """
**Bull Put Spread** (Credit Spread) - Moderately bullish

- **Setup**: Sell higher strike put, buy lower strike put
- **Max Profit**: Net credit received
- **Max Loss**: Strike width - Credit
- **Best When**: Bullish bias with elevated IV
        """,
        "bear_call_spread": """
**Bear Call Spread** (Credit Spread) - Moderately bearish

- **Setup**: Sell lower strike call, buy higher strike call
- **Max Profit**: Net credit received
- **Max Loss**: Strike width - Credit
- **Best When**: Bearish bias with elevated IV
        """,
        "iron_condor": """
**Iron Condor** - Neutral, range-bound

- **Setup**: Bull put spread + Bear call spread
- **Max Profit**: Net credit if price stays in range
- **Max Loss**: Width of wider spread - Credit
- **Best When**: High IV, expecting low volatility
        """,
        "straddle": """
**Long Straddle** - Expecting big move, direction uncertain

- **Setup**: Buy ATM call + ATM put
- **Max Profit**: Unlimited
- **Max Loss**: Total premium paid
- **Best When**: Low IV before expected volatility event
        """,
        "strangle": """
**Long Strangle** - Expecting big move, cheaper than straddle

- **Setup**: Buy OTM call + OTM put
- **Max Profit**: Unlimited
- **Max Loss**: Total premium paid
- **Best When**: Low IV, expecting large move
        """,
    }

    base_explanation = explanations.get(strategy, f"**{strategy}** - Custom strategy")

    # Add context from prediction
    iv_rank = prediction.get("iv_rank")
    if iv_rank:
        if iv_rank > 70:
            base_explanation += f"\n\n*Current IV Rank: {iv_rank:.0f}% (High - favor selling premium)*"
        elif iv_rank < 30:
            base_explanation += f"\n\n*Current IV Rank: {iv_rank:.0f}% (Low - favor buying options)*"
        else:
            base_explanation += f"\n\n*Current IV Rank: {iv_rank:.0f}% (Normal)*"

    return base_explanation
