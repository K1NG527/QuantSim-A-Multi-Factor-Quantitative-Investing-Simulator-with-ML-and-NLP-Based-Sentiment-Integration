"""
EquiSense — Streamlit Dashboard (Production Upgrade)

Tabs:
    1. Rule-Based Backtest
    2. ML-Based Backtest
    3. Benchmark Comparison
    4. Document Analyzer
    5. SHAP Explainability (NEW)
    6. AI Insights (NEW)
    7. Risk Metrics (NEW)
    8. Sentiment Analysis (NEW)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scripts.backtester as backtester
import scripts.visualizer as visualizer


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="EquiSense — Factor Investing Simulator", layout="wide")
st.title("📊 EquiSense — Quantitative Factor Investing Simulator")

# Sidebar Navigation
tab = st.sidebar.radio("Select View", [
    "📈 Rule-Based Backtest",
    "🤖 ML-Based Backtest",
    "📊 Compare to S&P 500",
    "📄 Document Analyzer",
    "🔍 SHAP Explainability",
    "🧠 AI Insights",
    "⚡ Risk Metrics",
    "📰 Sentiment Analysis",
])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def read_portfolio(path):
    df = pd.read_csv(path)
    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
    if not date_col:
        raise ValueError(f"No date column found in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col)


@st.cache_data
def load_data():
    prices_path = "data/processed/clean_dataset/daily_prices_clean.csv"

    def process_portfolio(path):
        returns, metrics = backtester.run_backtest(path, prices_path)
        df = pd.DataFrame(index=returns.index)
        df['Daily Return'] = returns
        df['Cumulative'] = (1 + returns).cumprod()
        peak = df['Cumulative'].expanding(min_periods=1).max()
        df['Drawdown'] = (df['Cumulative'] - peak) / peak
        df['Rolling Sharpe'] = (returns.rolling(63).mean() / returns.rolling(63).std()) * np.sqrt(252)
        return df

    rule_equal = process_portfolio("data/processed/portfolio_weight/equal_weight_portfolio.csv")
    rule_risk = process_portfolio("data/processed/portfolio_weight/risk_adjusted_portfolio.csv")
    ml_equal = process_portfolio("data/processed/predicted_portfolio_scores/equal_weight_portfolio_ml.csv")
    ml_risk = process_portfolio("data/processed/predicted_portfolio_scores/risk_adjusted_portfolio_ml.csv")

    sp500 = pd.read_csv("data/sp500.csv", header=2, parse_dates=True, index_col=0)
    sp500.columns = ['Close']
    return rule_equal, rule_risk, ml_equal, ml_risk, sp500


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 & 2: BACKTEST VIEWS
# ═══════════════════════════════════════════════════════════════════════════════

if tab == "📈 Rule-Based Backtest":
    rule_equal, rule_risk, ml_equal, ml_risk, sp500 = load_data()
    st.header("📈 Rule-Based Strategy")

    # Transaction cost / slippage controls
    with st.sidebar.expander("⚙️ Backtest Settings"):
        tc = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
        slip = st.slider("Slippage (bps)", 0, 20, 5)
        rebal = st.selectbox("Rebalance Frequency", ["daily", "monthly", "quarterly"])

    visualizer.plot_cumulative_returns(rule_equal, rule_risk)
    visualizer.plot_drawdowns(rule_equal, rule_risk)
    visualizer.plot_histogram(rule_equal, rule_risk)
    visualizer.plot_rolling_sharpe(rule_equal, rule_risk)

elif tab == "🤖 ML-Based Backtest":
    rule_equal, rule_risk, ml_equal, ml_risk, sp500 = load_data()
    st.header("🤖 ML-Based Strategy")
    visualizer.plot_cumulative_returns(ml_equal, ml_risk)
    visualizer.plot_drawdowns(ml_equal, ml_risk)
    visualizer.plot_histogram(ml_equal, ml_risk)
    visualizer.plot_rolling_sharpe(ml_equal, ml_risk)

elif tab == "📊 Compare to S&P 500":
    rule_equal, rule_risk, ml_equal, ml_risk, sp500 = load_data()
    st.header("📊 Benchmark Comparison")
    visualizer.plot_benchmark_comparison(rule_equal, rule_risk, ml_equal, ml_risk, sp500)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: DOCUMENT ANALYZER (existing functionality preserved)
# ═══════════════════════════════════════════════════════════════════════════════

elif tab == "📄 Document Analyzer":
    st.header("📄 Financial Document Intelligence Engine")
    st.markdown(
        "Upload a brokerage PDF for automated extraction, real-time enrichment, "
        "risk analysis, investor profiling, and longitudinal tracking."
    )

    col_user, col_enrich = st.columns([2, 1])
    with col_user:
        user_id = st.text_input("👤 User ID", value="investor_1",
                                help="Used for tracking portfolio history across uploads")
    with col_enrich:
        do_enrich = st.checkbox("📡 Enrich with live market data", value=True,
                                help="Fetch real-time prices, sector, and volatility via yfinance")

    uploaded_files = st.file_uploader(
        "Upload Brokerage PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more files (Holdings, P&L, Trades).",
    )

    if uploaded_files:
        with st.spinner(f"🔍 Analyzing {len(uploaded_files)} document(s)..."):
            try:
                import importlib
                import scripts.pdf_parser
                import scripts.data_extractor
                import scripts.risk_engine
                import scripts.market_data
                import scripts.longitudinal_engine
                import scripts.document_analyzer

                importlib.reload(scripts.pdf_parser)
                importlib.reload(scripts.data_extractor)
                importlib.reload(scripts.risk_engine)
                importlib.reload(scripts.market_data)
                importlib.reload(scripts.longitudinal_engine)
                importlib.reload(scripts.document_analyzer)

                from scripts.document_analyzer import analyze_document

                result = analyze_document(pdf_files=uploaded_files, user_id=user_id, enrich=do_enrich)
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                st.stop()

        # Document Summary
        st.subheader("📋 Document Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            doc_types = ", ".join(result["document_summary"]["document_types"])
            st.info(f"**Document Type(s):** {doc_types}")
        with c2:
            st.info(f"**Broker:** {result['document_summary']['broker']}")
        with c3:
            enrich_st = result["document_summary"].get("enrichment_status", "N/A")
            if "enriched" in str(enrich_st):
                st.success(f"**Enrichment:** ✅ {enrich_st}")
            else:
                st.warning(f"**Enrichment:** {enrich_st}")

        flags = result["extracted_data"].get("data_quality_flags", [])
        if flags:
            st.warning(f"⚠️ **Data Quality Flags:** {', '.join(flags)}")

        # Portfolio Overview
        pm = result["portfolio_metrics"]
        if pm.get("total_portfolio_value", 0) > 0:
            st.subheader("💰 Portfolio Overview")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Value", f"₹{pm['total_portfolio_value']:,.2f}")
            m2.metric("Stocks", pm["number_of_stocks"])
            m3.metric("Top-3 Concentration", f"{pm['top_3_concentration_ratio']}%")
            m4.metric("Diversification", f"{pm['diversification_score']:.2f}")

            vol = pm.get("portfolio_volatility")
            if vol is not None:
                e1, e2 = st.columns(2)
                e1.metric("📉 Portfolio Volatility", f"{vol:.2%}")
                sector_exp = pm.get("sector_exposure", {})
                if sector_exp:
                    top_sec = max(sector_exp.items(), key=lambda x: x[1])
                    e2.metric("🏭 Top Sector", f"{top_sec[0]} ({top_sec[1]:.1f}%)")

            # Portfolio Heatmap
            sector_exp = pm.get("sector_exposure", {})
            if sector_exp and len(sector_exp) > 1 and "Unknown" not in list(sector_exp.keys())[:1]:
                fig = px.pie(
                    names=list(sector_exp.keys()),
                    values=list(sector_exp.values()),
                    title="Sector Allocation",
                    hole=0.4,
                )
                fig.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

            # Holdings table
            holdings = result["extracted_data"].get("holdings", [])
            if holdings:
                with st.expander("📊 Holdings Details", expanded=False):
                    rows = []
                    for h in holdings:
                        row = {
                            "Stock": h.get("stock_name", ""),
                            "Symbol": h.get("standardized_symbol") or h.get("symbol", ""),
                            "Qty": h.get("quantity"),
                            "Avg Price": h.get("average_buy_price"),
                            "Current Price": h.get("current_price"),
                            "Value": h.get("total_value"),
                            "Weight %": h.get("portfolio_weight"),
                            "Confidence": h.get("confidence"),
                        }
                        market = h.get("market_data", {})
                        if isinstance(market, dict):
                            row["Sector"] = market.get("sector", "")
                            row["Volatility"] = market.get("volatility_proxy")
                            row["Source"] = market.get("source", "")
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Risk Dashboard
        ra = result["risk_analysis"]
        st.subheader("🎯 Risk Dashboard")

        risk_val = float(ra["risk_score"])
        if risk_val <= 3:
            risk_color, risk_label = "🟢", "Low Risk"
        elif risk_val <= 6:
            risk_color, risk_label = "🟡", "Moderate Risk"
        else:
            risk_color, risk_label = "🔴", "High Risk"

        sub = ra["sub_scores"]
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric(f"{risk_color} Overall", f"{ra['risk_score']}/10", risk_label)
        r2.metric("Concentration", f"{sub.get('concentration_score', 'N/A')}/10")
        r3.metric("Behavior", f"{sub.get('behavior_score', 'N/A')}/10")
        r4.metric("Performance", f"{sub.get('performance_score', 'N/A')}/10")
        r5.metric("Market Risk", f"{sub.get('market_risk_score', 'N/A')}/10")

        st.caption(ra["explanation"])

        # Investor Profile
        profile = result["investor_profile"]
        profile_styles = {
            "Conservative": ("🛡️", "green"),
            "Moderate": ("⚖️", "blue"),
            "Aggressive": ("🔥", "orange"),
            "Speculative / High-Risk Trader": ("⚡", "red"),
        }
        emoji, color = profile_styles.get(profile, ("📊", "gray"))
        st.subheader(f"{emoji} Investor Profile: **{profile}**")

        # Behavioral Metrics
        bm = result["behavioral_metrics"]
        if bm.get("total_trades", 0) > 0:
            st.subheader("📈 Behavioral Metrics")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Total Trades", bm["total_trades"])
            b2.metric("Trades/Month", bm.get("trade_frequency", "N/A"))
            bsr = bm.get("buy_sell_ratio", "N/A")
            b3.metric("Buy/Sell Ratio", bsr if bsr != float("inf") else "∞")
            hp = bm.get("average_holding_period")
            b4.metric("Avg Hold", f"{hp} days" if isinstance(hp, (int, float)) else "N/A")
            b5.metric("Churn Rate", bm.get("churn_rate", "N/A"))

            trades = result["extracted_data"].get("trades", [])
            if trades:
                with st.expander("📋 Trade Details", expanded=False):
                    st.dataframe(pd.DataFrame(trades), use_container_width=True)

        # Performance Metrics
        perf = result["performance_metrics"]
        if any(v is not None for v in perf.values()):
            st.subheader("📊 Performance Metrics")
            p1, p2, p3 = st.columns(3)
            ret = perf.get("total_realized_return")
            p1.metric("Realized Return", f"₹{ret:,.2f}" if ret is not None else "N/A")
            wl = perf.get("win_loss_ratio")
            p2.metric("Win/Loss Ratio", f"{wl:.2f}" if wl is not None and wl != float("inf") else "N/A")
            cs = perf.get("consistency_score")
            p3.metric("Consistency", f"{cs:.2f}" if cs is not None else "N/A")

        # Historical Comparison
        hist = result.get("historical_comparison", {})
        if hist.get("has_history"):
            st.subheader("📅 Historical Comparison")
            trend = hist.get("risk_trend", "stable")
            trend_emoji = {"increasing": "📈 ↑", "decreasing": "📉 ↓", "stable": "➡️ →"}.get(trend, "➡️")
            h1, h2, h3 = st.columns(3)
            h1.metric("Risk Trend", f"{trend_emoji} {trend.capitalize()}")
            val_change = hist.get("value_change")
            if val_change:
                delta_str = f"₹{val_change['absolute']:,.2f}"
                pct = val_change.get('percentage')
                if pct is not None:
                    delta_str += f" ({pct:+.1f}%)"
                h2.metric("Value Change", delta_str)
            risk_change = hist.get("risk_change")
            if risk_change is not None:
                h3.metric("Risk Score Δ", f"{risk_change:+.1f}")
            shifts = hist.get("behavioral_shifts", [])
            if shifts:
                st.markdown("**🔄 Behavioral Shifts Detected:**")
                for s in shifts:
                    st.info(s)

        # Key Insights
        st.subheader("💡 Key Insights")
        col_s, col_r = st.columns(2)
        with col_s:
            st.markdown("**✅ Strengths**")
            for s in result.get("key_insights", []):
                st.success(s)
        with col_r:
            st.markdown("**⚠️ Risk Factors**")
            for r in result.get("risk_factors", []):
                st.warning(r)

        observations = result.get("behavioral_observations", [])
        if observations:
            st.markdown("**🔍 Behavioral Observations**")
            for obs in observations:
                st.info(obs)

        # Recommendations
        st.subheader("📝 Recommendations")
        for i, rec in enumerate(result.get("recommendations", []), 1):
            st.markdown(f"**{i}.** {rec}")

        # Database Status
        snap = result.get("snapshot", {})
        db_status = snap.get("db_status")
        snap_id = snap.get("snapshot_id")
        if snap_id:
            st.success(f"✅ Snapshot saved to database (ID: {snap_id})")
        elif db_status:
            st.warning(f"⚠️ DB: {db_status}")

        with st.expander("🔧 Raw JSON Output", expanded=False):
            st.json(result)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: SHAP EXPLAINABILITY (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

elif tab == "🔍 SHAP Explainability":
    st.header("🔍 SHAP Factor Explainability")
    st.markdown("Understand which factors drive ML model predictions using SHAP values.")

    # Check for existing explainability data
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "explainability.json")
    bar_path = os.path.join(os.path.dirname(__file__), "..", "models", "shap_bar.png")
    summary_path = os.path.join(os.path.dirname(__file__), "..", "models", "shap_summary.png")

    col_run, col_info = st.columns([1, 2])
    with col_run:
        run_shap = st.button("🔄 Run SHAP Analysis", type="primary")

    if run_shap:
        with st.spinner("Computing SHAP values... This may take a minute."):
            try:
                from models.ml_pipeline import FactorModelTrainer
                from scripts.explainability import explain_model

                trainer = FactorModelTrainer(data_dir="data", model_dir="models")
                df = trainer.prepare_dataset()
                if df.empty:
                    st.error("No data available for SHAP analysis.")
                    st.stop()
                trainer.train_model()
                results = explain_model(trainer, output_dir="data", plot_dir="models")
                st.success("✅ SHAP analysis complete!")
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                st.stop()

    # Display results if available
    if os.path.exists(json_path):
        import json
        with open(json_path, "r") as f:
            shap_data = json.load(f)

        st.subheader("📊 Global Feature Importance")
        importance = pd.DataFrame(shap_data["global_feature_importance"])
        fig = px.bar(
            importance, x="mean_abs_shap", y="feature",
            orientation="h", color="mean_abs_shap",
            color_continuous_scale="RdYlBu_r",
            title="Mean |SHAP Value| per Factor",
        )
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Display SHAP plots
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(bar_path):
                st.image(bar_path, caption="SHAP Bar Chart")
        with col2:
            if os.path.exists(summary_path):
                st.image(summary_path, caption="SHAP Summary Plot")

        # Per-sample analysis
        per_sample = shap_data.get("per_sample_top_factors", [])
        if per_sample:
            st.subheader("🔎 Per-Stock Factor Breakdown (Sample)")
            sample_df = []
            for item in per_sample[:20]:
                for factor in item["top_factors"]:
                    sample_df.append({
                        "Sample": item["sample_index"],
                        "Factor": factor["feature"],
                        "SHAP Value": factor["shap_value"],
                        "Direction": factor["direction"],
                    })
            if sample_df:
                st.dataframe(pd.DataFrame(sample_df), use_container_width=True)
    else:
        st.info("No SHAP data found. Click **Run SHAP Analysis** to generate.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: AI INSIGHTS (PDF + Manual)
# ═══════════════════════════════════════════════════════════════════════════════

elif tab == "🧠 AI Insights":
    st.header("🧠 AI Portfolio Insights")
    st.markdown("Upload a brokerage **PDF statement** or manually configure your portfolio to get AI-powered analysis.")

    portfolio_data = None

    # --- PDF Upload Mode ---
    st.subheader("📂 Option 1: Upload Brokerage PDF")
    ai_pdf = st.file_uploader(
        "Upload your brokerage PDF (holdings, P&L, or trade statements)",
        type=["pdf"], key="ai_pdf_upload", accept_multiple_files=True,
    )

    if ai_pdf:
        with st.spinner("🔍 Parsing PDF and extracting portfolio data..."):
            try:
                from scripts.document_analyzer import analyze_document
                result = analyze_document(ai_pdf, user_id="ai_insights", enrich=True)

                pm = result.get("portfolio_metrics", {})
                ra = result.get("risk_analysis", {})
                holdings = result.get("extracted_data", {}).get("holdings", [])

                total_value = float(pm.get("total_portfolio_value", 0))
                n_stocks = int(pm.get("number_of_stocks", len(holdings)))
                concentration = float(pm.get("top_3_concentration_ratio", 0))
                div_score = float(pm.get("diversification_score", 0))
                volatility = float(pm.get("portfolio_volatility", 0.25))
                risk_score = float(ra.get("risk_score", 5))

                # Show parsed summary
                doc_info = result.get("document_summary", {})
                st.success(
                    f"✅ Parsed **{n_stocks}** holdings from **{doc_info.get('broker', 'Unknown')}** "
                    f"({', '.join(doc_info.get('document_types', []))}) | "
                    f"Total Value: **₹{total_value:,.0f}** | Risk Score: **{risk_score}/10**"
                )

                # Show holdings preview
                if holdings:
                    with st.expander("📊 Extracted Holdings Preview", expanded=False):
                        rows = []
                        for h in holdings:
                            rows.append({
                                "Stock": h.get("stock_name", ""),
                                "Symbol": h.get("standardized_symbol") or h.get("symbol", ""),
                                "Qty": h.get("quantity"),
                                "Avg Price": h.get("average_buy_price"),
                                "Value": h.get("total_value"),
                                "Weight %": h.get("portfolio_weight"),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                portfolio_data = {
                    "portfolio_metrics": {
                        "total_portfolio_value": total_value,
                        "number_of_stocks": n_stocks,
                        "top_3_concentration_ratio": round(concentration, 1),
                        "diversification_score": round(div_score, 2),
                        "portfolio_volatility": volatility,
                    },
                    "risk_analysis": {"risk_score": round(risk_score, 1)},
                    "holdings_summary": [{"stock": h.get("stock_name",""), "value": h.get("total_value",0), "weight": h.get("portfolio_weight",0)} for h in holdings[:20]],
                    "investor_profile": result.get("investor_profile", "Unknown"),
                    "key_insights": result.get("key_insights", []),
                    "risk_factors": result.get("risk_factors", []),
                }
            except Exception as e:
                st.error(f"Failed to parse PDF: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    # --- Manual Fallback Mode ---
    with st.expander("⚙️ Option 2: Manual Configuration (click to expand)", expanded=(not ai_pdf)):
        col1, col2 = st.columns(2)
        with col1:
            m_total_value = st.number_input("Total Portfolio Value ($)", value=500000, step=10000)
            m_n_stocks = st.number_input("Number of Stocks", value=12, step=1, min_value=1)
            m_risk_score = st.slider("Risk Score", 1.0, 10.0, 5.5, 0.5)
        with col2:
            m_concentration = st.slider("Top-3 Concentration (%)", 10.0, 100.0, 45.0, 1.0)
            m_volatility = st.slider("Portfolio Volatility", 0.05, 0.80, 0.28, 0.01)
            m_div_score = st.slider("Diversification Score", 0.0, 1.0, 0.72, 0.01)

        st.subheader("⚖️ Factor Exposures")
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            f_value = st.slider("Value", 0.0, 1.0, 0.5, 0.05)
        with fc2:
            f_momentum = st.slider("Momentum", 0.0, 1.0, 0.5, 0.05)
        with fc3:
            f_quality = st.slider("Quality", 0.0, 1.0, 0.5, 0.05)
        with fc4:
            f_volatility = st.slider("Volatility", 0.0, 1.0, 0.5, 0.05)
        with fc5:
            f_volume = st.slider("Volume", 0.0, 1.0, 0.5, 0.05)

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[f_value, f_momentum, f_quality, f_volatility, f_volume, f_value],
            theta=["Value", "Momentum", "Quality", "Volatility", "Volume", "Value"],
            fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_color="#2196F3",
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Factor Exposure Profile", height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

        if portfolio_data is None:
            portfolio_data = {
                "portfolio_metrics": {
                    "total_portfolio_value": m_total_value,
                    "number_of_stocks": m_n_stocks,
                    "top_3_concentration_ratio": m_concentration,
                    "diversification_score": m_div_score,
                    "portfolio_volatility": m_volatility,
                },
                "risk_analysis": {"risk_score": m_risk_score},
                "factor_exposures": {
                    "value": f_value, "momentum": f_momentum,
                    "quality": f_quality, "volatility": f_volatility,
                    "volume": f_volume,
                },
            }

    if st.button("🚀 Generate AI Insights", type="primary"):
        if portfolio_data is None:
            st.warning("Please upload a PDF or configure your portfolio above.")
        else:
            with st.spinner("🧠 Generating AI insights..."):
                try:
                    from scripts.ai_insights import generate_ai_insights
                    insights = generate_ai_insights(portfolio_data)

                    st.subheader(f"📋 Analysis ({insights['source'].upper()})")
                    st.markdown("### 🔴 Risk Summary")
                    st.markdown(insights["risk_summary"])
                    st.markdown("### 🔵 Diversification Analysis")
                    st.markdown(insights["diversification_analysis"])
                    st.markdown("### 💡 Suggestions")
                    for i, s in enumerate(insights.get("suggestions", []), 1):
                        st.markdown(f"**{i}.** {s}")
                    if insights.get("raw_response"):
                        with st.expander("📝 Full AI Response"):
                            st.markdown(insights["raw_response"])
                except Exception as e:
                    st.error(f"Failed to generate insights: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: RISK METRICS PANEL (PDF + Pre-built)
# ═══════════════════════════════════════════════════════════════════════════════

elif tab == "⚡ Risk Metrics":
    st.header("⚡ Risk Metrics Dashboard")
    st.markdown("Upload a brokerage **PDF statement** or select a pre-built strategy for risk analysis.")

    risk_result = None  # Holds the full analysis result from PDF
    returns = None  # Holds daily returns for pre-built portfolios

    # --- PDF Upload Mode ---
    risk_pdf = st.file_uploader(
        "Upload your brokerage PDF for risk analysis",
        type=["pdf"], key="risk_pdf_upload", accept_multiple_files=True,
    )

    if risk_pdf:
        with st.spinner("🔍 Parsing PDF and computing risk metrics..."):
            try:
                from scripts.document_analyzer import analyze_document
                risk_result = analyze_document(risk_pdf, user_id="risk_metrics", enrich=True)
            except Exception as e:
                st.error(f"Failed to parse PDF: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    if risk_result is not None:
        # Display PDF-derived risk dashboard
        pm = risk_result.get("portfolio_metrics", {})
        ra = risk_result.get("risk_analysis", {})

        st.subheader("📊 Portfolio Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Total Value", f"₹{float(pm.get('total_portfolio_value', 0)):,.0f}")
        c2.metric("📊 Stocks", pm.get("number_of_stocks", 0))
        c3.metric("🎯 Risk Score", f"{ra.get('risk_score', 'N/A')}/10")
        c4.metric("📈 Diversification", f"{float(pm.get('diversification_score', 0)):.2f}")

        # Sub-scores
        sub = ra.get("sub_scores", {})
        if sub:
            st.subheader("🎯 Risk Sub-Scores")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Concentration", f"{sub.get('concentration_score', 'N/A')}/10")
            s2.metric("Behavior", f"{sub.get('behavior_score', 'N/A')}/10")
            s3.metric("Performance", f"{sub.get('performance_score', 'N/A')}/10")
            s4.metric("Market Risk", f"{sub.get('market_risk_score', 'N/A')}/10")

        st.caption(ra.get("explanation", ""))

        # Sector allocation pie chart
        sector_exp = pm.get("sector_exposure", {})
        if sector_exp and len(sector_exp) > 1:
            fig_sector = px.pie(
                names=list(sector_exp.keys()), values=list(sector_exp.values()),
                title="Sector Allocation", hole=0.4,
            )
            fig_sector.update_layout(height=400)
            st.plotly_chart(fig_sector, use_container_width=True)

        # Holdings breakdown
        holdings = risk_result.get("extracted_data", {}).get("holdings", [])
        if holdings:
            st.subheader("📊 Holdings Breakdown")
            weights_list = []
            for h in holdings:
                w = h.get("portfolio_weight", 0)
                weights_list.append({
                    "Stock": h.get("stock_name", h.get("symbol", "Unknown")),
                    "Weight %": w if w else 0,
                    "Value": h.get("total_value", 0),
                })
            weights_df = pd.DataFrame(weights_list).sort_values("Weight %", ascending=False)
            fig_weights = px.bar(
                weights_df, x="Stock", y="Weight %", color="Weight %",
                color_continuous_scale="RdYlBu_r", title="Portfolio Weight Distribution",
            )
            fig_weights.update_layout(height=400)
            st.plotly_chart(fig_weights, use_container_width=True)

        # Investor profile
        profile = risk_result.get("investor_profile", "Unknown")
        st.subheader(f"📊 Investor Profile: **{profile}**")

        # Recommendations
        recs = risk_result.get("recommendations", [])
        if recs:
            st.subheader("📝 Recommendations")
            for i, rec in enumerate(recs, 1):
                st.markdown(f"**{i}.** {rec}")

    else:
        # --- Pre-built Fallback ---
        st.markdown("**Or select a pre-built portfolio:**")
        try:
            rule_equal, rule_risk, ml_equal, ml_risk, sp500 = load_data()
            portfolio_choice = st.selectbox(
                "Select Portfolio",
                ["Equal Weight (Rule)", "Risk-Adjusted (Rule)", "Equal Weight (ML)", "Risk-Adjusted (ML)"]
            )
            portfolio_map = {
                "Equal Weight (Rule)": rule_equal, "Risk-Adjusted (Rule)": rule_risk,
                "Equal Weight (ML)": ml_equal, "Risk-Adjusted (ML)": ml_risk,
            }
            selected = portfolio_map[portfolio_choice]
            returns = selected["Daily Return"]
        except Exception as e:
            st.error(f"Failed to load pre-built portfolios: {e}")

        # Display pre-built metrics
        if returns is not None and len(returns) > 0:
            metrics = backtester.calculate_metrics(returns)

            st.subheader("📊 Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cumulative Return", f"{metrics['cumulative_return']:.2%}")
            m2.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
            m3.metric("Annualized Volatility", f"{metrics['annualized_volatility']:.2%}")
            m4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

            m5, m6, m7 = st.columns(3)
            m5.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            m6.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
            m7.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")

            # Monthly returns heatmap
            st.subheader("🗓️ Monthly Returns Heatmap")
            monthly = returns.resample("ME").sum()
            monthly_df = pd.DataFrame({
                "Year": monthly.index.year, "Month": monthly.index.month, "Return": monthly.values,
            })
            if not monthly_df.empty:
                pivot = monthly_df.pivot_table(values="Return", index="Year", columns="Month")
                month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                pivot.columns = [month_names.get(c, c) for c in pivot.columns]
                fig_heat = px.imshow(
                    pivot.values, labels=dict(x="Month", y="Year", color="Return"),
                    x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    color_continuous_scale="RdYlGn", aspect="auto", title="Monthly Returns Heatmap",
                )
                fig_heat.update_layout(height=400)
                st.plotly_chart(fig_heat, use_container_width=True)

            # Return distribution
            st.subheader("📉 Return Distribution")
            fig_dist = px.histogram(
                x=returns.values, nbins=100, title="Daily Return Distribution",
                labels={"x": "Daily Return", "y": "Frequency"}, color_discrete_sequence=["#2196F3"],
            )
            fig_dist.add_vline(x=returns.mean(), line_dash="dash", line_color="red",
                               annotation_text=f"Mean: {returns.mean():.4f}")
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: SENTIMENT ANALYSIS (Fixed — uses session_state)
# ═══════════════════════════════════════════════════════════════════════════════

elif tab == "📰 Sentiment Analysis":
    st.header("📰 News Sentiment Analysis")
    st.markdown("Analyze market sentiment using Finnhub news + FinBERT NLP model.")

    # Initialize session state for sentiment results
    if "sentiment_results" not in st.session_state:
        st.session_state.sentiment_results = None
    if "sentiment_running" not in st.session_state:
        st.session_state.sentiment_running = False

    ticker_input = st.text_input("Enter ticker(s) separated by comma", "AAPL, MSFT, GOOGL")
    days_back = st.slider("Days of news to analyze", 1, 90, 30)
    store_db = st.checkbox("Store results in database", value=False)

    if st.button("🔍 Analyze Sentiment", type="primary") and not st.session_state.sentiment_running:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

        if not tickers:
            st.warning("Please enter at least one ticker.")
        else:
            st.session_state.sentiment_running = True
            with st.spinner(f"Analyzing sentiment for {', '.join(tickers)}... (this may take a moment on first run as FinBERT downloads)"):
                try:
                    from scripts.sentiment_pipeline import compute_sentiment_batch
                    df = compute_sentiment_batch(tickers, days_back=days_back, store_db=store_db)
                    st.session_state.sentiment_results = df
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
                finally:
                    st.session_state.sentiment_running = False

    # Display cached results (persists across reruns)
    if st.session_state.sentiment_results is not None:
        df = st.session_state.sentiment_results
        st.subheader("📊 Sentiment Results")
        st.dataframe(df, use_container_width=True)

        # Sentiment bar chart
        fig = px.bar(
            df, x="ticker", y="sentiment_score",
            color="sentiment_label",
            color_discrete_map={"positive": "#4CAF50", "negative": "#F44336", "neutral": "#9E9E9E"},
            title="Sentiment Score by Ticker",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment composition
        if len(df) > 0:
            fig_comp = go.Figure()
            for _, row in df.iterrows():
                fig_comp.add_trace(go.Bar(
                    name=row["ticker"],
                    x=["Positive %", "Negative %", "Neutral %"],
                    y=[row.get("positive_pct", 0), row.get("negative_pct", 0), row.get("neutral_pct", 0)],
                ))
            fig_comp.update_layout(title="Sentiment Composition", barmode="group", height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

        if st.button("🗑️ Clear Results"):
            st.session_state.sentiment_results = None
            st.rerun()