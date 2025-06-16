"""
Bharat QuanDex - The Quant's Command Center
Final Integrated Version

This Streamlit application provides an interactive user interface for
running backtests, screening for momentum stocks, and performing detailed
statistical analysis on stock pairs.
"""
import streamlit as st
import polars as pl
from datetime import date
from loguru import logger
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# --- Setup & Configuration ---
# Use a non-interactive backend for thread safety in Streamlit
matplotlib.use("Agg") 

# Add project root to path to allow imports from our package
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import all necessary modules from our backend
from quandex_core.config import config
from quandex_core.strategy_blueprints.pairs_arbitrage import PairsArbitrageStrategy
from quandex_core.strategy_blueprints.momentum_surge import run_momentum_screen
from quandex_core.portfolio_logic.backtest_engine import BacktestEngine
import pyfolio as pf

# --- Page Configuration (must be the first st command) ---
st.set_page_config(
    page_title="Bharat QuanDex",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main App Title ---
st.title("🇮🇳 Bharat QuanDex")
st.caption("Quantitative Analysis & Backtesting Dashboard")

# --- Cached Data & Backend Functions ---
@st.cache_data(ttl=3600)  # Cache symbols for 1 hour
def get_available_symbols():
    """Gets a list of all available stock symbols from the database."""
    try:
        symbols = config.data.conn.execute("SELECT DISTINCT symbol FROM processed_equity_data ORDER BY symbol").pl()['symbol'].to_list()
        # Return symbols if found, otherwise use the robust fallback list from config
        return symbols if symbols else config.market.fallback_symbols
    except Exception as e:
        logger.error(f"Error fetching symbols, using fallback: {e}")
        return config.market.fallback_symbols

@st.cache_data(max_entries=10) # Cache the last 10 backtest runs
def run_pairs_backtest_cached(_params):
    """Runs the pairs trading backtest and caches the result."""
    s1, s2, start, end, win, z, cap = _params
    try:
        strategy = PairsArbitrageStrategy(s1, s2, win, z)
        engine = BacktestEngine(strategy, start, end, cap)
        history_df = engine.run_simulation()
        
        if history_df.is_empty():
            return None, "No overlapping trading data found for the selected stocks and date range."
        
        returns = history_df.to_pandas().set_index('date')['portfolio_value'].pct_change().dropna()
        
        if returns.empty:
            return None, "No trades were executed. Try adjusting the strategy parameters (e.g., a lower Z-Score)."

        # Generate the pyfolio plot and return the figure object
        pf.create_full_tear_sheet(returns)
        fig = plt.gcf()
        plt.close(fig) # Important to close the global plot
        return fig, None
    except Exception as e:
        logger.exception("Pairs backtest failed:")
        return None, f"An unexpected error occurred: {e}"

# --- UI TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Pairs Arbitrage Backtester", "🔥 Momentum Screener", "🛠️ Analysis Tools"])


# ==============================================================================
# === TAB 1: PAIRS ARBITRAGE BACKTESTER ========================================
# ==============================================================================
with tab1:
    st.header("Pairs Arbitrage Strategy Backtest")

    # The sidebar is shared, but we can have specific controls per tab if needed
    with st.sidebar:
        st.header("Backtester Configuration")
        all_symbols_sb = get_available_symbols()
        s1_idx = all_symbols_sb.index("RELIANCE.NS") if "RELIANCE.NS" in all_symbols_sb else 0
        s1 = st.selectbox("Select Stock 1", all_symbols_sb, index=s1_idx, key="s1_backtest")
        
        s2_options = [s for s in all_symbols_sb if s != s1]
        s2_default = "TCS.NS" if "TCS.NS" in s2_options else s2_options[0] if s2_options else None
        s2_idx = s2_options.index(s2_default) if s2_default else 0
        s2 = st.selectbox("Select Stock 2", s2_options, index=s2_idx, key="s2_backtest")

        start = st.date_input("Start Date", date(2022, 1, 1), key="start_backtest")
        end = st.date_input("End Date", date(2023, 12, 31), key="end_backtest")
        win = st.slider("Spread Window (Days)", 10, 120, 60, 5, key="win_backtest")
        z = st.slider("Z-Score Entry", 1.0, 3.0, 2.0, 0.1, key="z_backtest")
        cap = st.number_input("Initial Capital (INR)", 10000, 10000000, 100000, 10000, key="cap_backtest")
    
    if st.button("🚀 Run Pairs Backtest", type="primary", use_container_width=True):
        params = (s1, s2, start, end, win, z, cap)
        with st.spinner("Running Pairs Backtest... This can take a moment."):
            fig_result, err_msg = run_pairs_backtest_cached(params)
            # Store results in session state to persist them
            st.session_state.pairs_results = (fig_result, err_msg)
    
    # Display results if they exist in the session state
    if 'pairs_results' in st.session_state:
        fig_result, err_msg = st.session_state.pairs_results
        if err_msg:
            st.error(err_msg, icon="🚨")
        elif fig_result:
            st.pyplot(fig_result)
        else:
            st.warning("Backtest ran, but something went wrong and no plot was generated.")


# ==============================================================================
# === TAB 2: MOMENTUM SCREENER =================================================
# ==============================================================================
with tab2:
    st.header("Momentum Surge Screener")
    st.write("Find the top stocks based on their historical rate-of-change (ROC).")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    lookback = col1.slider("Momentum Lookback (Trading Days)", 21, 252, 126, 21)
    top_n = col2.number_input("Number of Stocks to Show", 5, 50, 20)
    
    if col3.button("📈 Run Screener", type="primary", use_container_width=True):
        with st.spinner(f"Scanning market for top {top_n} stocks..."):
            st.session_state.momentum_results = run_momentum_screen(lookback, top_n)

    if 'momentum_results' in st.session_state:
        results = st.session_state.momentum_results
        if results is not None and not results.is_empty():
            st.success(f"Found {len(results)} top momentum stocks.")
            st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.error("Screener returned no results. Please ensure the database is populated with data.", icon="🚨")


# ==============================================================================
# === TAB 3: ANALYSIS TOOLS (ENHANCED VERSION) =================================
# ==============================================================================
with tab3:
    st.header("Cointegration & Pair Analysis Tool")
    st.write("Perform a deep statistical analysis on a stock pair to evaluate its suitability for a mean-reversion strategy.")

    all_symbols_tools = get_available_symbols()
    tool_col1, tool_col2 = st.columns(2)
    
    with tool_col1:
        s1_coint_idx = all_symbols_tools.index("ICICIBANK.NS") if "ICICIBANK.NS" in all_symbols_tools else 0
        coint_s1 = st.selectbox("Select Stock A", all_symbols_tools, index=s1_coint_idx, key="coint_s1")
        coint_start = st.date_input("Analysis Start Date", date(2022, 1, 1), key="coint_start")

    with tool_col2:
        s2_coint_options = [s for s in all_symbols_tools if s != coint_s1]
        s2_coint_default = "HDFCBANK.NS" if "HDFCBANK.NS" in s2_coint_options else s2_coint_options[0] if s2_coint_options else None
        s2_coint_idx = s2_coint_options.index(s2_coint_default) if s2_coint_default else 0
        coint_s2 = st.selectbox("Select Stock B", s2_coint_options, index=s2_coint_idx, key="coint_s2")
        coint_end = st.date_input("Analysis End Date", date(2023, 12, 31), key="coint_end")

    if st.button("🔬 Perform Full Analysis", type="primary", use_container_width=True):
        if not coint_s1 or not coint_s2:
            st.warning("Please select two different stocks.")
        else:
            with st.spinner(f"Analyzing pair {coint_s1} / {coint_s2}..."):
                try:
                    # 1. Fetch Data
                    query = "SELECT date, symbol, close FROM processed_equity_data WHERE symbol IN (?, ?) AND date BETWEEN ? AND ?"
                    data = config.data.conn.execute(query, [coint_s1, coint_s2, coint_start, coint_end]).pl()
                    
                    if data.height < 60:
                        st.error("Not enough data (< 60 days) in the selected range for a meaningful analysis.", icon="🚨")
                    else:
                        # 2. Pivot & Analyze
                        df1 = data.filter(pl.col("symbol") == coint_s1).rename({"close": f"{coint_s1}_close"})
                        df2 = data.filter(pl.col("symbol") == coint_s2).rename({"close": f"{coint_s2}_close"})
                        pivoted_data = df1.join(df2, on="date", how="inner").sort("date")
                        
                        analyzer = PairsArbitrageStrategy(coint_s1, coint_s2)
                        analyzer.perform_full_analysis(pivoted_data)

                        # Store results for display
                        st.session_state.analysis_results = {
                            'analyzer': analyzer,
                            'plot_data': pivoted_data
                        }

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    logger.exception("Cointegration check failed from dashboard:")

    # Display analysis results if they exist in session state
    if 'analysis_results' in st.session_state:
        analyzer = st.session_state.analysis_results['analyzer']
        plot_data = st.session_state.analysis_results['plot_data']

        st.subheader("Analysis Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        # Before formatting, we ensure we have a plain Python float.
        # The .item() method is the safest way to extract a single value from a Polars object.
        # We also add a check for None in case the calculation failed.
        correlation_value = analyzer.correlation.item() if hasattr(analyzer.correlation, 'item') else analyzer.correlation
        
        summary_col1.metric("Cointegration P-Value", f"{analyzer.cointegration_p_value:.4f}")
        summary_col2.metric("Price Correlation", f"{correlation_value:.4f}" if correlation_value is not None else "N/A")
        summary_col3.metric("Reversion Half-Life", f"{analyzer.half_life:.1f} days" if analyzer.half_life else "N/A")

        if analyzer.is_cointegrated:
            st.success(f"Conclusion: The pair IS LIKELY cointegrated (p-value < 0.05).", icon="✅")
        else:
            st.warning(f"Conclusion: The pair is NOT likely cointegrated (p-value > 0.05).", icon="⚠️")

        st.subheader("Normalized Price Series")
        plot_df = plot_data.with_columns(
            (pl.col(f"{analyzer.stock1}_close") / plot_data[f'{analyzer.stock1}_close'][0]).alias("Stock A (Normalized)"),
            (pl.col(f"{analyzer.stock2}_close") / plot_data[f'{analyzer.stock2}_close'][0]).alias("Stock B (Normalized)")
        )
        st.line_chart(plot_df.to_pandas().set_index("date")[["Stock A (Normalized)", "Stock B (Normalized)"]])