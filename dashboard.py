"""
Bharat QuanDex - Multi-Strategy Command Center

This Streamlit application provides an interactive UI for backtesting
multiple quantitative strategies, including Pairs Arbitrage and Momentum.
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
matplotlib.use("Agg")
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from quandex_core.config import config
from quandex_core.strategy_blueprints.pairs_arbitrage import PairsArbitrageStrategy
from quandex_core.strategy_blueprints.momentum_surge import MomentumStrategy, run_momentum_screen
from quandex_core.portfolio_logic.backtest_engine import BacktestEngine
import pyfolio as pf
import duckdb

# --- Helper & Backend Functions ---
def get_db_connection():
    return duckdb.connect(str(config.data.duckdb_path), read_only=True)

@st.cache_data(ttl=3600)
def get_available_symbols():
    try:
        with get_db_connection() as conn:
            symbols = conn.execute("SELECT DISTINCT symbol FROM processed_equity_data ORDER BY symbol").pl()['symbol'].to_list()
        return symbols if symbols else config.market.fallback_symbols
    except Exception: 
        return config.market.fallback_symbols

@st.cache_data(max_entries=10)
def run_backtest_cached(strategy_choice, strategy_params, backtest_params):
    """Run backtest with given config and return plotly figure and error message if any."""
    try:
        # Convert frozenset parameters back to dictionaries
        strategy_params = dict(strategy_params)
        backtest_params = dict(backtest_params)
        
        # Extract parameters
        start = backtest_params['start']
        end = backtest_params['end']
        cap = backtest_params['capital']

        # Instantiate the correct strategy based on user choice
        if strategy_choice == "Pairs Arbitrage":
            # FIX: Use correct parameter names for PairsArbitrageStrategy
            strategy = PairsArbitrageStrategy(
                stock1_symbol=strategy_params['s1'],
                stock2_symbol=strategy_params['s2'],
                window_size=strategy_params['window_size'],
                z_entry_threshold=strategy_params['z_entry_threshold']
            )
        elif strategy_choice == "Momentum":
            # Ensure universe has at least 1 stock
            universe = strategy_params['universe']
            if len(universe) == 0:
                return None, "Momentum strategy requires at least 1 stock in the universe"
                
            strategy = MomentumStrategy(
                all_symbols=universe,
                lookback_days=strategy_params['lookback'],
                top_n_pct=strategy_params['top_pct'],
                rebalance_period_days=strategy_params['rebalance_days']
            )
        else:
            return None, f"Unknown strategy: {strategy_choice}"

        engine = BacktestEngine(strategy, start, end, cap)
        history_df = engine.run_simulation()
        
        if history_df.is_empty(): 
            return None, "No data for this period or symbols."
        
        returns = history_df.to_pandas().set_index('date')['portfolio_value'].pct_change().dropna()
        if returns.empty: 
            return None, "No trades were executed."
        
        pf.create_full_tear_sheet(returns)
        fig = plt.gcf()
        plt.close(fig)
        return fig, None
    except Exception as e:
        logger.exception(f"{strategy_choice} backtest failed:")
        return None, f"An unexpected error occurred: {e}"

# --- UI Rendering ---
st.set_page_config(page_title="Bharat QuanDex", page_icon="🇮🇳", layout="wide")
st.title("🇮🇳 Bharat QuanDex")
st.caption("Multi-Strategy Quantitative Analysis Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.header("Strategy Configuration")
    
    # --- 1. Master Strategy Selector ---
    strategy_choice = st.selectbox("Select Strategy", ["Pairs Arbitrage", "Momentum"])
    
    # --- 2. Dynamic UI based on selection ---
    if strategy_choice == "Pairs Arbitrage":
        st.subheader("Pairs Arbitrage Settings")
        all_symbols = get_available_symbols()
        
        # Find safe indices for defaults
        reliance_index = all_symbols.index("RELIANCE.NS") if "RELIANCE.NS" in all_symbols else 0
        tcs_index = all_symbols.index("TCS.NS") if "TCS.NS" in all_symbols else 0
        
        s1 = st.selectbox("Stock 1", all_symbols, index=reliance_index)
        s2_options = [s for s in all_symbols if s != s1]
        
        # Find safe index for TCS in filtered list
        tcs_safe_index = s2_options.index("TCS.NS") if "TCS.NS" in s2_options else 0
        s2 = st.selectbox("Stock 2", s2_options, index=tcs_safe_index)
        
        win = st.slider("Spread Window (Days)", 10, 120, 60)
        z = st.slider("Z-Score Entry", 1.0, 3.0, 2.0, 0.1)
        
        strategy_params = {'s1': s1, 's2': s2, 'window_size': win, 'z_entry_threshold': z}

    elif strategy_choice == "Momentum":
        st.subheader("Momentum Settings")
        available_symbols = get_available_symbols()
        
        # Create filtered default list - removed LT.NS which was causing errors
        default_universe = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
            "ICICIBANK.NS", "ITC.NS", "HINDUNILVR.NS", "SBIN.NS", "AXISBANK.NS"
        ]
        valid_defaults = [s for s in default_universe if s in available_symbols]
        
        momentum_universe = st.multiselect(
            "Select Stock Universe",
            options=available_symbols,
            default=valid_defaults,
            key="momentum_universe"  # Unique key to avoid widget conflicts
        )
        
        # Fallback to valid defaults if nothing selected
        if not momentum_universe:
            st.warning("No stocks selected! Using default universe.")
            momentum_universe = valid_defaults
            
        lookback = st.slider("Momentum Lookback (Days)", 21, 252, 126)
        top_pct = st.slider("Top % of Universe to Buy", 0.1, 0.5, 0.2, 0.05)
        rebal = st.slider("Rebalance Period (Trading Days)", 5, 63, 21)

        # FIX: Convert list to tuple for hashability
        strategy_params = {
            'universe': tuple(momentum_universe),  # Convert list to tuple
            'lookback': lookback, 
            'top_pct': top_pct, 
            'rebalance_days': rebal
        }

    st.subheader("General Settings")
    start = st.date_input("Start Date", date(2022, 1, 1))
    end = st.date_input("End Date", date(2023, 12, 31))
    cap = st.number_input("Initial Capital (INR)", 10000, 10000000, 100000, 10000)
    
    backtest_params = {
        'start': start, 
        'end': end, 
        'capital': cap
    }
    
    run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# --- Main Content Area ---
st.header("Backtest Performance")

if run_button:
    # Create immutable keys for caching
    # FIX: Wrap in tuple() to handle date objects
    strategy_key = frozenset(tuple(strategy_params.items()))
    backtest_key = frozenset(tuple(backtest_params.items()))
    
    with st.spinner(f"Running {strategy_choice} backtest..."):
        try:
            fig_result, err_msg = run_backtest_cached(
                strategy_choice,
                strategy_key,
                backtest_key
            )
            st.session_state.results = (fig_result, err_msg)
        except Exception as e:
            st.error(f"Backtest initialization failed: {str(e)}", icon="🚨")
            st.session_state.results = (None, str(e))

# Display results if they exist in the session state
if 'results' in st.session_state:
    fig_result, err_msg = st.session_state.results
    if err_msg:
        st.error(err_msg, icon="🚨")
    elif fig_result:
        st.pyplot(fig_result)
        st.success("Backtest analysis complete!", icon="✅")
    else:
        st.warning("Backtest ran, but no results were generated.")
else:
    st.info("Configure your strategy in the sidebar and click 'Run Backtest'.", icon="👈")