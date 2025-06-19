"""
Bharat QuanDex - Generalized, Multi-Strategy Backtesting Engine

This module simulates various trading strategies over historical data, handling
portfolio management, dynamic position sizing, and performance tracking.
"""
import polars as pl
from loguru import logger
from datetime import date
import duckdb

from quandex_core.config import config
from quandex_core.market_insights.trading_holidays import is_trading_day
from quandex_core.market_insights.transaction_costs import calculate_equity_trade_cost
from quandex_core.portfolio_logic.risk_models import calculate_fixed_fractional_position_size
# We import the strategy classes themselves for type checking
from quandex_core.strategy_blueprints.pairs_arbitrage import PairsArbitrageStrategy
from quandex_core.strategy_blueprints.momentum_surge import MomentumStrategy


class BacktestEngine:
    """
    Simulates a strategy's performance against historical market data.
    This version is generalized to handle different strategy types.
    """
    def __init__(self, strategy, start_date: date, end_date: date, initial_capital: float):
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.cash = self.initial_capital
        # --- Positions are now fully dynamic ---
        self.positions = {}
        self.portfolio_history = []
        self.conn = duckdb.connect(str(config.data.duckdb_path))
        self.strategy_type = type(self.strategy).__name__
        logger.info(f"Initialized Generalized Backtest Engine for strategy: {self.strategy_type}")

    def _fetch_data(self) -> pl.DataFrame:
        """
        Fetches and prepares market data based on the strategy's requirements.
        """
        if self.strategy_type == "PairsArbitrageStrategy":
            symbols = [self.strategy.stock1, self.strategy.stock2]
            logger.info(f"Fetching data for Pair: {symbols}")
            query = "SELECT date, symbol, close FROM processed_equity_data WHERE symbol = ANY (?) AND date BETWEEN ? AND ?"
            data = self.conn.execute(query, [symbols, self.start_date, self.end_date]).pl()
            # Pairs strategy needs 'wide' data with columns for each stock
            return data.pivot(index="date", columns="symbol", values="close").rename({s: f"{s}_close" for s in symbols}).drop_nulls()

        elif self.strategy_type == "MomentumStrategy":
            symbols = self.strategy.all_symbols
            logger.info(f"Fetching data for Momentum universe of {len(symbols)} stocks.")
            query = "SELECT date, symbol, close FROM processed_equity_data WHERE symbol = ANY (?) AND date BETWEEN ? AND ?"
            data = self.conn.execute(query, [symbols, self.start_date, self.end_date]).pl()
            # Momentum strategy needs 'long' data
            return data

        else:
            raise NotImplementedError(f"Data fetching not implemented for strategy: {self.strategy_type}")

    def _execute_trade(self, stock_symbol: str, price: float, quantity: int, trade_type: str):
        # This function is now more generic
        trade_value = price * abs(quantity)
        cost = calculate_equity_trade_cost(trade_value, is_buy=(trade_type == 'BUY'))
        
        if trade_type == 'BUY':
            if self.cash < (trade_value + cost):
                logger.warning(f"Insufficient cash for BUY of {quantity} {stock_symbol}. Skipping.")
                return
            self.cash -= (trade_value + cost)
            self.positions[stock_symbol] = self.positions.get(stock_symbol, 0) + quantity
            logger.debug(f"EXECUTED BUY: {quantity} of {stock_symbol} @ {price:.2f}")
            logger.debug(f"Trade: {trade_type} {quantity} {stock_symbol} at {price}, cash after: {self.cash}, positions: {self.positions}")
        
        elif trade_type == 'SELL':
            current_qty = self.positions.get(stock_symbol, 0)
            if quantity > current_qty:
                logger.warning(f"Trying to sell {quantity} of {stock_symbol}, but only hold {current_qty}. Selling all.")
                quantity = current_qty
            
            self.cash += (price * quantity - cost)
            self.positions[stock_symbol] -= quantity
            if self.positions[stock_symbol] == 0:
                del self.positions[stock_symbol] # Clean up position if fully sold
            logger.debug(f"EXECUTED SELL: {quantity} of {stock_symbol} @ {price:.2f}")
            logger.debug(f"Trade: {trade_type} {quantity} {stock_symbol} at {price}, cash after: {self.cash}, positions: {self.positions}")

    def run_simulation(self) -> pl.DataFrame:
        price_data = self._fetch_data()
        if price_data.is_empty(): return pl.DataFrame()

        logger.debug(f"Fetched price data shape: {price_data.shape}")
        logger.debug(f"Price data head:\n{price_data.head(5)}")

        signals_df = self.strategy.generate_signals(price_data)
        logger.info(f"Starting simulation for {self.strategy_type}...")

        logger.debug(f"Signals DataFrame shape: {signals_df.shape}")
        logger.debug(f"Signals DataFrame head:\n{signals_df.head(5)}")
        if 'signal' in signals_df.columns:
            logger.debug(f"Signal value counts: {signals_df['signal'].value_counts()}")

        # For momentum, we need a way to look up prices efficiently. Let's pivot the long data.
        if self.strategy_type == "MomentumStrategy":
            daily_prices_wide = price_data.pivot(index="date", columns="symbol", values="close")
            combined_data = daily_prices_wide.join(signals_df, on='date', how='left').sort('date')
        else: # For pairs, signals_df already has the prices
            combined_data = signals_df.sort('date')

        # --- Main Simulation Loop ---
        for day_data in combined_data.iter_rows(named=True):
            current_date = day_data['date']
            
            # Update portfolio value
            portfolio_value = self.cash
            for stock, quantity in self.positions.items():
                price = day_data.get(f'{stock}_close') or day_data.get(stock)
                if quantity != 0 and price is not None:
                    portfolio_value += quantity * price
            self.portfolio_history.append({'date': current_date, 'portfolio_value': portfolio_value})

            logger.debug(f"Date: {current_date}, Portfolio Value: {portfolio_value}, Positions: {self.positions}")

            # --- Strategy-Specific Execution Logic ---
            if self.strategy_type == "PairsArbitrageStrategy":
                signal = day_data.get('signal')
                s1 = self.strategy.stock1
                s2 = self.strategy.stock2
                s1_price = day_data.get(f"{s1}_close")
                s2_price = day_data.get(f"{s2}_close")
                position_size = int(portfolio_value * 0.5 / s1_price) if s1_price else 0  # Example: 50% capital per leg

                if signal == "BUY_SPREAD" and s1_price and s2_price:
                    # Long s1, short s2
                    if self.positions.get(s1, 0) == 0:
                        self._execute_trade(s1, s1_price, position_size, 'BUY')
                    if self.positions.get(s2, 0) == 0:
                        self._execute_trade(s2, s2_price, position_size, 'SELL')
                elif signal == "SELL_SPREAD" and s1_price and s2_price:
                    # Short s1, long s2
                    if self.positions.get(s1, 0) == 0:
                        self._execute_trade(s1, s1_price, position_size, 'SELL')
                    if self.positions.get(s2, 0) == 0:
                        self._execute_trade(s2, s2_price, position_size, 'BUY')
                elif signal == "EXIT":
                    # Close all positions
                    for stock in [s1, s2]:
                        qty = self.positions.get(stock, 0)
                        price = day_data.get(f"{stock}_close")
                        if qty > 0 and price:
                            self._execute_trade(stock, price, qty, 'SELL')
                        elif qty < 0 and price:
                            self._execute_trade(stock, price, -qty, 'BUY')
            
            elif self.strategy_type == "MomentumStrategy":
                target_portfolio = day_data.get('target_portfolio')
                if target_portfolio is not None: # This is a rebalance day
                    logger.info(f"{current_date}: Rebalancing portfolio. Target has {len(target_portfolio)} stocks.")
                    
                    # 1. Sell stocks that are no longer in the target portfolio
                    stocks_to_sell = [stock for stock in self.positions if stock not in target_portfolio]
                    for stock in stocks_to_sell:
                        price = day_data.get(stock)
                        if price is not None:
                            self._execute_trade(stock, price, self.positions[stock], 'SELL')
                    
                    # 2. Buy new stocks that are in the target portfolio but not in our holdings
                    # Using equal-weight position sizing
                    if len(target_portfolio) > 0:
                        capital_per_stock = portfolio_value / len(target_portfolio)
                        for stock in target_portfolio:
                            if stock not in self.positions:
                                price = day_data.get(stock)
                                if price is not None and price > 0:
                                    quantity = int(capital_per_stock / price)
                                    if quantity > 0:
                                        self._execute_trade(stock, price, quantity, 'BUY')

        logger.info("Backtest simulation finished.")
                # --- NEW: Return both portfolio history and the signals data ---
        return pl.DataFrame(self.portfolio_history).sort('date'),signals_df
    

# --- Self-Test Block ---
if __name__ == '__main__':
    print("--- Testing Generalized Backtest Engine ---")
    
    # Define parameters for a MOMENTUM test run
    start_test = date(2022, 1, 1)
    end_test = date(2023, 12, 31)
    capital = 100000.0
    
    # Using a small universe for a fast test
    test_universe = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS", "HINDUNILVR.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS"]
    
    # 1. Initialize the Momentum Strategy
    momentum_strategy = MomentumStrategy(
        all_symbols=test_universe,
        lookback_days=126,
        top_n_pct=0.20, # Top 20% -> 2 stocks
        rebalance_period_days=63 # Rebalance every quarter
    )
    
    # 2. Initialize the Backtest Engine with the momentum strategy
    engine = BacktestEngine(
        strategy=momentum_strategy,
        start_date=start_test,
        end_date=end_test,
        initial_capital=capital
    )
    
    # 3. Run the backtest
    try:
        final_returns = engine.run_simulation() # We call run_simulation directly
        if not final_returns.is_empty():
            print("\n--- Momentum Backtest Performance Summary ---")
            final_value = final_returns['portfolio_value'].last()
            total_return_pct = ((final_value / capital) - 1) * 100
            print(f"Final Portfolio Value: {final_value:,.2f}")
            print(f"Total Return: {total_return_pct:.2f}%")
        else:
            print("Momentum backtest did not produce results.")
            
    except Exception as e:
        logger.exception("Momentum backtest run failed:")

    print("\n--- Test Complete ---")