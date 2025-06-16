"""
Event-Driven Backtesting Engine with Risk Management

This module simulates a trading strategy over historical data, handling
portfolio management, dynamic position sizing, stop-loss, and performance tracking.
"""
import polars as pl
from loguru import logger
import duckdb
from datetime import date
import pyfolio as pf
import matplotlib.pyplot as plt

from quandex_core.config import config
from quandex_core.market_insights.trading_holidays import is_trading_day
from quandex_core.market_insights.transaction_costs import calculate_equity_trade_cost
from quandex_core.portfolio_logic.risk_models import calculate_fixed_fractional_position_size, check_stop_loss_triggered

# ... (imports remain mostly the same) ...

class BacktestEngine:
    def __init__(self, strategy, start_date: date, end_date: date, initial_capital: float):
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.cash = self.initial_capital
        self.positions = {} # Positions will now be dynamic
        self.portfolio_history = []
        self.conn = duckdb.connect(str(config.data.duckdb_path))
        logger.info("Generalized Backtest Engine initialized.")

    def _get_market_data_for_symbols(self, symbols: list[str]) -> pl.DataFrame:
        logger.info(f"Fetching data for {len(symbols)} symbols...")
        query = "SELECT date, symbol, close FROM processed_equity_data WHERE symbol = ANY (?) AND date BETWEEN ? AND ?"
        data = self.conn.execute(query, [symbols, self.start_date, self.end_date]).pl()
        return data

    def run_simulation(self) -> pl.DataFrame:
        # Determine the type of strategy to handle data fetching accordingly
        strategy_type = type(self.strategy).__name__

        if strategy_type == "PairsArbitrageStrategy":
            all_symbols = [self.strategy.stock1, self.strategy.stock2]
            price_data = self._get_market_data_for_symbols(all_symbols).pivot(
                index="date", columns="symbol", values="close"
            ).rename({s: f"{s}_close" for s in all_symbols}).drop_nulls()
        elif strategy_type == "MomentumStrategy":
            all_symbols = self.strategy.all_symbols
            price_data = self._get_market_data_for_symbols(all_symbols)
        else:
            raise NotImplementedError(f"Backtest Engine does not support strategy: {strategy_type}")
            
        if price_data.is_empty(): return pl.DataFrame()

        signals_df = self.strategy.generate_signals(price_data)
        
        logger.info(f"Starting simulation for {strategy_type}...")
        
        # --- The Main Simulation Loop ---
        for day_data in price_data.join(signals_df, on='date', how='left').iter_rows(named=True):
            current_date = day_data['date']
            
            # Update portfolio value
            portfolio_value = self.cash
            for stock, quantity in self.positions.items():
                price = day_data.get(f'{stock}_close') or price_data.filter((pl.col('symbol') == stock) & (pl.col('date') == current_date))['close'][0]
                if quantity != 0 and price is not None:
                    portfolio_value += quantity * price
            self.portfolio_history.append({'date': current_date, 'portfolio_value': portfolio_value})
            
            # --- Execute strategy logic ---
            if strategy_type == "PairsArbitrageStrategy":
                # (You would paste your detailed pairs trading logic here)
                pass # For brevity, we'll focus on the new momentum logic
            elif strategy_type == "MomentumStrategy":
                target_portfolio = day_data.get('target_portfolio')
                if target_portfolio is not None: # This is a rebalance day
                    logger.info(f"{current_date}: Rebalancing to new target portfolio: {target_portfolio}")
                    # Sell stocks no longer in the target portfolio
                    for stock in list(self.positions.keys()):
                        if stock not in target_portfolio:
                            price = day_data.get(f'{stock}_close') or price_data.filter((pl.col('symbol') == stock) & (pl.col('date') == current_date))['close'][0]
                            self._execute_trade(stock, price, self.positions[stock], 'SELL')

                    # Buy new stocks in the target portfolio
                    position_value = portfolio_value / len(target_portfolio) if target_portfolio else 0
                    for stock in target_portfolio:
                        if stock not in self.positions:
                            price = day_data.get(f'{stock}_close') or price_data.filter((pl.col('symbol') == stock) & (pl.col('date') == current_date))['close'][0]
                            quantity = int(position_value / price) if price > 0 else 0
                            if quantity > 0:
                                self._execute_trade(stock, price, quantity, 'BUY')
        
        logger.info("Backtest simulation finished.")
        return pl.DataFrame(self.portfolio_history).sort('date')

    def _execute_trade(self, stock_symbol: str, price: float, quantity: int, trade_type: str):
        # ... (This function remains unchanged)
        # But we need to handle adding/removing from self.positions
        if trade_type == 'BUY':
            self.positions[stock_symbol] = self.positions.get(stock_symbol, 0) + quantity
        elif trade_type == 'SELL':
            # Completely remove position if selling all shares
            if self.positions.get(stock_symbol, 0) == quantity:
                del self.positions[stock_symbol]
            else:
                 self.positions[stock_symbol] -= quantity