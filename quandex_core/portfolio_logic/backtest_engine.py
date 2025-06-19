"""
Bharat QuanDex - Generalized, Multi-Strategy Backtesting Engine (Final Version)
"""
import polars as pl
from loguru import logger
from datetime import date
import duckdb

from quandex_core.config import config
from quandex_core.market_insights.trading_holidays import is_trading_day
from quandex_core.market_insights.transaction_costs import calculate_equity_trade_cost
from quandex_core.portfolio_logic.risk_models import calculate_fixed_fractional_position_size, check_stop_loss_triggered
from quandex_core.strategy_blueprints.pairs_arbitrage import PairsArbitrageStrategy
from quandex_core.strategy_blueprints.momentum_surge import MomentumStrategy

class BacktestEngine:
    def __init__(self, strategy, start_date: date, end_date: date, initial_capital: float):
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_history = []
        self.conn = duckdb.connect(str(config.data.duckdb_path), read_only=True)
        self.strategy_type = type(self.strategy).__name__
        logger.info(f"Initialized Generalized Backtest Engine for strategy: {self.strategy_type}")

    def _get_price_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        symbols = self.strategy.all_symbols if hasattr(self.strategy, 'all_symbols') else [self.strategy.stock1, self.strategy.stock2]
        logger.info(f"Fetching data for {len(symbols)} symbols...")
        query = "SELECT date, symbol, close FROM processed_equity_data WHERE symbol = ANY (?) AND date BETWEEN ? AND ?"
        long_data = self.conn.execute(query, [symbols, self.start_date, self.end_date]).pl()
        wide_data = long_data.pivot(index="date", columns="symbol", values="close").sort("date")
        return long_data, wide_data

    def _execute_trade(self, stock_symbol: str, price: float, quantity: int, trade_type: str):
        trade_value = price * abs(quantity)
        cost = calculate_equity_trade_cost(trade_value, is_buy=(trade_type == 'BUY'))
        if trade_type == 'BUY':
            if self.cash < (trade_value + cost):
                logger.warning(f"Insufficient cash for BUY of {quantity} {stock_symbol}. Skipping.")
                return
            self.cash -= (trade_value + cost)
            self.positions[stock_symbol] = self.positions.get(stock_symbol, 0) + quantity
        elif trade_type == 'SELL':
            current_qty = self.positions.get(stock_symbol, 0)
            if quantity > current_qty:
                quantity = current_qty
            self.cash += (price * quantity - cost)
            self.positions[stock_symbol] -= quantity
            if self.positions[stock_symbol] == 0:
                del self.positions[stock_symbol]
        logger.debug(f"EXECUTED {trade_type}: {quantity} of {stock_symbol} @ {price:.2f}")

    def run_simulation(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        long_price_data, wide_price_data = self._get_price_data()
        if wide_price_data.is_empty(): return pl.DataFrame(), pl.DataFrame()

        data_for_signals = wide_price_data if self.strategy_type == "PairsArbitrageStrategy" else long_price_data
        signals_df = self.strategy.generate_signals(data_for_signals)
        
        combined_data = wide_price_data.join(signals_df, on='date', how='left')
        logger.info(f"Starting simulation for {self.strategy_type}...")
        
        # --- State variables for the loop ---
        pair_position_state = 'OUT'
        pair_entry_prices = {}

        for day_data in combined_data.iter_rows(named=True):
            current_date = day_data['date']
            
            # --- Generic Portfolio Value Calculation ---
            portfolio_value = self.cash
            for stock, quantity in self.positions.items():
                price = day_data.get(stock)
                if quantity != 0 and price is not None:
                    portfolio_value += quantity * price
            self.portfolio_history.append({'date': current_date, 'portfolio_value': portfolio_value})

            # --- Strategy-Specific Execution Logic ---
            if self.strategy_type == "PairsArbitrageStrategy":
                signal = day_data.get('signal')
                if (signal == 'EXIT') and pair_position_state != 'OUT':
                    logger.info(f"{current_date}: Exiting pairs position based on SIGNAL.")
                    for stock, quantity in self.positions.copy().items():
                        self._execute_trade(stock, day_data[f'{stock}_close'], abs(quantity), 'SELL' if quantity > 0 else 'BUY')
                    pair_position_state = 'OUT'
                elif signal == 'BUY_SPREAD' and pair_position_state == 'OUT':
                    s1_qty = calculate_fixed_fractional_position_size(portfolio_value, day_data[f'{self.strategy.stock1}_close'])
                    if s1_qty > 0:
                        self._execute_trade(self.strategy.stock1, day_data[f'{self.strategy.stock1}_close'], s1_qty, 'BUY')
                        self._execute_trade(self.strategy.stock2, day_data[f'{self.strategy.stock2}_close'], s1_qty, 'SELL')
                        pair_position_state = 'LONG_SPREAD'
                elif signal == 'SELL_SPREAD' and pair_position_state == 'OUT':
                    s1_qty = calculate_fixed_fractional_position_size(portfolio_value, day_data[f'{self.strategy.stock1}_close'])
                    if s1_qty > 0:
                        self._execute_trade(self.strategy.stock1, day_data[f'{self.strategy.stock1}_close'], s1_qty, 'SELL')
                        self._execute_trade(self.strategy.stock2, day_data[f'{self.strategy.stock2}_close'], s1_qty, 'BUY')
                        pair_position_state = 'SHORT_SPREAD'

            elif self.strategy_type == "MomentumStrategy":
                target_portfolio = day_data.get('target_portfolio')
                if target_portfolio is not None: # This is a rebalance day
                    logger.info(f"{current_date}: Rebalancing portfolio to {len(target_portfolio)} target stocks.")
                    
                    stocks_to_sell = [stock for stock in self.positions if stock not in target_portfolio]
                    for stock in stocks_to_sell:
                        if day_data.get(stock) is not None:
                            self._execute_trade(stock, day_data[stock], self.positions[stock], 'SELL')
                    
                    if len(target_portfolio) > 0:
                        capital_per_stock = portfolio_value / len(target_portfolio)
                        for stock in target_portfolio:
                            if stock not in self.positions:
                                price = day_data.get(stock)
                                if price is not None and price > 0:
                                    quantity = int(capital_per_stock / price)
                                    if quantity > 0: self._execute_trade(stock, price, quantity, 'BUY')

        logger.info("Backtest simulation finished.")
        return pl.DataFrame(self.portfolio_history).sort('date'), signals_df