"""
Bharat QuanDex - Generalized, Multi-Strategy Backtesting Engine (Production Version)
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
    """
    Generalized backtesting engine supporting multiple strategies (Pairs Arbitrage, Momentum, etc).
    Handles robust data access, trade execution, and portfolio tracking.
    """
    def __init__(self, strategy, start_date: date, end_date: date, initial_capital: float):
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_history = []
        self.strategy_type = type(self.strategy).__name__
        logger.info(f"Initialized Backtest Engine for strategy: {self.strategy_type}")

    def _get_price_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Fetches price data for the strategy's universe using the single, shared database connection.
        """
        symbols = self.strategy.all_symbols if hasattr(self.strategy, 'all_symbols') else [self.strategy.stock1, self.strategy.stock2]
        logger.info(f"Fetching data for {len(symbols)} symbols using the shared connection.")

        # Use the single, globally shared connection from the config.
        query = "SELECT date, symbol, close FROM processed_equity_data WHERE symbol = ANY (?) AND date BETWEEN ? AND ?"
        long_data = config.data.conn.execute(query, [symbols, self.start_date, self.end_date]).pl()

        if long_data.is_empty():
            logger.warning("No price data found for selected universe/date range.")
            return pl.DataFrame(), pl.DataFrame()
        
        # Using correct, modern Polars pivot syntax.
        wide_data = long_data.pivot(index="date", columns="symbol", values="close").sort("date")
        
        # For pairs strategies, rename columns to {symbol}_close if needed
        if self.strategy_type == "PairsArbitrageStrategy":
            rename_map = {col: f"{col}_close" for col in wide_data.columns if col != "date"}
            wide_data = wide_data.rename(rename_map)
        
        return long_data, wide_data

    def _execute_trade(self, stock_symbol: str, price: float, quantity: int, trade_type: str):
        """
        Executes a buy or sell trade, updating cash and positions. Ignores invalid trades.
        """
        if price is None or quantity <= 0:
            logger.debug(f"Trade skipped: {trade_type} {quantity} {stock_symbol} @ {price}")
            return
        trade_value = price * quantity
        cost = calculate_equity_trade_cost(trade_value, is_buy=(trade_type == 'BUY'))
        if trade_type == 'BUY':
            if self.cash < (trade_value + cost):
                logger.warning(f"Insufficient cash for BUY of {quantity} {stock_symbol}. Skipping.")
                return
            self.cash -= (trade_value + cost)
            self.positions[stock_symbol] = self.positions.get(stock_symbol, 0) + quantity
        elif trade_type == 'SELL':
            current_qty = self.positions.get(stock_symbol, 0)
            if current_qty == 0:
                logger.warning(f"No position to SELL for {stock_symbol}. Skipping.")
                return
            if quantity > current_qty:
                quantity = current_qty
            self.cash += (price * quantity - cost)
            self.positions[stock_symbol] = current_qty - quantity
            if self.positions[stock_symbol] == 0:
                del self.positions[stock_symbol]
        logger.debug(f"EXECUTED {trade_type}: {quantity} of {stock_symbol} @ {price:.2f}")

    def run_simulation(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Runs the backtest simulation for the selected strategy.
        Returns:
            - Portfolio value history (pl.DataFrame)
            - Signals DataFrame (pl.DataFrame)
        """
        long_price_data, wide_price_data = self._get_price_data()
        if (self.strategy_type == "PairsArbitrageStrategy" and wide_price_data.is_empty()) or \
           (self.strategy_type == "MomentumStrategy" and long_price_data.is_empty()):
            logger.warning("No data available for simulation.")
            return pl.DataFrame(), pl.DataFrame()

        # Prepare data for signals
        data_for_signals = wide_price_data if self.strategy_type == "PairsArbitrageStrategy" else long_price_data
        signals_df = self.strategy.generate_signals(data_for_signals)
        # Join on date for simulation
        combined_data = data_for_signals.join(signals_df, on='date', how='left').sort("date")
        logger.info(f"Starting simulation for {self.strategy_type}...")

        # State for pairs
        pair_position_state = 'OUT'
        pair_entry_prices = {}

        for day_data in combined_data.iter_rows(named=True):
            current_date = day_data['date']
            # Portfolio value calculation
            portfolio_value = self.cash
            for stock, quantity in self.positions.items():
                price = day_data.get(stock)
                if quantity != 0 and price is not None:
                    portfolio_value += quantity * price
            self.portfolio_history.append({'date': current_date, 'portfolio_value': portfolio_value})

            # --- Stop-Loss for Pairs ---
            if self.strategy_type == "PairsArbitrageStrategy" and pair_position_state != 'OUT':
                stop_loss_hit = False
                s1, s2 = self.strategy.stock1, self.strategy.stock2
                s1_price = day_data.get(f"{s1}_close")
                s2_price = day_data.get(f"{s2}_close")
                if s1_price is not None and s2_price is not None:
                    if pair_position_state == 'LONG_SPREAD':
                        s1_stop = check_stop_loss_triggered(pair_entry_prices.get(s1, s1_price), s1_price, 'LONG')
                        s2_stop = check_stop_loss_triggered(pair_entry_prices.get(s2, s2_price), s2_price, 'SHORT')
                        stop_loss_hit = s1_stop or s2_stop
                    elif pair_position_state == 'SHORT_SPREAD':
                        s1_stop = check_stop_loss_triggered(pair_entry_prices.get(s1, s1_price), s1_price, 'SHORT')
                        s2_stop = check_stop_loss_triggered(pair_entry_prices.get(s2, s2_price), s2_price, 'LONG')
                        stop_loss_hit = s1_stop or s2_stop
                if stop_loss_hit:
                    day_data['signal'] = 'EXIT'

            # --- Strategy-Specific Execution ---
            if self.strategy_type == "PairsArbitrageStrategy":
                signal = day_data.get('signal')
                s1, s2 = self.strategy.stock1, self.strategy.stock2
                s1_price = day_data.get(f"{s1}_close")
                s2_price = day_data.get(f"{s2}_close")
                if (signal == 'EXIT') and pair_position_state != 'OUT':
                    logger.info(f"{current_date}: Exiting pairs position.")
                    for stock, quantity in self.positions.copy().items():
                        price = day_data.get(f"{stock}_close")
                        if price is not None:
                            self._execute_trade(stock, price, abs(quantity), 'SELL' if quantity > 0 else 'BUY')
                    pair_position_state = 'OUT'
                elif signal == 'BUY_SPREAD' and pair_position_state == 'OUT' and s1_price and s2_price:
                    qty = calculate_fixed_fractional_position_size(portfolio_value, s1_price)
                    if qty > 0:
                        self._execute_trade(s1, s1_price, qty, 'BUY'); pair_entry_prices[s1] = s1_price
                        self._execute_trade(s2, s2_price, qty, 'SELL'); pair_entry_prices[s2] = s2_price
                        pair_position_state = 'LONG_SPREAD'
                elif signal == 'SELL_SPREAD' and pair_position_state == 'OUT' and s1_price and s2_price:
                    qty = calculate_fixed_fractional_position_size(portfolio_value, s1_price)
                    if qty > 0:
                        self._execute_trade(s1, s1_price, qty, 'SELL'); pair_entry_prices[s1] = s1_price
                        self._execute_trade(s2, s2_price, qty, 'BUY'); pair_entry_prices[s2] = s2_price
                        pair_position_state = 'SHORT_SPREAD'

            elif self.strategy_type == "MomentumStrategy":
                target_portfolio = day_data.get('target_portfolio')
                if target_portfolio is not None:
                    logger.info(f"{current_date}: Rebalancing portfolio to {target_portfolio}")
                    # Sell stocks not in target
                    for stock_to_sell in [s for s in self.positions if s not in target_portfolio]:
                        price = day_data.get(stock_to_sell)
                        if price is not None:
                            self._execute_trade(stock_to_sell, price, self.positions[stock_to_sell], 'SELL')
                    # Buy new stocks
                    if target_portfolio:
                        cap_per_stock = portfolio_value / len(target_portfolio)
                        for stock_to_buy in [s for s in target_portfolio if s not in self.positions]:
                            price = day_data.get(stock_to_buy)
                            if price is not None and price > 0:
                                qty = int(cap_per_stock / price)
                                if qty > 0:
                                    self._execute_trade(stock_to_buy, price, qty, 'BUY')

        logger.info("Backtest simulation finished.")
        return pl.DataFrame(self.portfolio_history).sort('date'), signals_df