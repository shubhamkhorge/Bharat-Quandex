import polars as pl
from loguru import logger
from datetime import date
import numpy as np

from quandex_core.config import config

class MomentumStrategy:
    """
    A strategy that buys a portfolio of the top N momentum stocks from a
    given universe and rebalances on a periodic basis.
    """
    def __init__(self, all_symbols: list[str], lookback_days: int = 126, top_n_pct: float = 0.10, rebalance_period_days: int = 21):
        """
        Initializes the momentum strategy.

        Args:
            all_symbols (list[str]): The universe of symbols to consider for the strategy.
            lookback_days (int): The period for calculating the rate-of-change (momentum).
            top_n_pct (float): The percentage of the universe to go long on (e.g., 0.10 for top 10%).
            rebalance_period_days (int): How often to rebalance the portfolio (e.g., 21 for monthly).
        """
        self.all_symbols = all_symbols
        self.lookback = lookback_days
        # Ensure we always have at least 1 stock in our portfolio
        self.top_n = max(1, int(len(all_symbols) * top_n_pct))
        self.rebalance_period = rebalance_period_days
        logger.info(f"Initialized MomentumStrategy: Target portfolio size is Top {self.top_n} stocks from a universe of {len(all_symbols)}.")

    def generate_signals(self, market_data: pl.DataFrame) -> pl.DataFrame:
        logger.info("Generating momentum signals for the entire period...")

        # 1. Calculate momentum for all stocks in the universe
        momentum_df = market_data.sort("date").with_columns(
            (
                (pl.col("close") - pl.col("close").shift(self.lookback)) /
                pl.col("close").shift(self.lookback)
            ).over("symbol").alias("momentum_roc")
        )

        # 2. Determine the rebalance dates (e.g., every 21 trading days)
        all_dates = market_data['date'].unique().sort()
        rebalance_dates = all_dates[::self.rebalance_period]

        # 3. For each rebalance date, find the top N stocks
        target_portfolios = []
        for rebal_date in rebalance_dates:
            # Get a snapshot of all stocks on that specific day
            current_snapshot = momentum_df.filter(pl.col('date') == rebal_date).drop_nulls('momentum_roc')
            
            # Rank by momentum and get the list of top symbols
            target_portfolio_list = (
                current_snapshot.sort('momentum_roc', descending=True)
                .head(self.top_n)
                ['symbol']
                .to_list()
            )
            
            target_portfolios.append({
                'date': rebal_date,
                'target_portfolio': target_portfolio_list
            })
        
        if not target_portfolios:
            logger.warning("Could not generate any target portfolios. Check data and date range.")
            return pl.DataFrame({'date': [], 'target_portfolio': []})

        rebalance_signals_df = pl.DataFrame(target_portfolios).with_columns(pl.col("date").cast(pl.Date))
        
        # 4. Create a full signal series by forward-filling the target portfolio
        signals_df = (
            all_dates.to_frame(name="date")
            .join(rebalance_signals_df, on='date', how='left')
            .with_columns(
                pl.col("target_portfolio").fill_null(strategy="forward")
            )
            .filter(pl.col("target_portfolio").is_not_null())
        )
        
        logger.info(f"Successfully generated {len(rebalance_dates)} rebalance signals.")
        return signals_df


def run_momentum_screen(lookback_days: int = 126, top_n: int = 20) -> pl.DataFrame:
    """
    Scans all processed equity data to find the top N stocks with the
    highest rate of change (momentum) over a given lookback period.

    Args:
        lookback_days (int): The number of trading days to look back for the momentum calculation.
                             Approximately 126 days = 6 months.
        top_n (int): The number of top-performing stocks to return.

    Returns:
        pl.DataFrame: A DataFrame containing the top N stocks, sorted by
                      their momentum, including their last close price and the
                      momentum value. Returns an empty DataFrame on error.
    """
    logger.info(f"Running momentum screen with {lookback_days}-day lookback for top {top_n} stocks.")
    try:
        conn = config.data.conn
        
        # Query all necessary data. We only need symbol, date, and close.
        all_data = conn.execute("SELECT symbol, date, close FROM processed_equity_data").pl()

        if all_data.is_empty():
            logger.warning("Processed equity data is empty. Cannot run momentum screen.")
            return pl.DataFrame()

        # Calculate Rate of Change (Momentum) using Polars window functions
        # This is significantly faster than looping.
        momentum_df = all_data.sort("date").with_columns(
            (
                (pl.col("close") - pl.col("close").shift(lookback_days)) /
                pl.col("close").shift(lookback_days)
            ).over("symbol").alias("momentum_roc")
        )

        # Get the latest momentum value for each stock
        latest_momentum = (
            momentum_df.drop_nulls("momentum_roc")
            .group_by("symbol")
            .last() # Gets the last row for each symbol, which is the most recent
        )

        # Sort by momentum and take the top N
        top_performers = (
            latest_momentum.sort("momentum_roc", descending=True)
            .head(top_n)
            .select(["symbol", "date", "close", "momentum_roc"]) # Select and reorder columns
        )
        
        logger.info(f"Successfully screened {latest_momentum.height} symbols and found top {top_n}.")
        return top_performers

    except Exception as e:
        logger.exception(f"An error occurred in the momentum screen: {e}")
        return pl.DataFrame()

# --- Self-Test Block ---
if __name__ == '__main__':
    print("--- Testing Momentum Strategy Class ---")
    
    # Create mock universe and data
    mock_universe = [f"STOCK_{i}" for i in range(50)]
    dates = pl.date_range(date(2023, 1, 1), date(2023, 6, 30), "1d", eager=True)
    
    # Create mock data where some stocks trend up and others down
    mock_data_list = []
    for i, stock in enumerate(mock_universe):
        # First 10 stocks are high momentum
        trend = (i - 25) * 0.001 
        prices = 100 + pl.arange(0, len(dates), eager=True) * trend + np.random.normal(0, 0.5, len(dates))
        mock_data_list.append(pl.DataFrame({'date': dates, 'symbol': stock, 'close': prices}))
    
    full_mock_data = pl.concat(mock_data_list)
    
    # Initialize and run the strategy
    strategy = MomentumStrategy(
        all_symbols=mock_universe,
        lookback_days=60,
        top_n_pct=0.10, # Top 10% -> 5 stocks
        rebalance_period_days=30
    )
    
    signals = strategy.generate_signals(full_mock_data)
    
    print(f"\nGenerated a total of {signals.height} daily signals.")
    print("Showing the portfolio composition on a few rebalance days:")
    
    # Corrected line: use unique() instead of drop_duplicates()
    rebalance_signals = signals.unique(subset=['target_portfolio'])
    print(rebalance_signals.head(3))

    print("\n--- Test Complete ---")