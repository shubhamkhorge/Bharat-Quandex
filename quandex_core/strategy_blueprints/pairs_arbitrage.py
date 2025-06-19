"""
Pairs Arbitrage Strategy Blueprint

This module provides a class to comprehensively analyze a pair of securities
for cointegration and generate trading signals based on their price spread.
"""

import polars as pl
from loguru import logger
from statsmodels.tsa.stattools import adfuller
import numpy as np

class PairsArbitrageStrategy:
    """
    Encapsulates the logic and statistical analysis for a pairs trading strategy.
    """
    def __init__(self, stock1_symbol: str, stock2_symbol: str, window_size: int = 60, z_entry_threshold: float = 2.0, z_exit_threshold: float = 0.5):
        """
        Initializes the strategy with its parameters.
        """
        self.stock1 = stock1_symbol
        self.stock2 = stock2_symbol
        self.window = window_size
        self.z_entry = z_entry_threshold
        self.z_exit = z_exit_threshold
        
        # --- NEW: Attributes to store rich analysis results ---
        self.is_cointegrated = None
        self.cointegration_p_value = None
        self.correlation = None
        self.half_life = None
        self.latest_z_score = None
        # --------------------------------------------------------

        logger.info(f"Initialized PairsArbitrageStrategy for {self.stock1}-{self.stock2}")

    def _calculate_half_life(self, spread: pl.Series) -> float:
        """
        Calculates the half-life of mean reversion for a given time series.
        """
        # Create a lagged version of the spread
        spread_lag = spread.shift(1).drop_nulls()
        delta_spread = (spread - spread.shift(1)).drop_nulls()
        
        # The regression model requires a constant, so we add a column of ones.
        # We need to work with numpy for the Ordinary Least Squares regression.
        df = pl.DataFrame({'delta_spread': delta_spread, 'spread_lag': spread_lag})
        X = df['spread_lag'].to_numpy()
        Y = df['delta_spread'].to_numpy()
        X = np.column_stack((np.ones(len(X)), X)) # Add constant for intercept
        
        # Calculate the regression coefficient (lambda)
        try:
            # Using numpy's least squares for simplicity and speed
            regression_slope = np.linalg.lstsq(X, Y, rcond=None)[0][1]
            # The half-life is log(2) / -lambda
            half_life = np.log(2) / -regression_slope
            return half_life
        except np.linalg.LinAlgError:
            return 0.0 # Return 0 if regression fails

    def perform_full_analysis(self, prices_df: pl.DataFrame):
        """
        Performs a comprehensive analysis of the pair, calculating cointegration,
        correlation, half-life, and other key statistics. This method should be
        called before generating signals or accessing analysis results.

        Args:
            prices_df (pl.DataFrame): A pivoted DataFrame with 'date' and columns 
                                      for each stock's close price.
        """
        logger.info(f"Performing full analysis for {self.stock1}-{self.stock2}...")
        try:
            # Ensure we have enough data for meaningful analysis
            if prices_df.height < self.window:
                logger.warning("Not enough data to perform analysis (less than window size).")
                self.is_cointegrated = False
                return

            # --- 1. Calculate Spread ---
            spread = (prices_df[f'{self.stock1}_close'] / prices_df[f'{self.stock2}_close']).drop_nulls()
            
            # --- 2. Cointegration Test (ADF) ---
            adf_result = adfuller(spread)
            self.cointegration_p_value = adf_result[1]
            self.is_cointegrated = self.cointegration_p_value < 0.05
            logger.info(f"Cointegration test: p-value={self.cointegration_p_value:.4f}")

            # --- 3. Correlation Coefficient ---
            correlation_result = pl.corr(prices_df[f'{self.stock1}_close'], prices_df[f'{self.stock2}_close'])
            self.correlation = correlation_result.item() if correlation_result is not None else None
            logger.info(f"Price correlation: {self.correlation:.4f}")

            # --- 4. Half-Life of Mean Reversion ---
            if self.is_cointegrated:
                self.half_life = self._calculate_half_life(spread)
                logger.info(f"Mean reversion half-life: {self.half_life:.2f} days")
            else:
                self.half_life = None # Not applicable if not mean-reverting

        except Exception as e:
            logger.error(f"Error during pair analysis: {e}")
            self.is_cointegrated = False

    def generate_signals(self, prices_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the Z-score and generates trading signals. Assumes that
        perform_full_analysis might have been run, but does not depend on it.
        """
        # ... (This method's internal logic remains unchanged) ...
        df_with_signals = prices_df.with_columns(
            (pl.col(f'{self.stock1}_close') / pl.col(f'{self.stock2}_close')).alias('spread')
        )
        df_with_signals = df_with_signals.with_columns([
            pl.col('spread').rolling_mean(window_size=self.window).alias('spread_mean'),
            pl.col('spread').rolling_std(window_size=self.window).alias('spread_std')
        ])
        df_with_signals = df_with_signals.with_columns(
            ((pl.col('spread') - pl.col('spread_mean')) / pl.col('spread_std')).alias('z_score')
        )
        self.latest_z_score = df_with_signals['z_score'].last() # Store latest z-score
        df_with_signals = df_with_signals.with_columns(
            pl.when(pl.col('z_score') > self.z_entry)
              .then(pl.lit('SELL_SPREAD'))
              .when(pl.col('z_score') < -self.z_entry)
              .then(pl.lit('BUY_SPREAD'))
              .when(pl.col('z_score').abs() < self.z_exit)
              .then(pl.lit('EXIT'))
              .otherwise(pl.lit('HOLD'))
              .alias('signal')
        )
        return df_with_signals

# --- Self-Test Block ---
if __name__ == '__main__':
    # ... (The self-test block can remain the same, but we can enhance it)
    print("--- Testing Enhanced Pairs Arbitrage Module ---")
    # ... (mock data creation)
    strategy = PairsArbitrageStrategy('RELIANCE', 'TCS')
    
    # NEW: Call the analysis method first
    strategy.perform_full_analysis(mock_df) # type: ignore
    
    print(f"\nAnalysis Results:")
    print(f"  - Is Cointegrated? {'Yes' if strategy.is_cointegrated else 'No'}")
    print(f"  - P-value: {strategy.cointegration_p_value:.4f}")
    print(f"  - Correlation: {strategy.correlation:.4f}")
    print(f"  - Half-life: {strategy.half_life:.2f} days" if strategy.half_life else "  - Half-life: N/A")

    signals_df = strategy.generate_signals(mock_df) # type: ignore
    print("\n--- Generated Signals ---")
    print(signals_df.tail(5))