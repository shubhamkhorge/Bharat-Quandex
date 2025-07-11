"""
Core configuration module for Bharat QuanDex
Centralized settings management with environment variable support
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import duckdb
from loguru import logger 

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

@dataclass
class DataConfig:
    """Data storage and processing configuration"""
    raw_data_path: Optional[Path] = None
    processed_data_path: Optional[Path] = None
    duckdb_path: Optional[Path] = None
    duckdb_memory_limit: str = "4GB"
    backup_days: int = 30
    max_file_size_mb: int = 500
    conn: Optional[duckdb.DuckDBPyConnection] = None
    
    def __post_init__(self):
        # Set absolute paths based on project root
        if self.raw_data_path is None:
            self.raw_data_path = PROJECT_ROOT / "data_vault" / "raw_feeds"
        if self.processed_data_path is None:
            self.processed_data_path = PROJECT_ROOT / "data_vault" / "market_boards"
        if self.duckdb_path is None:
            self.duckdb_path = PROJECT_ROOT / "data_vault" / "market_boards" / "quandex.duckdb"
        
        # Ensure directories exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB connection
        if self.conn is None:
            self.conn = duckdb.connect(str(self.duckdb_path))
            self.conn.execute(f"SET memory_limit='{self.duckdb_memory_limit}'")
            logger.info(f"DuckDB connection initialized at {self.duckdb_path}")

@dataclass
class MarketConfig:
    """Indian market specific configuration"""
    # Trading hours (IST)
    market_open: str = "09:15"
    market_close: str = "15:30"
    
    # Major indices
    major_indices: List[str] = None
    
    # Transaction costs (as percentages)
    brokerage: float = 0.01  # 0.01%
    stt_equity: float = 0.025  # 0.025% on sell side
    exchange_charges: float = 0.00325  # NSE charges
    gst: float = 18.0  # 18% on brokerage + exchange charges
    sebi_charges: float = 0.0001  # 0.0001%
    stamp_duty: float = 0.003  # 0.003% on buy side
    fallback_symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS"
    ])
    
    # Market holidays (will be loaded dynamically)
    trading_holidays: List[str] = None

    
    
    def __post_init__(self):
        if self.major_indices is None:
            self.major_indices = [
                "^NSEI",  # Nifty 50
                "^NSEBANK",  # Bank Nifty
                "^NSMIDCP",  # Nifty Midcap 100
                "^NSSMLCP",  # Nifty Smallcap 100
            ]
        
        if self.trading_holidays is None:
            self.trading_holidays = []
    
    def calculate_total_transaction_cost(self, trade_value: float, is_buy: bool = True) -> float:
        """Calculate total transaction cost for a trade"""
        costs = 0.0
        
        # Brokerage
        costs += trade_value * (self.brokerage / 100)
        
        # STT (only on sell for equity)
        if not is_buy:
            costs += trade_value * (self.stt_equity / 100)
        
        # Exchange charges
        exchange_cost = trade_value * (self.exchange_charges / 100)
        costs += exchange_cost
        
        # GST on brokerage + exchange charges
        taxable_amount = (trade_value * (self.brokerage / 100)) + exchange_cost
        costs += taxable_amount * (self.gst / 100)
        
        # SEBI charges
        costs += trade_value * (self.sebi_charges / 100)
        
        # Stamp duty (only on buy)
        if is_buy:
            costs += trade_value * (self.stamp_duty / 100)
        
        return costs

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0  # 1 Lakh INR
    start_date: str = "2020-01-01"
    end_date: str = "2025-06-18"
    benchmark: str = "^NSEI"
    rebalance_frequency: str = "daily"  # daily, weekly, monthly, quarterly
    
    # Risk management
    max_position_size: float = 0.05  # 5% of portfolio
    stop_loss: float = 0.05  # 5% stop loss
    max_drawdown: float = 0.15  # 15% max drawdown

@dataclass
class MLConfig:
    """Machine learning configuration"""
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    n_trials: int = 100  # for Optuna optimization
    
    # Feature engineering
    lookback_periods: List[int] = None
    technical_indicators: List[str] = None
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50, 200]
        
        if self.technical_indicators is None:
            self.technical_indicators = [
                "SMA", "EMA", "RSI", "MACD", "BB", "STOCH"
            ]

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    host: str = "localhost"
    port: int = 8501
    theme: str = "light"
    auto_refresh: bool = True
    refresh_interval: int = 300  # 5 minutes

@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    proxy: Optional[str] = None  # Proxy URL if needed
    user_agents: Optional[List[str]] = None
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0
    
    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.data = DataConfig()
        self.market = MarketConfig()
        self.backtest = BacktestConfig()
        self.ml = MLConfig()
        self.dashboard = DashboardConfig()
        self.scraping = ScrapingConfig()
        
        # Override with environment variables if present
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        # Data paths
        nse_data_path = os.getenv("NSE_DATA_PATH")
        if nse_data_path:
            self.data.raw_data_path = Path(nse_data_path)
        
        processed_data_path = os.getenv("PROCESSED_DATA_PATH")
        if processed_data_path:
            self.data.processed_data_path = Path(processed_data_path)
        
        # Dashboard settings
        dashboard_host = os.getenv("DASHBOARD_HOST")
        if dashboard_host:
            self.dashboard.host = dashboard_host
        
        dashboard_port = os.getenv("DASHBOARD_PORT")
        if dashboard_port:
            self.dashboard.port = int(dashboard_port)
        
        # Backtest settings
        initial_capital = os.getenv("INITIAL_CAPITAL")
        if initial_capital:
            self.backtest.initial_capital = float(initial_capital)

# Global configuration instance
config = Config()

# Convenience functions
def get_data_path(filename: str = "") -> Path:
    """Get path to data file"""
    if config.data.processed_data_path is None:
        raise ValueError("Processed data path not configured")
    return config.data.processed_data_path / filename

def get_raw_data_path(filename: str = "") -> Path:
    """Get path to raw data file"""
    if config.data.raw_data_path is None:
        raise ValueError("Raw data path not configured")
    return config.data.raw_data_path / filename

def is_trading_day(date_str: str) -> bool:
    """Check if given date is a trading day"""
    # Simple implementation - can be enhanced with actual holiday calendar
    from datetime import datetime
    
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # Skip weekends
        if date_obj.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Skip holidays (basic implementation)
        if date_str in config.market.trading_holidays:
            return False
        
        return True
    except ValueError:
        return False

        

if __name__ == "__main__":
    # Test configuration
    print("Bharat QuanDex Configuration")
    print("=" * 40)
    print(f"Raw data path: {config.data.raw_data_path}")
    print(f"Processed data path: {config.data.processed_data_path}")
    print(f"Initial capital: ₹{config.backtest.initial_capital:,.2f}")
    print(f"Market open: {config.market.market_open}")
    print(f"Dashboard: {config.dashboard.host}:{config.dashboard.port}")
    
    # Test transaction cost calculation
    trade_value = 10000  # ₹10,000 trade
    buy_cost = config.market.calculate_total_transaction_cost(trade_value, is_buy=True)
    sell_cost = config.market.calculate_total_transaction_cost(trade_value, is_buy=False)
    
    print(f"\nTransaction costs for ₹{trade_value:,.2f} trade:")
    print(f"Buy cost: ₹{buy_cost:.2f} ({buy_cost/trade_value*100:.3f}%)")
    print(f"Sell cost: ₹{sell_cost:.2f} ({sell_cost/trade_value*100:.3f}%)")
    print(f"Total round-trip: ₹{buy_cost + sell_cost:.2f} ({(buy_cost + sell_cost)/trade_value*100:.3f}%)")