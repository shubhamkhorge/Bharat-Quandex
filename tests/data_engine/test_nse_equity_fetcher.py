import pytest
from unittest.mock import MagicMock, call, patch, ANY
import pandas as pd
import polars as pl
import duckdb
import yfinance as yf
from pathlib import Path
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, Future

# Import the class to be tested
from quandex_core.data_engine.nse_equity_fetcher import NSEDataFetcher
from quandex_core.config import Config, DataConfig # For mocking config

# --- Global Mocks & Fixtures ---

@pytest.fixture
def mock_data_config_attrs():
    """Provides a dictionary of attributes for a mock DataConfig."""
    return {
        'raw_data_path': Path("/tmp/raw"),
        'processed_data_path': Path("/tmp/processed"),
        'duckdb_path': Path("/tmp/db/test.duckdb"),
        'duckdb_memory_limit': '128MB'
    }

@pytest.fixture
def mock_global_config(mocker, mock_data_config_attrs):
    """Mocks the global config object in quandex_core.config."""
    mock_cfg = MagicMock(spec=Config)
    mock_cfg.data = MagicMock(spec=DataConfig)
    for attr, value in mock_data_config_attrs.items():
        setattr(mock_cfg.data, attr, value)

    # Ensure PROJECT_ROOT is also sensible if used by the class directly
    mock_cfg.PROJECT_ROOT = Path("/tmp")

    # Patch the global 'config' instance used by the module
    # This assumes nse_equity_fetcher.py uses 'from quandex_core.config import config'
    return mocker.patch('quandex_core.data_engine.nse_equity_fetcher.config', mock_cfg)


@pytest.fixture
def mock_duckdb_connection(mocker):
    """Mocks duckdb.connect and the connection object."""
    mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)
    mock_conn.execute.return_value = mock_conn # for fluent interface like .execute().fetchall()
    mock_conn.fetchall.return_value = []
    mock_conn.fetchone.return_value = None
    mock_conn.pl.return_value = pl.DataFrame() # For .pl() calls
    mock_conn.df.return_value = pd.DataFrame() # For .df() calls

    # Mock context manager methods
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    mocker.patch('duckdb.connect', return_value=mock_conn)
    return mock_conn

@pytest.fixture
def mock_yfinance_ticker(mocker):
    """Mocks yfinance.Ticker and its history method."""
    mock_ticker_instance = MagicMock(spec=yf.Ticker)
    mock_ticker_instance.history.return_value = pd.DataFrame() # Default empty history
    mock_yf_ticker = mocker.patch('yfinance.Ticker', return_value=mock_ticker_instance)
    return mock_yf_ticker, mock_ticker_instance # Return both class and instance mock

@pytest.fixture
def mock_thread_pool_executor(mocker):
    """Mocks concurrent.futures.ThreadPoolExecutor."""
    mock_executor_instance = MagicMock(spec=ThreadPoolExecutor)
    mock_executor_instance.map.return_value = [] # Default empty results from map

    # For submit and future.result()
    mock_future = MagicMock(spec=Future)
    mock_future.result.return_value = None # Default future result
    mock_executor_instance.submit.return_value = mock_future

    mock_executor_class = mocker.patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_executor_instance)
    return mock_executor_class, mock_executor_instance


@pytest.fixture
def mock_time_sleep(mocker):
    """Mocks time.sleep."""
    return mocker.patch('time.sleep')

@pytest.fixture
def mock_datetime_now(mocker):
    """Mocks datetime.datetime.now to return a fixed datetime."""
    fixed_now = datetime.datetime(2023, 1, 1, 12, 0, 0)
    mock_dt = MagicMock()
    mock_dt.now.return_value = fixed_now
    return mocker.patch('datetime.datetime', mock_dt)


# --- Test Class ---

class TestNSEDataFetcher:

    @pytest.fixture(autouse=True)
    def setup_method_mocks(self, mock_global_config, mock_duckdb_connection, mock_yfinance_ticker, mock_thread_pool_executor, mock_time_sleep, mock_datetime_now):
        """Auto-applies common mocks to all test methods in this class."""
        self.mock_global_config = mock_global_config
        self.mock_conn = mock_duckdb_connection
        self.mock_yf_ticker_class, self.mock_yf_ticker_instance = mock_yfinance_ticker
        self.mock_executor_class, self.mock_executor_instance = mock_thread_pool_executor
        self.mock_time_sleep = mock_time_sleep
        self.mock_datetime_now = mock_datetime_now

    def test_init_success(self, mocker):
        """Test NSEDataFetcher initialization success path."""
        mock_load_symbols = mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        mock_init_db = mocker.patch.object(NSEDataFetcher, '_initialize_database')

        fetcher = NSEDataFetcher()

        # Assert config attributes are used (indirectly via mocks if not directly stored)
        assert fetcher.DUCKDB_PATH == self.mock_global_config.data.duckdb_path
        assert fetcher.DUCKDB_MEMORY_LIMIT == self.mock_global_config.data.duckdb_memory_limit

        # Assert duckdb.connect was called
        duckdb.connect.assert_called_once_with(
            database=str(self.mock_global_config.data.duckdb_path),
            read_only=False,
            config={'memory_limit': self.mock_global_config.data.duckdb_memory_limit}
        )
        assert fetcher.conn == self.mock_conn

        # Assert initial methods are called
        mock_load_symbols.assert_called_once()
        mock_init_db.assert_called_once()

        # Assert connection check was performed
        self.mock_conn.execute.assert_any_call("SELECT 1") # Initial connection check

    def test_init_connection_retry(self, mocker):
        """Test connection retry logic during __init__."""
        mock_load_symbols = mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        mock_init_db = mocker.patch.object(NSEDataFetcher, '_initialize_database')

        # Simulate initial connection failure, then success
        self.mock_conn.execute.side_effect = [
            duckdb.IOException("Initial connection failed"), # For "SELECT 1"
            MagicMock(), # For subsequent "SELECT 1" after reconnect
            MagicMock()  # For any other execute calls by _initialize_database etc.
        ]

        # Ensure duckdb.connect is called multiple times
        # First call is by fixture, subsequent by _ensure_fresh_connection
        # We need to make sure the connect mock can be called multiple times and returns fresh conn mocks
        # or the same mock if that's okay. For this test, returning the same mock is fine.

        fetcher = NSEDataFetcher(max_retries=1, retry_delay=0.01)

        assert duckdb.connect.call_count >= 2 # Initial + at least one retry

        # Check calls to SELECT 1 (initial check, then retry check)
        select_1_calls = [c for c in self.mock_conn.execute.call_args_list if c[0][0] == "SELECT 1"]
        assert len(select_1_calls) >= 2

        mock_load_symbols.assert_called_once()
        mock_init_db.assert_called_once() # Should be called after successful (re)connection
        self.mock_time_sleep.assert_called_with(0.01) # Check retry delay

    def test_init_connection_final_failure(self, mocker):
        """Test __init__ when DB connection ultimately fails after retries."""
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols') # Won't be called
        mocker.patch.object(NSEDataFetcher, '_initialize_database') # Won't be called

        self.mock_conn.execute.side_effect = duckdb.IOException("Connection always fails")

        with pytest.raises(duckdb.IOException, match="Failed to connect to DuckDB after 2 retries"):
            NSEDataFetcher(max_retries=2, retry_delay=0.01)

        assert duckdb.connect.call_count == 1 + 2 # Initial attempt + 2 retries
        self.mock_time_sleep.assert_has_calls([call(0.01), call(0.01)])


    def test_load_nifty_symbols_success(self, mocker):
        """Test _load_nifty_symbols success with symbols from DB."""
        mocker.patch.object(NSEDataFetcher, '_initialize_database') # Avoid its call during init

        # Mock DB return for symbols
        db_symbols = [("RELIANCE.NS",), ("TCS.NS",)]
        self.mock_conn.execute.return_value.fetchall.return_value = db_symbols

        fetcher = NSEDataFetcher() # _load_nifty_symbols is called in init

        expected_symbols = ["RELIANCE.NS", "TCS.NS"]
        assert fetcher.nifty_500_symbols == expected_symbols
        self.mock_conn.execute.assert_called_with("SELECT DISTINCT symbol FROM nifty_500_symbols WHERE series = 'EQ'")


    def test_load_nifty_symbols_db_error_fallback(self, mocker):
        """Test _load_nifty_symbols fallback when DB query fails."""
        mocker.patch.object(NSEDataFetcher, '_initialize_database')

        # Simulate DB error, then empty result for the check if table exists
        self.mock_conn.execute.side_effect = [
            duckdb.CatalogException("Table not found or query error"), # For initial symbol load
            MagicMock(fetchall=MagicMock(return_value=[])), # For checking if nifty_500_symbols exists
            MagicMock() # For other calls
        ]

        fetcher = NSEDataFetcher()

        # Should use fallback symbols from config
        assert fetcher.nifty_500_symbols == self.mock_global_config.market.fallback_symbols

        # Verify it tried to read from DB first
        self.mock_conn.execute.assert_any_call("SELECT DISTINCT symbol FROM nifty_500_symbols WHERE series = 'EQ'")
        # Verify it tried to check if table exists (or similar logic for fallback)
        # The actual query for fallback might vary, e.g., checking table existence.
        # Based on current code, it catches the error and uses fallback.
        # A check for table existence might be:
        self.mock_conn.execute.assert_any_call("SELECT table_name FROM information_schema.tables WHERE table_name = 'nifty_500_symbols'")


    def test_initialize_database_success(self, mocker):
        """Test _initialize_database successfully creates tables."""
        # Prevent _load_nifty_symbols from running as it's not the focus here
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols', return_value=None)

        fetcher = NSEDataFetcher() # _initialize_database is called in init

        # Check for CREATE TABLE statements
        # The exact SQL can be long, so check for key parts or use ANY for the SQL string
        # if the full string is too brittle for the test.

        # Example: Check if the raw_equity_data table creation was attempted
        raw_equity_sql_found = False
        processed_equity_sql_found = False
        symbols_sql_found = False

        for c in self.mock_conn.execute.call_args_list:
            sql_command = c[0][0].upper() # Get the SQL command string
            if "CREATE TABLE IF NOT EXISTS RAW_EQUITY_DATA" in sql_command:
                raw_equity_sql_found = True
            if "CREATE TABLE IF NOT EXISTS PROCESSED_EQUITY_DATA" in sql_command:
                processed_equity_sql_found = True
            if "CREATE TABLE IF NOT EXISTS NIFTY_500_SYMBOLS" in sql_command:
                symbols_sql_found = True

        assert raw_equity_sql_found, "CREATE TABLE for raw_equity_data not called"
        assert processed_equity_sql_found, "CREATE TABLE for processed_equity_data not called"
        assert symbols_sql_found, "CREATE TABLE for nifty_500_symbols not called"

    def test_initialize_database_retry_and_success(self, mocker):
        """Test _initialize_database retry logic on duckdb.Error."""
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')

        # Simulate duckdb.Error during table creation, then success
        self.mock_conn.execute.side_effect = [
            MagicMock(), # For SELECT 1 in init
            duckdb.Error("Failed to create table first time"), # For first CREATE TABLE call
            duckdb.Error("Failed to create table second time"),# For second CREATE TABLE call (retry)
            MagicMock(), # Success for CREATE TABLE (third attempt)
            MagicMock(), # For subsequent CREATE TABLE calls
            MagicMock(), # For subsequent CREATE TABLE calls
            MagicMock(), # For any other calls
        ]

        fetcher = NSEDataFetcher(max_db_retries=3, db_retry_delay=0.01)

        # Check that execute was called multiple times for table creation attempts
        # This count is tricky because there are multiple CREATE TABLE statements.
        # We need to ensure that at least one of them went through the retry cycle.
        # The side_effect list assumes the first CREATE TABLE is the one failing.

        assert self.mock_conn.execute.call_count > 3 # Initial SELECT 1 + 3 attempts for the first table + others

        # Check that time.sleep was called for retries
        self.mock_time_sleep.assert_has_calls([call(0.01), call(0.01)])

        # Check that connect was NOT called again (retries are on execute, not full re-connect for _init_db)
        # duckdb.connect call count should be 1 (from fixture setup) unless _ensure_fresh_connection was triggered
        # For this specific test, _ensure_fresh_connection isn't the primary target for retry, _initialize_db is.
        # If _initialize_database itself calls _ensure_fresh_connection on error, then connect count could increase.
        # Based on current nse_equity_fetcher.py, _initialize_database does not call _ensure_fresh_connection.
        assert duckdb.connect.call_count == 1


    def test_initialize_database_final_failure(self, mocker):
        """Test _initialize_database failure after all retries."""
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')

        self.mock_conn.execute.side_effect = duckdb.Error("Persistent DB error")

        with pytest.raises(duckdb.Error, match="Failed to initialize database after 3 retries: Persistent DB error"):
            NSEDataFetcher(max_db_retries=3, db_retry_delay=0.01)

        self.mock_time_sleep.assert_has_calls([call(0.01), call(0.01), call(0.01)])


    # More tests will follow for other methods:
    # _fetch_single_stock, _fetch_symbol_with_retry, fetch_parallel, etc.

    def test_fetch_single_stock_success(self, mocker):
        """Test _fetch_single_stock successfully fetches and processes data."""
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')

        fetcher = NSEDataFetcher()

        # Sample yfinance history output
        sample_data = {
            'Open': [100, 101], 'High': [102, 103], 'Low': [99, 100],
            'Close': [101, 102], 'Volume': [1000, 1100],
            'Dividends': [0, 0], 'Stock Splits': [0, 0]
        }
        sample_index = pd.to_datetime(['2023-01-01', '2023-01-02']).tz_localize('Asia/Kolkata')
        mock_history_df = pd.DataFrame(sample_data, index=sample_index)
        self.mock_yf_ticker_instance.history.return_value = mock_history_df

        symbol = "TEST.NS"
        result_df = fetcher._fetch_single_stock(symbol, start_date_str="2023-01-01", end_date_str="2023-01-03")

        self.mock_yf_ticker_class.assert_called_once_with(symbol)
        self.mock_yf_ticker_instance.history.assert_called_once_with(
            start="2023-01-01", end="2023-01-03", interval="1d", auto_adjust=True, prepost=False
        )

        assert not result_df.empty
        assert "symbol" in result_df.columns
        assert result_df["symbol"].iloc[0] == symbol
        assert "fetched_at" in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['date'])

        # Check column renaming
        expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'symbol', 'fetched_at']
        assert all(col in result_df.columns for col in expected_cols)
        assert len(result_df.columns) == len(expected_cols)


    def test_fetch_single_stock_empty_history(self, mocker):
        """Test _fetch_single_stock when yfinance returns empty DataFrame."""
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        self.mock_yf_ticker_instance.history.return_value = pd.DataFrame() # Empty data

        symbol = "EMPTY.NS"
        result_df = fetcher._fetch_single_stock(symbol, "2023-01-01", "2023-01-03")

        assert result_df is None # Or an empty DataFrame, depending on implementation. Current returns None.

    def test_fetch_single_stock_yfinance_exception(self, mocker):
        """Test _fetch_single_stock exception handling for yfinance errors."""
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        self.mock_yf_ticker_instance.history.side_effect = Exception("yfinance API error")

        symbol = "ERROR.NS"
        result_df = fetcher._fetch_single_stock(symbol, "2023-01-01", "2023-01-03")

        assert result_df is None # Should handle exception and return None
        # Optionally, check for logging if implemented

    def test_fetch_single_stock_start_date_filtering(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        sample_data = {
            'Open': [100, 101, 102], 'High': [102, 103, 104], 'Low': [99, 100, 101],
            'Close': [101, 102, 103], 'Volume': [1000, 1100, 1200]
        }
        sample_index = pd.to_datetime(['2022-12-30', '2023-01-01', '2023-01-02']).tz_localize('Asia/Kolkata')
        mock_history_df = pd.DataFrame(sample_data, index=sample_index)
        self.mock_yf_ticker_instance.history.return_value = mock_history_df

        result_df = fetcher._fetch_single_stock("TEST.NS", start_date_str="2023-01-01", end_date_str="2023-01-03")

        assert len(result_df) == 2
        assert result_df['date'].min() == pd.Timestamp("2023-01-01")

    # Tests for _fetch_symbol_with_retry
    def test_fetch_symbol_with_retry_success_first_try(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        mock_data = pd.DataFrame({'Close': [10]})
        mocker.patch.object(fetcher, '_fetch_single_stock', return_value=mock_data)

        result = fetcher._fetch_symbol_with_retry("TEST.NS", "2023-01-01", "2023-01-02", max_retries=2, delay=0.01)

        assert result is mock_data
        fetcher._fetch_single_stock.assert_called_once_with("TEST.NS", "2023-01-01", "2023-01-02")
        self.mock_time_sleep.assert_not_called()

    def test_fetch_symbol_with_retry_success_after_retries(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        mock_data = pd.DataFrame({'Close': [10]})
        mocker.patch.object(fetcher, '_fetch_single_stock', side_effect=[None, None, mock_data]) # Fails twice, then succeeds

        result = fetcher._fetch_symbol_with_retry("TEST.NS", "2023-01-01", "2023-01-02", max_retries=3, delay=0.01)

        assert result is mock_data
        assert fetcher._fetch_single_stock.call_count == 3
        self.mock_time_sleep.assert_has_calls([call(0.01), call(0.01)])


    def test_fetch_symbol_with_retry_failure_after_all_retries(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        mocker.patch.object(fetcher, '_fetch_single_stock', return_value=None) # Always fails

        result = fetcher._fetch_symbol_with_retry("TEST.NS", "2023-01-01", "2023-01-02", max_retries=2, delay=0.01)

        assert result is None
        assert fetcher._fetch_single_stock.call_count == 3 # Initial + 2 retries
        self.mock_time_sleep.assert_has_calls([call(0.01), call(0.01)])

    # --- Tests for fetch_parallel ---
    def test_fetch_parallel_basic_flow(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()
        fetcher.nifty_500_symbols = ["TEST1.NS", "TEST2.NS", "TEST3.NS"]

        # Mock _fetch_symbol_with_retry results
        mock_df1 = pd.DataFrame({'symbol': ['TEST1.NS'], 'date': [pd.Timestamp('2023-01-01')], 'close': [10]})
        mock_df2 = pd.DataFrame({'symbol': ['TEST2.NS'], 'date': [pd.Timestamp('2023-01-01')], 'close': [20]})
        # TEST3.NS will return None (simulating fetch failure)

        # Mock ThreadPoolExecutor.map behavior or submit behavior
        # If using map:
        # self.mock_executor_instance.map.return_value = [mock_df1, mock_df2, None]

        # If using submit:
        future1 = Future()
        future1.set_result(mock_df1)
        future2 = Future()
        future2.set_result(mock_df2)
        future3 = Future()
        future3.set_result(None)
        self.mock_executor_instance.submit.side_effect = [future1, future2, future3]


        # Mock database query for latest dates (return empty initially for full fetch)
        self.mock_conn.execute().pl.return_value = pl.DataFrame({'symbol': [], 'latest_date': []})

        all_data_dfs = fetcher.fetch_parallel(start_date_str="2023-01-01", end_date_str="2023-01-02")

        assert self.mock_executor_class.called_once_with(max_workers=fetcher.MAX_WORKERS)

        # Check calls to _fetch_symbol_with_retry via executor.submit
        # The arguments to _fetch_symbol_with_retry are (symbol, effective_start_date, end_date_str)
        # effective_start_date should be "2023-01-01" for all symbols as latest_dates_map is empty
        expected_submit_calls = [
            call(fetcher._fetch_symbol_with_retry, "TEST1.NS", "2023-01-01", "2023-01-02"),
            call(fetcher._fetch_symbol_with_retry, "TEST2.NS", "2023-01-01", "2023-01-02"),
            call(fetcher._fetch_symbol_with_retry, "TEST3.NS", "2023-01-01", "2023-01-02"),
        ]
        self.mock_executor_instance.submit.assert_has_calls(expected_submit_calls, any_order=True) # any_order due to dict iteration

        assert len(all_data_dfs) == 2
        assert mock_df1 in all_data_dfs
        assert mock_df2 in all_data_dfs

        # Verify query for latest dates
        self.mock_conn.execute.assert_any_call(
            f"SELECT symbol, MAX(date) as latest_date FROM {fetcher.RAW_TABLE_NAME} GROUP BY symbol"
        )

    def test_fetch_parallel_with_latest_dates(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()
        fetcher.nifty_500_symbols = ["RELIANCE.NS", "TCS.NS"]

        # Mock latest dates from DB
        latest_dates_df = pl.DataFrame({
            'symbol': ["RELIANCE.NS", "TCS.NS"],
            'latest_date': [datetime.date(2023, 1, 10), datetime.date(2023, 1, 5)]
        })
        self.mock_conn.execute().pl.return_value = latest_dates_df

        # Mock _fetch_symbol_with_retry
        mock_reliance_data = pd.DataFrame({'symbol': ['RELIANCE.NS']})
        mock_tcs_data = pd.DataFrame({'symbol': ['TCS.NS']})

        future_reliance = Future(); future_reliance.set_result(mock_reliance_data)
        future_tcs = Future(); future_tcs.set_result(mock_tcs_data)
        self.mock_executor_instance.submit.side_effect = [future_reliance, future_tcs]


        fetcher.fetch_parallel(start_date_str="2023-01-01", end_date_str="2023-01-15")

        # Expected start_date for RELIANCE.NS is 2023-01-11 (latest_date + 1 day)
        # Expected start_date for TCS.NS is 2023-01-06
        expected_calls = [
            call(fetcher._fetch_symbol_with_retry, "RELIANCE.NS", "2023-01-11", "2023-01-15"),
            call(fetcher._fetch_symbol_with_retry, "TCS.NS", "2023-01-06", "2023-01-15"),
        ]
        self.mock_executor_instance.submit.assert_has_calls(expected_calls, any_order=True)

    def test_fetch_parallel_no_symbols(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()
        fetcher.nifty_500_symbols = [] # No symbols to fetch

        results = fetcher.fetch_parallel("2023-01-01", "2023-01-02")
        assert results == []
        self.mock_executor_instance.submit.assert_not_called()

    # --- Tests for save_to_duckdb ---
    def test_save_to_duckdb_empty_list(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        fetcher.save_to_duckdb([]) # Empty list of dataframes
        self.mock_conn.register.assert_not_called()
        self.mock_conn.execute.assert_not_called() # Besides init calls

    def test_save_to_duckdb_success_with_pk(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        # Mock that primary key exists
        self.mock_conn.execute().fetchone.return_value = ("symbol, date",) # Simulate PK exists

        df1 = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01']), 'symbol': ['TEST1.NS'], 'open': [100]
        })
        df2 = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-02']), 'symbol': ['TEST2.NS'], 'open': [200]
        })
        all_data_dfs = [df1, df2]
        combined_df = pd.concat(all_data_dfs, ignore_index=True)

        fetcher.save_to_duckdb(all_data_dfs)

        self.mock_conn.register.assert_called_once_with('temp_df_to_insert', combined_df)

        expected_sql = f"""
            INSERT INTO {fetcher.RAW_TABLE_NAME}
            SELECT * FROM temp_df_to_insert
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                dividends = excluded.dividends,
                stock_splits = excluded.stock_splits,
                fetched_at = excluded.fetched_at;
        """
        # Check if execute was called with SQL containing the core part of the upsert
        # Using ANY or a custom matcher might be better if whitespace/formatting is an issue.
        # For now, simple string comparison after normalizing whitespace.

        # Find the call that matches the INSERT ... ON CONFLICT
        insert_sql_call_found = False
        for actual_call in self.mock_conn.execute.call_args_list:
            actual_sql = ' '.join(actual_call[0][0].split()) # Normalize whitespace
            expected_sql_normalized = ' '.join(expected_sql.split())
            if expected_sql_normalized in actual_sql:
                 insert_sql_call_found = True
                 break
        assert insert_sql_call_found, "Upsert SQL not executed as expected"

        self.mock_conn.commit.assert_called_once()
        self.mock_conn.unregister.assert_called_once_with('temp_df_to_insert')

    def test_save_to_duckdb_no_pk_recreates_table(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database') # Mock this out for this specific test
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        # Mock primary key check: first fetchone returns None (no PK), then after recreation it might return PK
        # The logic for PK check:
        # 1. Checks PRAGMA table_info for 'pk' column > 0 or duckdb_constraints for primary key type.
        # Let's mock the duckdb_constraints path as it's more direct.
        # First call (check PK): return empty, meaning no PK.
        # Subsequent calls (after table recreation, if any): may or may not be relevant for this test.
        self.mock_conn.execute().fetchall.side_effect = [
            [], # No PK found initially from duckdb_constraints
            [], # For PRAGMA table_info as a fallback (empty means no PK column)
            # ... any other fetchall calls
        ]

        # Mock _initialize_database to be callable if needed after drop
        mock_init_db_after_drop = mocker.patch.object(fetcher, '_initialize_database')


        df1 = pd.DataFrame({'date': [pd.Timestamp('2023-01-01')], 'symbol': ['TEST.NS'], 'close': [100]})

        # Mock data already in table (to be backed up)
        backup_data_df = pd.DataFrame({'date': [pd.Timestamp('2022-12-30')], 'symbol': ['OLD.NS'], 'close': [90]})
        # This is tricky: the execute().df() needs to be specific to the "SELECT * FROM raw_equity_data" call
        # We need a more sophisticated side_effect for self.mock_conn.execute()

        # Let's refine the mock_conn.execute side_effect
        def execute_side_effect_for_pk_recreate(query, *args, **kwargs):
            mock_sub_conn = MagicMock() # Represents the cursor/relation object
            if "SELECT constraint_text FROM duckdb_constraints()" in query: # Check for PK
                mock_sub_conn.fetchall.return_value = [] # No PK
                mock_sub_conn.fetchone.return_value = None
            elif "PRAGMA table_info('raw_equity_data')" in query: # Fallback PK check
                 mock_sub_conn.fetchall.return_value = [] # No PK column
                 mock_sub_conn.fetchone.return_value = None
            elif f"SELECT * FROM {fetcher.RAW_TABLE_NAME}" in query: # Backup table
                mock_sub_conn.df.return_value = backup_data_df
            else: # Default behavior for other queries
                mock_sub_conn.fetchall.return_value = []
                mock_sub_conn.fetchone.return_value = None
                mock_sub_conn.df.return_value = pd.DataFrame()
                mock_sub_conn.pl.return_value = pl.DataFrame()
            return mock_sub_conn

        self.mock_conn.execute.side_effect = execute_side_effect_for_pk_recreate

        fetcher.save_to_duckdb([df1])

        # Verify table drop and recreation sequence
        self.mock_conn.execute.assert_any_call(f"DROP TABLE IF EXISTS {fetcher.RAW_TABLE_NAME}_backup")
        self.mock_conn.execute.assert_any_call(f"ALTER TABLE {fetcher.RAW_TABLE_NAME} RENAME TO {fetcher.RAW_TABLE_NAME}_backup")

        # _initialize_database should be called to recreate the table structure correctly
        mock_init_db_after_drop.assert_called_once_with(force_create_raw=True) # Check it's called to recreate raw table

        # Verify data insertion into new table (from backup and new data)
        # This part is complex due to multiple register/execute for backup and new data.
        # Check for registration of backup data and new data
        # It should register 'backup_data_df_relation' and 'temp_df_to_insert'

        # Simplified check: ensure main INSERT INTO (not ON CONFLICT) is called for the new data
        # And also for the backup data
        insert_backup_sql_found = False
        insert_new_sql_found = False
        final_sql_calls = [c[0][0].upper() for c in self.mock_conn.execute.call_args_list]

        for sql_command in final_sql_calls:
            if f"INSERT INTO {fetcher.RAW_TABLE_NAME.upper()} SELECT * FROM BACKUP_DATA_DF_RELATION" in sql_command:
                insert_backup_sql_found = True
            if f"INSERT INTO {fetcher.RAW_TABLE_NAME.upper()} SELECT * FROM TEMP_DF_TO_INSERT" in sql_command and "ON CONFLICT" not in sql_command:
                insert_new_sql_found = True

        assert insert_backup_sql_found, "Insert from backup table not found"
        assert insert_new_sql_found, "Direct insert for new data (after table recreate) not found"

        self.mock_conn.execute.assert_any_call(f"DROP TABLE IF EXISTS {fetcher.RAW_TABLE_NAME}_backup")
        self.mock_conn.commit.assert_called() # Should be called at least once


    def test_save_to_duckdb_binder_exception_triggers_reinit(self, mocker):
        """Test that a BinderException during INSERT triggers table re-initialization."""
        mocker.patch.object(NSEDataFetcher, '_initialize_database') # Mock the original init
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        # Mock that PK exists initially
        self.mock_conn.execute().fetchone.return_value = ("symbol, date",)

        df1 = pd.DataFrame({'date': [pd.Timestamp('2023-01-01')], 'symbol': ['TEST.NS'], 'close': [100]})

        # Make the INSERT INTO ... ON CONFLICT fail with BinderException
        # Then, the subsequent plain INSERT (after reinit) should succeed.
        # This requires careful side_effect management on execute

        mock_init_db_on_binder = mocker.spy(fetcher, '_initialize_database')

        def execute_side_effect_for_binder(query, *args, **kwargs):
            mock_sub_conn = MagicMock()
            if "INSERT INTO" in query and "ON CONFLICT" in query:
                # First attempt (upsert) throws BinderException
                if execute_side_effect_for_binder.first_upsert_call:
                     execute_side_effect_for_binder.first_upsert_call = False
                     raise duckdb.BinderException("Simulated Binder Error: schema mismatch")
                else: # Subsequent calls (e.g. after reinit)
                    pass # Normal execution
            elif "SELECT constraint_text FROM duckdb_constraints()" in query: # PK check
                 mock_sub_conn.fetchall.return_value = [("PRIMARY KEY (symbol, date)",)]
                 mock_sub_conn.fetchone.return_value = ("PRIMARY KEY (symbol, date)",)
            # Add other conditions as needed for PRAGMA, SELECT * FROM backup etc if relevant
            return mock_sub_conn
        execute_side_effect_for_binder.first_upsert_call = True

        self.mock_conn.execute.side_effect = execute_side_effect_for_binder

        fetcher.save_to_duckdb([df1])

        # _initialize_database should have been called due to BinderException
        mock_init_db_on_binder.assert_called_once_with(force_create_raw=True)

        # After re-initialization, a direct INSERT should have been attempted.
        # The self.mock_conn.execute would be called with a plain INSERT.
        # We need to ensure the execute mock allows this to pass.
        # The test checks that _initialize_database was called, which is the main point.
        # Also check that the data was registered.
        self.mock_conn.register.assert_called_with('temp_df_to_insert', ANY)
        self.mock_conn.commit.assert_called_once()


# Continue with process_raw_data, update_symbols_list, etc.
# This is becoming a very long file.
# For the remaining tests, I'll focus on the core logic and mocking interactions.
# The detailed SQL verification can be simplified to checking key phrases.

    # --- Test for process_raw_data ---
    def test_process_raw_data(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        fetcher.process_raw_data()

        # Check that the main CREATE OR REPLACE TABLE query was executed
        create_replace_sql_found = False
        delete_raw_sql_found = False
        for actual_call in self.mock_conn.execute.call_args_list:
            sql_command = ' '.join(actual_call[0][0].split()).upper() # Normalize and uppercase
            if f"CREATE OR REPLACE TABLE {fetcher.PROCESSED_TABLE_NAME.upper()}" in sql_command and "AS SELECT" in sql_command:
                create_replace_sql_found = True
            if f"DELETE FROM {fetcher.RAW_TABLE_NAME.upper()} WHERE DATE <=" in sql_command: # Check for delete from raw
                delete_raw_sql_found = True

        assert create_replace_sql_found, "CREATE OR REPLACE TABLE for processed_equity_data not called correctly."
        assert delete_raw_sql_found, f"DELETE FROM {fetcher.RAW_TABLE_NAME} not called correctly."
        self.mock_conn.commit.assert_called() # Should be at least one commit

    # --- Test for update_symbols_list ---
    def test_update_symbols_list(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mock_load_symbols_method = mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        # Reset the mock for _load_nifty_symbols because it's called in __init__
        mock_load_symbols_method.reset_mock()

        new_symbols_df = pd.DataFrame({
            'symbol': ['NEW1.NS', 'NEW2.NS'],
            'series': ['EQ', 'EQ'],
            # Add other required columns as per nifty_500_symbols table structure
            'company_name': ['New Company 1', 'New Company 2'],
            'isin_number': ['INE001A00001', 'INE001A00002'],
            'industry': ['Tech', 'Finance'],
            'meta_data': [None, None] # Assuming meta_data can be null or dict
        })

        fetcher.update_symbols_list(new_symbols_df)

        self.mock_conn.register.assert_called_once_with('new_symbols_df_relation', new_symbols_df)

        # Check for the INSERT OR IGNORE SQL
        insert_ignore_sql_found = False
        for actual_call in self.mock_conn.execute.call_args_list:
            sql_command = ' '.join(actual_call[0][0].split()).upper()
            if f"INSERT OR IGNORE INTO {fetcher.SYMBOLS_TABLE_NAME.upper()}" in sql_command and "SELECT * FROM NEW_SYMBOLS_DF_RELATION" in sql_command:
                insert_ignore_sql_found = True
                break
        assert insert_ignore_sql_found, "INSERT OR IGNORE for new symbols not executed."

        self.mock_conn.commit.assert_called() # Should be called after insert
        self.mock_conn.unregister.assert_called_once_with('new_symbols_df_relation')
        mock_load_symbols_method.assert_called_once() # _load_nifty_symbols should be re-called

    # --- Tests for incremental_update and full_refresh ---
    def test_incremental_update(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols') # Called in init

        fetcher = NSEDataFetcher()
        # Reset mocks for methods called during the main flow of incremental_update
        mock_fetch_parallel = mocker.patch.object(fetcher, 'fetch_parallel', return_value=[pd.DataFrame({'a': [1]})]) # Return some data to save
        mock_save_to_duckdb = mocker.patch.object(fetcher, 'save_to_duckdb')
        mock_process_raw_data = mocker.patch.object(fetcher, 'process_raw_data')
        mock_cleanup = mocker.patch.object(fetcher, 'cleanup') # Ensure cleanup is called

        fetcher.nifty_500_symbols = ["TEST.NS"] # Ensure there are symbols to process

        fetcher.incremental_update(start_date_str="2023-01-01", end_date_str="2023-01-31")

        mock_fetch_parallel.assert_called_once_with(
            start_date_str="2023-01-01",
            end_date_str="2023-01-31",
            symbols=None # Or fetcher.nifty_500_symbols depending on implementation detail
        )
        mock_save_to_duckdb.assert_called_once_with(mock_fetch_parallel.return_value)
        mock_process_raw_data.assert_called_once()
        mock_cleanup.assert_called_once()

    def test_incremental_update_no_data_fetched(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        mock_fetch_parallel = mocker.patch.object(fetcher, 'fetch_parallel', return_value=[]) # No data fetched
        mock_save_to_duckdb = mocker.patch.object(fetcher, 'save_to_duckdb')
        mock_process_raw_data = mocker.patch.object(fetcher, 'process_raw_data')
        mock_cleanup = mocker.patch.object(fetcher, 'cleanup')

        fetcher.nifty_500_symbols = ["TEST.NS"]
        fetcher.incremental_update("2023-01-01", "2023-01-31")

        mock_fetch_parallel.assert_called_once()
        mock_save_to_duckdb.assert_not_called() # Should not save if no data
        mock_process_raw_data.assert_not_called() # Should not process if no data saved
        mock_cleanup.assert_called_once() # Cleanup should still run


    def test_full_refresh(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        mock_fetch_parallel = mocker.patch.object(fetcher, 'fetch_parallel', return_value=[pd.DataFrame({'b': [2]})])
        mock_save_to_duckdb = mocker.patch.object(fetcher, 'save_to_duckdb')
        mock_process_raw_data = mocker.patch.object(fetcher, 'process_raw_data')
        mock_cleanup = mocker.patch.object(fetcher, 'cleanup')


        fetcher.nifty_500_symbols = ["TEST.NS"]

        fetcher.full_refresh(start_date_str="2022-01-01", end_date_str="2022-12-31")

        # Check for DELETE statements
        delete_raw_found = False
        delete_processed_found = False
        for actual_call in self.mock_conn.execute.call_args_list:
            sql_command = ' '.join(actual_call[0][0].split()).upper()
            if f"DELETE FROM {fetcher.RAW_TABLE_NAME.upper()}" == sql_command:
                delete_raw_found = True
            if f"DELETE FROM {fetcher.PROCESSED_TABLE_NAME.upper()}" == sql_command:
                delete_processed_found = True

        assert delete_raw_found, f"DELETE FROM {fetcher.RAW_TABLE_NAME} not called."
        assert delete_processed_found, f"DELETE FROM {fetcher.PROCESSED_TABLE_NAME} not called."

        mock_fetch_parallel.assert_called_once_with(
            start_date_str="2022-01-01",
            end_date_str="2022-12-31",
            symbols=None # Or fetcher.nifty_500_symbols
        )
        mock_save_to_duckdb.assert_called_once_with(mock_fetch_parallel.return_value)
        mock_process_raw_data.assert_called_once()
        mock_cleanup.assert_called_once()

    # --- Test for _ensure_fresh_connection ---
    def test_ensure_fresh_connection_success(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        # Reset mocks for connection related calls as init already called them
        self.mock_conn.reset_mock()
        duckdb.connect.reset_mock() # Reset the module level mock

        # First call to execute ("SELECT 1") should succeed
        self.mock_conn.execute.return_value = MagicMock() # Simulate successful execute

        fetcher._ensure_fresh_connection()

        self.mock_conn.execute.assert_called_once_with("SELECT 1")
        duckdb.connect.assert_not_called() # No new connection needed
        self.mock_conn.close.assert_not_called()

    def test_ensure_fresh_connection_failure_and_reconnect(self, mocker):
        mock_init_db_method = mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        # Reset mocks from __init__
        self.mock_conn.reset_mock()
        duckdb.connect.reset_mock() # Reset the module level mock
        mock_init_db_method.reset_mock() # Reset this as well as it's called by reconnect logic

        # First "SELECT 1" fails, second (after reconnect) succeeds.
        # Need a fresh mock connection object for the reconnect
        new_mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)
        new_mock_conn.execute.return_value = new_mock_conn # for "SELECT 1"

        self.mock_conn.execute.side_effect = [
            duckdb.Error("Connection is closed or invalid"), # Initial check fails
            MagicMock() # This won't be on self.mock_conn but new_mock_conn
        ]
        duckdb.connect.return_value = new_mock_conn # connect() will now return the new mock

        fetcher._ensure_fresh_connection(max_retries=1, delay=0.01)

        self.mock_conn.execute.assert_called_once_with("SELECT 1") # Original connection attempt
        self.mock_conn.close.assert_called_once() # Original connection closed

        duckdb.connect.assert_called_once_with(
            database=str(fetcher.DUCKDB_PATH),
            read_only=False,
            config={'memory_limit': fetcher.DUCKDB_MEMORY_LIMIT}
        )
        assert fetcher.conn == new_mock_conn # Fetcher's conn attribute updated
        new_mock_conn.execute.assert_called_once_with("SELECT 1") # New connection check
        mock_init_db_method.assert_called_once() # _initialize_database called after reconnect
        self.mock_time_sleep.assert_called_once_with(0.01) # Retry delay

    def test_ensure_fresh_connection_final_failure(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database') # Mock this as it's called in retry loop
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        self.mock_conn.reset_mock()
        duckdb.connect.reset_mock()

        self.mock_conn.execute.side_effect = duckdb.Error("Connection always fails")
        # Also make connect itself fail to simulate total failure
        duckdb.connect.side_effect = [self.mock_conn] + [duckdb.Error("Cannot even connect")] * 2 # first call in init, then failures


        with pytest.raises(duckdb.Error, match="Failed to ensure fresh DB connection after 2 retries"):
            fetcher._ensure_fresh_connection(max_retries=2, delay=0.01)

        assert self.mock_time_sleep.call_count == 2


    # --- Test for get_processed_data ---
    def test_get_processed_data_success(self, mocker):
        mock_ensure_fresh = mocker.patch.object(NSEDataFetcher, '_ensure_fresh_connection')
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        mock_result_df = pl.DataFrame({'symbol': ['TEST.NS'], 'close': [100]})
        self.mock_conn.execute().pl.return_value = mock_result_df

        start_date = "2023-01-01"
        end_date = "2023-01-31"
        symbols = ["TEST.NS", "ANOTHER.NS"]

        result = fetcher.get_processed_data(start_date, end_date, symbols)

        mock_ensure_fresh.assert_called_once()
        self.mock_conn.commit.assert_called_once() # From context manager

        expected_query_part_symbols = "AND symbol IN ('TEST.NS', 'ANOTHER.NS')"
        expected_query_part_dates = "date >= '2023-01-01' AND date <= '2023-01-31'"

        # Check that the execute call contains the key parts of the query
        query_found = False
        for actual_call in self.mock_conn.execute.call_args_list:
            sql = ' '.join(actual_call[0][0].split()) # Normalize
            if f"SELECT * FROM {fetcher.PROCESSED_TABLE_NAME}" in sql and \
               expected_query_part_dates in sql and \
               expected_query_part_symbols in sql:
                query_found = True
                break
        assert query_found, "Query for get_processed_data not as expected."
        assert result.equals(mock_result_df)

    def test_get_processed_data_no_symbols(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_ensure_fresh_connection')
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        fetcher.get_processed_data("2023-01-01", "2023-01-31", symbols=None) # No symbols

        query_found = False
        for actual_call in self.mock_conn.execute.call_args_list:
            sql = actual_call[0][0]
            if f"SELECT * FROM {fetcher.PROCESSED_TABLE_NAME}" in sql and "AND symbol IN" not in sql:
                query_found = True
                break
        assert query_found, "Query should not contain 'AND symbol IN' if no symbols provided."


    # --- Test for cleanup ---
    def test_cleanup(self, mocker):
        mocker.patch.object(NSEDataFetcher, '_initialize_database')
        mocker.patch.object(NSEDataFetcher, '_load_nifty_symbols')
        fetcher = NSEDataFetcher()

        mock_pl_clear = mocker.patch('polars.clear_caches') # Or free_memory based on version

        # Mock SHOW VIEWS and SHOW TABLES to return some temp items
        # Ensure execute mock is reset if it was set with complex side effects earlier
        self.mock_conn.reset_mock() # Reset conn to default behavior for this test

        def show_side_effect(query_string):
            query_string_upper = query_string.upper()
            rel = MagicMock(spec=duckdb.DuckDBPyRelation) # Make it spec-compliant
            if "SHOW VIEWS" in query_string_upper:
                rel.fetchall.return_value = [("temp_my_view",), ("another_view",)]
            elif "SHOW TABLES" in query_string_upper:
                # Only return tables that start with 'temp_' or are specific backup tables
                # The actual code iterates and checks startswith('temp_') or specific names
                rel.fetchall.return_value = [("temp_my_table",), ("raw_equity_data_backup",), ("actual_table",)]
            else: # Default for CHECKPOINT, etc.
                rel.fetchall.return_value = []
            return rel
        self.mock_conn.execute.side_effect = show_side_effect


        fetcher.cleanup() # Call with the new side_effect

        self.mock_conn.execute.assert_any_call("DROP VIEW IF EXISTS temp_my_view")
        self.mock_conn.execute.assert_any_call("DROP TABLE IF EXISTS temp_my_table")
        self.mock_conn.execute.assert_any_call(f"DROP TABLE IF EXISTS {fetcher.RAW_TABLE_NAME}_backup")
        self.mock_conn.execute.assert_any_call(f"DROP TABLE IF EXISTS {fetcher.PROCESSED_TABLE_NAME}_backup")

        # Check that non-temp tables/views are not dropped
        # This requires checking all execute calls and ensuring no "DROP ... another_view" or "DROP ... actual_table"
        drop_calls_args = [c[0][0] for c in self.mock_conn.execute.call_args_list if "DROP" in c[0][0]]
        assert not any("another_view" in call_arg for call_arg in drop_calls_args)
        assert not any("actual_table" in call_arg for call_arg in drop_calls_args)


        self.mock_conn.execute.assert_any_call("CHECKPOINT")
        self.mock_conn.close.assert_called_once()
        mock_pl_clear.assert_called_once()

```
