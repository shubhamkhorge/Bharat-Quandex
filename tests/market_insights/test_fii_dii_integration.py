import pytest
import asyncio
import polars as pl
from polars.testing import assert_frame_equal
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import pandas # For mocking fetchdf

import duckdb
import tempfile # Moved import to top
import os # For removing temp file

from quandex_core.market_insights.fii_dii_tracker import NSE_FII_DII_Scraper
from quandex_core.market_insights.fii_dii_tracker import main as run_tracker_main
from quandex_core.market_insights.fii_dii_tracker import logger as fii_dii_logger
from quandex_core import config as global_config_module # For mocking
import logging # For Loguru propagation to caplog

# This fixture will apply to all tests in this file.
@pytest.fixture(autouse=True)
def setup_loguru_to_caplog_propagation(caplog):
    class PropagateHandler(logging.Handler):
        def emit(self, record): # This 'record' is a standard logging.LogRecord
            # The message (record.msg or record.getMessage()) is already formatted by Loguru
            # due to the format="{message}" in logger.add()
            # We just need to let caplog capture it by handling it with a standard logger.
            # Pytest's caplog should automatically capture records emitted to loggers
            # if this handler is part of the chain.
            # The key is that Loguru passes a standard LogRecord to this handler.
            logging.getLogger(record.name).handle(record)

    # Add the propagation handler to Loguru.
    # format="{message}" ensures record.msg in the handler is the clean message.
    handler_id = fii_dii_logger.add(PropagateHandler(), format="{message}", level="DEBUG")
    yield
    fii_dii_logger.remove(handler_id) # Clean up

@pytest.fixture
def mock_config_for_integration(mocker):
    """Mocks config for NSE_FII_DII_Scraper, ensuring db_path can be overridden."""
    mock_cfg = MagicMock()
    # This path will be overridden by individual tests if they use in-memory DB
    mock_cfg.data.duckdb_path = "dummy_integration_fii_dii.db"
    mock_cfg.scraping.user_agents = ["test_integration_user_agent"]
    mock_cfg.scraping.nse_fii_dii_home_url = "https://mock_home_url_integration.com"
    mock_cfg.scraping.nse_fii_dii_api_url = "https://mock_api_url_integration.com/api"
    mock_cfg.scraping.nse_fii_dii_html_url = "https://mock_html_url_integration.com/html"

    # Set specific scraping parameters for test environment
    mock_cfg.scraping.max_retries = 1  # Fast retries for tests
    mock_cfg.scraping.retry_delay = 0.01 # Very short delay
    mock_cfg.scraping.request_timeout = 5   # Short timeout

    mock_cfg.market.trading_holidays = [] # Keep simple for tests

    # Patch the global config object used by the module
    mocker.patch('quandex_core.market_insights.fii_dii_tracker.config', mock_cfg)
    return mock_cfg

@pytest.fixture
def sample_fii_dii_data():
    """Provides a sample Polars DataFrame mimicking scraped FII/DII data."""
    return pl.DataFrame({
        "date": [date(2024, 8, 26), date(2024, 8, 27)],
        "fii_buy_cr": [1000.50, 1200.00],
        "fii_sell_cr": [500.25, 600.00],
        "fii_net_cr": [500.25, 600.00],
        "dii_buy_cr": [700.00, 800.00],
        "dii_sell_cr": [300.75, 400.00],
        "dii_net_cr": [399.25, 400.00]
    })

class TestFiiDiiScrapingToDatabaseFlow:

    # tempfile import was here, it's already moved to the top. This search block is just for context.

    @pytest.mark.asyncio
    async def test_successful_data_flow_to_in_memory_db(self, mock_config_for_integration, sample_fii_dii_data, mocker):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db_file: # delete=False
            db_path = tmp_db_file.name

        try:
            if os.path.exists(db_path): # Ensure DuckDB creates the file
                os.remove(db_path)
            mock_config_for_integration.data.duckdb_path = db_path # Ensure config leads to temp file DB

            OriginalScraperClass = NSE_FII_DII_Scraper
            created_instances = []

            def class_wrapper(*args, **kwargs):
                # The global config (mock_config_for_integration) should be active here
                # So, instance.db_path will be db_path (from tempfile) via __init__
                instance = OriginalScraperClass(*args, **kwargs)
                instance.scrape = AsyncMock(return_value=sample_fii_dii_data)
                # Spy on the actual update_database method of this instance
                mocker.spy(instance, 'update_database')
                created_instances.append(instance)
                return instance

            mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper', side_effect=class_wrapper)

            # Run the main orchestrator function
            await run_tracker_main()

            assert len(created_instances) == 1, "Scraper instance was not created as expected"
            scraper_instance_used_by_main = created_instances[0]

            scraper_instance_used_by_main.scrape.assert_awaited_once()
            scraper_instance_used_by_main.update_database.assert_called_once_with(sample_fii_dii_data)

            # Now, connect to the same temporary file database to verify
            conn = None
            try:
                conn = duckdb.connect(database=db_path, read_only=False) # Use db_path from tempfile

                # Query the institutional_flows table
                db_data = conn.execute("SELECT * FROM institutional_flows ORDER BY date").pl()
                assert_frame_equal(db_data, sample_fii_dii_data, check_dtype=True)

                # Query the v_institutional_trends view
                # This view requires some data to be non-empty.
                view_data = conn.execute("SELECT * FROM v_institutional_trends").pl()
                assert not view_data.is_empty()
                assert "fii_30d_roll" in view_data.columns # Corrected column name
                assert "dii_30d_roll" in view_data.columns # Corrected column name
            finally: # This finally corresponds to the try block for db connection
                if conn:
                    conn.close()
        finally:
            if os.path.exists(db_path): # Clean up the temp file
                os.remove(db_path)
            # tmp_db_file is automatically closed/deleted when 'with' block exits

    @pytest.mark.asyncio
    async def test_flow_when_scrape_returns_empty(self, mock_config_for_integration, mocker):
        mock_config_for_integration.data.duckdb_path = ":memory:"

        OriginalScraperClass = NSE_FII_DII_Scraper
        created_instances = []

        def class_wrapper(*args, **kwargs):
            instance = OriginalScraperClass(*args, **kwargs)
            instance.db_path = ":memory:" # Ensure this instance uses in-memory for its update_database
            instance.scrape = AsyncMock(return_value=pl.DataFrame()) # Simulate scrape returning empty DataFrame
            mocker.spy(instance, 'update_database') # Spy on its update_database
            created_instances.append(instance)
            return instance

        mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper', side_effect=class_wrapper)
        # Patch the global logger used by update_database internal logging
        mock_logger_warning = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.warning')

        await run_tracker_main()

        assert len(created_instances) == 1, "Scraper instance was not created as expected"
        scraper_instance_used_by_main = created_instances[0]

        scraper_instance_used_by_main.scrape.assert_awaited_once()

        # Check that update_database was called with an empty DataFrame
        scraper_instance_used_by_main.update_database.assert_called_once()
        call_arg = scraper_instance_used_by_main.update_database.call_args[0][0]
        assert isinstance(call_arg, pl.DataFrame), "Argument to update_database was not a Polars DataFrame"
        assert call_arg.is_empty(), "Argument to update_database was not an empty DataFrame"

        # Check logs for appropriate message from update_database when it receives empty data.
        # The method logs a warning: "No data to update"
        assert any(
            "No data to update" in call_args[0][0]
            for call_args in mock_logger_warning.call_args_list
            if call_args[0] # Ensure there are positional arguments in the call
        ), "Expected log message 'No data to update' not found"


class TestMainOrchestratorFunction:
    # Removed import from here: from quandex_core.market_insights.fii_dii_tracker import logger as fii_dii_logger

    @pytest.mark.asyncio
    async def test_main_successful_run(self, mock_config_for_integration, sample_fii_dii_data, mocker, caplog): # Added caplog back
        # log_messages = []
        # def capture_loguru_records(message_obj):
        #     log_messages.append(str(message_obj))
        # logger_id = fii_dii_logger.add(capture_loguru_records, level="INFO")

        # Mock the NSE_FII_DII_Scraper class itself or its instance methods if instance is created in main
        mock_scraper_instance = MagicMock(spec=NSE_FII_DII_Scraper)
        mock_scraper_instance.db_path = ":memory:" # Ensure the mock instance has this attribute
        mock_scraper_instance.scrape = AsyncMock(return_value=sample_fii_dii_data)
        mock_scraper_instance.update_database.return_value = True

        # Patch the constructor to return our mocked instance
        mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper', return_value=mock_scraper_instance)

        # Mock duckdb.connect for the final log message part in main()
        mock_db_conn_final_log = MagicMock()
        # main() expects fetchdf() to return a Pandas DataFrame
        mock_db_conn_final_log.execute().fetchdf.return_value = pandas.DataFrame({
            'date': [date(2024, 1, 1)], 'fii_net_cr': [0.0], 'dii_net_cr': [0.0]
        })
        # Patch duckdb.connect specifically for the context of fii_dii_tracker.main's final block
        mocker.patch('quandex_core.market_insights.fii_dii_tracker.duckdb.connect',
                     return_value=MagicMock(__enter__=MagicMock(return_value=mock_db_conn_final_log)))

        # Mock time.time for predictable duration logging
        # Values made distinct to trace consumption for duration calculation.
        # Expecting duration = 3002.5 - 1000.0 = 2002.5
        mocker.patch('time.time', side_effect=[1000.0, 2000.0, 3002.5, 4000.0, 5000.0, 6000.0])

        await run_tracker_main()

        mock_scraper_instance.scrape.assert_awaited_once()
        # update_database is mocked on the instance, so its actual logic (and logging) won't run
        # We only check it was called and that main's overall logging is correct.
        mock_scraper_instance.update_database.assert_called_once_with(sample_fii_dii_data)

        # fii_dii_logger.remove(logger_id) # Clean up sink
        # full_log_text = "\n".join(log_messages)

        assert "🚀 Starting FII/DII tracker" in caplog.text # Use caplog
        assert "FII/DII tracker update completed in 2.50 seconds." in caplog.text # Use caplog


    @pytest.mark.asyncio
    async def test_main_scrape_fails(self, mock_config_for_integration, mocker, caplog): # Added caplog back
        # log_messages = []
        # def capture_loguru_records(message_obj):
        #     log_messages.append(str(message_obj))
        # logger_id = fii_dii_logger.add(capture_loguru_records, level="INFO") # Capture INFO and ERROR

        mock_scraper_instance = MagicMock(spec=NSE_FII_DII_Scraper)
        mock_scraper_instance.scrape = AsyncMock(return_value=None) # Simulate scrape failure

        mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper', return_value=mock_scraper_instance)
        # Expecting duration = 3001.0 - 1000.0 = 2001.0
        mocker.patch('time.time', side_effect=[1000.0, 2000.0, 2500.0, 3001.0, 4000.0])

        await run_tracker_main()

        mock_scraper_instance.scrape.assert_awaited_once()
        mock_scraper_instance.update_database.assert_not_called() # Should not call if scrape returns None

        # fii_dii_logger.remove(logger_id)
        # full_log_text = "\n".join(log_messages)

        assert "🚀 Starting FII/DII tracker" in caplog.text # Use caplog
        assert "Scraping failed completely" in caplog.text # Use caplog
        assert "FII/DII tracker update completed in 1.00 seconds." in caplog.text # Use caplog


    @pytest.mark.asyncio
    async def test_main_database_update_fails(self, mock_config_for_integration, sample_fii_dii_data, mocker, caplog): # Added caplog back
        # log_messages = []
        # def capture_loguru_records(message_obj):
        #     log_messages.append(str(message_obj))
        # logger_id = fii_dii_logger.add(capture_loguru_records, level="INFO")

        mock_scraper_instance = MagicMock(spec=NSE_FII_DII_Scraper)
        mock_scraper_instance.db_path = ":memory:" # Ensure the mock instance has this attribute
        mock_scraper_instance.scrape = AsyncMock(return_value=sample_fii_dii_data)
        mock_scraper_instance.update_database.return_value = False # Simulate DB update failure

        mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper', return_value=mock_scraper_instance)

        # Mock duckdb.connect for the final log message part in main()
        mock_db_conn_for_final_log = MagicMock()
        # main() expects fetchdf() to return a Pandas DataFrame
        mock_db_conn_for_final_log.execute().fetchdf.return_value = pandas.DataFrame({
            'date': [date(2024, 1, 1)], 'fii_net_cr': [0.0], 'dii_net_cr': [0.0]
        })
        mocker.patch('quandex_core.market_insights.fii_dii_tracker.duckdb.connect',
                     return_value=MagicMock(__enter__=MagicMock(return_value=mock_db_conn_for_final_log)))

        # Expecting duration = 3001.5 - 1000.0 = 2001.5
        mocker.patch('time.time', side_effect=[1000.0, 2000.0, 3001.5, 4000.0, 5000.0, 6000.0])

        await run_tracker_main()

        mock_scraper_instance.scrape.assert_awaited_once()
        mock_scraper_instance.update_database.assert_called_once_with(sample_fii_dii_data)

        # fii_dii_logger.remove(logger_id)
        # full_log_text = "\n".join(log_messages)

        assert "🚀 Starting FII/DII tracker" in caplog.text # Use caplog
        assert "FII/DII tracker update completed in 1.50 seconds." in caplog.text # Use caplog
