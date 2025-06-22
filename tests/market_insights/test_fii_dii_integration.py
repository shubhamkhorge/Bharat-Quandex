import pytest
import asyncio
import polars as pl
from polars.testing import assert_frame_equal
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

import duckdb

from quandex_core.market_insights.fii_dii_tracker import NSE_FII_DII_Scraper
from quandex_core.market_insights.fii_dii_tracker import main as run_tracker_main
from quandex_core import config as global_config_module # For mocking

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

    @pytest.mark.asyncio
    async def test_successful_data_flow_to_in_memory_db(self, mock_config_for_integration, sample_fii_dii_data, mocker):
        # Initialize scraper and override db_path to use in-memory DuckDB
        # _initialize_db_schema is not a method in the provided source code of NSE_FII_DII_Scraper
        # Its __init__ does not call such a method.
        # DB schema is handled directly in update_database method or by direct duckdb calls in __init__.
        scraper = NSE_FII_DII_Scraper()

        scraper.db_path = ":memory:"
        # Call _initialize_db_schema manually now with the in-memory path implicitly used by duckdb.connect
        # Or, ensure update_database calls it if not exists
        # The current NSE_FII_DII_Scraper calls _initialize_db_schema in its __init__.
        # So, we need to ensure that call uses the in-memory path.
        # The cleanest way: set config.data.duckdb_path to :memory: via the fixture,
        # or directly patch scraper.db_path *before* any DB connection is made by it.

        # Re-initialize with patched config for in-memory to be used by _initialize_db_schema
        mock_config_for_integration.data.duckdb_path = ":memory:"
        # This re-init is a bit complex due to __init__ side effects.
        # A better approach: pass db_path to constructor or have a method to set it post-init before connecting.
        # For this test, let's assume update_database will create table if not exists.
        # Or, we can connect to ":memory:" and pass the connection object to update_database if it allowed.

        # Simpler: Let update_database handle the connection and table creation.
        # We just need to ensure the scraper instance uses the in-memory path for its operations.
        # The fixture `mock_config_for_integration` is already patched into the module.
        # We just need to ensure the scraper instance uses the in-memory path for this test.
        scraper.db_path = ":memory:" # This is the critical part for update_database

        # Mock the scrape method to return our sample data
        # Since scrape is async, its mock should be awaitable if called with await
        mock_scrape_method = AsyncMock(return_value=sample_fii_dii_data)
        mocker.patch.object(scraper, 'scrape', mock_scrape_method)

        # Spy on update_database to ensure it's called by main
        update_db_spy = mocker.spy(scraper, 'update_database')

        # Run the main orchestrator function
        await run_tracker_main()

        mock_scrape_method.assert_awaited_once()
        update_db_spy.assert_called_once_with(sample_fii_dii_data)

        # Now, connect to the same in-memory database to verify
        conn = None
        try:
            conn = duckdb.connect(database=":memory:", read_only=False)

            # Query the institutional_flows table
            db_data = conn.execute("SELECT * FROM institutional_flows ORDER BY date").pl()
            assert_frame_equal(db_data, sample_fii_dii_data, check_dtype=True)

            # Query the v_institutional_trends view
            # This view requires some data to be non-empty.
            view_data = conn.execute("SELECT * FROM v_institutional_trends").pl()
            assert not view_data.is_empty()
            assert "rolling_fii_net_sum" in view_data.columns
            assert "rolling_dii_net_sum" in view_data.columns

        finally:
            if conn:
                conn.close()

    @pytest.mark.asyncio
    async def test_flow_when_scrape_returns_empty(self, mock_config_for_integration, mocker):
        # Scraper setup
        # _initialize_db_schema is not a method in the provided source code of NSE_FII_DII_Scraper
        scraper = NSE_FII_DII_Scraper()
        scraper.db_path = ":memory:" # Use in-memory to avoid file system writes

        mock_scrape_method = AsyncMock(return_value=pl.DataFrame()) # Empty DataFrame
        mocker.patch.object(scraper, 'scrape', mock_scrape_method)

        update_db_spy = mocker.spy(scraper, 'update_database')
        mock_logger_info = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.info')

        await run_tracker_main()

        mock_scrape_method.assert_awaited_once()
        update_db_spy.assert_called_once_with(pl.DataFrame())

        # Check logs for appropriate message (e.g., "No data to update" or similar from update_database)
        # This depends on update_database's logging when it receives empty data.
        # The current update_database returns False and logs "No data provided to update_database."
        assert any("No data provided to update_database" in call_args[0][0] for call_args in mock_logger_info.call_args_list)


class TestMainOrchestratorFunction:

    @pytest.mark.asyncio
    async def test_main_successful_run(self, mock_config_for_integration, sample_fii_dii_data, mocker, caplog):
        # Mock the NSE_FII_DII_Scraper class itself or its instance methods if instance is created in main
        mock_scraper_instance = MagicMock(spec=NSE_FII_DII_Scraper)
        mock_scraper_instance.db_path = ":memory:" # Ensure the mock instance has this attribute
        mock_scraper_instance.scrape = AsyncMock(return_value=sample_fii_dii_data)
        mock_scraper_instance.update_database.return_value = True

        # Patch the constructor to return our mocked instance
        mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper', return_value=mock_scraper_instance)

        # Mock duckdb.connect for the final log message part in main()
        mock_db_conn_final_log = MagicMock()
        mock_db_conn_final_log.execute().fetchdf.return_value = pl.DataFrame({
            'date': [date(2024, 1, 1)], 'fii_net_cr': [0.0], 'dii_net_cr': [0.0]
        })
        # Patch duckdb.connect specifically for the context of fii_dii_tracker.main's final block
        mocker.patch('quandex_core.market_insights.fii_dii_tracker.duckdb.connect',
                     return_value=MagicMock(__enter__=MagicMock(return_value=mock_db_conn_final_log)))

        # Mock time.time for predictable duration logging
        mocker.patch('time.time', side_effect=[1000.0, 1002.5]) # Start time, end time

        await run_tracker_main()

        mock_scraper_instance.scrape.assert_awaited_once()
        mock_scraper_instance.update_database.assert_called_once_with(sample_fii_dii_data)

        # Check logs (using caplog fixture from pytest)
        assert "Starting FII/DII tracker update..." in caplog.text
        assert "FII/DII data scraped successfully." in caplog.text
        assert "Database updated successfully with FII/DII data." in caplog.text
        assert "FII/DII tracker update completed in 2.50 seconds." in caplog.text


    @pytest.mark.asyncio
    async def test_main_scrape_fails(self, mock_config_for_integration, mocker, caplog):
        mock_scraper_instance = MagicMock(spec=NSE_FII_DII_Scraper)
        mock_scraper_instance.scrape = AsyncMock(return_value=None) # Simulate scrape failure

        mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper', return_value=mock_scraper_instance)
        mocker.patch('time.time', side_effect=[1000.0, 1001.0])

        await run_tracker_main()

        mock_scraper_instance.scrape.assert_awaited_once()
        mock_scraper_instance.update_database.assert_not_called() # Should not call if scrape returns None

        assert "Scraping failed completely" in caplog.text # Changed this line
        assert "Scraping FII/DII data failed or returned no data." in caplog.text # This line might be redundant if the one above is the true "complete" failure log. Or it might be from scraper internal logs.
        assert "FII/DII tracker update completed in 1.00 seconds." in caplog.text # Duration check


    @pytest.mark.asyncio
    async def test_main_database_update_fails(self, mock_config_for_integration, sample_fii_dii_data, mocker, caplog):
        mock_scraper_instance = MagicMock(spec=NSE_FII_DII_Scraper)
        mock_scraper_instance.db_path = ":memory:" # Ensure the mock instance has this attribute
        mock_scraper_instance.scrape = AsyncMock(return_value=sample_fii_dii_data)
        mock_scraper_instance.update_database.return_value = False # Simulate DB update failure

        mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper', return_value=mock_scraper_instance)

        # Mock duckdb.connect for the final log message part in main()
        mock_db_conn_for_final_log = MagicMock()
        mock_db_conn_for_final_log.execute().fetchdf.return_value = pl.DataFrame({
            'date': [date(2024, 1, 1)], 'fii_net_cr': [0.0], 'dii_net_cr': [0.0]
        })
        mocker.patch('quandex_core.market_insights.fii_dii_tracker.duckdb.connect',
                     return_value=MagicMock(__enter__=MagicMock(return_value=mock_db_conn_for_final_log)))

        mocker.patch('time.time', side_effect=[1000.0, 1001.5])

        await run_tracker_main()

        mock_scraper_instance.scrape.assert_awaited_once()
        mock_scraper_instance.update_database.assert_called_once_with(sample_fii_dii_data)

        assert "Starting FII/DII tracker update..." in caplog.text
        assert "FII/DII data scraped successfully." in caplog.text
        assert "Failed to update database with FII/DII data." in caplog.text
        assert "FII/DII tracker update completed in 1.50 seconds." in caplog.text
