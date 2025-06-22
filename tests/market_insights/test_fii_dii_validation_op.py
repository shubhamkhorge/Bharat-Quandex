import pytest
import asyncio
import polars as pl
from datetime import date
from unittest.mock import MagicMock, patch, AsyncMock, ANY

import duckdb # For duckdb.IOException etc.
import requests # For requests.exceptions

from quandex_core.market_insights.fii_dii_tracker import main as run_tracker_main
from quandex_core.market_insights.fii_dii_tracker import NSE_FII_DII_Scraper
from quandex_core import config as global_config_module
import logging # For Loguru propagation

# Attempt to import PlaywrightError, mock if not available
try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:
    PlaywrightError = type('PlaywrightError', (Exception,), {})
    PlaywrightTimeoutError = type('PlaywrightTimeoutError', (PlaywrightError,), {})

from quandex_core.market_insights.fii_dii_tracker import logger as fii_dii_logger_val # Import for patching sink


@pytest.fixture(autouse=True)
def setup_loguru_to_caplog_propagation_validation(caplog):
    class PropagateHandler(logging.Handler):
        def emit(self, record): # This 'record' is a standard logging.LogRecord
            logging.getLogger(record.name).handle(record)

    # Add the propagation handler to Loguru.
    # format="{message}" ensures record.msg in the handler is the clean message.
    handler_id = fii_dii_logger_val.add(PropagateHandler(), format="{message}", level="DEBUG")
    yield
    fii_dii_logger_val.remove(handler_id)

@pytest.fixture(autouse=True) # Auto-use for all tests in this module
def mock_global_config_for_validation_op(mocker):
    """Mocks global config to use in-memory DB and specific URLs for tests."""
    mock_cfg = MagicMock()
    mock_cfg.data.duckdb_path = ":memory:"
    mock_cfg.scraping.user_agents = ["test_validation_op_user_agent"]
    mock_cfg.scraping.nse_fii_dii_home_url = "https://www.nseindia.com"
    mock_cfg.scraping.nse_fii_dii_api_url = "https://www.nseindia.com/api/fiidiiTradeReact"
    mock_cfg.scraping.nse_fii_dii_html_url = "https://www.nseindia.com/market-data/fii-dii-activity"
    mock_cfg.market.trading_holidays = []

    mocker.patch('quandex_core.market_insights.fii_dii_tracker.config', mock_cfg)
    return mock_cfg


@pytest.mark.asyncio
class TestFiiDiiFailureScenarios:

    async def test_simulate_javascript_disabled_impact_playwright(self, mocker, caplog): # Added caplog back
        """
        Simulates a scenario where API fails, and Playwright fetches HTML
        that is missing critical data (as if JS was disabled or didn't render content),
        leading to fallback to mock data.
        """
        # 1. Mock API to fail
        mock_session_instance = MagicMock(spec=requests.Session)
        mock_session_instance.get.side_effect = requests.exceptions.RequestException("API Network Error")
        mocker.patch('requests.Session', return_value=mock_session_instance)

        # 2. Mock Playwright to return "deficient" HTML
        mock_async_playwright_cm = AsyncMock()
        mock_playwright_instance = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_async_playwright_cm.__aenter__.return_value = mock_playwright_instance
        mock_playwright_instance.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        # HTML content that _parse_html_table would likely fail to parse or return None/empty from
        # For example, the main div is present, but the table inside it is missing.
        deficient_html_content = "<html><body><div id='fiiDiiFlow'></div></body></html>"
        mock_page.content.return_value = deficient_html_content

        mocker.patch('quandex_core.market_insights.fii_dii_tracker.async_playwright', return_value=mock_async_playwright_cm)
        mocker.patch('asyncio.sleep', new_callable=AsyncMock) # For any retries

        # 3. Spy on _generate_mock_data (it will be called on the instance created by main)
        # Patch methods directly on the NSE_FII_DII_Scraper class so main() uses a real instance
        # with these specific methods mocked.
        mock_generated_data = pl.DataFrame({"date": [date.today()], "fii_buy_cr": [10.0]})

        patch_scrape_api = mocker.patch.object(
            NSE_FII_DII_Scraper, 'scrape_with_api',
            new_callable=AsyncMock, return_value=None
        )
        # For this test, scrape_with_playwright should also return None due to "deficient" HTML
        patch_scrape_pw = mocker.patch.object(
            NSE_FII_DII_Scraper, 'scrape_with_playwright',
            new_callable=AsyncMock, return_value=None # Simulate parsing deficient HTML leads to None
        )
        patch_generate_mock = mocker.patch.object(
            NSE_FII_DII_Scraper, '_generate_mock_data',
            return_value=mock_generated_data
        )
        patch_update_db = mocker.patch.object(
            NSE_FII_DII_Scraper, 'update_database',
            return_value=True # Assume DB update itself is fine
        )

        # 4. Execution
        await run_tracker_main()

        # 5. Verification
        patch_scrape_api.assert_called_once()
        patch_scrape_pw.assert_awaited_once()
        patch_generate_mock.assert_called_once()
        patch_update_db.assert_called_once_with(mock_generated_data)

        # (Original duplicate call to run_tracker_main() and assertions were here - now removed)

        assert "API failed, trying Playwright..." in caplog.text # Logged by scraper.scrape()
        assert "All scraping methods failed, using mock data" in caplog.text # Logged by scraper.scrape()

    # Removed @pytest.mark.asyncio as this is a synchronous test
    def test_simulate_duckdb_read_only_scenario(self, mock_global_config_for_validation_op, mocker):
        """
        Tests that update_database handles errors correctly if DuckDB is effectively read-only
        (e.g., CREATE TABLE or INSERT fails).
        """
        # _initialize_db_schema does not exist on NSE_FII_DII_Scraper.
        # Instantiation of the scraper does not perform DB operations.
        scraper = NSE_FII_DII_Scraper()

        # This test directly calls update_database.
        # The global mock mock_global_config_for_validation_op ensures config.data.duckdb_path is :memory:
        # So, scraper.db_path (set in __init__ from config) will be :memory:.
        # Forcing it to ensure:
        scraper.db_path = ":memory:"

        sample_df = pl.DataFrame({"date": [date(2024, 1, 1)], "fii_buy_cr": [100.0]})

        # Mock duckdb.connect to return a connection that fails on execute for DDL/DML
        mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn) # Make it a context manager

        def exit_calls_close(*args):
            mock_conn.close() # Call the mock close method
            return None # __exit__ should return None or bool

        mock_conn.__exit__ = MagicMock(side_effect=exit_calls_close)

        def execute_behavior(sql_query, *args, **kwargs):
            query_upper = sql_query.strip().upper() # Added .strip()
            if query_upper.startswith("CREATE TABLE") or \
               query_upper.startswith("INSERT INTO") or \
               query_upper.startswith("CREATE OR REPLACE VIEW"):
                raise duckdb.IOException("Simulated Read-Only DB: Cannot write")
            # Allow other queries like PRAGMA or SELECT to pass if any were made (though not expected here)
            mock_relation = MagicMock()
            mock_relation.fetchall.return_value = []
            return mock_relation

        mock_conn.execute.side_effect = execute_behavior
        mock_conn.register = MagicMock() # Allow register to work
        mock_conn.close = MagicMock()    # Allow close to be called

        mocker.patch('duckdb.connect', return_value=mock_conn)
        # Patch the global logger used by update_database
        mock_logger_error = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.error')

        result = scraper.update_database(sample_df)

        assert result is False
        duckdb.connect.assert_called_once_with(scraper.db_path, read_only=False) # Match positional db_path
        mock_conn.register.assert_not_called() # Should not be called if execute fails before it

        # Check that execute was called with an attempt to create table (which then failed)
        assert any("CREATE TABLE IF NOT EXISTS institutional_flows" in call_args[0][0] for call_args in mock_conn.execute.call_args_list)

        mock_logger_error.assert_called_once()
        logged_message = mock_logger_error.call_args[0][0]
        assert "Database update failed:" in logged_message # Match actual log format part 1
        assert "Simulated Read-Only DB: Cannot write" in logged_message # Match actual log format part 2 (exception stringified)
        mock_conn.close.assert_called_once() # Ensure connection is closed
