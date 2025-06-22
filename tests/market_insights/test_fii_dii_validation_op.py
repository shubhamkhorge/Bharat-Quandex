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

# Attempt to import PlaywrightError, mock if not available
try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:
    PlaywrightError = type('PlaywrightError', (Exception,), {})
    PlaywrightTimeoutError = type('PlaywrightTimeoutError', (PlaywrightError,), {})


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

    async def test_simulate_javascript_disabled_impact_playwright(self, mocker, caplog):
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
        # We need to patch the class that `main` instantiates.
        mock_scraper_class = mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper')
        mock_scraper_instance = mock_scraper_class.return_value

        mock_scraper_instance.scrape_with_api.return_value = None # Consistent with API mock
        mock_scraper_instance.scrape_with_playwright = AsyncMock(return_value=None) # Reflects that deficient HTML leads to None

        # Let the original _generate_mock_data run or mock its return for simplicity here
        mock_generated_data = pl.DataFrame({"date": [date.today()], "fii_buy_cr": [10.0]})
        mock_scraper_instance._generate_mock_data.return_value = mock_generated_data

        mock_scraper_instance.update_database.return_value = True # Assume DB update is fine with mock data
        mock_scraper_instance.db_path = ":memory:" # Ensure the instance uses in-memory

        # 4. Execution
        await run_tracker_main()

        # 5. Verification
        mock_scraper_instance.scrape_with_api.assert_called_once()
        mock_scraper_instance.scrape_with_playwright.assert_awaited_once()
        mock_scraper_instance._generate_mock_data.assert_called_once()
        mock_scraper_instance.update_database.assert_called_once_with(mock_generated_data)

        assert "API scraping failed" in caplog.text # Or similar from actual run
        assert "Playwright scraping failed or returned no data" in caplog.text # Or similar
        assert "All scraping methods failed. Falling back to mock data." in caplog.text


    def test_simulate_duckdb_read_only_scenario(self, mock_global_config_for_validation_op, mocker):
        """
        Tests that update_database handles errors correctly if DuckDB is effectively read-only
        (e.g., CREATE TABLE or INSERT fails).
        """
        # We need a scraper instance, but its __init__ might try to connect.
        # Patch _initialize_db_schema to prevent DB ops during init for this test.
        with patch.object(NSE_FII_DII_Scraper, '_initialize_db_schema', return_value=None):
            scraper = NSE_FII_DII_Scraper()

        # This test directly calls update_database, so we control the db_path here if needed,
        # but the global mock should already set it to :memory: or a test specific one.
        # Forcing it to ensure:
        scraper.db_path = ":memory:"

        sample_df = pl.DataFrame({"date": [date(2024, 1, 1)], "fii_buy_cr": [100.0]})

        # Mock duckdb.connect to return a connection that fails on execute for DDL/DML
        mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)

        def execute_behavior(sql_query, *args, **kwargs):
            query_upper = sql_query.upper()
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
        mock_logger_error = mocker.patch.object(scraper.logger, 'error')

        result = scraper.update_database(sample_df)

        assert result is False
        duckdb.connect.assert_called_once_with(database=scraper.db_path, read_only=False)
        mock_conn.register.assert_called_once_with("temp_fii_dii", sample_df) # Registration attempt

        # Check that execute was called with an attempt to create table (which then failed)
        assert any("CREATE TABLE IF NOT EXISTS institutional_flows" in call_args[0][0] for call_args in mock_conn.execute.call_args_list)

        mock_logger_error.assert_called_once()
        assert "Error during database update" in mock_logger_error.call_args[0][0]
        assert "Simulated Read-Only DB: Cannot write" in str(mock_logger_error.call_args[0][1]) # Check exception in log
        mock_conn.close.assert_called_once() # Ensure connection is closed
