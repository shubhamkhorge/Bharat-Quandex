import pytest
import asyncio
import polars as pl
from polars.testing import assert_frame_equal
from datetime import date
from unittest.mock import MagicMock, patch, AsyncMock, ANY

import duckdb
import requests # For requests.exceptions
from requests.cookies import RequestsCookieJar # Import for mocking session cookies

from quandex_core.market_insights.fii_dii_tracker import main as run_tracker_main
from quandex_core.market_insights.fii_dii_tracker import NSE_FII_DII_Scraper # To access constants like API_URL
from quandex_core import config as global_config_module # For mocking

# Attempt to import PlaywrightError, mock if not available
try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:
    PlaywrightError = type('PlaywrightError', (Exception,), {})
    PlaywrightTimeoutError = type('PlaywrightTimeoutError', (PlaywrightError,), {})

# Sample data to be returned by mocks
MOCK_API_JSON_SUCCESS = {
    "fiiDii": [
        {"tradedDate": "26-Aug-2024", "fiiBuy": "1,000.50", "fiiSell": "500.25", "diiBuy": "700.00", "diiSell": "300.75"},
        {"tradedDate": "27-Aug-2024", "fiiBuy": "1,200.00", "fiiSell": "600.00", "diiBuy": "800.00", "diiSell": "400.00"}
    ]
}
EXPECTED_DF_FROM_API_SUCCESS = pl.DataFrame({
    "date": [date(2024, 8, 26), date(2024, 8, 27)],
    "fii_buy_cr": [1000.50, 1200.00], "fii_sell_cr": [500.25, 600.00], "fii_net_cr": [500.25, 600.00],
    "dii_buy_cr": [700.00, 800.00], "dii_sell_cr": [300.75, 400.00], "dii_net_cr": [399.25, 400.00]
})

MOCK_HTML_CONTENT_SUCCESS = """
<html><body>
    <div id="fiiDiiFlow"> <!-- Target selector by scraper -->
        <table>
          <thead>
            <tr><th>Date</th><th>FII Gross Purchase (Rs Cr)</th><th>FII Gross Sales (Rs Cr)</th><th>FII Net Purchase / Sales (Rs Cr)</th>
                <th>DII Gross Purchase (Rs Cr)</th><th>DII Gross Sales (Rs Cr)</th><th>DII Net Purchase / Sales (Rs Cr)</th></tr>
          </thead>
          <tbody>
            <tr><td>25-Aug-2024</td><td>2,000.10</td><td>1,500.00</td><td>500.10</td><td>1,700.20</td><td>1,300.00</td><td>400.20</td></tr>
          </tbody>
        </table>
    </div>
</body></html>
"""
EXPECTED_DF_FROM_HTML_SUCCESS = pl.DataFrame({
    "date": [date(2024, 8, 25)],
    "fii_buy_cr": [2000.10], "fii_sell_cr": [1500.00], "fii_net_cr": [500.10],
    "dii_buy_cr": [1700.20], "dii_sell_cr": [1300.00], "dii_net_cr": [400.20]
})


@pytest.fixture(autouse=True) # Auto-use for all tests in this module
def mock_global_config_for_e2e(mocker):
    """Mocks global config to use in-memory DB and specific URLs for all E2E tests."""
    mock_cfg = MagicMock()
    mock_cfg.data.duckdb_path = ":memory:" # Critical for E2E tests
    mock_cfg.scraping.user_agents = ["test_e2e_user_agent"]
    mock_cfg.scraping.max_retries = 1  # Explicitly set for tests
    mock_cfg.scraping.retry_delay = 0.01 # Explicitly set for tests
    # Use actual URLs from NSE_FII_DII_Scraper constants for matching in mocks
    mock_cfg.scraping.nse_fii_dii_home_url = "https://www.nseindia.com"
    mock_cfg.scraping.nse_fii_dii_api_url = "https://www.nseindia.com/api/fiidiiTradeReact"
    mock_cfg.scraping.nse_fii_dii_html_url = "https://www.nseindia.com/market-data/fii-dii-activity"
    mock_cfg.market.trading_holidays = [] # Simplify by having no holidays for mock data generation tests

    # Patch this config into the module where `main` and `NSE_FII_DII_Scraper` will see it
    # Note: db_path will be overridden in tests that need specific DB setups (like temp files)
    mocker.patch('quandex_core.market_insights.fii_dii_tracker.config', mock_cfg)
    return mock_cfg

import tempfile # For creating temporary database files
import logging # For Loguru propagation
from quandex_core.market_insights.fii_dii_tracker import logger as fii_dii_logger_e2e # Import for patching sink

# This fixture will apply to all tests in this file/class.
@pytest.fixture(autouse=True)
def setup_loguru_to_caplog_propagation_e2e(caplog):
    class PropagateHandler(logging.Handler):
        def emit(self, record): # This 'record' is a standard logging.LogRecord
            logging.getLogger(record.name).handle(record)

    # Add the propagation handler to Loguru.
    # format="{message}" ensures record.msg in the handler is the clean message.
    handler_id = fii_dii_logger_e2e.add(PropagateHandler(), format="{message}", level="DEBUG")
    yield
    fii_dii_logger_e2e.remove(handler_id)

import os # For removing temp file before DuckDB uses it

@pytest.mark.asyncio
class TestFiiDiiE2EWorkflow:

    async def test_e2e_api_success_path(self, mock_global_config_for_e2e, mocker, caplog): # mock_global_config_for_e2e to set config
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db_file: # delete=False initially
            db_path = tmp_db_file.name

        try:
            # Ensure DuckDB creates the file, not just uses an empty one from tempfile
            if os.path.exists(db_path):
                os.remove(db_path)

            # Override the global config's db_path for this test
            mock_global_config_for_e2e.data.duckdb_path = db_path

            # 1. Setup Mocks
            mock_session_instance = MagicMock(spec=requests.Session)
            mock_session_instance.headers = {}
            mock_session_instance.cookies = MagicMock(spec=RequestsCookieJar)
            mock_session_instance.cookies.get_dict.return_value = {"session_cookie_e2e": "val_from_session_mock"}


            def mock_get_router(url, **kwargs):
                response = MagicMock(spec=requests.Response)
                response.cookies = RequestsCookieJar()
                response.cookies.set("bm_sv", "mock_bm_sv_cookie")

                if url == "https://www.nseindia.com":
                    response.status_code = 200
                elif url == "https://www.nseindia.com/api/fiidiiTradeReact":
                    response.status_code = 200
                    response.json.return_value = MOCK_API_JSON_SUCCESS
                    response.content = str(MOCK_API_JSON_SUCCESS).encode()
                else:
                    response.status_code = 404
                return response

            mock_session_instance.get.side_effect = mock_get_router
            mocker.patch('requests.Session', return_value=mock_session_instance)

            mock_async_playwright_cm = AsyncMock()
            mock_async_playwright_cm.__aenter__.side_effect = Exception("Playwright should not be called in API success path")
            mocker.patch('quandex_core.market_insights.fii_dii_tracker.async_playwright', return_value=mock_async_playwright_cm)

            # 2. Execution
            await run_tracker_main()

            # 3. Verification
            conn = None
            try:
                conn = duckdb.connect(database=db_path, read_only=False) # Use temp db_path
                db_data = conn.execute("SELECT * FROM institutional_flows ORDER BY date").pl()

                # Convert date columns to actual date objects if they are strings in expected
                expected_df = EXPECTED_DF_FROM_API_SUCCESS.with_columns(pl.col("date").cast(pl.Date))
                assert_frame_equal(db_data, expected_df, check_dtype=True)

                view_data = conn.execute("SELECT * FROM v_institutional_trends").pl()
                assert not view_data.is_empty()
                assert len(view_data) == len(expected_df) # Simple check for now
            finally:
                if conn:
                    conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

        assert "Fetched FII/DII data from API" in caplog.text # Actual log message
        # The following assertion was problematic as this exact phrase is not in core logs
        # assert "Database updated successfully with FII/DII data." in caplog.text


    async def test_e2e_api_fail_playwright_success_path(self, mock_global_config_for_e2e, mocker, caplog):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db_file: # delete=False initially
            db_path = tmp_db_file.name

        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            mock_global_config_for_e2e.data.duckdb_path = db_path

            # 1. Setup Mocks
            # Mock requests.Session.get to fail for API calls
            mock_session_instance = MagicMock(spec=requests.Session)
            mock_session_instance.headers = {}
            mock_session_instance.cookies = MagicMock(spec=RequestsCookieJar)
            mock_session_instance.cookies.get_dict.return_value = {"session_cookie_e2e_playwright": "val_from_session_mock"}
            def mock_get_router_api_fail(url, **kwargs):
                response = MagicMock(spec=requests.Response)
                response.cookies = RequestsCookieJar()
                response.cookies.set("bm_sv", "mock_bm_sv_cookie")

                if url == "https://www.nseindia.com":
                    response.status_code = 200
                elif url == "https://www.nseindia.com/api/fiidiiTradeReact":
                    response.status_code = 500
                    response.raise_for_status.side_effect = requests.exceptions.HTTPError("API Server Error")
                else:
                    response.status_code = 404
                return response
            mock_session_instance.get.side_effect = mock_get_router_api_fail
            mocker.patch('requests.Session', return_value=mock_session_instance)

            # Mock Playwright to succeed
            mock_async_playwright_cm = AsyncMock()
            mock_playwright_instance = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_async_playwright_cm.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            mock_browser.new_context.return_value = mock_context
            mock_context.new_page.return_value = mock_page
            mock_page.content.return_value = MOCK_HTML_CONTENT_SUCCESS

            mocker.patch('quandex_core.market_insights.fii_dii_tracker.async_playwright', return_value=mock_async_playwright_cm)
            mocker.patch('asyncio.sleep', new_callable=AsyncMock) # For retries in playwright

            # 2. Execution
            await run_tracker_main()

            # 3. Verification
            conn = None
            try:
                conn = duckdb.connect(database=db_path, read_only=False) # Use temp db_path
                db_data = conn.execute("SELECT * FROM institutional_flows ORDER BY date").pl()
                expected_df = EXPECTED_DF_FROM_HTML_SUCCESS.with_columns(pl.col("date").cast(pl.Date))
                assert_frame_equal(db_data, expected_df, check_dtype=True)

                view_data = conn.execute("SELECT * FROM v_institutional_trends").pl()
                assert not view_data.is_empty()
            finally:
                if conn:
                    conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

            assert "API failed, trying Playwright..." in caplog.text # Actual log message
            # The following assertions were problematic as these exact phrases are not in core logs
            # assert "FII/DII data scraped successfully via Playwright." in caplog.text
            # assert "Database updated successfully with FII/DII data." in caplog.text

    async def test_e2e_all_fail_mock_data_path(self, mocker, caplog):
        # This test uses :memory: DB via mock_global_config_for_e2e by default,
        # and update_database is fully mocked, so no temp file needed here.

        # 1. Setup Mocks
        # Mock requests.Session.get to fail for API calls
        mock_session_instance = MagicMock(spec=requests.Session)
        mock_session_instance.get.side_effect = requests.exceptions.RequestException("API Network Error")
        mocker.patch('requests.Session', return_value=mock_session_instance)

        # Mock Playwright to fail
        mock_async_playwright_cm = AsyncMock()
        mock_playwright_instance = AsyncMock()
        mock_browser = AsyncMock() # Define mock_browser here
        mock_async_playwright_cm.__aenter__.return_value = mock_playwright_instance
        mock_playwright_instance.chromium.launch.return_value = mock_browser # Now mock_browser is defined
        mock_browser.new_context.side_effect = PlaywrightError("Playwright context error")

        mocker.patch('quandex_core.market_insights.fii_dii_tracker.async_playwright', return_value=mock_async_playwright_cm)
        mocker.patch('asyncio.sleep', new_callable=AsyncMock)

        # Spy on _generate_mock_data to ensure it's called (it's a method of the instance)
        # We need to patch it on the instance that will be created inside main()
        # Patch methods directly on the NSE_FII_DII_Scraper class
        # This allows the actual scraper.scrape() logic to run using these mocked sub-methods.
        mock_generated_data = pl.DataFrame({"date": [date.today()], "fii_buy_cr": [1.0]})

        patch_scrape_api = mocker.patch.object(
            NSE_FII_DII_Scraper, 'scrape_with_api',
            new_callable=AsyncMock, return_value=None
        )
        patch_scrape_pw = mocker.patch.object(
            NSE_FII_DII_Scraper, 'scrape_with_playwright',
            new_callable=AsyncMock, return_value=None
        )
        patch_generate_mock = mocker.patch.object(
            NSE_FII_DII_Scraper, '_generate_mock_data',
            return_value=mock_generated_data # _generate_mock_data is synchronous
        )
        # The update_database method on the instance created by main will be called.
        # We need to ensure it's properly configured (e.g. db_path) and returns True.
        # Patching update_database on the class means any instance will use this mock.
        patch_update_db = mocker.patch.object(
            NSE_FII_DII_Scraper, 'update_database',
            return_value=True # update_database is synchronous
        )

        # 2. Execution
        await run_tracker_main()

        # 3. Verification
        patch_scrape_api.assert_called_once()
        patch_scrape_pw.assert_awaited_once()
        patch_generate_mock.assert_called_once()
        patch_update_db.assert_called_once_with(mock_generated_data)

        # Check logs
        # The log "All scraping methods failed. Falling back to mock data." comes from scraper.scrape()
        # The log "Database updated successfully with FII/DII data." is from the test assertion logic,
        # based on update_database returning True. The actual update_database logs "Updated X records".
        # The main() function does not log "Database updated successfully..."
        # Check for the specific error log message from the scraper's scrape method
        expected_log_message = "All scraping methods failed, using mock data" # Message from scraper.scrape() when fallback occurs
        assert expected_log_message in caplog.text
        # If patch_update_db returns True, main() considers it a success.
        # The specific log "Database updated successfully..." is not from the core code.
        # We can assert that the overall process completes and the mock update was called.

        # For this test, since update_database is fully mocked to return True,
        # we don't verify the DB content here. The focus is on the fallback logic.
        # If we wanted to verify DB content with mock data, update_database would need to be
        # a spy on the real method, and db_path handled (e.g. temp file).
        # For simplicity, this E2E test focuses on the fallback to _generate_mock_data and its data being passed to update_database.
        # A separate integration test (like test_successful_data_flow_to_in_memory_db) already confirms
        # that update_database correctly writes to DB.
