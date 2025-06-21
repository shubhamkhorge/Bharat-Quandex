import pytest
import asyncio
import polars as pl
from polars.testing import assert_frame_equal
from datetime import date
from unittest.mock import MagicMock, patch, AsyncMock, ANY

import duckdb
import requests # For requests.exceptions

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
    # Use actual URLs from NSE_FII_DII_Scraper constants for matching in mocks
    mock_cfg.scraping.nse_fii_dii_home_url = NSE_FII_DII_Scraper.HOME_URL
    mock_cfg.scraping.nse_fii_dii_api_url = NSE_FII_DII_Scraper.API_URL
    mock_cfg.scraping.nse_fii_dii_html_url = NSE_FII_DII_Scraper.HTML_URL
    mock_cfg.market.trading_holidays = [] # Simplify by having no holidays for mock data generation tests

    # Patch this config into the module where `main` and `NSE_FII_DII_Scraper` will see it
    mocker.patch('quandex_core.market_insights.fii_dii_tracker.config', mock_cfg)
    return mock_cfg


@pytest.mark.asyncio
class TestFiiDiiE2EWorkflow:

    async def test_e2e_api_success_path(self, mocker, caplog):
        # 1. Setup Mocks
        mock_session_instance = MagicMock(spec=requests.Session)

        def mock_get_router(url, **kwargs):
            response = MagicMock()
            if url == NSE_FII_DII_Scraper.HOME_URL:
                response.status_code = 200
                response.cookies.get_dict.return_value = {"bm_sv": "mock_bm_sv_cookie"}
            elif url == NSE_FII_DII_Scraper.API_URL:
                response.status_code = 200
                response.json.return_value = MOCK_API_JSON_SUCCESS
                response.content = str(MOCK_API_JSON_SUCCESS).encode() # if content is checked
            else:
                response.status_code = 404
            return response

        mock_session_instance.get.side_effect = mock_get_router
        mocker.patch('requests.Session', return_value=mock_session_instance)

        # Ensure Playwright path is not taken by making it fail if called
        mock_async_playwright_cm = AsyncMock()
        mock_async_playwright_cm.__aenter__.side_effect = Exception("Playwright should not be called in API success path")
        mocker.patch('quandex_core.market_insights.fii_dii_tracker.async_playwright', return_value=mock_async_playwright_cm)

        # 2. Execution
        await run_tracker_main()

        # 3. Verification
        conn = None
        try:
            conn = duckdb.connect(database=":memory:", read_only=False)
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

        assert "FII/DII data scraped successfully via API." in caplog.text
        assert "Database updated successfully with FII/DII data." in caplog.text


    async def test_e2e_api_fail_playwright_success_path(self, mocker, caplog):
        # 1. Setup Mocks
        # Mock requests.Session.get to fail for API calls
        mock_session_instance = MagicMock(spec=requests.Session)
        def mock_get_router_api_fail(url, **kwargs):
            response = MagicMock()
            if url == NSE_FII_DII_Scraper.HOME_URL: # Home URL for API still needs to work for cookie
                response.status_code = 200
                response.cookies.get_dict.return_value = {"bm_sv": "mock_bm_sv_cookie"}
            elif url == NSE_FII_DII_Scraper.API_URL:
                response.status_code = 500 # Simulate API error
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
            conn = duckdb.connect(database=":memory:", read_only=False)
            db_data = conn.execute("SELECT * FROM institutional_flows ORDER BY date").pl()
            expected_df = EXPECTED_DF_FROM_HTML_SUCCESS.with_columns(pl.col("date").cast(pl.Date))
            assert_frame_equal(db_data, expected_df, check_dtype=True)

            view_data = conn.execute("SELECT * FROM v_institutional_trends").pl()
            assert not view_data.is_empty()
        finally:
            if conn:
                conn.close()

        assert "API scraping failed or returned no data. Attempting Playwright." in caplog.text
        assert "FII/DII data scraped successfully via Playwright." in caplog.text
        assert "Database updated successfully with FII/DII data." in caplog.text

    async def test_e2e_all_fail_mock_data_path(self, mocker, caplog):
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
        # So, we patch the class, then check the method on the instance returned by the patched class
        mock_scraper_class = mocker.patch('quandex_core.market_insights.fii_dii_tracker.NSE_FII_DII_Scraper')
        mock_scraper_instance = mock_scraper_class.return_value # This is the instance that main will use

        # Configure the mocked instance's methods
        mock_scraper_instance.scrape_with_api.return_value = None # Ensure this is how API failure is propagated
        mock_scraper_instance.scrape_with_playwright = AsyncMock(return_value=None) # Ensure this is how PW failure is propagated

        # The actual _generate_mock_data will run on this instance
        # We can spy on it if we let the original method run, or mock its return value.
        # Let's spy on the original method of the *actual* (but config-mocked) scraper class for this test
        # to verify it's called AND it populates the DB.
        # This requires careful layering of mocks.
        # Alternative: Let the patched mock_scraper_instance handle _generate_mock_data
        mock_generated_data = pl.DataFrame({"date": [date.today()], "fii_buy_cr": [1.0]}) # Simplified mock data
        mock_scraper_instance._generate_mock_data.return_value = mock_generated_data

        # Ensure update_database is also on the mocked instance
        mock_scraper_instance.update_database.return_value = True # Assume DB update itself works with generated data
        mock_scraper_instance.db_path = ":memory:" # Ensure mocked instance uses in-memory for its update_database


        # 2. Execution
        await run_tracker_main()

        # 3. Verification
        mock_scraper_instance.scrape_with_api.assert_called_once()
        mock_scraper_instance.scrape_with_playwright.assert_awaited_once()
        mock_scraper_instance._generate_mock_data.assert_called_once()
        mock_scraper_instance.update_database.assert_called_once_with(mock_generated_data)

        assert "All scraping methods failed. Falling back to mock data." in caplog.text
        assert "Database updated successfully with FII/DII data." in caplog.text # Since update_database is mocked to True

        # Verify data in DB (this part assumes the *actual* update_database logic is somewhat tested
        # by the mocked instance's behavior, or we'd need a more complex setup to have the *real*
        # update_database run with the mock_generated_data against a real in-memory DB).
        # For simplicity, this E2E test focuses on the fallback to _generate_mock_data and its data being passed to update_database.
        # A separate integration test (like test_successful_data_flow_to_in_memory_db) already confirms
        # that update_database correctly writes to DB.
```
