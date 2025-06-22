import pytest
import asyncio # For async sleep and playwright tests
import polars as pl
from polars.testing import assert_frame_equal
from datetime import date
from unittest.mock import MagicMock, patch, AsyncMock, ANY # AsyncMock for playwright, ANY for assertions

import requests # For exception types

from quandex_core.market_insights.fii_dii_tracker import NSE_FII_DII_Scraper
from quandex_core import config as global_config_module # For mocking

# Attempt to import PlaywrightError, mock if not available
try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:
    PlaywrightError = type('PlaywrightError', (Exception,), {})
    PlaywrightTimeoutError = type('PlaywrightTimeoutError', (PlaywrightError,), {})


@pytest.fixture
def mock_config_for_scraper(mocker):
    """Mocks config for NSE_FII_DII_Scraper instantiation."""
    mock_cfg = MagicMock()
    mock_cfg.data.duckdb_path = "dummy_fii_dii_scraping.db"
    mock_cfg.scraping.user_agents = ["test_user_agent_scraping"]
    mock_cfg.scraping.nse_fii_dii_home_url = "https://mock_home_url.com"
    mock_cfg.scraping.nse_fii_dii_api_url = "https://mock_api_url.com/api"
    mock_cfg.scraping.nse_fii_dii_html_url = "https://mock_html_url.com/html"
    mock_cfg.market.trading_holidays = []
    # Set retry and timeout parameters for fast tests
    mock_cfg.scraping.max_retries = 1
    mock_cfg.scraping.retry_delay = 0.01
    mock_cfg.scraping.request_timeout = 5
    mocker.patch('quandex_core.market_insights.fii_dii_tracker.config', mock_cfg)
    return mock_cfg

@pytest.fixture
def scraper_instance(mock_config_for_scraper):
    """Provides an instance of NSE_FII_DII_Scraper with mocked config."""
    # _initialize_db_schema is not a method in the provided source code of NSE_FII_DII_Scraper
    # Its __init__ does not call such a method.
    # DB schema is handled directly in update_database method.
    scraper = NSE_FII_DII_Scraper() # Parameters like retries will be picked from mocked config
    return scraper

class TestScrapeWithApi:

    @pytest.fixture
    def mock_requests_session(self, mocker):
        mock_session_instance = MagicMock(spec=requests.Session)
        mock_session_instance.headers = {} # Ensure headers attribute exists and is a dict
        # Mock the session's cookies attribute directly for use after home page fetch
        mock_session_instance.cookies = MagicMock()
        mock_session_instance.cookies.get_dict.return_value = {"bm_sv_session_cookie": "mock_val"}


        # Mock for home URL response (cookies obtained from response object are less critical if session manages them)
        mock_home_response = MagicMock(spec=requests.Response)
        mock_home_response.status_code = 200
        # mock_home_response.cookies = MagicMock()
        # mock_home_response.cookies.get_dict.return_value = {"bm_sv_resp_cookie": "mock_val"}
        # The actual home_response.cookies.get_dict() is not directly used by scrape_with_api,
        # it relies on session object's cookie management after the call.

        # Mock for API URL response (JSON data)
        mock_api_response = MagicMock(spec=requests.Response)
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {"fiiDii": [{"tradedDate": "26-Aug-2024", "fiiBuy": "100"}]}
        mock_api_response.content = b'{"fiiDii": [{"tradedDate": "26-Aug-2024", "fiiBuy": "100"}]}'

        # Default: get returns home then api response
        mock_session_instance.get.side_effect = [mock_home_response, mock_api_response]

        mocker.patch('requests.Session', return_value=mock_session_instance)
        return mock_session_instance, mock_home_response, mock_api_response

    @pytest.mark.asyncio
    async def test_successful_api_fetch(self, scraper_instance, mock_requests_session, mocker):
        mock_session, mock_home_response_config, mock_api_response_config = mock_requests_session

        # Configure side_effect for this specific test run if needed, or rely on fixture's default
        # The fixture already sets up a side_effect: [mock_home_response, mock_api_response]
        # mock_session is mock_session_instance from the fixture.

        sample_processed_df = pl.DataFrame({"date": [date(2024, 8, 26)], "fii_buy_cr": [100.0]})
        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data', return_value=sample_processed_df)

        # Await the async method
        result_df = await scraper_instance.scrape_with_api()

        assert mock_session.get.call_count == 2
        # Note: ANY from unittest.mock might be needed if headers/cookies are complex/dynamic
        from unittest.mock import ANY
        # The actual calls in scrape_with_api do not pass headers explicitly to session.get()
        # as the session object is expected to hold them.
        mock_session.get.assert_any_call(scraper_instance.home_url, timeout=scraper_instance.timeout)
        mock_session.get.assert_any_call(scraper_instance.api_url, timeout=scraper_instance.timeout)

        # mock_api_response_config is the mock response object for the API call from the fixture
        mock_process_api.assert_called_once_with(mock_api_response_config.json())
        assert_frame_equal(result_df, sample_processed_df)

    @pytest.mark.asyncio
    async def test_api_returns_401_unauthorized(self, scraper_instance, mock_requests_session, mocker):
        mock_session, _, mock_api_resp_config = mock_requests_session
        mock_api_resp_config.status_code = 401
        # If .json() is called on a 401, it might raise JSONDecodeError or requests.exceptions.HTTPError before that.
        # Let's assume it raises HTTPError due to response.raise_for_status()
        mock_api_resp_config.raise_for_status.side_effect = requests.exceptions.HTTPError("Unauthorized")
        # And if .json() were still called, make it fail clearly
        mock_api_resp_config.json.side_effect = requests.exceptions.JSONDecodeError("dummy", "dummy", 0)


        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data')
        # Patch global logger used by the scraper method
        mock_logger_error = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.error')

        result_df = await scraper_instance.scrape_with_api()

        assert result_df is None
        mock_process_api.assert_not_called() # Because raise_for_status fails first
        mock_logger_error.assert_called()
        # The actual log is "API scrape failed: {e}" and e includes "Unauthorized"
        # Or, if it's specifically handling 401: "API access denied. Check your headers and cookies."
        # Current code logs "API access denied..." for 401, then "API scrape failed..." from outer try-except.
        # Let's check for the more specific one if it's logged first.
        # The current code: if response.status_code == 401: logger.error("API access denied...") then returns None.
        # So the general "API scrape failed: {e}" in the outer except block won't be hit for the 401 itself.
        assert any("API access denied" in call_args[0][0] for call_args in mock_logger_error.call_args_list)


    @pytest.mark.asyncio
    async def test_api_http_error(self, scraper_instance, mock_requests_session, mocker):
        mock_session, mock_home_response_config_fixture, mock_api_response_config_fixture = mock_requests_session

        # Ensure home response is fine for this test's specific side_effect
        mock_home_response_obj = MagicMock(spec=requests.Response)
        mock_home_response_obj.status_code = 200
        # Ensure cookies attribute is a mock that can handle get_dict
        mock_home_response_obj.cookies = MagicMock()
        mock_home_response_obj.cookies.get_dict.return_value = {"bm_sv": "mock_cookie_val_http_error"}

        # API call will raise HTTPError directly from session.get()
        mock_session.get.side_effect = [
            mock_home_response_obj,
            requests.exceptions.HTTPError("Server Error")
        ]

        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data')
        mock_logger_error = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.error')

        result_df = await scraper_instance.scrape_with_api()

        assert result_df is None
        mock_process_api.assert_not_called()
        mock_logger_error.assert_called()
        # The log will be "API scrape failed: Server Error"
        assert any("API scrape failed: Server Error" in call_args[0][0] for call_args in mock_logger_error.call_args_list)


    @pytest.mark.asyncio
    async def test_api_returns_empty_content(self, scraper_instance, mock_requests_session, mocker):
        mock_session, _, mock_api_resp_config = mock_requests_session
        mock_api_resp_config.status_code = 200 # Success status
        mock_api_resp_config.content = b''    # Empty content
        # .json() on empty content raises JSONDecodeError
        mock_api_resp_config.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "doc", 0)
        # raise_for_status should pass for 200
        mock_api_resp_config.raise_for_status = MagicMock()


        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data')
        # Patch global logger
        mock_logger_error = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.error') # For "Empty API response"
        mock_logger_exception = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.exception') # For general "API data processing failed"

        result_df = await scraper_instance.scrape_with_api()

        # In current code, if content is empty, it logs "Empty API response received" and returns None.
        # _process_api_data is not called.
        assert result_df is None
        mock_process_api.assert_not_called()
        mock_logger_error.assert_called()
        assert any("Empty API response received" in call_args[0][0] for call_args in mock_logger_error.call_args_list)


    @pytest.mark.asyncio
    async def test_homepage_fetch_fails(self, scraper_instance, mock_requests_session, mocker):
        mock_session, _, _ = mock_requests_session
        # First call to session.get (for home_url) raises an exception
        mock_session.get.side_effect = requests.exceptions.RequestException("Homepage connection error")

        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data')
        # Patch global logger
        mock_logger_warning = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.warning') # For "Session setup failed"
        mock_logger_error = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.error') # For outer "API scrape failed"

        result_df = await scraper_instance.scrape_with_api()

        assert result_df is None
        mock_session.get.assert_called_once() # Only first call to home_url
        mock_process_api.assert_not_called()

        # Check for "Session setup failed"
        # print(f"DEBUG: mock_logger_warning.call_args_list for test_homepage_fetch_fails: {mock_logger_warning.call_args_list}")
        # The logged message is f"Home page fetch for session setup failed: {e}", where e is "Homepage connection error"
        mock_logger_warning.assert_any_call("Home page fetch for session setup failed: Homepage connection error")
        # Check for the outer "API scrape failed" log is not strictly necessary if the specific one is caught.
        # The current code: if home page fails, it logs warning then returns None. Outer "API scrape failed" is not hit.
        # So, only mock_logger_warning should have calls.
        mock_logger_error.assert_not_called() # Explicitly assert error logger is not called for this path

# Placeholder for TestScrapeWithPlaywright
# These tests are async and require more complex mocking of playwright's async context managers
class TestScrapeWithPlaywright:
    @pytest.mark.asyncio
    async def test_successful_playwright_fetch(self, scraper_instance, mocker):
        mock_async_playwright_cm = AsyncMock() # For 'async with async_playwright() as p:'
        mock_playwright_instance = AsyncMock() # For 'p'
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_async_playwright_cm.__aenter__.return_value = mock_playwright_instance
        mock_playwright_instance.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        mock_html_content = "<html><body><table><tr><td>Data</td></tr></table></body></html>"
        mock_page.content.return_value = mock_html_content

        mocker.patch('quandex_core.market_insights.fii_dii_tracker.async_playwright', return_value=mock_async_playwright_cm)

        sample_parsed_df = pl.DataFrame({"parsed_col": [1]})
        mock_parse_html = mocker.patch.object(scraper_instance, '_parse_html_table', return_value=sample_parsed_df)

        # Mock asyncio.sleep
        mocker.patch('asyncio.sleep', new_callable=AsyncMock)


        result_df = await scraper_instance.scrape_with_playwright()

        mock_playwright_instance.chromium.launch.assert_called_once()
        mock_browser.new_context.assert_called_once()
        mock_context.new_page.assert_called_once()
        # Corrected assertion for goto to include wait_until and specific timeout if not using ANY for it
        mock_page.goto.assert_called_once_with(
            scraper_instance.html_url,
            wait_until="domcontentloaded",
            timeout=40000 # Match actual call
        )
        mock_page.wait_for_selector.assert_called_once_with(scraper_instance.HTML_TABLE_SELECTOR, timeout=ANY) # Assuming 15000 is fine with ANY
        mock_page.content.assert_called_once()
        mock_browser.close.assert_called_once()

        mock_parse_html.assert_called_once() # Argument would be a BeautifulSoup object
        assert_frame_equal(result_df, sample_parsed_df)

    # Further Playwright tests (failure cases) would follow similar mocking patterns
    # - table not found (wait_for_selector times out or _parse_html_table returns None)
    # - page.goto fails (raises error, check retries via asyncio.sleep)
    # - other playwright errors


# Placeholder for TestScrapeOrchestration
class TestScrapeOrchestration:
    @pytest.mark.asyncio
    async def test_api_success(self, scraper_instance, mocker):
        sample_df = pl.DataFrame({'api_data': [1]})
        mock_api = mocker.patch.object(scraper_instance, 'scrape_with_api', return_value=sample_df)
        mock_playwright = mocker.patch.object(scraper_instance, 'scrape_with_playwright', new_callable=AsyncMock) # AsyncMock for async method
        mock_generate = mocker.patch.object(scraper_instance, '_generate_mock_data')

        result = await scraper_instance.scrape()

        assert_frame_equal(result, sample_df)
        mock_api.assert_called_once()
        mock_playwright.assert_not_called()
        mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_fail_playwright_success(self, scraper_instance, mocker):
        sample_df = pl.DataFrame({'pw_data': [1]})
        mocker.patch.object(scraper_instance, 'scrape_with_api', return_value=None)
        mock_playwright = mocker.patch.object(scraper_instance, 'scrape_with_playwright', new_callable=AsyncMock, return_value=sample_df)
        mock_generate = mocker.patch.object(scraper_instance, '_generate_mock_data')

        result = await scraper_instance.scrape()

        assert_frame_equal(result, sample_df)
        mock_playwright.assert_called_once()
        mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_fail_playwright_fail_mock_data_used(self, scraper_instance, mocker):
        sample_df = pl.DataFrame({'mock_data': [1]})
        mocker.patch.object(scraper_instance, 'scrape_with_api', return_value=None) # This is fine as scrape_with_api is async
        mocker.patch.object(scraper_instance, 'scrape_with_playwright', new_callable=AsyncMock, return_value=None)
        mock_generate = mocker.patch.object(scraper_instance, '_generate_mock_data', return_value=sample_df)
        # Patch global logger - the actual log is logger.error when mock data is used by scrape()
        mock_logger_error = mocker.patch('quandex_core.market_insights.fii_dii_tracker.logger.error')

        result = await scraper_instance.scrape()

        assert_frame_equal(result, sample_df)
        mock_generate.assert_called_once()
        # The log "All scraping methods failed. Falling back to mock data." is now logger.error in scrape()
        # The actual log in scrape() after API and Playwright fail is:
        # logger.error("All scraping methods failed, using mock data")
        # However, the test assertion was for "All scraping methods failed. Falling back to mock data."
        # Let's check the current `scrape` method's log:
        # `logger.error("All scraping methods failed, using mock data")` - this is what it logs before calling _generate_mock_data
        # So, the assertion text needs to match this.
        mock_logger_error.assert_any_call("All scraping methods failed, using mock data")
