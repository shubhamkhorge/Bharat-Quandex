import pytest
import asyncio # For async sleep and playwright tests
import polars as pl
from polars.testing import assert_frame_equal
from datetime import date
from unittest.mock import MagicMock, patch, AsyncMock # AsyncMock for playwright

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

        # Mock for home URL response (cookies)
        mock_home_response = MagicMock()
        mock_home_response.status_code = 200
        mock_home_response.cookies.get_dict.return_value = {"bm_sv": "mock_cookie_val"}

        # Mock for API URL response (JSON data)
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {"fiiDii": [{"tradedDate": "26-Aug-2024", "fiiBuy": "100"}]}
        mock_api_response.content = b'{"fiiDii": [{"tradedDate": "26-Aug-2024", "fiiBuy": "100"}]}' # if .content is used

        # Default: get returns home then api response
        mock_session_instance.get.side_effect = [mock_home_response, mock_api_response]

        mocker.patch('requests.Session', return_value=mock_session_instance)
        return mock_session_instance, mock_home_response, mock_api_response

    def test_successful_api_fetch(self, scraper_instance, mock_requests_session, mocker):
        mock_session, _, mock_api_resp = mock_requests_session
        sample_processed_df = pl.DataFrame({"date": [date(2024, 8, 26)], "fii_buy_cr": [100.0]})

        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data', return_value=sample_processed_df)

        result_df = scraper_instance.scrape_with_api()

        assert mock_session.get.call_count == 2
        mock_session.get.assert_any_call(scraper_instance.home_url, headers=ANY, timeout=ANY)
        mock_session.get.assert_any_call(scraper_instance.api_url, headers=ANY, timeout=ANY)

        mock_process_api.assert_called_once_with(mock_api_resp.json())
        assert_frame_equal(result_df, sample_processed_df)

    def test_api_returns_401_unauthorized(self, scraper_instance, mock_requests_session, mocker):
        mock_session, _, mock_api_resp = mock_requests_session
        mock_api_resp.status_code = 401
        mock_api_resp.json.side_effect = requests.exceptions.JSONDecodeError("dummy", "dummy", 0) # If it tries to parse json on error

        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data')
        mock_logger_warning = mocker.patch.object(scraper_instance.logger, 'warning')

        result_df = scraper_instance.scrape_with_api()

        assert result_df is None
        mock_process_api.assert_not_called()
        mock_logger_warning.assert_called()
        assert "Unauthorized (401)" in mock_logger_warning.call_args[0][0]


    def test_api_http_error(self, scraper_instance, mock_requests_session, mocker):
        mock_session, _, mock_api_resp = mock_requests_session
        mock_api_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        # Make the second call to get (API call) raise this by configuring side_effect carefully
        mock_home_response = mock_session.get.side_effect[0] # Keep the first successful home response
        mock_session.get.side_effect = [mock_home_response, requests.exceptions.HTTPError("Server Error")]


        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data')
        mock_logger_error = mocker.patch.object(scraper_instance.logger, 'error')

        result_df = scraper_instance.scrape_with_api()

        assert result_df is None
        mock_process_api.assert_not_called()
        mock_logger_error.assert_called()
        assert "HTTP error during FII/DII API call" in mock_logger_error.call_args[0][0]

    def test_api_returns_empty_content(self, scraper_instance, mock_requests_session, mocker):
        mock_session, _, mock_api_resp = mock_requests_session
        mock_api_resp.content = b''
        mock_api_resp.json.side_effect = requests.exceptions.JSONDecodeError("dummy", "dummy", 0) # Empty content leads to JSON error

        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data')
        mock_logger_warning = mocker.patch.object(scraper_instance.logger, 'warning')

        result_df = scraper_instance.scrape_with_api()

        assert result_df is None
        # _process_api_data might be called with None or raise error if json() fails badly
        # Based on current code, if json() fails, it logs and returns None.
        # So _process_api_data won't be called with valid data.
        # If json() returns None or empty dict, then _process_api_data would be called.
        # Let's assume json() fails as above.
        mock_process_api.assert_not_called() # Or called with None/{} if json() doesn't raise
        mock_logger_warning.assert_called()
        assert "empty response or failed to decode JSON" in mock_logger_warning.call_args[0][0]


    def test_homepage_fetch_fails(self, scraper_instance, mock_requests_session, mocker):
        mock_session, _, _ = mock_requests_session
        mock_session.get.side_effect = requests.exceptions.RequestException("Homepage connection error")

        mock_process_api = mocker.patch.object(scraper_instance, '_process_api_data')
        mock_logger_error = mocker.patch.object(scraper_instance.logger, 'error')

        result_df = scraper_instance.scrape_with_api()

        assert result_df is None
        mock_session.get.assert_called_once() # Only first call to home_url
        mock_process_api.assert_not_called()
        mock_logger_error.assert_called()
        assert "Failed to fetch FII/DII home page" in mock_logger_error.call_args[0][0]

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
        mock_page.goto.assert_called_once_with(scraper_instance.html_url, timeout=ANY)
        mock_page.wait_for_selector.assert_called_once_with(scraper_instance.HTML_TABLE_SELECTOR, timeout=ANY)
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
        mocker.patch.object(scraper_instance, 'scrape_with_api', return_value=None)
        mocker.patch.object(scraper_instance, 'scrape_with_playwright', new_callable=AsyncMock, return_value=None)
        mock_generate = mocker.patch.object(scraper_instance, '_generate_mock_data', return_value=sample_df)
        mock_logger_info = mocker.patch.object(scraper_instance.logger, 'info')

        result = await scraper_instance.scrape()

        assert_frame_equal(result, sample_df)
        mock_generate.assert_called_once()
        mock_logger_info.assert_any_call("All scraping methods failed. Falling back to mock data.")
