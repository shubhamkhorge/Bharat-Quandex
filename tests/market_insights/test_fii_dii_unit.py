import pytest
import polars as pl
from polars.testing import assert_frame_equal
from datetime import date, timedelta, datetime
from unittest.mock import MagicMock, patch

from quandex_core.market_insights.fii_dii_tracker import NSE_FII_DII_Scraper
from quandex_core import config as global_config_module # For mocking

# Attempt to import BeautifulSoup, mock if not available
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = MagicMock()


@pytest.fixture
def mock_config_for_scraper(mocker):
    """Mocks config for NSE_FII_DII_Scraper instantiation."""
    mock_cfg = MagicMock()
    mock_cfg.data.duckdb_path = "dummy_fii_dii.db"
    mock_cfg.scraping.user_agents = ["test_user_agent"]
    mock_cfg.market.trading_holidays = [] # Empty for mock data generation simplicity
    mocker.patch('quandex_core.market_insights.fii_dii_tracker.config', mock_cfg)
    return mock_cfg

@pytest.fixture
def scraper_instance(mock_config_for_scraper):
    """Provides an instance of NSE_FII_DII_Scraper with mocked config."""
    # _initialize_db_schema is not a method in the provided source code of NSE_FII_DII_Scraper
    # Its __init__ does not call such a method.
    # DB schema is handled directly in update_database method.
    scraper = NSE_FII_DII_Scraper()
    return scraper

class TestProcessApiData:

    def test_standard_list_of_dicts_input(self, scraper_instance):
        api_data = [
            {"tradedDate": "26-Aug-2024", "fiiBuy": "1,000.50", "fiiSell": "500.25", "diiBuy": "700.00", "diiSell": "300.75"},
            {"tradedDate": "27-Aug-2024", "fiiBuy": "1,200.00", "fiiSell": "600.00", "diiBuy": "800.00", "diiSell": "400.00"}
        ]
        expected_df = pl.DataFrame({
            "date": [date(2024, 8, 26), date(2024, 8, 27)],
            "fii_buy_cr": [1000.50, 1200.00],
            "fii_sell_cr": [500.25, 600.00],
            "fii_net_cr": [500.25, 600.00],
            "dii_buy_cr": [700.00, 800.00],
            "dii_sell_cr": [300.75, 400.00],
            "dii_net_cr": [399.25, 400.00]
        })
        processed_df = scraper_instance._process_api_data(api_data)
        assert_frame_equal(processed_df, expected_df, check_dtype=True)

    def test_fii_dii_key_input(self, scraper_instance):
        api_data = {"fiiDii": [
            {"tradedDate": "26-Aug-2024", "fiiBuy": "100.0", "diiSell": "50.0"}
        ]}
        # Expect only relevant columns, others will be null and then filled by calculations if possible
        # fii_sell_cr, dii_buy_cr will be 0.0 as they are not present.
        expected_df = pl.DataFrame({
            "date": [date(2024, 8, 26)],
            "fii_buy_cr": [100.0], "fii_sell_cr": [0.0], "fii_net_cr": [100.0],
            "dii_buy_cr": [0.0], "dii_sell_cr": [50.0], "dii_net_cr": [-50.0]
        })
        processed_df = scraper_instance._process_api_data(api_data)
        assert_frame_equal(processed_df, expected_df, check_dtype=True)

    def test_data_key_input(self, scraper_instance):
        api_data = {"data": [
            {"tradedDate": "27-Aug-2024", "fiiSell": "200.0", "diiBuy": "150.0"}
        ]}
        expected_df = pl.DataFrame({
            "date": [date(2024, 8, 27)],
            "fii_buy_cr": [0.0], "fii_sell_cr": [200.0], "fii_net_cr": [-200.0],
            "dii_buy_cr": [150.0], "dii_sell_cr": [0.0], "dii_net_cr": [150.0]
        })
        processed_df = scraper_instance._process_api_data(api_data)
        assert_frame_equal(processed_df, expected_df, check_dtype=True)

    def test_alternative_column_names(self, scraper_instance):
        api_data = [
            {"tradeDate": "28-Aug-2024", "fiiPurchaseValue": "100.0", "fiiSalesValue": "50.0",
             "diiPurchaseValue": "70.0", "diiSalesValue": "30.0"}
        ]
        expected_df = pl.DataFrame({
            "date": [date(2024, 8, 28)],
            "fii_buy_cr": [100.0], "fii_sell_cr": [50.0], "fii_net_cr": [50.0],
            "dii_buy_cr": [70.0], "dii_sell_cr": [30.0], "dii_net_cr": [40.0]
        })
        processed_df = scraper_instance._process_api_data(api_data)
        assert_frame_equal(processed_df, expected_df, check_dtype=True)

    def test_net_calculation(self, scraper_instance):
        api_data = [{"tradedDate": "29-Aug-2024", "fiiBuy": "100", "fiiSell": "150", "diiBuy": "200", "diiSell": "50"}]
        processed_df = scraper_instance._process_api_data(api_data)
        assert processed_df["fii_net_cr"][0] == -50.0
        assert processed_df["dii_net_cr"][0] == 150.0

    def test_empty_input(self, scraper_instance):
        assert scraper_instance._process_api_data([]) is None
        assert scraper_instance._process_api_data({}) is None

        # Test with empty list inside dict structure
        assert scraper_instance._process_api_data({"fiiDii": []}) is None
        assert scraper_instance._process_api_data({"data": []}) is None


    def test_unexpected_input_type(self, scraper_instance):
        assert scraper_instance._process_api_data("not a dict or list") is None
        assert scraper_instance._process_api_data(123) is None

    def test_date_parsing_error(self, scraper_instance):
        api_data = [{"tradedDate": "Invalid Date", "fiiBuy": "100"}]
        processed_df = scraper_instance._process_api_data(api_data)
        # strict=False in strptime means it will produce NaT for unparseable dates
        assert processed_df is not None
        assert processed_df["date"][0] is None # Expect NaT (None in Polars for Date)
        assert processed_df["fii_buy_cr"][0] == 100.0

    def test_missing_all_value_columns(self, scraper_instance):
        api_data = [{"tradedDate": "26-Aug-2024"}] # Only date, no FII/DII values
        expected_df = pl.DataFrame({
            "date": [date(2024, 8, 26)],
            "fii_buy_cr": [0.0], "fii_sell_cr": [0.0], "fii_net_cr": [0.0],
            "dii_buy_cr": [0.0], "dii_sell_cr": [0.0], "dii_net_cr": [0.0]
        })
        processed_df = scraper_instance._process_api_data(api_data)
        assert_frame_equal(processed_df, expected_df, check_dtype=True)

    def test_numeric_parsing_with_empty_strings(self, scraper_instance):
        api_data = [{"tradedDate": "26-Aug-2024", "fiiBuy": "", "fiiSell": "50.0"}]
        processed_df = scraper_instance._process_api_data(api_data)
        # Empty strings for numeric should become 0.0 after cleaning and casting
        assert processed_df["fii_buy_cr"][0] == 0.0
        assert processed_df["fii_sell_cr"][0] == 50.0
        assert processed_df["fii_net_cr"][0] == -50.0

# Placeholder for TestParseHtmlTable - requires BeautifulSoup and HTML samples
class TestParseHtmlTable:
    # These tests would require more involved setup with HTML strings and possibly mocking requests
    # For now, just basic structure if BeautifulSoup is not fully mocked/available
    def test_valid_html_table(self, scraper_instance, mocker):
        if BeautifulSoup == mocker.MagicMock(): # Skip if bs4 not available and not deeply mocked
            pytest.skip("BeautifulSoup not available or not deeply mocked for HTML parsing test")

        html_content = """
        <table>
          <thead>
            <tr><th>Date</th><th>FII Gross Purchase (Rs Cr)</th><th>FII Gross Sales (Rs Cr)</th><th>DII Gross Purchase (Rs Cr)</th><th>DII Gross Sales (Rs Cr)</th></tr>
          </thead>
          <tbody>
            <tr><td>26 Aug 2024</td><td>1,000.50</td><td>500.25</td><td>700.00</td><td>300.75</td></tr>
          </tbody>
        </table>
        """
        mock_table = BeautifulSoup(html_content, "html.parser").find("table")

        expected_df = pl.DataFrame({
            "date": [date(2024, 8, 26)],
            "fii_buy_cr": [1000.50], "fii_sell_cr": [500.25], "fii_net_cr": [500.25],
            "dii_buy_cr": [700.00], "dii_sell_cr": [300.75], "dii_net_cr": [399.25]
        })

        parsed_df = scraper_instance._parse_html_table(mock_table)
        assert parsed_df is not None
        assert_frame_equal(parsed_df.select(expected_df.columns), expected_df, check_dtype=True, rtol=1e-3)

    def test_empty_html_table(self, scraper_instance, mocker):
        if BeautifulSoup == mocker.MagicMock():
            pytest.skip("BeautifulSoup not available or not deeply mocked")
        html_content = "<table><thead><tr><th>Header</th></tr></thead><tbody></tbody></table>"
        mock_table = BeautifulSoup(html_content, "html.parser").find("table")
        parsed_df = scraper_instance._parse_html_table(mock_table)
        assert parsed_df is None # Or empty DF with schema, depends on implementation

class TestGenerateMockData:
    def test_generate_mock_data_output(self, scraper_instance):
        mock_df = scraper_instance._generate_mock_data(num_days=10, skip_weekends=True)

        assert isinstance(mock_df, pl.DataFrame)
        assert not mock_df.is_empty()
        assert len(mock_df) <= 10 # Could be less if weekends are skipped

        expected_cols = ["date", "fii_buy_cr", "fii_sell_cr", "fii_net_cr", "dii_buy_cr", "dii_sell_cr", "dii_net_cr"]
        for col in expected_cols:
            assert col in mock_df.columns

        assert mock_df["date"].dtype == pl.Date
        for col in expected_cols[1:]: # Numeric columns
            assert mock_df[col].dtype == pl.Float64
            # Check that net values are buy - sell
            if "fii_net_cr" in col:
                assert_frame_equal(mock_df.select(pl.col("fii_buy_cr") - pl.col("fii_sell_cr")).rename({"fii_buy_cr":"fii_net_cr"}), mock_df.select("fii_net_cr"))
            if "dii_net_cr" in col:
                 assert_frame_equal(mock_df.select(pl.col("dii_buy_cr") - pl.col("dii_sell_cr")).rename({"dii_buy_cr":"dii_net_cr"}), mock_df.select("dii_net_cr"))

        # Check if weekends are skipped (if num_days is large enough to include a weekend)
        if len(mock_df) < 10 and len(mock_df) > 5: # Heuristic: if some days were skipped for a 10-day request
            for r_date in mock_df["date"]:
                assert r_date.weekday() < 5 # Monday is 0, Sunday is 6

    def test_generate_mock_data_no_days(self, scraper_instance):
        mock_df = scraper_instance._generate_mock_data(num_days=0)
        assert mock_df.is_empty()
        # Schema should still be correct even if empty
        expected_cols = ["date", "fii_buy_cr", "fii_sell_cr", "fii_net_cr", "dii_buy_cr", "dii_sell_cr", "dii_net_cr"]
        for col in expected_cols:
            assert col in mock_df.columns


@pytest.fixture
def mock_duckdb_connection(mocker):
    mock_conn = MagicMock(spec=duckdb.DuckDBPyConnection)
    mock_conn.execute.return_value = mock_conn # for fluent interface
    mock_conn.fetchall.return_value = []
    mock_conn.fetchone.return_value = None
    mocker.patch('duckdb.connect', return_value=mock_conn)
    return mock_conn

class TestUpdateDatabase:
    def test_update_database_successful(self, scraper_instance, mock_duckdb_connection, mocker):
        sample_df = pl.DataFrame({
            "date": [date(2024, 8, 26)], "fii_buy_cr": [100.0], "fii_sell_cr": [50.0],
            "fii_net_cr": [50.0], "dii_buy_cr": [70.0], "dii_sell_cr": [30.0], "dii_net_cr": [40.0]
        })

        # Spy on execute to check SQL calls
        execute_spy = mocker.spy(mock_duckdb_connection, 'execute')
        register_spy = mocker.spy(mock_duckdb_connection, 'register')

        result = scraper_instance.update_database(sample_df)

        assert result is True
        duckdb.connect.assert_called_once_with(database=scraper_instance.db_path, read_only=False)
        register_spy.assert_called_once_with("temp_fii_dii", sample_df)

        # Check for key SQL statements (can be more specific if needed)
        sql_calls = [call_args[0][0] for call_args in execute_spy.call_args_list]

        assert any("CREATE TABLE IF NOT EXISTS institutional_flows" in sql for sql in sql_calls)
        assert any("INSERT INTO institutional_flows" in sql and "ON CONFLICT(date) DO UPDATE SET" in sql for sql in sql_calls)
        assert any("CREATE OR REPLACE VIEW v_institutional_trends AS" in sql for sql in sql_calls)
        mock_duckdb_connection.close.assert_called_once()


    def test_update_database_empty_dataframe(self, scraper_instance, mock_duckdb_connection):
        empty_df = pl.DataFrame()
        result = scraper_instance.update_database(empty_df)

        assert result is False
        mock_duckdb_connection.connect.assert_not_called() # connect on module, not instance
        duckdb.connect.assert_not_called()
        mock_duckdb_connection.execute.assert_not_called()

    def test_update_database_connection_error(self, scraper_instance, mocker):
        sample_df = pl.DataFrame({"date": [date(2024, 8, 26)], "fii_buy_cr": [100.0]}) # Min data
        mocker.patch('duckdb.connect', side_effect=Exception("DB Connection Failed"))

        # Mock logger to check for error logging
        mock_logger_error = mocker.patch.object(scraper_instance.logger, 'error')

        result = scraper_instance.update_database(sample_df)
        assert result is False
        mock_logger_error.assert_called_once()
        assert "Failed to connect to DuckDB" in mock_logger_error.call_args[0][0]


    def test_update_database_execution_error(self, scraper_instance, mock_duckdb_connection, mocker):
        sample_df = pl.DataFrame({"date": [date(2024, 8, 26)], "fii_buy_cr": [100.0]})
        mock_duckdb_connection.execute.side_effect = Exception("SQL Execution Failed")

        mock_logger_error = mocker.patch.object(scraper_instance.logger, 'error')

        result = scraper_instance.update_database(sample_df)
        assert result is False
        mock_logger_error.assert_called_once()
        assert "Error during database update" in mock_logger_error.call_args[0][0]
        mock_duckdb_connection.close.assert_called_once() # Ensure connection is closed even on error

# More tests to come for _generate_mock_data and update_database
