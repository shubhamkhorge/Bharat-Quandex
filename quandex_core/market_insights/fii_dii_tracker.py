"""
Bharat QuanDex - FII/DII Institutional Flow Tracker

This module scrapes daily cash market activity for Foreign and Domestic
Institutional Investors from a public source, cleans the data, and saves it
to the DuckDB database.
"""
import polars as pl
from loguru import logger
import requests
import duckdb
from datetime import datetime
import sys
from pathlib import Path

# --- Setup Project Path ---
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from quandex_core.config import config

# --- Constants ---
# Using a well-known source for FII/DII data.
DATA_URL = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def setup_database_table(conn: duckdb.DuckDBPyConnection):
    """Ensures the institutional_flows table exists with the correct schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS institutional_flows (
            date DATE PRIMARY KEY,
            fii_buy_cr DOUBLE,
            fii_sell_cr DOUBLE,
            fii_net_cr DOUBLE,
            dii_buy_cr DOUBLE,
            dii_sell_cr DOUBLE,
            dii_net_cr DOUBLE
        );
    """)
    logger.info("Database table 'institutional_flows' is set up.")

def parse_and_clean_data(html_content: str) -> pl.DataFrame:
    """
    Parses the HTML content to find and clean the FII/DII data table.
    
    Args:
        html_content: The raw HTML string of the webpage.

    Returns:
        A Polars DataFrame with the cleaned data for the latest day, or None if failed.
    """
    try:
        # Polars' read_html finds all tables on the page. We need to find the correct one.
        tables = pl.read_html(html_content)
        
        # Heuristic: The correct table has a specific structure. Let's find it.
        # It usually has columns like 'Date', 'Gross Purchase', 'Gross Sales'.
        data_table = None
        for table in tables:
            if 'Gross Purchase' in table.columns and 'Gross Sales' in table.columns:
                data_table = table
                break
        
        if data_table is None:
            logger.error("Could not find the FII/DII data table on the webpage.")
            return None
        
        # The data we need is typically in the first two rows (FII and DII for the latest day)
        latest_data = data_table.head(2)

        # --- Data Cleaning and Structuring ---
        # 1. Extract date
        date_str = latest_data.filter(pl.col("Category") == "FIIs")["Date"][0]
        trade_date = datetime.strptime(date_str, "%B %d, %Y").date()

        # 2. Extract FII and DII data
        fii_row = latest_data.filter(pl.col("Category") == "FIIs")
        dii_row = latest_data.filter(pl.col("Category") == "DIIs")
        
        if fii_row.is_empty() or dii_row.is_empty():
            logger.error("Could not find both FII and DII rows in the extracted table.")
            return None
            
        # 3. Create a structured dictionary
        # Values are in "Rs. Crores", we need to remove commas and convert to float
        def clean_value(val_series):
            return val_series.str.replace_all(",", "").cast(pl.Float64)[0]

        structured_data = {
            "date": trade_date,
            "fii_buy_cr": clean_value(fii_row["Gross Purchase"]),
            "fii_sell_cr": clean_value(fii_row["Gross Sales"]),
            "fii_net_cr": clean_value(fii_row["Net Purchase / Sales"]),
            "dii_buy_cr": clean_value(dii_row["Gross Purchase"]),
            "dii_sell_cr": clean_value(dii_row["Gross Sales"]),
            "dii_net_cr": clean_value(dii_row["Net Purchase / Sales"]),
        }
        
        return pl.DataFrame([structured_data])

    except Exception as e:
        logger.exception(f"Failed to parse or clean data: {e}")
        return None

def run_scraper():
    """
    Main function to run the scraper, process the data, and save to the database.
    """
    logger.info("--- Starting FII/DII Institutional Flow Scraper ---")
    
    # 1. Fetch Webpage
    try:
        logger.info(f"Fetching data from {DATA_URL}")
        response = requests.get(DATA_URL, headers=HEADERS, timeout=15)
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch webpage: {e}")
        return

    # 2. Parse and Clean Data
    cleaned_df = parse_and_clean_data(response.text)
    
    if cleaned_df is None or cleaned_df.is_empty():
        logger.error("Scraping process failed. No data to save.")
        return

    logger.info("Successfully parsed data for date:")
    print(cleaned_df)

    # 3. Save to Database
    try:
        with duckdb.connect(str(config.data.duckdb_path)) as conn:
            setup_database_table(conn)
            
            # Use "INSERT OR IGNORE" to handle conflicts if data for the date already exists
            conn.execute(f"""
                INSERT OR IGNORE INTO institutional_flows
                SELECT * FROM cleaned_df;
            """)
            logger.success("Data successfully saved to DuckDB.")
    except Exception as e:
        logger.error(f"Failed to save data to DuckDB: {e}")
        
    logger.info("--- Scraper run finished ---")


if __name__ == '__main__':
    run_scraper()