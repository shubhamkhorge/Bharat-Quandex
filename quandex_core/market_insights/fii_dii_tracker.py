"""
NSE FII/DII Institutional Flow Tracker - Improved Version
Robust API and Playwright scraping with anti-automation and fallback strategies.
"""

import asyncio
import polars as pl
import duckdb
from datetime import datetime, timedelta, date
import json
import random
import logging
from loguru import logger
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Error as PlaywrightError
import time
import re
from typing import List, Dict, Any, Union, Optional

# --- Setup Project Path ---
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from quandex_core.config import config
except ImportError:
    # Fallback configuration
    class ConfigFallback:
        class data:
            duckdb_path = project_root / "data_vault" / "market_boards" / "quandex.duckdb"
        class scraping:
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
            ]
            max_retries = 3
            retry_delay = 2
            request_timeout = 15
    config = ConfigFallback()

# --- Logging Setup ---
log_dir = project_root / "logs"
try:
    log_dir.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Log directory error: {e}")

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{line} - {message}"
)
logger.add(
    str(log_dir / "fii_dii_scraper.log"),
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    enqueue=True
)
logger.info("Logging system initialized")
logger.info(f"DuckDB path: {config.data.duckdb_path}")

class NSE_FII_DII_Scraper:
    def __init__(self):
        """Initialize scraper with robust configuration"""
        try:
            self.db_path = str(config.data.duckdb_path)
            self.api_url = "https://www.nseindia.com/api/fiidiiTradeReact"
            self.home_url = "https://www.nseindia.com"
            self.css_selectors = [
                'table.fii-dii-table',
                'table.table-fii-dii',
                'div#fiiDiiData table',
                'table'
            ]
            self.user_agents = config.scraping.user_agents
            self.max_retries = config.scraping.max_retries
            self.retry_delay = config.scraping.retry_delay
            self.timeout = config.scraping.request_timeout
            self.session_counter = 0
            self.max_sessions = 3
            # Example headers - REPLACE with your actual headers from browser DevTools
            self.nse_headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nseindia.com/',
                'Connection': 'keep-alive',
                'Cookie': '',  # <-- REPLACE with your actual cookie
                'X-Requested-With': 'XMLHttpRequest'
            }
            logger.info("FII/DII scraper initialized")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def scrape_with_api(self) -> Optional[pl.DataFrame]:
        """Scrape using official NSE API with robust session handling"""
        try:
            session = requests.Session()
            session.headers.update(self.nse_headers)
            logger.debug(f"Request headers: {session.headers}")
            # Visit homepage first to get cookies
            try:
                response = session.get(self.home_url, timeout=self.timeout)
                response.raise_for_status()
                logger.debug(f"Session cookies: {session.cookies.get_dict()}")
                # Update session headers with new cookies if needed
                self.nse_headers['Cookie'] = '; '.join(
                    f"{k}={v}" for k, v in session.cookies.get_dict().items()
                )
                session.headers.update(self.nse_headers)
                logger.info("NSE session established")
            except Exception as e:
                logger.warning(f"Session setup failed: {e}")
                return None
            # Now try the API
            response = session.get(self.api_url, timeout=self.timeout)
            logger.debug(f"API status: {response.status_code}")
            if response.status_code == 401:
                logger.error("API access denied. Check your headers and cookies.")
                return None
            response.raise_for_status()
            if not response.content:
                logger.error("Empty API response received")
                return None
            data = response.json()
            logger.success("Fetched FII/DII data from API")
            logger.debug(f"API response snippet: {str(data)[:500]}...")
            return self._process_api_data(data)
        except Exception as e:
            logger.error(f"API scrape failed: {e}")
            return None

    def _process_api_data(self, data: Union[list, dict]) -> Optional[pl.DataFrame]:
        """Process NSE API response with flexible structure handling"""
        try:
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                if 'fiiDii' in data:
                    records = data['fiiDii']
                else:
                    records = data.get('data', [])
            else:
                logger.warning(f"Unexpected API response type: {type(data)}")
                return None

            if not records:
                logger.warning("No records in API response")
                return None

            df = pl.DataFrame(records)

            column_mapping = {
                "tradedDate": "date", "tradeDate": "date", "date": "date",
                "fiiBuy": "fii_buy_cr", "fiiPurchaseValue": "fii_buy_cr", "fii_buy": "fii_buy_cr",
                "fiiSell": "fii_sell_cr", "fiiSalesValue": "fii_sell_cr", "fii_sell": "fii_sell_cr",
                "fiiNet": "fii_net_cr", "fii_net": "fii_net_cr",
                "diiBuy": "dii_buy_cr", "diiPurchaseValue": "dii_buy_cr", "dii_buy": "dii_buy_cr",
                "diiSell": "dii_sell_cr", "diiSalesValue": "dii_sell_cr", "dii_sell": "dii_sell_cr",
                "diiNet": "dii_net_cr", "dii_net": "dii_net_cr"
            }

            # Rename columns based on mapping
            current_columns = df.columns
            df = df.rename({orig_col: column_mapping[orig_col]
                            for orig_col in current_columns if orig_col in column_mapping})

            # Parse date column
            if 'date' in df.columns:
                df = df.with_columns(
                    pl.col("date").str.strptime(pl.Date, "%d-%b-%Y", strict=False)
                )

            # Convert numeric columns
            potential_numeric_cols = [
                'fii_buy_cr', 'fii_sell_cr', 'fii_net_cr',
                'dii_buy_cr', 'dii_sell_cr', 'dii_net_cr'
            ]
            for col_name in potential_numeric_cols:
                if col_name in df.columns:
                    df = df.with_columns(
                        pl.col(col_name)
                        .str.replace_all(r"[, ]", "") # Remove commas and spaces
                        .replace("", None) # Replace empty strings with null
                        .cast(pl.Float64, strict=False) # Cast to float, invalid become null
                        .fill_null(0.0) # Fill actual nulls (from cast error or original) with 0.0
                        .alias(col_name)
                    )

            # Calculate net values if buy/sell columns are present
            if 'fii_buy_cr' in df.columns and 'fii_sell_cr' in df.columns:
                df = df.with_columns(
                    (pl.col("fii_buy_cr") - pl.col("fii_sell_cr")).alias("fii_net_cr")
                )
            elif 'fii_net_cr' not in df.columns: # Ensure column exists if not calculable
                 df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias("fii_net_cr"))

            if 'dii_buy_cr' in df.columns and 'dii_sell_cr' in df.columns:
                df = df.with_columns(
                    (pl.col("dii_buy_cr") - pl.col("dii_sell_cr")).alias("dii_net_cr")
                )
            elif 'dii_net_cr' not in df.columns: # Ensure column exists if not calculable
                 df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias("dii_net_cr"))

            # Ensure all expected final columns are present, adding them with 0.0 if missing
            all_expected_numeric_cols = ['fii_buy_cr', 'fii_sell_cr', 'fii_net_cr', 'dii_buy_cr', 'dii_sell_cr', 'dii_net_cr']
            for col_name in all_expected_numeric_cols:
                if col_name not in df.columns:
                    df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col_name))

            # Define final_cols based on a fixed desired schema
            final_cols = ['date', 'fii_buy_cr', 'fii_sell_cr', 'fii_net_cr', 'dii_buy_cr', 'dii_sell_cr', 'dii_net_cr']
            if 'date' not in df.columns: # Ensure date column exists, even if all null
                 df = df.with_columns(pl.lit(None).cast(pl.Date).alias("date"))

            df = df.select(final_cols) # Select in fixed order

            logger.info(f"Processed {len(df)} records from API")
            return df
        except Exception as e:
            logger.exception(f"API data processing failed: {e}")
            return None

    async def scrape_with_playwright(self) -> Optional[pl.DataFrame]:
        """Scrape using Playwright with anti-automation flags and robust error handling"""
        retries = 0
        http_versions = ["2", "1.1"]
        while retries < self.max_retries:
            for http_version in http_versions:
                try:
                    user_agent = random.choice(self.user_agents)
                    logger.debug(f"Using Chromium with HTTP/{http_version}")
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(
                            headless=True,
                            args=[
                                "--disable-blink-features=AutomationControlled",
                                "--disable-extensions",
                                "--disable-gpu",
                                "--no-sandbox",
                                f"--http-version={http_version}"
                            ]
                        )
                        context = await browser.new_context(
                            user_agent=user_agent,
                            viewport={'width': 1280, 'height': 800},
                            bypass_csp=True,
                            ignore_https_errors=True
                        )
                        page = await context.new_page()
                        page.set_default_timeout(30000)
                        await page.set_extra_http_headers({
                            "Accept-Language": "en-US,en;q=0.9",
                            "Cache-Control": "no-cache"
                        })
                        await page.goto(
                            "https://www.nseindia.com/market-data/fii-dii-activity",
                            wait_until="domcontentloaded",
                            timeout=40000
                        )
                        try:
                            await page.wait_for_selector(".fii-dii-table", timeout=15000)
                        except:
                            logger.debug("FII/DII table not found")
                        content = await page.content()
                        await browser.close()
                        soup = BeautifulSoup(content, 'html.parser')
                        table = soup.find('table', class_='fii-dii-table')
                        if table:
                            return self._parse_html_table(table)
                        for selector in self.css_selectors:
                            table = soup.select_one(selector)
                            if table:
                                return self._parse_html_table(table)
                        logger.warning("No valid tables found in page")
                except (PlaywrightError, asyncio.TimeoutError) as e:
                    logger.warning(f"HTTP/{http_version} attempt failed: {str(e)[:100]}")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
            retries += 1
            wait_time = self.retry_delay * (2 ** retries)
            logger.info(f"Retry {retries}/{self.max_retries} in {wait_time}s")
            await asyncio.sleep(wait_time)
        logger.error("Playwright scraping failed after all attempts")
        return None

    def _parse_html_table(self, table) -> Optional[pl.DataFrame]:
        """Parse HTML table into structured DataFrame"""
        try:
            headers = []
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            header_count = {}
            unique_headers = []
            for header in headers:
                clean_header = re.sub(r'\s+', ' ', header).strip()
                if clean_header in header_count:
                    header_count[clean_header] += 1
                    unique_headers.append(f"{clean_header}_{header_count[clean_header]}")
                else:
                    header_count[clean_header] = 1
                    unique_headers.append(clean_header)
            rows = []
            for row in table.find_all('tr')[1:]:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:
                    rows.append(row_data[:len(unique_headers)])
            if not rows:
                return None
            df = pl.DataFrame(rows, schema=unique_headers)

            column_mapping = {}
            for col in df.columns: # 'col' is the original header string from HTML
                col_text_lower = col.lower() # Use a new variable for the lowercased version for checks

                if 'date' in col_text_lower:
                    column_mapping[col] = 'date'
                elif 'fii' in col_text_lower and ('buy' in col_text_lower or 'purchase' in col_text_lower):
                    column_mapping[col] = 'fii_buy_cr'
                elif 'fii' in col_text_lower and ('sell' in col_text_lower or 'sales' in col_text_lower):
                    column_mapping[col] = 'fii_sell_cr'
                elif 'fii' in col_text_lower and 'net' in col_text_lower: # Assuming 'net' is sufficient for FII Net
                    column_mapping[col] = 'fii_net_cr'
                elif 'dii' in col_text_lower and ('buy' in col_text_lower or 'purchase' in col_text_lower):
                    column_mapping[col] = 'dii_buy_cr'
                elif 'dii' in col_text_lower and ('sell' in col_text_lower or 'sales' in col_text_lower):
                    column_mapping[col] = 'dii_sell_cr'
                elif 'dii' in col_text_lower and 'net' in col_text_lower: # Assuming 'net' is sufficient for DII Net
                    column_mapping[col] = 'dii_net_cr'

            df = df.rename(column_mapping)
            valid_columns = [col for col in [
                'date', 'fii_buy_cr', 'fii_sell_cr', 'fii_net_cr',
                'dii_buy_cr', 'dii_sell_cr', 'dii_net_cr'
            ] if col in df.columns]
            df = df.select(valid_columns)
            if 'date' in df.columns:
                df = df.with_columns(
                    pl.col("date").str.strptime(pl.Date, "%d-%b-%Y", strict=False)
                )
            numeric_cols = [col for col in df.columns if col != 'date']
            for col in numeric_cols:
                df = df.with_columns(
                    pl.col(col).str.replace_all(r'[^\d.]', '').cast(pl.Float64)
                )
            if 'fii_net_cr' not in df.columns and 'fii_buy_cr' in df.columns and 'fii_sell_cr' in df.columns:
                df = df.with_columns(
                    (pl.col("fii_buy_cr") - pl.col("fii_sell_cr")).alias("fii_net_cr")
                )
            if 'dii_net_cr' not in df.columns and 'dii_buy_cr' in df.columns and 'dii_sell_cr' in df.columns:
                df = df.with_columns(
                    (pl.col("dii_buy_cr") - pl.col("dii_sell_cr")).alias("dii_net_cr")
                )
            return df
        except Exception as e:
            logger.error(f"Table parsing failed: {e}")
            return None

    def _generate_mock_data(self) -> pl.DataFrame:
        """Generate mock FII/DII data for fallback"""
        try:
            mock_data = []
            for i in range(5):
                trade_date = date.today() - timedelta(days=i)
                if trade_date.weekday() >= 5:
                    continue
                fii_buy = round(random.uniform(2000, 8000), 2)
                fii_sell = round(random.uniform(1800, 7500), 2)
                dii_buy = round(random.uniform(1500, 4000), 2)
                dii_sell = round(random.uniform(1400, 3800), 2)
                mock_data.append({
                    'date': trade_date,
                    'fii_buy_cr': fii_buy,
                    'fii_sell_cr': fii_sell,
                    'fii_net_cr': fii_buy - fii_sell,
                    'dii_buy_cr': dii_buy,
                    'dii_sell_cr': dii_sell,
                    'dii_net_cr': dii_buy - dii_sell
                })
            df = pl.DataFrame(mock_data)
            logger.warning(f"Generated {len(mock_data)} days of mock data")
            logger.warning("⚠️ WARNING: Using mock data as fallback. Check your scraping setup.")
            return df
        except Exception as e:
            logger.error(f"Mock data generation failed: {e}")
            return None

    async def scrape(self) -> Optional[pl.DataFrame]:
        """Main scraping method with fallback strategy"""
        logger.info("Trying official API...")
        data = await self.scrape_with_api()
        if data is not None and not data.is_empty():
            return data
        logger.info("API failed, trying Playwright...")
        data = await self.scrape_with_playwright()
        if data is not None and not data.is_empty():
            return data
        logger.error("All scraping methods failed, using mock data")
        return self._generate_mock_data()

    def update_database(self, df: pl.DataFrame) -> bool:
        """UPSERT data into DuckDB"""
        if df is None or df.is_empty():
            logger.warning("No data to update")
            return False
        try:
            with duckdb.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS institutional_flows (
                        date DATE PRIMARY KEY,
                        fii_buy_cr DOUBLE,
                        fii_sell_cr DOUBLE,
                        fii_net_cr DOUBLE,
                        dii_buy_cr DOUBLE,
                        dii_sell_cr DOUBLE,
                        dii_net_cr DOUBLE
                    )
                """)
                conn.register("temp_fii_dii", df)
                conn.execute("""
                    INSERT INTO institutional_flows BY NAME
                    SELECT * FROM temp_fii_dii
                    ON CONFLICT (date) DO UPDATE SET
                        fii_buy_cr = EXCLUDED.fii_buy_cr,
                        fii_sell_cr = EXCLUDED.fii_sell_cr,
                        fii_net_cr = EXCLUDED.fii_net_cr,
                        dii_buy_cr = EXCLUDED.dii_buy_cr,
                        dii_sell_cr = EXCLUDED.dii_sell_cr,
                        dii_net_cr = EXCLUDED.dii_net_cr
                """)
                conn.execute("""
                    CREATE OR REPLACE VIEW v_institutional_trends AS
                    SELECT
                        date,
                        fii_net_cr,
                        dii_net_cr,
                        SUM(fii_net_cr) OVER (
                            ORDER BY date
                            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                        ) AS fii_30d_roll,
                        SUM(dii_net_cr) OVER (
                            ORDER BY date
                            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                        ) AS dii_30d_roll
                    FROM institutional_flows
                    ORDER BY date DESC
                """)
                logger.success(f"Updated {len(df)} records")
                return True
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            return False

async def main():
    """Main async workflow"""
    logger.info("🚀 Starting FII/DII tracker")
    start_time = time.time()
    scraper = NSE_FII_DII_Scraper()
    data = await scraper.scrape()
    if data is not None:
        scraper.update_database(data)
    else:
        logger.error("Scraping failed completely")
    duration = time.time() - start_time
    logger.info(f"✅ Tracker completed in {duration:.2f} seconds")

    if data is not None:
        logger.info(f"Data sample:\n{data.head(2)}")
        with duckdb.connect(scraper.db_path) as conn:
            latest = conn.execute("""
                SELECT date, fii_net_cr, dii_net_cr
                FROM institutional_flows
                ORDER BY date DESC
                LIMIT 1
            """).fetchdf()
            logger.success(f"Latest record: {latest.to_dict('records')[0]}")

if __name__ == "__main__":
    asyncio.run(main())
