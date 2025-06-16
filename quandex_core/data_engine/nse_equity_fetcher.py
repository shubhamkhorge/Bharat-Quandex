"""
Ultra-Fast NSE Equity Data Fetcher with Polars and DuckDB Integration
Optimized for Indian market data with parallel processing and incremental updates
"""

import polars as pl
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger
import time
import concurrent.futures
import duckdb
import numpy as np
from tqdm import tqdm

from ..config import config, get_raw_data_path, get_data_path

class NSEDataFetcher:
    """High-performance NSE data fetcher with parallel processing and DuckDB integration"""

    def __init__(self):
        self.raw_path = config.data.raw_data_path
        self.processed_path = config.data.processed_data_path

        # Initialize or refresh connection
        if not hasattr(config.data, 'conn') or config.data.conn is None:
            config.data.__post_init__()

        self.conn = config.data.conn

        # Test connection
        try:
            self.conn.execute("SELECT 1").fetchall()
        except:
            logger.warning("Database connection stale, reconnecting...")
            config.data.conn = duckdb.connect(str(config.data.duckdb_path))
            self.conn = config.data.conn

        # Load symbols
        self.nifty_500_symbols = self._load_nifty_symbols()
        logger.info(f"NSE Data Fetcher initialized with {len(self.nifty_500_symbols)} symbols")
        logger.info(f"DuckDB path: {config.data.duckdb_path}")

        # Initialize database tables if needed
        self._initialize_database()

    def _load_nifty_symbols(self) -> List[str]:
        """Load Nifty 500 symbols with fallback mechanism"""
        try:
            # Try loading from DuckDB first
            result = self.conn.execute("SELECT symbol FROM nse_symbols").fetchall()
            if result:
                return [row[0] for row in result]
        except:
            pass

        # Fallback to static list
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
            "HDFC.NS", "ITC.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "SBIN.NS",
            "ASIANPAINT.NS", "AXISBANK.NS", "LT.NS", "MARUTI.NS", "BAJFINANCE.NS",
            "WIPRO.NS", "ONGC.NS", "SUNPHARMA.NS", "NTPC.NS", "TITAN.NS"
        ]

    def _initialize_database(self,max_retries=3):
        """Initialize DuckDB tables with error handling"""
        if not hasattr(self, '_init_attempts'):
            self._init_attempts = 0

        if self._init_attempts >= max_retries:
            logger.error("Maximum database initialization attempts reached")
            return False
        try:
            # Create raw data table with proper constraints
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_equity_data (
                date DATE NOT NULL,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT,
                symbol VARCHAR NOT NULL,
                fetched_at TIMESTAMP,
                CONSTRAINT pk_raw_data PRIMARY KEY (date, symbol)
            )
            """)

            # Create processed data table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_equity_data (
                date DATE,
                symbol VARCHAR,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT,
                daily_return FLOAT,
                sma_20 FLOAT,
                sma_50 FLOAT,
                sma_200 FLOAT,
                rsi_14 FLOAT,
                macd FLOAT,
                macd_signal FLOAT,
                bollinger_upper FLOAT,
                bollinger_lower FLOAT,
                atr_14 FLOAT,
                PRIMARY KEY (date, symbol)
            )
            """)

            # Create symbols table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nse_symbols (
                symbol STRING PRIMARY KEY,
                name STRING,
                sector STRING,
                last_updated TIMESTAMP
            )
            """)
            logger.info("Database tables initialized with proper constraints")
            return True
        except duckdb.Error as e:
            self._init_attempts += 1
            logger.error(f"Database initialization failed: {str(e)}")
            try:
            # Try to reconnect only if we haven't exceeded max attempts
                if self._init_attempts < max_retries:
                    config.data.conn = duckdb.connect(str(config.data.duckdb_path))
                    self.conn = config.data.conn
                    logger.info("Reconnected to DuckDB - retrying initialization")
                    return self._initialize_database(max_retries)  # Retry with same max_retries
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect: {str(reconnect_error)}")
                return False
        return False

    def _fetch_symbol_with_retry(self, symbol: str, start_date: Optional[datetime.date], max_retries=3, delay=1) ->Optional[pl.DataFrame]:
        """A simple retry wrapper that now takes a start_date."""
        for attempt in range(max_retries):
            try:
                # Pass the start_date directly to the fetcher
                return self._fetch_single_stock(symbol, start_date)
            except Exception as e:
                # No more DuckDB errors here, just potential network/yfinance errors
                logger.error(f"Error fetching {symbol} on attempt {attempt+1}: {e}")
                time.sleep(delay * (attempt + 1))
        logger.error(f"Failed to fetch {symbol} after {max_retries} retries.")
        return None

    def _fetch_single_stock(self, symbol: str, start_date: Optional[datetime.date]) -> Optional[pl.DataFrame]:
        """
        Simplified fetcher. It NO LONGER connects to DuckDB.
        It just takes a symbol and an optional start_date.
        """
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now().date()
            
            # If no start_date provided, fetch 2 years of historical data
            if start_date is None:
                start_date = (end_date - timedelta(days=365*2))

            # yfinance handles a None start_date by fetching max history
            hist = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                prepost=False
            )

            if hist.empty:
                return None

            df = pl.from_pandas(hist.reset_index())

            df = df.rename({
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume"
            })

            df = df.with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit(datetime.now().replace(microsecond=0)).alias("fetched_at"),
                pl.col("date").dt.date().cast(pl.Date)
            )
            
            if start_date:
                df = df.filter(pl.col("date") >= start_date)

            # Cast to proper types (can be done here or after concatenation)
            df = df.with_columns(
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Int64)
            )

            return df
        except Exception as e:
            # Re-raising allows the retry wrapper to catch it
            raise e
    def fetch_parallel(self, symbols: List[str], max_workers=8) -> pl.DataFrame:
        """
        Fetch data in parallel.
        Now includes an upfront query to get all start dates.
        """
        all_data = []

        # --- STEP 1: UPFRONT DATABASE READ ---
        logger.info("Querying database for last known dates for all symbols...")
        try:
            latest_dates_df = self.conn.execute("""
                SELECT symbol, MAX(date) as last_date
                FROM raw_equity_data
                WHERE symbol = ANY (?)
                GROUP BY symbol
            """, [symbols]).pl()
            
            # Convert to a convenient dictionary for quick lookups
            latest_dates_map = {row['symbol']: row['last_date'] for row in latest_dates_df.to_dicts()}
            logger.info(f"Found latest dates for {len(latest_dates_map)} symbols.")

        except duckdb.Error as e:
            logger.error(f"Could not get latest dates from DB, will perform full fetch. Error: {e}")
            latest_dates_map = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {}
            for symbol in symbols:
                # --- STEP 2: PASS INFO TO THREADS ---
                last_date = latest_dates_map.get(symbol)
                # Fetch from the day after the last known date
                start_date = (last_date + timedelta(days=1)) if last_date else None
                
                # The worker no longer needs a DB connection
                future = executor.submit(self._fetch_symbol_with_retry, symbol, start_date)
                future_to_symbol[future] = symbol

            for future in tqdm(concurrent.futures.as_completed(future_to_symbol),
                              total=len(symbols),
                              desc="Fetching Data"):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None and not df.is_empty():
                        all_data.append(df)

                except Exception as e:
                    logger.error(f"Future for {symbol} resulted in an error: {str(e)}")

            for symbol in symbols:
                last_date = latest_dates_map.get(symbol)
                start_date = (last_date + timedelta(days=1)) if last_date else None
                future = executor.submit(self._fetch_symbol_with_retry, symbol, start_date)
                future_to_symbol[future] = symbol

        return pl.concat(all_data, how="vertical_relaxed") if all_data else pl.DataFrame()
            
    def save_to_duckdb(self, df: pl.DataFrame, table_name="raw_equity_data"):
        """Save DataFrame to DuckDB table with conflict resolution"""
        if df.is_empty():
            return

        # Only keep columns that match the table schema
        expected_cols = ["date", "open", "high", "low", "close", "volume", "symbol", "fetched_at"]
        df = df.select([col for col in expected_cols if col in df.columns])

        # Register DataFrame as temporary view
        self.conn.register("temp_data", df)

        # Use proper DuckDB upsert syntax with ON CONFLICT
        try:
            # Check if table exists and has constraints
            result = self.conn.execute(f"""
                SELECT column_name
                FROM information_schema.key_column_usage
                WHERE table_name = '{table_name}'
                AND constraint_name LIKE 'primary%'
            """).fetchall()

            if not result and table_name == "raw_equity_data":
                logger.warning("Table lacks primary key constraints - recreating table")
                # Backup existing data
                backup = self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()

                # Recreate table with proper constraints
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.conn.execute(f"""
                CREATE TABLE {table_name} (
                    date DATE NOT NULL,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume BIGINT,
                    symbol VARCHAR NOT NULL,
                    fetched_at TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
                """)

                # Restore backup data
                if not backup.empty:
                    self.conn.register("backup_data", backup)
                    self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM backup_data")

            # Now attempt the upsert
            self.conn.execute(f"""
                INSERT INTO {table_name} ({', '.join(df.columns)})
                SELECT {', '.join(df.columns)} FROM temp_data
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    fetched_at = EXCLUDED.fetched_at
            """)

            logger.info(f"Saved {len(df)} records to {table_name}")

        except duckdb.BinderException as e:
            if "no UNIQUE/PRIMARY KEY" in str(e).lower():
                logger.error("Table missing constraints - will attempt to fix")
                # If we get here, the previous attempt to fix constraints failed
                # Implement more robust table recreation logic
                self._initialize_database()  # Force reinitialization
                self.save_to_duckdb(df, table_name)  # Retry
            else:
                raise
        except Exception as e:
            logger.error(f"Error saving to DuckDB: {str(e)}")
            raise

    def process_raw_data(self):
        """
        Process raw data into analytical features using a single, atomic,
        and robust CREATE OR REPLACE TABLE command.
        """
        logger.info("Processing raw data into analytical features...")

        # This single, complex query will be used to build our final table directly.
        self.conn.execute("""
        DELETE FROM raw_equity_data 
        WHERE date IS NULL OR symbol IS NULL
        """)

        processing_query = """
        CREATE OR REPLACE TABLE processed_equity_data AS
        WITH
        raw_data AS (
            SELECT * FROM raw_equity_data
        ),
        daily_returns AS (
            SELECT
                date,
                symbol,
                (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) AS daily_return
            FROM raw_data
        ),
        moving_averages AS (
            SELECT
                date,
                symbol,
                AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
                AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50,
                AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS sma_200,
                STDDEV(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS stdev_20
            FROM raw_data
        ),
        rsi_base AS (
            SELECT
                date,
                symbol,
                close - LAG(close) OVER (PARTITION BY symbol ORDER BY date) as price_change
            FROM raw_data
        ),
        rsi_gains_losses AS (
            SELECT
                date,
                symbol,
                CASE WHEN price_change > 0 THEN price_change ELSE 0 END AS gain,
                CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END AS loss
            FROM rsi_base
        ),
        rsi AS (
            SELECT
                date,
                symbol,
                100.0 - (100.0 / (1.0 + (
                    AVG(gain) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) /
                    NULLIF(AVG(loss) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 0)
                ))) AS rsi_14
            FROM rsi_gains_losses
        )
        -- Final SELECT to join all the calculated features
        SELECT
            rd.date,
            rd.symbol,
            rd.open,
            rd.high,
            rd.low,
            rd.close,
            rd.volume,
            dr.daily_return,
            ma.sma_20,
            ma.sma_50,
            ma.sma_200,
            rsi.rsi_14,
            -- Placeholder for MACD and ATR as they require more complex subqueries
            NULL::DOUBLE AS macd,
            NULL::DOUBLE AS macd_signal,
            (ma.sma_20 + (2 * ma.stdev_20)) AS bollinger_upper,
            (ma.sma_20 - (2 * ma.stdev_20)) AS bollinger_lower,
            NULL::DOUBLE AS atr_14
        FROM raw_data rd
        LEFT JOIN daily_returns dr ON rd.date = dr.date AND rd.symbol = dr.symbol
        LEFT JOIN moving_averages ma ON rd.date = ma.date AND rd.symbol = ma.symbol
        LEFT JOIN rsi ON rd.date = rsi.date AND rd.symbol = rsi.symbol;
        """

        try:
            # Execute the single, atomic command
            self.conn.execute(processing_query)

            count = self.conn.execute("SELECT COUNT(*) FROM processed_equity_data").fetchone()[0]
            logger.info(f"Successfully created 'processed_equity_data' table with {count} records.")
            return True
        except Exception as e:
            logger.error(f"Error processing raw data with CREATE TABLE AS: {e}")
            logger.exception("Traceback:")
            return False

    def update_symbols_list(self):
        """Update NSE symbols list from external source (mock implementation)"""
        # In production, this would fetch from NSE website or API
        new_symbols = [
            {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automobile"},
            {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Conglomerate"}
        ]

        # Insert new symbols
        for symbol_data in new_symbols:
            self.conn.execute(f"""
            INSERT OR IGNORE INTO nse_symbols (symbol, name, sector, last_updated)
            VALUES (
                '{symbol_data['symbol']}',
                '{symbol_data['name']}',
                '{symbol_data['sector']}',
                '{datetime.now().replace(microsecond=0)}'
            )
            """)

        # Refresh in-memory list
        self.nifty_500_symbols = self._load_nifty_symbols()
        logger.info(f"Updated symbols list: {len(self.nifty_500_symbols)} symbols")

    def incremental_update(self, days_to_update=1):
        """Safer incremental update"""
        logger.info("Starting safe incremental update")

        try:
            # Get symbols safely
            symbols = self.nifty_500_symbols.copy()
            if not symbols:
                logger.warning("No symbols available for update")
                return False

            # Process in smaller batches
            batch_size = 10
            all_new_data = []

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{len(symbols)//batch_size + 1}")

                new_data = self.fetch_parallel(batch)
                if not new_data.is_empty():
                    self.save_to_duckdb(new_data)
                    all_new_data.append(new_data)

                # Small delay between batches
                time.sleep(1)

            if all_new_data:
                self.process_raw_data()
                return True

            return False

        except Exception as e:
            logger.exception("Critical error in incremental update")
            return False

    def full_refresh(self, symbols: List[str] = None):
        """Perform full historical data refresh"""
        symbols_to_fetch = symbols or self.nifty_500_symbols
        logger.info(f"Starting full refresh for {len(symbols_to_fetch)} symbols")

        # Clear existing data
        self.conn.execute("DELETE FROM raw_equity_data WHERE symbol IN ({})".format(
            ','.join([f"'{s}'" for s in symbols_to_fetch])
        ))

        # Fetch all historical data
        new_data = self.fetch_parallel(symbols_to_fetch)

        if new_data.is_empty():
            logger.error("No data fetched during full refresh")
            return False

        # Save to database
        self.save_to_duckdb(new_data)

        # Process data
        self.process_raw_data()

        logger.info(f"Full refresh complete: {len(new_data)} records")
        return True

    def _ensure_fresh_connection(self):
        """Ensure we have a fresh, working connection"""
        try:
            # Test current connection
            self.conn.execute("SELECT 1").fetchone()
            return True
        except:
            # Connection is stale, create a new one
            logger.warning("Connection stale, creating fresh connection")
            try:
                self.conn.close()
            except:
                pass
            
            # Create new connection
            self.conn = duckdb.connect(str(config.data.duckdb_path))
            config.data.conn = self.conn  # Update the shared connection
            
            # Re-initialize tables if needed
            self._initialize_database()
            return True    

    def get_processed_data(self, symbol: str, start_date: str, end_date: str) -> pl.DataFrame:
        """Retrieve processed data for analysis with connection fix"""
    
        # Ensure fresh connection
        self._ensure_fresh_connection()
        
        # Ensure dates are in correct format
        try:
            if not isinstance(start_date, str) or not start_date.strip():
                start_date = "2000-01-01"
            elif not start_date.strip().count('-') == 2:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    
            if not isinstance(end_date, str) or not end_date.strip():
                end_date = datetime.now().strftime("%Y-%m-%d")
            elif not end_date.strip().count('-') == 2:
                end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            # Format for DuckDB query
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")    
        except ValueError:
            logger.warning(f"Invalid date format. Using default date range.")
            start_date = "2000-01-01"
            end_date = datetime.now().strftime("%Y-%m-%d")
    
        # Force a commit before querying
        try:
            self.conn.commit()
        except:
            pass
    
        query = f"""
        SELECT *
        FROM processed_equity_data
        WHERE
            symbol = '{symbol}' AND
            date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """
        
        try:
            result = self.conn.execute(query).pl()
            logger.info(f"Retrieved {len(result)} records for {symbol}")
            return result
        except Exception as e:
            logger.error(f"Error retrieving processed data: {e}")
            return pl.DataFrame()


    def cleanup(self):
        """Clean up resources safely"""
        logger.info("Cleaning up resources")
        try:
            # DuckDB cleanup 
            if hasattr(self, 'conn') and self.conn:
                try:
                    # Get list of all tables and views
                    all_objects = []
                    try:
                        # Method 1: Try with name column
                        all_objects = self.conn.execute("""
                            SELECT table_name, table_type FROM information_schema.tables
                            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                        """).fetchall()
                    except Exception as e:
                        logger.warning(f"Could not list objects for cleanup: {str(e)}")
                    # Method 2: Alternative query if the above fails
                        try:
                            all_objects = []
                             # Get tables
                            tables = self.conn.execute("SELECT name, 'TABLE' as type FROM duckdb_tables()").fetchall()
                            all_objects.extend(tables)
                             # Get views
                            views = self.conn.execute("SELECT table_name as name, 'VIEW' as type FROM duckdb_views()").fetchall()
                            all_objects.extend(views)
                        except Exception:
                        # Method 3: Simple table listing
                            tables = self.conn.execute("SHOW TABLES").fetchall()
                            all_objects = [(t[0], 'TABLE') for t in tables]

                    # Drop all temporary objects
                    for obj in all_objects:
                        obj_name, obj_type = obj[0],obj[1]
                        if obj_name.startswith('temp_'):
                            try:
                                if obj_type.lower() == 'view':
                                    self.conn.execute(f"DROP VIEW IF EXISTS {obj_name}")
                                else:
                                    self.conn.execute(f"DROP TABLE IF EXISTS {obj_name}")
                            except Exception as e:
                                logger.warning(f"Error dropping {obj_type} {obj_name}: {str(e)}")
                     # Commit and checkpoint if possible
                    try:
                        if self.conn:
                            self.conn.commit()
                            self.conn.execute("CHECKPOINT")
                    except Exception as e:
                        logger.warning(f"Error during commit/checkpoint: {str(e)}")

                except Exception as db_err:
                    logger.warning(f"DuckDB cleanup failed: {str(db_err)}")
                    
            # Polars cleanup with version awareness
            try:
                if hasattr(pl, 'clear_cache'):
                    pl.clear_cache()
                elif hasattr(pl, 'free_cache'):
                    pl.free_cache()
                else:
                    logger.info("Polars cache clearing not available in this version")
            except ImportError:
                pass 
            except Exception as pl_err:
                logger.warning(f"Polars cleanup failed: {str(pl_err)}")

        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")
        
        finally:
            # Close all open connections
            try:
                if hasattr(self, 'conn') and self.conn:
                    try:
                        self.conn.close()
                        logger.info("Closed DuckDB connection in finally block")
                    except Exception as e:
                        logger.warning(f"Error closing connection in finally block: {str(e)}")
                    self.conn = None
            except Exception as e:
                logger.warning(f"Error in finally block: {str(e)}")

                 # Clear other resources
            self.nifty_500_symbols = None
            logger.info("Cleanup completed")
            

def main():
    """Main function to run a targeted and verified data refresh."""
    logger.info("--- Starting Definitive Data Refresh ---")
    
    try:
        # We will connect directly here to avoid any ambiguity
        conn = duckdb.connect(str(config.data.duckdb_path))
        print(f"DuckDB connection successful")
    except Exception as e:
        print(f"DuckDB connection failed: {str(e)}")
        return

    # Instantiate the fetcher with the direct connection
    fetcher = NSEDataFetcher()
    fetcher.conn = conn

    # Define the exact symbols we need for our backtest
    symbols_to_test = ["RELIANCE.NS", "TCS.NS"]
    
    try:
        # --- 1. Perform a guaranteed FULL refresh for these symbols ---
        print(f"\n1. Performing a FULL data refresh for: {symbols_to_test}")
        # This will DELETE old data for these symbols and fetch everything fresh.
        refresh_successful = fetcher.full_refresh(symbols=symbols_to_test)
        
        if not refresh_successful:
            raise RuntimeError("Full refresh process failed to return a success status.")

        # --- 2. Force the database to save all changes to disk ---
        print("\n2. Forcing database CHECKPOINT to persist all changes...")
        fetcher.conn.execute("CHECKPOINT;")
        print("--> CHECKPOINT successful.")

        # --- 3. Verify the data exists IN THE SAME SCRIPT ---
        print("\n3. Verifying data in 'processed_equity_data' table...")
        
        # We query the count of records for our specific symbols
        verification_query = """
        SELECT symbol, COUNT(*) as record_count
        FROM processed_equity_data
        WHERE symbol = ANY (?)
        GROUP BY symbol
        """
        result_df = fetcher.conn.execute(verification_query, [symbols_to_test]).pl()

        if result_df.is_empty():
            raise ValueError("Verification FAILED: No records found for test symbols after refresh.")
        
        print("Verification successful. Data found in processed table:")
        print(result_df)

        print("\n--- Data Refresh Complete and Verified ---")

    except Exception as e:
        logger.exception("An error occurred during the data refresh and verification process:")
    finally:
        # Always clean up resources
        fetcher.cleanup()

if __name__ == "__main__":
    main()