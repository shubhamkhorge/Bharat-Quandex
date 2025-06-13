"""
DuckDB Connection and Table Diagnostic Script
Run this script to diagnose the database state
"""

import duckdb
from pathlib import Path
import sys

def diagnose_database():
    """Comprehensive database diagnosis"""
    
    # Path to your database (adjust if needed)
    db_path = Path('data_vault') / 'market_boards' / 'quandex.duckdb'
    
    print(f"🔍 Diagnosing database at: {db_path.absolute()}")
    print(f"📁 Database file exists: {db_path.exists()}")
    
    if db_path.exists():
        print(f"📏 Database file size: {db_path.stat().st_size} bytes")
    
    try:
        # Create a fresh connection
        con = duckdb.connect(str(db_path))
        print("✅ Successfully connected to database")
        
        # List all tables
        print("\n📋 All tables in database:")
        tables = con.execute("SHOW TABLES;").fetchall()
        if tables:
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("   ❌ No tables found!")
        
        # Check if processed_equity_data exists
        print("\n🔍 Checking processed_equity_data table:")
        try:
            count = con.execute("SELECT COUNT(*) FROM processed_equity_data;").fetchone()[0]
            print(f"   ✅ Table exists with {count} records")
            
            # Show sample data
            print("\n📊 Sample data from processed_equity_data:")
            sample = con.execute("""
                SELECT date, symbol, close, sma_50, rsi_14 
                FROM processed_equity_data 
                WHERE symbol IN ('RELIANCE.NS', 'TCS.NS')
                ORDER BY date DESC 
                LIMIT 5
            """).fetchall()
            
            if sample:
                for row in sample:
                    print(f"   {row}")
            else:
                print("   ⚠️ No data for RELIANCE.NS or TCS.NS found")
                
        except duckdb.CatalogException as e:
            print(f"   ❌ Table does not exist: {e}")
        
        # Check raw_equity_data
        print("\n🔍 Checking raw_equity_data table:")
        try:
            count = con.execute("SELECT COUNT(*) FROM raw_equity_data;").fetchone()[0]
            print(f"   ✅ Table exists with {count} records")
            
            # Show symbols
            symbols = con.execute("SELECT DISTINCT symbol FROM raw_equity_data;").fetchall()
            print(f"   📈 Symbols: {[s[0] for s in symbols]}")
            
        except duckdb.CatalogException as e:
            print(f"   ❌ Table does not exist: {e}")
        
        # Check database settings
        print("\n⚙️ Database settings:")
        try:
            autocommit = con.execute("SELECT current_setting('autocommit');").fetchone()[0]
            print(f"   Autocommit: {autocommit}")
        except:
            print("   Could not get autocommit setting")
        
        # Force commit and checkpoint
        print("\n💾 Forcing commit and checkpoint:")
        try:
            con.commit()
            con.execute("CHECKPOINT;")
            print("   ✅ Commit and checkpoint successful")
        except Exception as e:
            print(f"   ⚠️ Commit/checkpoint failed: {e}")
        
        con.close()
        print("\n🔌 Connection closed")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    return True

def test_config_connection():
    """Test the config module connection"""
    print("\n🔧 Testing config module connection...")
    
    try:
        # Add parent directory to path to import config
        parent_dir = Path.cwd().parent
        if str(parent_dir) not in sys.path:
            sys.path.append(str(parent_dir))
        
        from quandex_core.config import config
        print("✅ Successfully imported config")
        
        print(f"📁 Config database path: {config.data.duckdb_path}")
        
        # Test the config connection
        if hasattr(config.data, 'conn') and config.data.conn:
            print("🔌 Config has active connection")
            
            try:
                tables = config.data.conn.execute("SHOW TABLES;").fetchall()
                print(f"📋 Tables via config connection: {[t[0] for t in tables]}")
            except Exception as e:
                print(f"❌ Config connection query failed: {e}")
        else:
            print("❌ Config has no active connection")
            
    except ImportError as e:
        print(f"❌ Could not import config: {e}")
    except Exception as e:
        print(f"❌ Config test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting DuckDB diagnosis...\n")
    
    # Test direct connection
    success = diagnose_database()
    
    # Test config connection
    test_config_connection()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ Diagnosis complete - check results above")
    else:
        print("❌ Diagnosis found issues - check errors above")