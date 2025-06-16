"""
Portfolio Risk Management Models

This module provides functions for sophisticated risk control, including
dynamic position sizing and stop-loss calculations.
"""
from loguru import logger

def calculate_fixed_fractional_position_size(
    portfolio_capital: float,
    stock_price: float,
    risk_fraction: float = 0.02
) -> int:
    """
    Calculates the number of shares to trade based on a fixed fractional
    risk model (e.g., risk no more than 2% of capital on a single stock).

    Args:
        portfolio_capital (float): The total current value of the portfolio.
        stock_price (float): The current price of the stock to be traded.
        risk_fraction (float): The fraction of the portfolio to allocate to this position.

    Returns:
        int: The calculated number of shares to trade. Returns 0 if inputs are invalid.
    """
    if stock_price <= 0 or portfolio_capital <= 0:
        return 0

    # Determine the total cash value to allocate to this position
    capital_to_allocate = portfolio_capital * risk_fraction
    
    # Calculate the number of shares that can be bought with the allocated capital
    num_shares = int(capital_to_allocate / stock_price)
    
    logger.debug(f"Position Size Calc: Capital={portfolio_capital:,.0f}, Price={stock_price:,.2f}, RiskFrac={risk_fraction:.2f} -> Shares={num_shares}")
    return num_shares

def check_stop_loss_triggered(
    entry_price: float,
    current_price: float,
    position_type: str, # 'LONG' or 'SHORT'
    stop_loss_pct: float = 0.05
) -> bool:
    """
    Checks if a position should be liquidated due to a stop-loss trigger.

    Args:
        entry_price (float): The price at which the position was entered.
        current_price (float): The current market price of the stock.
        position_type (str): The direction of the trade, 'LONG' or 'SHORT'.
        stop_loss_pct (float): The percentage loss at which to trigger the stop.

    Returns:
        bool: True if the stop-loss level has been breached, False otherwise.
    """
    if entry_price <= 0:
        return False
        
    if position_type == 'LONG':
        # For a long position, trigger if the price drops below the stop level
        stop_price = entry_price * (1 - stop_loss_pct)
        if current_price < stop_price:
            logger.warning(f"STOP-LOSS TRIGGERED (LONG): Entry={entry_price:.2f}, Current={current_price:.2f}, Stop Level={stop_price:.2f}")
            return True
            
    elif position_type == 'SHORT':
        # For a short position, trigger if the price rises above the stop level
        stop_price = entry_price * (1 + stop_loss_pct)
        if current_price > stop_price:
            logger.warning(f"STOP-LOSS TRIGGERED (SHORT): Entry={entry_price:.2f}, Current={current_price:.2f}, Stop Level={stop_price:.2f}")
            return True
            
    return False

# --- Self-Test Block ---
if __name__ == '__main__':
    print("--- Testing Risk Management Models ---")

    # 1. Test Position Sizing
    print("\n--- Testing Position Sizing ---")
    capital = 1000000.0
    price = 1500.0
    shares = calculate_fixed_fractional_position_size(capital, price, risk_fraction=0.05)
    print(f"With INR {capital:,.0f} capital, for a stock at INR {price:,.2f}, allocating 5% results in buying {shares} shares.")
    
    # 2. Test Stop Loss
    print("\n--- Testing Stop-Loss Logic ---")
    long_entry = 100.0
    short_entry = 200.0
    
    # Use the CORRECT function name: check_stop_loss_triggered
    print(f"Testing LONG position entered at {long_entry:.2f}:")
    print(f"  - Price at 102.0 (profit): Stop triggered? {check_stop_loss_triggered(long_entry, 102.0, 'LONG')}")
    print(f"  - Price at 96.0 (small loss): Stop triggered? {check_stop_loss_triggered(long_entry, 96.0, 'LONG')}")
    print(f"  - Price at 94.0 (big loss): Stop triggered? {check_stop_loss_triggered(long_entry, 94.0, 'LONG')}")
    
    # Use the CORRECT function name: check_stop_loss_triggered
    print(f"\nTesting SHORT position entered at {short_entry:.2f}:")
    print(f"  - Price at 195.0 (profit): Stop triggered? {check_stop_loss_triggered(short_entry, 195.0, 'SHORT')}")
    print(f"  - Price at 208.0 (small loss): Stop triggered? {check_stop_loss_triggered(short_entry, 208.0, 'SHORT')}")
    print(f"  - Price at 211.0 (big loss): Stop triggered? {check_stop_loss_triggered(short_entry, 211.0, 'SHORT')}")
    
    print("\n--- Test Complete ---")