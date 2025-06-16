"""
Transaction Cost Calculator

This module provides functions to realistically model the costs associated
with trading on Indian exchanges (NSE/BSE). It reads all cost parameters
from the central configuration.
"""

from quandex_core.config import config

def calculate_equity_trade_cost(trade_value: float, is_buy: bool) -> float:
    """
    Calculates the total transaction cost for a single equity trade.

    This function accounts for brokerage, STT, exchange charges, GST,
    SEBI charges, and stamp duty, based on the side of the trade.

    Args:
        trade_value (float): The total value of the trade (e.g., price * quantity).
        is_buy (bool): True if the trade is a buy, False if it is a sell.

    Returns:
        float: The total calculated cost for the trade.
    """
    if trade_value <= 0:
        return 0.0

    market_params = config.market
    total_cost = 0.0

    # 1. Brokerage
    brokerage_cost = trade_value * (market_params.brokerage / 100)
    total_cost += brokerage_cost

    # 2. Securities Transaction Tax (STT)
    # Applied on both buy and sell for equity delivery, but let's assume
    # intraday/derivatives logic might differ. For simplicity, your config
    # had it on sell-side, which is a common case.
    if not is_buy:
        total_cost += trade_value * (market_params.stt_equity / 100)

    # 3. Exchange Transaction Charges
    exchange_cost = trade_value * (market_params.exchange_charges / 100)
    total_cost += exchange_cost

    # 4. Goods and Services Tax (GST)
    # GST is applied on the sum of Brokerage and Exchange Charges.
    taxable_for_gst = brokerage_cost + exchange_cost
    total_cost += taxable_for_gst * (market_params.gst / 100)

    # 5. SEBI Turnover Charges
    total_cost += trade_value * (market_params.sebi_charges / 100)

    # 6. Stamp Duty
    # Applied only on the buy side.
    if is_buy:
        total_cost += trade_value * (market_params.stamp_duty / 100)

    return total_cost

def calculate_round_trip_cost(trade_value: float) -> dict:
    """
    Calculates the costs for a full round-trip trade (buy and then sell)
    of the same value.

    Args:
        trade_value (float): The value of the initial buy trade.

    Returns:
        dict: A dictionary containing buy_cost, sell_cost, total_cost,
              and total_cost_percent.
    """
    buy_cost = calculate_equity_trade_cost(trade_value, is_buy=True)
    sell_cost = calculate_equity_trade_cost(trade_value, is_buy=False)
    total_cost = buy_cost + sell_cost
    
    # The total percentage cost is relative to the initial capital outlay (buy side)
    # plus the capital received (sell side). A round trip involves 2 * trade_value turnover.
    total_cost_percent = (total_cost / (2 * trade_value)) * 100 if trade_value > 0 else 0.0

    return {
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "total_cost": total_cost,
        "total_cost_percent": total_cost_percent,
    }

# --- Self-Test Block ---
# This allows us to run the file directly to test its logic.
if __name__ == '__main__':
    print("--- Testing Transaction Cost Calculator ---")
    
    test_trade_value = 100000.0  # 1 Lakh INR
    
    print(f"\nCalculating costs for a single trade of INR {test_trade_value:,.2f}:")
    buy_cost_example = calculate_equity_trade_cost(test_trade_value, is_buy=True)
    print(f"  - Buy-side Cost:  INR {buy_cost_example:7.2f} ({buy_cost_example/test_trade_value*100:.4f}%)")

    sell_cost_example = calculate_equity_trade_cost(test_trade_value, is_buy=False)
    print(f"  - Sell-side Cost: INR {sell_cost_example:7.2f} ({sell_cost_example/test_trade_value*100:.4f}%)")

    print(f"\nCalculating costs for a round-trip trade of INR {test_trade_value:,.2f}:")
    round_trip_info = calculate_round_trip_cost(test_trade_value)
    
    total = round_trip_info['total_cost']
    total_pct = (total / test_trade_value) * 100
    
    print(f"  - Total Round-Trip Cost: INR {total:7.2f}")
    print(f"  - Cost as a percentage of ONE side of the trade: {total_pct:.4f}%")
    print("\nThis means a stock must appreciate by at least this percentage for a round-trip trade to break even.")
    print("\n--- Test Complete ---")