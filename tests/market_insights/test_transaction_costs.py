import pytest
from unittest.mock import MagicMock

from quandex_core.market_insights.transaction_costs import (
    calculate_equity_trade_cost,
    calculate_round_trip_cost,
)
# Import config to mock it.
# This assumes that transaction_costs.py imports config like:
# from quandex_core.config import config
from quandex_core import config as global_config_module


@pytest.fixture
def mock_market_config(mocker):
    """
    Provides a mock market config object and patches it into the global config module
    used by transaction_costs.py.
    """
    mock_cfg = MagicMock()
    # All percentages are divided by 100 in the actual calculation function
    # So, if we want 0.01%, we set the value to 0.01 here.
    # The function will do: trade_value * (mock_cfg.brokerage / 100)
    # This seems to be a slight misunderstanding in the prompt's manual calculation.
    # The config values should represent the actual percentage value, e.g., 0.01 for 0.01%.
    # The function then divides by 100.
    # Let's assume the config stores values like 0.01 for 0.01% (not 0.0001).
    # The original MarketConfig in config.py has values like:
    # self.brokerage = 0.0001 # 0.01% -> this means it stores the direct multiplier.
    # Let's follow the structure of MarketConfig where values are direct multipliers.

    mock_cfg.brokerage = 0.0001  # 0.01% (Brokerage rate)
    mock_cfg.max_brokerage_per_trade = 20 # Max Rs 20 for brokerage

    # STT rates (these are direct multipliers)
    mock_cfg.stt_delivery_buy_rate = 0.0      # STT on buy for delivery is 0
    mock_cfg.stt_delivery_sell_rate = 0.001  # 0.1% STT on sell for delivery

    mock_cfg.exchange_transaction_charge_nse = 0.0000345 # 0.00345% for NSE Equity

    mock_cfg.gst_rate = 0.18 # 18% GST

    # SEBI Turnover Fees: Rs. 10 per Crore (0.0001%)
    # The config stores it as 'sebi_turnover_per_crore = 10'
    # The calculation is (trade_value / 1_00_00_000) * config.market.sebi_turnover_per_crore
    mock_cfg.sebi_turnover_per_crore = 10.0 # Rs 10 per crore

    # Stamp Duty: 0.015% on buy for delivery, 0 on sell
    mock_cfg.stamp_duty_buy_equity = 0.00015 # 0.015%
    mock_cfg.stamp_duty_sell_equity = 0.0    # 0%

    # Patch the 'config.market' object accessed by the functions under test
    mocker.patch.object(global_config_module, 'config', MagicMock(market=mock_cfg))
    return mock_cfg

class TestCalculateEquityTradeCost:

    def test_zero_trade_value(self, mock_market_config):
        assert calculate_equity_trade_cost(0, is_buy=True) == 0.0
        assert calculate_equity_trade_cost(0, is_buy=False) == 0.0

    def test_buy_trade_costs(self, mock_market_config: MagicMock):
        trade_value = 10000.0

        # Manual calculation based on mock_market_config values:
        # Brokerage: min(10000 * 0.0001, 20) = min(1.0, 20) = 1.0
        brokerage_cost = 1.0
        # STT (Buy): 10000 * 0.0 = 0.0
        stt_cost = 0.0
        # Exchange Charges: 10000 * 0.0000345 = 0.345
        exchange_cost = 0.345
        # Taxable for GST: Brokerage + Exchange Charges = 1.0 + 0.345 = 1.345
        taxable_value_for_gst = brokerage_cost + exchange_cost
        # GST: 1.345 * 0.18 = 0.2421
        gst_cost = taxable_value_for_gst * mock_market_config.gst_rate
        # SEBI Charges: (10000 / 1_00_00_000) * 10 = 0.001 * 10 = 0.01
        sebi_cost = (trade_value / 1_00_00_000) * mock_market_config.sebi_turnover_per_crore
        # Stamp Duty (Buy): 10000 * 0.00015 = 1.5
        stamp_duty_cost = trade_value * mock_market_config.stamp_duty_buy_equity

        expected_total_buy_cost = brokerage_cost + stt_cost + exchange_cost + gst_cost + sebi_cost + stamp_duty_cost
        # 1.0 + 0.0 + 0.345 + 0.2421 + 0.01 + 1.5 = 3.0971

        actual_cost = calculate_equity_trade_cost(trade_value, is_buy=True)
        assert actual_cost == pytest.approx(expected_total_buy_cost)

    def test_sell_trade_costs(self, mock_market_config: MagicMock):
        trade_value = 10000.0

        # Manual calculation:
        # Brokerage: min(10000 * 0.0001, 20) = 1.0
        brokerage_cost = 1.0
        # STT (Sell): 10000 * 0.001 = 10.0
        stt_cost = trade_value * mock_market_config.stt_delivery_sell_rate
        # Exchange Charges: 10000 * 0.0000345 = 0.345
        exchange_cost = 0.345
        # Taxable for GST: Brokerage + Exchange Charges = 1.0 + 0.345 = 1.345
        taxable_value_for_gst = brokerage_cost + exchange_cost
        # GST: 1.345 * 0.18 = 0.2421
        gst_cost = taxable_value_for_gst * mock_market_config.gst_rate
        # SEBI Charges: (10000 / 1_00_00_000) * 10 = 0.01
        sebi_cost = (trade_value / 1_00_00_000) * mock_market_config.sebi_turnover_per_crore
        # Stamp Duty (Sell): 10000 * 0.0 = 0.0
        stamp_duty_cost = trade_value * mock_market_config.stamp_duty_sell_equity

        expected_total_sell_cost = brokerage_cost + stt_cost + exchange_cost + gst_cost + sebi_cost + stamp_duty_cost
        # 1.0 + 10.0 + 0.345 + 0.2421 + 0.01 + 0.0 = 11.5971

        actual_cost = calculate_equity_trade_cost(trade_value, is_buy=False)
        assert actual_cost == pytest.approx(expected_total_sell_cost)

    def test_buy_trade_costs_max_brokerage_scenario(self, mock_market_config: MagicMock):
        trade_value = 300000.0 # Value high enough to hit max brokerage

        # Manual calculation:
        # Brokerage: min(300000 * 0.0001, 20) = min(30.0, 20) = 20.0
        brokerage_cost = 20.0
        stt_cost = 0.0
        # Exchange Charges: 300000 * 0.0000345 = 10.35
        exchange_cost = 10.35
        taxable_value_for_gst = brokerage_cost + exchange_cost # 20.0 + 10.35 = 30.35
        gst_cost = taxable_value_for_gst * mock_market_config.gst_rate # 30.35 * 0.18 = 5.463
        sebi_cost = (trade_value / 1_00_00_000) * mock_market_config.sebi_turnover_per_crore # (300000/1e7)*10 = 0.3
        stamp_duty_cost = trade_value * mock_market_config.stamp_duty_buy_equity # 300000 * 0.00015 = 45.0

        expected_total_buy_cost = brokerage_cost + stt_cost + exchange_cost + gst_cost + sebi_cost + stamp_duty_cost
        # 20.0 + 0.0 + 10.35 + 5.463 + 0.3 + 45.0 = 81.113

        actual_cost = calculate_equity_trade_cost(trade_value, is_buy=True)
        assert actual_cost == pytest.approx(expected_total_buy_cost)


class TestCalculateRoundTripCost:

    def test_zero_trade_value(self, mock_market_config):
        result = calculate_round_trip_cost(0)
        assert result["buy_cost"] == 0.0
        assert result["sell_cost"] == 0.0
        assert result["total_cost"] == 0.0
        assert result["total_cost_percent"] == 0.0

    def test_positive_trade_value(self, mock_market_config: MagicMock):
        trade_value = 10000.0

        # Expected costs from previous tests for trade_value = 10000.0
        expected_buy_cost = 3.0971
        expected_sell_cost = 11.5971

        expected_total_cost = expected_buy_cost + expected_sell_cost # 3.0971 + 11.5971 = 14.6942

        # For round trip, trade_value is for one side (buy or sell).
        # Total transaction involved is 2 * trade_value if we consider both legs for percentage.
        # However, typical cost percentage is calculated on the total value of shares traded (buy value + sell value).
        # If buy and sell prices are the same, it's 2 * trade_value.
        # The function seems to use `trade_value` as the basis for one leg.
        # The prompt calculation used (total_cost / (2 * trade_value)) * 100
        # Let's assume the function calculates percentage based on `trade_value` representing one side of the transaction.
        # So, total value transacted for a round trip is 2 * trade_value.
        expected_total_cost_percent = (expected_total_cost / (2 * trade_value)) * 100
        # (14.6942 / 20000) * 100 = 0.073471

        result = calculate_round_trip_cost(trade_value)

        assert result["buy_cost"] == pytest.approx(expected_buy_cost)
        assert result["sell_cost"] == pytest.approx(expected_sell_cost)
        assert result["total_cost"] == pytest.approx(expected_total_cost)
        assert result["total_cost_percent"] == pytest.approx(expected_total_cost_percent)

```
