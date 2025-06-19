import pytest
import datetime

from quandex_core.market_insights.trading_holidays import is_trading_day, _TRADING_HOLIDAYS_SET

# Helper to create date objects
d = datetime.date

class TestIsTradingDay:

    # 1. Test Trading Days (Weekdays, not holidays)
    @pytest.mark.parametrize(
        "test_date",
        [
            d(2023, 1, 2),  # Monday
            d(2023, 6, 13), # Tuesday
            d(2023, 9, 21), # Thursday
            d(2024, 2, 1),  # Thursday
            d(2024, 7, 17), # Wednesday
            d(2024, 11, 11),# Monday
            d(2025, 4, 2),  # Wednesday
            d(2025, 8, 11), # Monday
            d(2025, 10, 3), # Friday
        ],
    )
    def test_regular_weekdays_are_trading_days(self, test_date):
        """Test that a known weekday that is not a holiday is a trading day."""
        assert test_date not in _TRADING_HOLIDAYS_SET, f"{test_date} is in holiday set but shouldn't be for this test."
        assert test_date.weekday() < 5, f"{test_date} is not a weekday."
        assert is_trading_day(test_date) is True

    # 2. Test Weekends
    @pytest.mark.parametrize(
        "test_date",
        [
            d(2023, 1, 7),  # Saturday
            d(2023, 1, 8),  # Sunday
            d(2024, 3, 23), # Saturday (Holi is 25th, this Sat before)
            d(2024, 3, 24), # Sunday (Holi is 25th, this Sun before)
            d(2025, 8, 16), # Saturday (Independence day is 15th, this is Sat after)
            d(2025, 8, 17), # Sunday (Independence day is 15th, this is Sun after)
            # Test a Saturday that is also listed as a holiday (e.g., Diwali if it falls on Sat)
            # From the list, 2023-11-11 is Diwali (Saturday) - this is not in the provided list,
            # let's find one from the list.
            # 2024-11-02 (Diwali Balipratipada) is a Saturday.
            d(2024, 11, 2), # Saturday, also a holiday in _TRADING_HOLIDAYS_SET
        ],
    )
    def test_weekends_are_not_trading_days(self, test_date):
        """Test that weekends (Saturday, Sunday) are not trading days."""
        assert test_date.weekday() >= 5, f"{test_date} is not a weekend."
        # For the special case where a weekend day is also a holiday
        if test_date == d(2024, 11, 2):
            assert test_date in _TRADING_HOLIDAYS_SET, f"{test_date} should be in holiday set for this specific case."
        assert is_trading_day(test_date) is False

    # 3. Test Holidays (Weekdays that are holidays)
    @pytest.mark.parametrize(
        "holiday_date",
        [
            # 2023 Holidays (from _TRADING_HOLIDAYS_SET, ensuring they are weekdays)
            d(2023, 1, 26),  # Republic Day, Thursday
            d(2023, 3, 7),   # Holi, Tuesday
            d(2023, 4, 4),   # Mahavir Jayanti, Tuesday
            d(2023, 4, 7),   # Good Friday, Friday
            d(2023, 4, 14),  # Dr.Baba Saheb Ambedkar Jayanti, Friday
            d(2023, 5, 1),   # Maharashtra Day, Monday
            d(2023, 6, 29),  # Bakri ID, Thursday
            d(2023, 8, 15),  # Independence Day, Tuesday
            d(2023, 9, 19),  # Ganesh Chaturthi, Tuesday
            d(2023, 10, 2),  # Mahatma Gandhi Jayanti, Monday
            d(2023, 10, 24), # Dussehra, Tuesday
            d(2023, 11, 27), # Guru Nanak Jayanti, Monday
            d(2023, 12, 25), # Christmas, Monday

            # 2024 Holidays (from _TRADING_HOLIDAYS_SET, ensuring they are weekdays)
            d(2024, 1, 26),  # Republic Day, Friday
            d(2024, 3, 8),   # Mahashivratri, Friday
            d(2024, 3, 25),  # Holi, Monday
            d(2024, 3, 29),  # Good Friday, Friday
            d(2024, 4, 11),  # Id-Ul-Fitr (Ramzan ID), Thursday
            d(2024, 4, 17),  # Shri Ram Navmi, Wednesday
            d(2024, 5, 1),   # Maharashtra Day, Wednesday
            d(2024, 6, 17),  # Bakri ID, Monday
            d(2024, 7, 17),  # Moharram, Wednesday
            d(2024, 8, 15),  # Independence Day/Parsi New Year, Thursday
            d(2024, 10, 2),  # Mahatma Gandhi Jayanti, Wednesday
            d(2024, 11, 1),  # Diwali Laxmi Pujan, Friday
            d(2024, 11, 15), # Gurunanak Jayanti, Friday
            d(2024, 12, 25), # Christmas, Wednesday

            # 2025 Holidays (from _TRADING_HOLIDAYS_SET, ensuring they are weekdays)
            d(2025, 1, 26),  # Republic Day, Sunday - This is a weekend, skip for this test type
            # d(2025, 1, 26) should be tested by weekend test.
            # Let's pick actual weekday holidays for 2025 from the list if available
            # The provided _TRADING_HOLIDAYS_SET only goes up to 2024.
            # I will add placeholders if 2025 holidays are not in the current list.
            # For now, I will use the ones I can find for 2024.
            # If I had 2025 holidays:
            # d(2025, 3, 14), # Holi (Example, if it were a weekday & in list)
            # d(2025, 12, 25), # Christmas, Thursday (Example, if in list)
            # Since 2025 holidays are not in the provided list, I will limit to 2023 & 2024.
            # Adding a check for 2025 Christmas if it were in the list to show intent:
            # d(2025, 12, 25), # Christmas, Thursday
        ],
    )
    def test_holidays_are_not_trading_days(self, holiday_date):
        """Test that known holidays (that are weekdays) are not trading days."""
        # This assertion confirms the test data is correctly set up
        assert holiday_date in _TRADING_HOLIDAYS_SET, f"Date {holiday_date} is not in _TRADING_HOLIDAYS_SET. Test data error."
        assert holiday_date.weekday() < 5, f"Holiday {holiday_date} is not a weekday. Test data error."
        assert is_trading_day(holiday_date) is False

    # Test for 2025 Christmas explicitly if it were in the list
    # This test will currently fail if 2025-12-25 is not in _TRADING_HOLIDAYS_SET
    # or demonstrate how it would be tested.
    # For now, I will assume it's not in the live set and skip or xfail it.
    @pytest.mark.skipif(d(2025, 12, 25) not in _TRADING_HOLIDAYS_SET, reason="2025-12-25 not in holiday list")
    def test_specific_future_holiday_example(self):
        holiday_2025_christmas = d(2025, 12, 25) # Thursday
        assert holiday_2025_christmas.weekday() < 5
        assert holiday_2025_christmas in _TRADING_HOLIDAYS_SET
        assert is_trading_day(holiday_2025_christmas) is False


    # 4. Edge Cases
    @pytest.mark.parametrize(
        "date_around_holiday, expected_is_trading_day",
        [
            # Around 2024-01-26 (Republic Day, Friday)
            (d(2024, 1, 25), True),  # Thursday before holiday (Trading day)
            (d(2024, 1, 29), True),  # Monday after holiday (Trading day, 27/28 are Sat/Sun)

            # Around 2024-03-25 (Holi, Monday)
            (d(2024, 3, 22), True),  # Friday before holiday (Trading day)
            (d(2024, 3, 26), True),  # Tuesday after holiday (Trading day)

            # Around 2023-12-25 (Christmas, Monday)
            (d(2023, 12, 22), True),  # Friday before holiday (Trading day)
            (d(2023, 12, 26), True),  # Tuesday after holiday (Trading day)

            # A day that is a holiday itself (should be False if it's a weekday holiday)
            (d(2024, 11, 1), False), # Diwali Laxmi Pujan, Friday (Holiday)
        ],
    )
    def test_edge_cases_around_holidays(self, date_around_holiday, expected_is_trading_day):
        """Test days immediately before or after a holiday."""
        # Ensure the holiday itself is correctly identified if it's part of the test data
        if date_around_holiday in _TRADING_HOLIDAYS_SET and date_around_holiday.weekday() < 5:
            assert is_trading_day(date_around_holiday) is False, f"Holiday {date_around_holiday} should not be a trading day."
        elif date_around_holiday.weekday() >=5: # Weekend
             assert is_trading_day(date_around_holiday) is False, f"Weekend {date_around_holiday} should not be a trading day."
        else: # Weekday, not a holiday
            assert is_trading_day(date_around_holiday) is expected_is_trading_day

        # More direct assertion for the parametrized test's intent
        assert is_trading_day(date_around_holiday) is expected_is_trading_day


    def test_holidays_from_source_set_are_not_trading_days(self):
        """Iterate over all holidays in _TRADING_HOLIDAYS_SET.
        If a holiday falls on a weekday, it should not be a trading day.
        If it falls on a weekend, it's already not a trading day (covered by weekend tests).
        """
        for holiday_date in _TRADING_HOLIDAYS_SET:
            if holiday_date.weekday() < 5: # It's a weekday holiday
                assert is_trading_day(holiday_date) is False, f"Weekday holiday {holiday_date} reported as trading day."
            else: # It's a weekend holiday
                assert is_trading_day(holiday_date) is False, f"Weekend holiday {holiday_date} reported as trading day (should be False due to weekend)."

```
