"""
Indian Market Holiday Calendar

This module provides a function to determine if a given date is a trading day
on the NSE/BSE. It accounts for weekends and a manually maintained list of
public holidays.
"""
from datetime import date, datetime

# Using a set for holidays provides a very fast O(1) lookup time.
# Format: YYYY-MM-DD
# Source: Official NSE Holiday List
_TRADING_HOLIDAYS_SET = {
    # --- 2023 Holidays ---
    "2023-01-26",  # Republic Day
    "2023-03-07",  # Holi
    "2023-03-30",  # Ram Navami
    "2023-04-04",  # Mahavir Jayanti
    "2023-04-07",  # Good Friday
    "2023-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2023-05-01",  # Maharashtra Day
    "2023-06-29",  # Bakri Id
    "2023-08-15",  # Independence Day
    "2023-09-19",  # Ganesh Chaturthi
    "2023-10-02",  # Mahatma Gandhi Jayanti
    "2023-10-24",  # Dussehra
    "2023-11-14",  # Diwali Balipratipada
    "2023-11-27",  # Guru Nanak Jayanti
    "2023-12-25",  # Christmas

    # --- 2024 Holidays ---
    "2024-01-26",  # Republic Day
    "2024-03-08",  # Mahashivratri
    "2024-03-25",  # Holi
    "2024-03-29",  # Good Friday
    "2024-04-11",  # Id-Ul-Fitr (Ramadan Eid)
    "2024-04-17",  # Ram Navami
    "2024-05-01",  # Maharashtra Day
    "2024-06-17",  # Bakri Id
    "2024-07-17",  # Moharram
    "2024-08-15",  # Independence Day/Parsi New Year
    "2024-10-02",  # Mahatma Gandhi Jayanti
    "2024-11-01",  # Diwali-Laxmi Pujan
    "2024-11-15",  # Gurunanak Jayanti
    "2024-12-25",  # Christmas

    # --- 2025 Holidays ---
    # Note: Republic Day (2025-01-26) is on a Sunday, so not included here
    "2025-02-26",  # Mahashivratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr (Ramzan Id)
    "2025-04-10",  # Shri Mahavir Jayanti
    "2025-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Mahatma Gandhi Jayanti / Dussehra
    "2025-10-21",  # Diwali-Laxmi Pujan
    "2025-10-22",  # Diwali Balipratipada
    "2025-11-05",  # Prakash Gurpurb Sri Guru Nanak Dev Jayanti
    "2025-12-25",  # Christmas
}

def is_trading_day(day: date) -> bool:
    """
    Checks if a given date is a trading day in the Indian market.

    Args:
        day (datetime.date): The date to check.

    Returns:
        bool: True if the date is a trading day, False otherwise.
    """
    # 1. Check for weekends: Monday is 0, Sunday is 6
    if day.weekday() >= 5:
        return False

    # 2. Check against the holiday set
    if day.strftime("%Y-%m-%d") in _TRADING_HOLIDAYS_SET:
        return False

    # 3. If it's not a weekend and not a holiday, it's a trading day
    return True

# --- Self-Test Block ---
if __name__ == '__main__':
    print("--- Testing Market Holiday Calendar ---")

    # A known trading day
    test_date_1 = date(2024, 6, 14)  # A Friday
    print(f"Is {test_date_1} a trading day? ... {is_trading_day(test_date_1)}")

    # A known weekend
    test_date_2 = date(2024, 6, 15)  # A Saturday
    print(f"Is {test_date_2} a trading day? ... {is_trading_day(test_date_2)}")

    # A known holiday
    test_date_3 = date(2024, 8, 15)  # Independence Day
    print(f"Is {test_date_3} a trading day? ... {is_trading_day(test_date_3)}")

    # Another known holiday
    test_date_4 = date(2023, 11, 14)  # Diwali Balipratipada
    print(f"Is {test_date_4} a trading day? ... {is_trading_day(test_date_4)}")

    # A known holiday in 2025
    test_date_5 = date(2025, 10, 21)  # Diwali-Laxmi Pujan, Tuesday
    print(f"Is {test_date_5} a trading day? ... {is_trading_day(test_date_5)}")

    # A known trading day in 2025
    test_date_6 = date(2025, 6, 16)  # A Monday
    print(f"Is {test_date_6} a trading day? ... {is_trading_day(test_date_6)}")

    print("\n--- Test Complete ---")