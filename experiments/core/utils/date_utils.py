"""
Unified date utility functions for epidemic forecasting.

This module consolidates date handling functionality from:
- rolling_agent_forecast.py
- offline_dataset_builder.py
- web_sources.py
"""

from datetime import datetime, timedelta
from typing import List


def dt(date_str: str) -> datetime:
    """Convert date string to datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def iso(d: datetime) -> str:
    """Convert datetime to ISO format string."""
    return d.strftime("%Y-%m-%d")


def weekly_dates_between(dates: List[str], start: str, end: str) -> List[str]:
    """Get weekly dates between start and end dates."""
    start_dt = dt(start)
    end_dt = dt(end)
    
    weekly_dates = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        weekly_dates.append(iso(current_dt))
        current_dt += timedelta(weeks=1)
    
    return weekly_dates


def month_to_weight(ymd: str) -> float:
    """Convert month to seasonal weight for influenza."""
    try:
        month = int(ymd.split('-')[1])
        if month in [12, 1, 2]:  # Winter
            return 1.5
        elif month in [3, 4, 5]:  # Spring
            return 0.8
        elif month in [6, 7, 8]:  # Summer
            return 0.3
        else:  # Fall
            return 1.2
    except:
        return 1.0


def month_weeks(mon_str: str) -> List[int]:
    """Get week numbers for a given month string (e.g., '2024-01')."""
    try:
        year, month = map(int, mon_str.split('-'))
        start_date = datetime(year, month, 1)
        
        # Find the first Monday of the month
        while start_date.weekday() != 0:  # Monday = 0
            start_date += timedelta(days=1)
        
        weeks = []
        current_date = start_date
        
        while current_date.month == month:
            week_num = current_date.isocalendar()[1]
            if week_num not in weeks:
                weeks.append(week_num)
            current_date += timedelta(weeks=1)
        
        return weeks
    except:
        return []


def week_of(date_str: str) -> int:
    """Get week number for a given date string."""
    try:
        date_obj = dt(date_str)
        return date_obj.isocalendar()[1]
    except:
        return 0
