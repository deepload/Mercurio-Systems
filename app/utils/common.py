"""
Common utility functions for the Mercurio AI platform.

This module provides shared utility functions used across
different parts of the application.
"""

import re
import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple


def format_currency(amount: float, include_cents: bool = True, currency_symbol: str = "$") -> str:
    """
    Format a number as currency.
    
    Args:
        amount: The amount to format
        include_cents: Whether to include cents in the formatted string
        currency_symbol: The currency symbol to use
        
    Returns:
        Formatted currency string
    """
    if include_cents:
        return f"{currency_symbol}{amount:,.2f}"
    else:
        return f"{currency_symbol}{int(amount):,}"


def format_percentage(value: float, include_sign: bool = True, decimal_places: int = 2) -> str:
    """
    Format a decimal as a percentage.
    
    Args:
        value: The decimal value to format as percentage
        include_sign: Whether to include the plus sign for positive values
        decimal_places: Number of decimal places to include
        
    Returns:
        Formatted percentage string
    """
    if include_sign and value > 0:
        return f"+{value:.{decimal_places}f}%"
    else:
        return f"{value:.{decimal_places}f}%"


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with an optional prefix.
    
    Args:
        prefix: Optional prefix to prepend to the ID
        
    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def get_date_range(start_date: Union[str, datetime], 
                  end_date: Union[str, datetime], 
                  as_str: bool = False,
                  date_format: str = "%Y-%m-%d") -> List[Union[datetime, str]]:
    """
    Get a list of dates between start_date and end_date (inclusive).
    
    Args:
        start_date: Start date
        end_date: End date
        as_str: Whether to return dates as strings
        date_format: Format for date strings if as_str is True
        
    Returns:
        List of dates between start_date and end_date
    """
    # Convert string dates to datetime if necessary
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, date_format)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, date_format)
    
    # Calculate date range
    delta = end_date - start_date
    dates = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    
    # Convert to strings if requested
    if as_str:
        return [date.strftime(date_format) for date in dates]
    return dates


def is_market_open(dt: datetime = None) -> bool:
    """
    Check if the US stock market is open at the given datetime.
    This is a simplified check that doesn't account for holidays.
    
    Args:
        dt: Datetime to check (defaults to current time)
        
    Returns:
        True if market is open, False otherwise
    """
    if dt is None:
        dt = datetime.now()
    
    # Check if it's a weekday (0 = Monday, 6 = Sunday)
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if time is between 9:30 AM and 4:00 PM Eastern Time
    # This is simplified and doesn't handle timezone conversion
    market_open_hour, market_open_minute = 9, 30
    market_close_hour, market_close_minute = 16, 0
    
    current_time = dt.time()
    market_open = datetime.time(market_open_hour, market_open_minute)
    market_close = datetime.time(market_close_hour, market_close_minute)
    
    return market_open <= current_time <= market_close


def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    Parse a timeframe string into value and unit.
    
    Args:
        timeframe: String representation of timeframe (e.g., "1d", "4h", "30m")
        
    Returns:
        Tuple of (value, unit)
    """
    pattern = r"(\d+)([a-zA-Z]+)"
    match = re.match(pattern, timeframe)
    
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    value = int(match.group(1))
    unit = match.group(2).lower()
    
    valid_units = ["s", "m", "h", "d", "w", "mo", "y"]
    if unit not in valid_units:
        raise ValueError(f"Invalid timeframe unit: {unit}")
    
    return value, unit


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert a timeframe string to seconds.
    
    Args:
        timeframe: String representation of timeframe (e.g., "1d", "4h", "30m")
        
    Returns:
        Number of seconds in the timeframe
    """
    value, unit = parse_timeframe(timeframe)
    
    # Convert to seconds
    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600
    elif unit == "d":
        return value * 86400
    elif unit == "w":
        return value * 604800
    elif unit == "mo":  # Approximation
        return value * 2592000
    elif unit == "y":  # Approximation
        return value * 31536000
    
    raise ValueError(f"Unhandled timeframe unit: {unit}")


def load_json_file(file_path: str, default: Any = None) -> Any:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if file doesn't exist or has invalid JSON
        
    Returns:
        Loaded JSON data or default value
    """
    if not os.path.exists(file_path):
        return default
    
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading JSON file {file_path}: {str(e)}")
        return default


def save_json_file(data: Any, file_path: str, indent: int = 4) -> bool:
    """
    Save data as JSON to a file.
    
    Args:
        data: Data to save
        file_path: Path where to save the JSON file
        indent: Indentation for pretty printing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=indent)
        return True
    except IOError as e:
        logging.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False
