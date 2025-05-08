"""
Mathematical utilities for options pricing and financial calculations.

This module provides functions for:
- Black-Scholes option pricing model
- Implied volatility calculations
- Greeks calculations (delta, gamma, theta, vega, rho)
- Other financial mathematics utilities
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import math
from typing import Optional, Tuple, Dict, Union


def bs_option_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0
) -> float:
    """
    Calculate option price using the Black-Scholes model.
    
    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Annual risk-free interest rate (as a decimal)
        volatility: Annual volatility of the underlying asset (as a decimal)
        option_type: Type of option ("call" or "put")
        dividend_yield: Continuous dividend yield (as a decimal)
        
    Returns:
        Theoretical option price according to Black-Scholes model
    """
    # Handle edge cases
    if time_to_expiry <= 0:
        if option_type.lower() == "call":
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)
    
    # Convert to lowercase for case-insensitive comparison
    option_type = option_type.lower()
    
    # Calculate d1 and d2
    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    # Calculate option price
    if option_type == "call":
        option_price = spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    elif option_type == "put":
        option_price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return option_price


def calculate_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    precision: float = 0.00001,
    max_iterations: int = 100
) -> Optional[float]:
    """
    Calculate implied volatility using a numerical method (Brent's method).
    
    Args:
        option_price: Market price of the option
        spot: Current price of the underlying asset
        strike: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Annual risk-free interest rate (as a decimal)
        option_type: Type of option ("call" or "put")
        dividend_yield: Continuous dividend yield (as a decimal)
        precision: Desired precision for the implied volatility
        max_iterations: Maximum number of iterations for the numerical method
        
    Returns:
        Implied volatility as a decimal, or None if calculation fails
    """
    # Handle edge cases
    if option_price <= 0 or time_to_expiry <= 0:
        return None
    
    # Define objective function for root finding
    def objective(volatility):
        theoretical_price = bs_option_price(
            spot, strike, time_to_expiry, risk_free_rate, 
            volatility, option_type, dividend_yield
        )
        return theoretical_price - option_price
    
    try:
        # Use Brent's method for root finding with reasonable bounds for volatility
        implied_vol = brentq(
            objective,
            0.0001,  # Lower bound (0.01%)
            5.0,     # Upper bound (500%)
            xtol=precision,
            maxiter=max_iterations
        )
        return implied_vol
    except (ValueError, RuntimeError):
        # If the solution is not in the specified interval or another error occurs
        return None


def calculate_option_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0
) -> Dict[str, float]:
    """
    Calculate option Greeks using the Black-Scholes model.
    
    Args:
        spot: Current price of the underlying asset
        strike: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Annual risk-free interest rate (as a decimal)
        volatility: Annual volatility of the underlying asset (as a decimal)
        option_type: Type of option ("call" or "put")
        dividend_yield: Continuous dividend yield (as a decimal)
        
    Returns:
        Dictionary containing delta, gamma, theta, vega, and rho
    """
    # Handle edge cases
    if time_to_expiry <= 0 or volatility <= 0:
        return {
            "delta": 1.0 if option_type.lower() == "call" and spot > strike else 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }
    
    # Calculate d1 and d2
    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    # Common calculations
    sqrt_time = np.sqrt(time_to_expiry)
    pdf_d1 = norm.pdf(d1)
    
    # Calculate delta
    if option_type.lower() == "call":
        delta = np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
    else:
        delta = np.exp(-dividend_yield * time_to_expiry) * (norm.cdf(d1) - 1)
    
    # Calculate gamma (same for calls and puts)
    gamma = np.exp(-dividend_yield * time_to_expiry) * pdf_d1 / (spot * volatility * sqrt_time)
    
    # Calculate theta
    term1 = -spot * np.exp(-dividend_yield * time_to_expiry) * pdf_d1 * volatility / (2 * sqrt_time)
    
    if option_type.lower() == "call":
        term2 = -risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        term3 = dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
    else:
        term2 = risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        term3 = -dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
    
    # Theta is expressed in value per year, divide by 365 to get daily theta
    theta = (term1 + term2 + term3) / 365
    
    # Calculate vega (same for calls and puts)
    # Vega is typically expressed as change per 1% volatility change (0.01)
    vega = spot * np.exp(-dividend_yield * time_to_expiry) * pdf_d1 * sqrt_time * 0.01
    
    # Calculate rho (sensitivity to interest rate change of 1%)
    if option_type.lower() == "call":
        rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) * 0.01
    else:
        rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) * 0.01
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }


def calculate_days_between_dates(start_date, end_date) -> int:
    """
    Calculate the number of days between two dates.
    
    Args:
        start_date: Start date as datetime object
        end_date: End date as datetime object
        
    Returns:
        Number of days between the dates
    """
    return (end_date - start_date).days


def calculate_annualized_return(initial_value: float, final_value: float, days: int) -> float:
    """
    Calculate annualized return given initial and final values.
    
    Args:
        initial_value: Initial investment value
        final_value: Final investment value
        days: Number of days for the investment period
        
    Returns:
        Annualized return as a decimal
    """
    if initial_value <= 0 or days <= 0:
        return 0.0
        
    total_return = final_value / initial_value - 1
    years = days / 365
    
    if years < 0.003:  # Avoid very short periods that can lead to extreme annualized returns
        return total_return
        
    annualized_return = (1 + total_return) ** (1 / years) - 1
    return annualized_return


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: Array of period returns (daily, weekly, etc.)
        risk_free_rate: Risk-free rate for the same period (as a decimal)
        annualization_factor: Factor to annualize returns (252 for daily, 52 for weekly, 12 for monthly)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) <= 1:
        return 0.0
        
    excess_returns = returns - risk_free_rate / annualization_factor
    mean_excess_return = np.mean(excess_returns)
    std_dev = np.std(excess_returns, ddof=1)  # Use sample standard deviation
    
    if std_dev == 0:
        return 0.0
        
    sharpe = mean_excess_return / std_dev * np.sqrt(annualization_factor)
    return sharpe


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sortino ratio for a series of returns.
    
    Args:
        returns: Array of period returns (daily, weekly, etc.)
        risk_free_rate: Risk-free rate for the same period (as a decimal)
        annualization_factor: Factor to annualize returns (252 for daily, 52 for weekly, 12 for monthly)
        
    Returns:
        Sortino ratio
    """
    if len(returns) <= 1:
        return 0.0
        
    excess_returns = returns - risk_free_rate / annualization_factor
    mean_excess_return = np.mean(excess_returns)
    
    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return float('inf')  # No negative returns
        
    downside_deviation = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(annualization_factor)
    
    if downside_deviation == 0:
        return 0.0
        
    sortino = mean_excess_return * annualization_factor / downside_deviation
    return sortino


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration from an equity curve.
    
    Args:
        equity_curve: Array of equity values over time
        
    Returns:
        Tuple containing (max_drawdown_pct, start_idx, end_idx)
    """
    # Handle empty or single-point equity curves
    if len(equity_curve) <= 1:
        return 0.0, 0, 0
        
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in percentage terms
    drawdown = (equity_curve - running_max) / running_max
    
    # Find the maximum drawdown
    max_drawdown = np.min(drawdown)
    
    # Find the index of the maximum drawdown
    end_idx = np.argmin(drawdown)
    
    # Find the index of the peak before the maximum drawdown
    peak_idx = np.argmax(equity_curve[:end_idx+1])
    
    return max_drawdown, peak_idx, end_idx


def simple_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate simple moving average of a data series.
    
    Args:
        data: Data series as a numpy array
        window: Window size for the moving average
        
    Returns:
        Array of simple moving averages
    """
    if len(data) < window:
        # Return array of NaNs if data is shorter than window
        return np.full(len(data), np.nan)
        
    return np.convolve(data, np.ones(window) / window, mode='valid')


def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """
    Calculate exponential moving average of a data series.
    
    Args:
        data: Data series as a numpy array
        span: Span for the EMA (approximately equivalent to window size in SMA)
        
    Returns:
        Array of exponential moving averages
    """
    if len(data) <= 1:
        return data.copy()
        
    # Calculate alpha from span (alpha = 2 / (span + 1))
    alpha = 2 / (span + 1)
    
    # Initialize EMA with the first value
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    # Calculate EMA
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
    return ema


def calculate_rsi(data: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI) for a data series.
    
    Args:
        data: Data series as a numpy array
        window: Window size for RSI calculation
        
    Returns:
        Array of RSI values
    """
    if len(data) <= window:
        return np.full(len(data), np.nan)
        
    # Calculate price changes
    deltas = np.diff(data)
    
    # Initialize arrays for gains and losses
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    
    # Separate gains and losses
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]
    
    # Calculate average gains and losses
    avg_gain = np.zeros_like(data)
    avg_loss = np.zeros_like(data)
    
    # First average
    avg_gain[window] = np.mean(gains[:window])
    avg_loss[window] = np.mean(losses[:window])
    
    # Calculate subsequent averages
    for i in range(window + 1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (window - 1) + gains[i-1]) / window
        avg_loss[i] = (avg_loss[i-1] * (window - 1) + losses[i-1]) / window
    
    # Calculate RS and RSI
    rs = avg_gain[window:] / np.where(avg_loss[window:] == 0, 0.0001, avg_loss[window:])
    rsi = 100 - (100 / (1 + rs))
    
    # Pad the beginning of the output array with NaNs
    result = np.full(len(data), np.nan)
    result[window:] = rsi
    
    return result
