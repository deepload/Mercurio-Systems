"""
Options backtesting service for Mercurio AI platform.

This module provides functionality to backtest options strategies using
historical market data and simulated options chains.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Type

from app.core.models.option import OptionContract, OptionPosition, OptionType
from app.strategies.options.base_options_strategy import BaseOptionsStrategy
from app.services.market_data import MarketDataService
from app.utils.math_utils import calculate_implied_volatility, bs_option_price
from app.utils.common import format_currency

logger = logging.getLogger(__name__)

class OptionsBacktester:
    """
    Service for backtesting options strategies with historical data.
    
    This class provides functionality to test options strategies against
    historical data by simulating option chains and executing strategy
    signals based on historical price movements.
    """
    
    def __init__(
        self,
        market_data_service: MarketDataService,
        initial_capital: float = 100000.0,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.05,
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None,
        output_directory: str = './backtest_results'
    ):
        """
        Initialize the options backtester.
        
        Args:
            market_data_service: Service for retrieving market data
            initial_capital: Starting capital for the backtest
            commission_per_contract: Commission cost per options contract
            slippage_pct: Simulated slippage percentage for executions
            data_start_date: Start date for backtest data (YYYY-MM-DD)
            data_end_date: End date for backtest data (YYYY-MM-DD)
            output_directory: Directory to save backtest results
        """
        self.market_data_service = market_data_service
        self.initial_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.slippage_pct = slippage_pct
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.output_directory = output_directory
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Backtest state variables
        self.equity = initial_capital
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        self.current_date = None
        self.latest_data = {}
        self.simulated_options_chains = {}
        
    async def run_backtest(
        self,
        strategy_class: Type[BaseOptionsStrategy],
        symbols: List[str],
        strategy_params: Dict[str, Any],
        timeframe: str = '1d',
        report_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest for the specified options strategy.
        
        Args:
            strategy_class: The strategy class to backtest
            symbols: List of symbols to trade
            strategy_params: Parameters for the strategy
            timeframe: Data timeframe (1d, 1h, etc.)
            report_name: Optional custom name for the report
            
        Returns:
            Dict: Backtest results
        """
        start_time = datetime.now()
        logger.info(f"Starting options backtest for {strategy_class.__name__} on {len(symbols)} symbols")
        
        # Reset backtest state
        self.equity = self.initial_capital
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        self.simulated_options_chains = {}
        
        # Get historical data for all symbols
        data_by_symbol = {}
        for symbol in symbols:
            try:
                data = await self.market_data_service.get_historical_data(
                    symbol=symbol,
                    start_date=self.data_start_date,
                    end_date=self.data_end_date,
                    timeframe=timeframe
                )
                
                if data is not None and not data.empty:
                    data_by_symbol[symbol] = data
                    logger.info(f"Got {len(data)} data points for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {str(e)}")
        
        if not data_by_symbol:
            logger.error("No data available for any symbols")
            return {"success": False, "error": "No data available"}
        
        # Align dates across all symbols
        common_dates = self._align_dates(data_by_symbol)
        if not common_dates:
            logger.error("No common dates found across symbols")
            return {"success": False, "error": "No common dates found"}
        
        # Initialize strategies for each symbol
        strategies = {}
        for symbol in data_by_symbol.keys():
            # Create a mock broker adapter for the strategy
            mock_broker = MockOptionsBacktestBroker(self)
            
            # Initialize strategy with custom parameters and mock broker
            params = {**strategy_params, "underlying_symbol": symbol, "account_size": self.initial_capital}
            strategies[symbol] = strategy_class(**params)
            strategies[symbol].broker = mock_broker
        
        # Run the backtest day by day
        for date in common_dates:
            self.current_date = date
            self.equity_curve.append({"date": date, "equity": self.equity})
            
            # Update latest data for each symbol
            for symbol, data in data_by_symbol.items():
                if date in data.index:
                    self.latest_data[symbol] = data.loc[:date].copy()
                    
                    # Generate simulated options chain if it doesn't exist
                    if symbol not in self.simulated_options_chains or \
                       date.strftime("%Y-%m-%d") not in self.simulated_options_chains[symbol]:
                        self._generate_simulated_options_chain(symbol, date)
            
            # Process existing positions - check for exit conditions
            await self._process_existing_positions(strategies)
            
            # Check for entry conditions for each symbol
            await self._check_entry_conditions(strategies)
        
        # Close all open positions at the end of the backtest
        await self._close_all_positions(strategies)
        
        # Generate backtest report
        backtest_duration = (datetime.now() - start_time).total_seconds()
        report = self._generate_backtest_report(strategy_class.__name__, symbols, strategy_params, backtest_duration)
        
        # Save report
        if report_name is None:
            report_name = f"{strategy_class.__name__}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_path = os.path.join(self.output_directory, f"{report_name}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logger.info(f"Backtest completed in {backtest_duration:.2f} seconds. Results saved to {report_path}")
        return report
    
    async def _process_existing_positions(self, strategies: Dict[str, BaseOptionsStrategy]) -> None:
        """
        Process existing positions to check for exit signals.
        
        Args:
            strategies: Dictionary of strategy instances by symbol
        """
        positions_to_remove = []
        
        for position in self.positions:
            symbol = position['symbol']
            
            if symbol in strategies:
                strategy = strategies[symbol]
                
                # Get latest market data
                market_data = self.latest_data.get(symbol, pd.DataFrame())
                
                # Check exit condition
                if await strategy.should_exit(market_data):
                    # Execute exit
                    exit_result = await strategy.execute_exit()
                    
                    if exit_result.get('success', False):
                        # Record trade
                        position['exit_date'] = self.current_date
                        position['exit_price'] = exit_result.get('exit_premium', 0)
                        position['profit_loss'] = exit_result.get('profit_loss', 0)
                        position['profit_loss_pct'] = exit_result.get('profit_loss_pct', 0)
                        
                        # Update equity
                        self.equity += position.get('profit_loss', 0)
                        
                        # Add to trade history
                        self.trade_history.append(position.copy())
                        
                        # Mark for removal
                        positions_to_remove.append(position)
                        
                        logger.info(f"Exited position {symbol} on {self.current_date}, P&L: {position['profit_loss']:.2f}")
        
        # Remove closed positions
        for position in positions_to_remove:
            if position in self.positions:
                self.positions.remove(position)
    
    async def _check_entry_conditions(self, strategies: Dict[str, BaseOptionsStrategy]) -> None:
        """
        Check for entry conditions for each symbol.
        
        Args:
            strategies: Dictionary of strategy instances by symbol
        """
        for symbol, strategy in strategies.items():
            # Skip if we already have a position for this symbol
            if any(p['symbol'] == symbol for p in self.positions):
                continue
            
            # Get latest market data
            market_data = self.latest_data.get(symbol, pd.DataFrame())
            
            # Check entry condition
            if await strategy.should_enter(market_data):
                # Execute entry
                entry_result = await strategy.execute_entry()
                
                if entry_result.get('success', False):
                    # Create position record
                    position = {
                        'symbol': symbol,
                        'strategy': strategy.__class__.__name__,
                        'entry_date': self.current_date,
                        'entry_price': entry_result.get('premium', 0),
                        'quantity': entry_result.get('quantity', 0),
                        'option_data': entry_result
                    }
                    
                    # Update equity
                    position_cost = position['quantity'] * position['entry_price'] * 100
                    self.equity -= position_cost
                    
                    # Add to positions list
                    self.positions.append(position)
                    
                    logger.info(f"Entered position {symbol} on {self.current_date}, cost: {position_cost:.2f}")
    
    async def _close_all_positions(self, strategies: Dict[str, BaseOptionsStrategy]) -> None:
        """
        Close all open positions at the end of the backtest.
        
        Args:
            strategies: Dictionary of strategy instances by symbol
        """
        for position in self.positions[:]:
            symbol = position['symbol']
            
            if symbol in strategies:
                strategy = strategies[symbol]
                
                # Execute exit
                exit_result = await strategy.execute_exit()
                
                if exit_result.get('success', False):
                    # Record trade
                    position['exit_date'] = self.current_date
                    position['exit_price'] = exit_result.get('exit_premium', 0)
                    position['profit_loss'] = exit_result.get('profit_loss', 0)
                    position['profit_loss_pct'] = exit_result.get('profit_loss_pct', 0)
                    
                    # Update equity
                    self.equity += position.get('profit_loss', 0)
                    
                    # Add to trade history
                    self.trade_history.append(position.copy())
                    
                    # Remove from positions
                    self.positions.remove(position)
                    
                    logger.info(f"Closed position {symbol} at end of backtest, P&L: {position['profit_loss']:.2f}")
    
    def _align_dates(self, data_by_symbol: Dict[str, pd.DataFrame]) -> List[datetime]:
        """
        Align dates across all symbols to ensure we have data for all symbols on each date.
        
        Args:
            data_by_symbol: Dictionary of DataFrames by symbol
            
        Returns:
            List[datetime]: List of common dates
        """
        all_dates = set()
        common_dates = set()
        
        # First pass: collect all dates
        for symbol, data in data_by_symbol.items():
            dates = set(data.index)
            all_dates.update(dates)
        
        # Second pass: find common dates
        for date in all_dates:
            if all(date in data.index for data in data_by_symbol.values()):
                common_dates.add(date)
        
        # Sort chronologically
        return sorted(list(common_dates))
    
    def _generate_simulated_options_chain(self, symbol: str, date: datetime) -> None:
        """
        Generate a simulated options chain for a symbol on a specific date.
        
        Args:
            symbol: The underlying symbol
            date: The date to generate options for
        """
        if symbol not in self.latest_data:
            return
        
        data = self.latest_data[symbol]
        
        if date not in data.index:
            return
        
        # Get the current price
        current_price = data.loc[date, 'close']
        
        # Initialize options chain dictionary for this symbol if it doesn't exist
        if symbol not in self.simulated_options_chains:
            self.simulated_options_chains[symbol] = {}
        
        date_str = date.strftime("%Y-%m-%d")
        
        # Generate option chains for different expiration dates
        expirations = [
            date + timedelta(days=30),  # 1 month
            date + timedelta(days=60),  # 2 months
            date + timedelta(days=90)   # 3 months
        ]
        
        chains = []
        
        # Calculate historical volatility
        if len(data) > 30:
            returns = data['close'].pct_change().dropna()
            historical_volatility = returns.rolling(window=30).std() * np.sqrt(252)
            current_volatility = historical_volatility.iloc[-1] if not historical_volatility.empty else 0.20
        else:
            current_volatility = 0.20  # Default if not enough data
        
        # Generate strike prices around current price
        strikes = []
        strike_pct_range = 0.20  # 20% range around current price
        price_increment = max(1.0, round(current_price * 0.025, 0))  # 2.5% increments
        
        lower_strike = max(1, round(current_price * (1 - strike_pct_range) / price_increment) * price_increment)
        upper_strike = round(current_price * (1 + strike_pct_range) / price_increment) * price_increment
        
        strike = lower_strike
        while strike <= upper_strike:
            strikes.append(strike)
            strike += price_increment
        
        # Generate options for each expiration and strike
        for expiry in expirations:
            expiry_date = expiry.strftime("%Y-%m-%d")
            days_to_expiry = (expiry - date).days
            
            for strike in strikes:
                # Generate call option
                iv_call = current_volatility * (1 + 0.1 * (abs(strike - current_price) / current_price))
                
                call_price = bs_option_price(
                    S=current_price,
                    K=strike,
                    T=days_to_expiry / 365,
                    r=0.02,  # Assumed risk-free rate
                    sigma=iv_call,
                    option_type='call'
                )
                
                # Generate Greeks (simplified)
                call_delta = max(0, min(1, (1 if current_price > strike else 0.5)))
                call_gamma = 0.03
                call_theta = -call_price * 0.01  # Approximate theta decay
                call_vega = call_price * 0.1
                
                bid_ask_spread = max(0.05, call_price * 0.10)  # 10% spread with minimum
                
                call = OptionContract(
                    symbol=f"{symbol}_{expiry_date}_C{strike}",
                    underlying=symbol,
                    option_type=OptionType.CALL,
                    strike=strike,
                    expiry_date=expiry_date,
                    bid=max(0.01, call_price - bid_ask_spread/2),
                    ask=call_price + bid_ask_spread/2,
                    last=call_price,
                    volume=int(100 * (1 - abs(strike - current_price) / current_price)),
                    open_interest=int(500 * (1 - abs(strike - current_price) / current_price)),
                    implied_volatility=iv_call,
                    delta=call_delta,
                    gamma=call_gamma,
                    theta=call_theta,
                    vega=call_vega,
                    rho=0.01
                )
                
                chains.append(call)
                
                # Generate put option
                iv_put = current_volatility * (1 + 0.1 * (abs(strike - current_price) / current_price))
                
                put_price = bs_option_price(
                    S=current_price,
                    K=strike,
                    T=days_to_expiry / 365,
                    r=0.02,  # Assumed risk-free rate
                    sigma=iv_put,
                    option_type='put'
                )
                
                # Generate Greeks (simplified)
                put_delta = max(-1, min(0, (-1 if current_price < strike else -0.5)))
                put_gamma = 0.03
                put_theta = -put_price * 0.01  # Approximate theta decay
                put_vega = put_price * 0.1
                
                bid_ask_spread = max(0.05, put_price * 0.10)  # 10% spread with minimum
                
                put = OptionContract(
                    symbol=f"{symbol}_{expiry_date}_P{strike}",
                    underlying=symbol,
                    option_type=OptionType.PUT,
                    strike=strike,
                    expiry_date=expiry_date,
                    bid=max(0.01, put_price - bid_ask_spread/2),
                    ask=put_price + bid_ask_spread/2,
                    last=put_price,
                    volume=int(100 * (1 - abs(strike - current_price) / current_price)),
                    open_interest=int(500 * (1 - abs(strike - current_price) / current_price)),
                    implied_volatility=iv_put,
                    delta=put_delta,
                    gamma=put_gamma,
                    theta=put_theta,
                    vega=put_vega,
                    rho=-0.01
                )
                
                chains.append(put)
        
        # Store the generated options chain
        self.simulated_options_chains[symbol][date_str] = chains
    
    def _generate_backtest_report(
        self,
        strategy_name: str,
        symbols: List[str],
        strategy_params: Dict[str, Any],
        duration_seconds: float
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive backtest report.
        
        Args:
            strategy_name: Name of the strategy
            symbols: List of symbols traded
            strategy_params: Strategy parameters
            duration_seconds: Duration of the backtest in seconds
            
        Returns:
            Dict: Backtest results
        """
        # Calculate performance metrics
        total_trades = len(self.trade_history)
        profitable_trades = sum(1 for t in self.trade_history if t.get('profit_loss', 0) > 0)
        
        if total_trades > 0:
            win_rate = profitable_trades / total_trades
        else:
            win_rate = 0
        
        total_profit = sum(t.get('profit_loss', 0) for t in self.trade_history)
        total_return_pct = (total_profit / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # Calculate max drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        running_max = 0
        drawdowns = []
        
        for equity in equity_values:
            if equity > running_max:
                running_max = equity
            
            if running_max > 0:
                drawdown_pct = (running_max - equity) / running_max * 100
                drawdowns.append(drawdown_pct)
        
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Create report
        report = {
            "strategy": strategy_name,
            "symbols": symbols,
            "parameters": strategy_params,
            "start_date": self.equity_curve[0]['date'] if self.equity_curve else None,
            "end_date": self.equity_curve[-1]['date'] if self.equity_curve else None,
            "initial_capital": self.initial_capital,
            "final_equity": self.equity,
            "total_return": total_profit,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": total_trades - profitable_trades,
            "win_rate": win_rate,
            "execution_time_seconds": duration_seconds,
            "trades": self.trade_history,
            "equity_curve": self.equity_curve
        }
        
        return report


class MockOptionsBacktestBroker:
    """
    Mock broker adapter for options backtesting.
    
    This class simulates broker operations during backtesting by interfacing
    with the OptionsBacktester to retrieve simulated options data.
    """
    
    def __init__(self, backtester: OptionsBacktester):
        """
        Initialize the mock broker.
        
        Args:
            backtester: The backtester instance that contains simulated data
        """
        self.backtester = backtester
        self.enable_options = True
        
    async def get_account(self) -> Dict[str, Any]:
        """
        Get mock account information.
        
        Returns:
            Dict: Account information with current equity
        """
        return {"buying_power": self.backtester.equity}
    
    async def get_option_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        option_type: Optional[OptionType] = None
    ) -> List[OptionContract]:
        """
        Get simulated options chain data.
        
        Args:
            symbol: Underlying symbol
            expiration_date: Optional specific expiration date
            option_type: Optional filter for option type
            
        Returns:
            List[OptionContract]: List of option contracts
        """
        if symbol not in self.backtester.simulated_options_chains:
            return []
        
        date_str = self.backtester.current_date.strftime("%Y-%m-%d")
        
        # Use the latest available options chain if current date doesn't have one
        available_dates = sorted(list(self.backtester.simulated_options_chains[symbol].keys()))
        
        if not available_dates:
            return []
        
        if date_str not in available_dates:
            date_str = max(d for d in available_dates if d <= date_str) if available_dates else None
            
        if not date_str:
            return []
            
        options_chain = self.backtester.simulated_options_chains[symbol][date_str]
        
        # Filter by expiration date if provided
        if expiration_date:
            options_chain = [o for o in options_chain if o.expiry_date == expiration_date]
            
        # Filter by option type if provided
        if option_type:
            options_chain = [o for o in options_chain if o.option_type == option_type]
            
        return options_chain
    
    async def place_option_order(
        self,
        option_symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simulate placing an option order.
        
        Args:
            option_symbol: Option contract symbol
            qty: Quantity of contracts
            side: Order side ("buy" or "sell")
            order_type: Order type
            time_in_force: Time in force
            limit_price: Limit price if applicable
            
        Returns:
            Dict: Order execution result
        """
        for symbol, chains in self.backtester.simulated_options_chains.items():
            for date_str, options in chains.items():
                for option in options:
                    if option.symbol == option_symbol:
                        # Apply slippage to price
                        price = option.ask if side == "buy" else option.bid
                        
                        if order_type == "limit":
                            # For limit orders, use the specified price if it's favorable
                            if side == "buy" and limit_price < option.ask:
                                price = limit_price
                            elif side == "sell" and limit_price > option.bid:
                                price = limit_price
                        
                        # Apply slippage
                        price = price * (1 + self.backtester.slippage_pct) if side == "buy" else price * (1 - self.backtester.slippage_pct)
                        
                        # Apply commission
                        commission = self.backtester.commission_per_contract * qty
                        
                        return {
                            "success": True,
                            "order_id": f"backtest_{option_symbol}_{side}_{self.backtester.current_date.strftime('%Y%m%d')}",
                            "symbol": option_symbol,
                            "side": side,
                            "qty": qty,
                            "price": price,
                            "commission": commission,
                            "status": "filled"
                        }
        
        return {"success": False, "error": f"Option {option_symbol} not found in simulated data"}
    
    async def place_option_strategy(
        self,
        strategy_type: str,
        underlying: str,
        legs: List[Dict[str, Any]],
        qty: int
    ) -> Dict[str, Any]:
        """
        Simulate placing a multi-leg options strategy order.
        
        Args:
            strategy_type: Type of strategy (e.g., "iron_condor")
            underlying: Underlying symbol
            legs: List of strategy legs
            qty: Quantity of strategies
            
        Returns:
            Dict: Order execution result
        """
        results = []
        
        # Execute each leg separately
        for leg in legs:
            result = await self.place_option_order(
                option_symbol=leg["symbol"],
                qty=leg["quantity"] if "quantity" in leg else qty,
                side=leg["side"],
                order_type="market"
            )
            
            results.append(result)
            
        # Check if all legs were executed successfully
        success = all(r.get("success", False) for r in results)
        
        if success:
            return {
                "success": True,
                "order_id": f"backtest_{strategy_type}_{underlying}_{self.backtester.current_date.strftime('%Y%m%d')}",
                "legs": results,
                "status": "filled"
            }
        else:
            # Find the first error
            error = next((r.get("error") for r in results if not r.get("success", False)), "Unknown error")
            return {"success": False, "error": error}
    
    async def get_option_positions(self) -> List[Dict[str, Any]]:
        """
        Get current option positions.
        
        Returns:
            List[Dict]: List of current option positions
        """
        positions = []
        
        for position in self.backtester.positions:
            option_data = position.get('option_data', {})
            
            if 'symbol' in option_data:
                positions.append({
                    "symbol": option_data.get('symbol'),
                    "option_type": option_data.get('option_type', 'call'),
                    "strike_price": option_data.get('strike', 0),
                    "expiration_date": option_data.get('expiry', ''),
                    "qty": position.get('quantity', 0),
                    "avg_entry_price": position.get('entry_price', 0),
                    "market_value": 0,  # Would be updated in real implementation
                    "cost_basis": position.get('entry_price', 0) * position.get('quantity', 0) * 100,
                    "unrealized_pl": 0  # Would be updated in real implementation
                })
        
        return positions
