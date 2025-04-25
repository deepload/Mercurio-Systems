"""
MercurioAI Portfolio Optimizer

This module provides portfolio optimization capabilities using
modern portfolio theory and alternative approaches.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization using various methods"""
    
    def __init__(self, method: str = 'mean_variance'):
        """
        Initialize portfolio optimizer
        
        Args:
            method: Optimization method ('mean_variance', 'equal_weight', 
                   'risk_parity', 'min_variance', 'max_sharpe')
        """
        self.method = method
        self.returns_data = None
        self.symbols = []
        self.weights = None
        self.risk_free_rate = 0.0  # Annualized risk-free rate
        
    def set_returns_data(self, returns_data: pd.DataFrame, annualized: bool = False):
        """
        Set returns data for optimization
        
        Args:
            returns_data: DataFrame of asset returns (columns are assets)
            annualized: Whether returns are already annualized
        """
        self.returns_data = returns_data
        self.symbols = list(returns_data.columns)
        
        # Convert to annualized if needed (assuming daily returns)
        if not annualized:
            # Number of trading days in a year (approximately)
            trading_days = 252
            self.annual_returns = returns_data.mean() * trading_days
            self.annual_cov = returns_data.cov() * trading_days
        else:
            self.annual_returns = returns_data.mean()
            self.annual_cov = returns_data.cov()
        
        # Default to equal weights
        self.weights = np.ones(len(self.symbols)) / len(self.symbols)
    
    def set_risk_free_rate(self, rate: float):
        """
        Set risk-free rate for optimization
        
        Args:
            rate: Annualized risk-free rate (e.g., 0.02 for 2%)
        """
        self.risk_free_rate = rate
    
    def optimize(self, 
                method: Optional[str] = None, 
                target_return: Optional[float] = None,
                target_risk: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights
        
        Args:
            method: Optimization method (overrides instance method)
            target_return: Target portfolio return (only for mean_variance)
            target_risk: Target portfolio risk (only for mean_variance)
            
        Returns:
            Dictionary of optimization results
        """
        if self.returns_data is None or len(self.returns_data) == 0:
            logger.error("Returns data not set")
            return {'error': 'Returns data not set'}
            
        # Use instance method if not specified
        if method is None:
            method = self.method
            
        # Run optimization
        if method == 'equal_weight':
            self._equal_weight()
        elif method == 'mean_variance':
            self._mean_variance(target_return, target_risk)
        elif method == 'min_variance':
            self._min_variance()
        elif method == 'max_sharpe':
            self._max_sharpe()
        elif method == 'risk_parity':
            self._risk_parity()
        else:
            logger.error(f"Unknown optimization method: {method}")
            return {'error': f"Unknown optimization method: {method}"}
            
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics()
        
        # Return results
        results = {
            'weights': dict(zip(self.symbols, self.weights.tolist())),
            'method': method,
            'metrics': metrics
        }
        
        return results
    
    def _equal_weight(self):
        """Equal weight allocation"""
        self.weights = np.ones(len(self.symbols)) / len(self.symbols)
    
    def _mean_variance(self, target_return: Optional[float] = None, target_risk: Optional[float] = None):
        """
        Mean-variance optimization
        
        Args:
            target_return: Target portfolio return (if None, maximize return)
            target_risk: Target portfolio volatility (if None, minimize risk)
        """
        n = len(self.symbols)
        
        if target_return is not None:
            # Minimize risk subject to target return
            def objective(weights):
                return self._portfolio_volatility(weights)
                
            def return_constraint(weights):
                return self._portfolio_return(weights) - target_return
                
            constraints = [
                {'type': 'eq', 'fun': return_constraint},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
        elif target_risk is not None:
            # Maximize return subject to target risk
            def objective(weights):
                return -self._portfolio_return(weights)
                
            def risk_constraint(weights):
                return self._portfolio_volatility(weights) - target_risk
                
            constraints = [
                {'type': 'eq', 'fun': risk_constraint},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
        else:
            # Maximize Sharpe ratio
            return self._max_sharpe()
        
        # All weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess
        initial_weights = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            self.weights = result['x']
        else:
            logger.warning(f"Optimization failed: {result['message']}")
            # Fall back to equal weights
            self.weights = np.ones(n) / n
    
    def _min_variance(self):
        """Minimum variance portfolio optimization"""
        n = len(self.symbols)
        
        # Objective: minimize portfolio variance
        def objective(weights):
            return self._portfolio_volatility(weights) ** 2
            
        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # All weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess
        initial_weights = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            self.weights = result['x']
        else:
            logger.warning(f"Optimization failed: {result['message']}")
            # Fall back to equal weights
            self.weights = np.ones(n) / n
    
    def _max_sharpe(self):
        """Maximum Sharpe ratio portfolio optimization"""
        n = len(self.symbols)
        
        # Objective: maximize Sharpe ratio (negative of Sharpe ratio to minimize)
        def objective(weights):
            portfolio_return = self._portfolio_return(weights)
            portfolio_volatility = self._portfolio_volatility(weights)
            
            # Avoid division by zero
            if portfolio_volatility == 0:
                return -999
                
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
            
        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # All weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess
        initial_weights = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            self.weights = result['x']
        else:
            logger.warning(f"Optimization failed: {result['message']}")
            # Fall back to equal weights
            self.weights = np.ones(n) / n
    
    def _risk_parity(self):
        """Risk parity portfolio optimization"""
        n = len(self.symbols)
        
        # Objective: minimize variance of risk contributions
        def objective(weights):
            weights = np.array(weights)
            portfolio_vol = self._portfolio_volatility(weights)
            
            # Risk contributions
            marginal_risk = np.dot(self.annual_cov, weights)
            risk_contributions = weights * marginal_risk / portfolio_vol
            
            # Target: equal risk contribution from each asset
            target_risk_contribution = portfolio_vol / n
            
            # Sum of squared deviations from target
            return np.sum((risk_contributions - target_risk_contribution) ** 2)
            
        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # All weights positive (typically not constrained to <= 1 in risk parity)
        bounds = tuple((0.01, None) for _ in range(n))
        
        # Initial guess
        initial_weights = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            # Normalize weights to sum to 1
            self.weights = result['x'] / np.sum(result['x'])
        else:
            logger.warning(f"Optimization failed: {result['message']}")
            # Fall back to equal weights
            self.weights = np.ones(n) / n
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio return
        
        Args:
            weights: Asset weights
            
        Returns:
            Portfolio return
        """
        return np.sum(self.annual_returns * weights)
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility
        
        Args:
            weights: Asset weights
            
        Returns:
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.annual_cov, weights)))
    
    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio metrics
        
        Returns:
            Dictionary of metrics
        """
        portfolio_return = self._portfolio_return(self.weights)
        portfolio_volatility = self._portfolio_volatility(self.weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Calculate diversification ratio
        asset_volatilities = np.sqrt(np.diag(self.annual_cov))
        weighted_volatilities = self.weights * asset_volatilities
        diversification_ratio = portfolio_volatility / np.sum(weighted_volatilities) if np.sum(weighted_volatilities) > 0 else 1
        
        # Calculate risk contributions
        marginal_risk = np.dot(self.annual_cov, self.weights)
        risk_contributions = self.weights * marginal_risk / portfolio_volatility if portfolio_volatility > 0 else np.zeros_like(self.weights)
        
        # Calculate maximum drawdown using historical data
        if self.returns_data is not None:
            portfolio_returns = self.returns_data.dot(self.weights)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max) - 1
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio,
            'risk_contributions': dict(zip(self.symbols, risk_contributions.tolist())),
            'max_drawdown': max_drawdown
        }
    
    def efficient_frontier(self, points: int = 20) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            points: Number of points on the frontier
            
        Returns:
            DataFrame with return, volatility, and sharpe for each point
        """
        # Get minimum and maximum returns
        min_return = min(self.annual_returns)
        max_return = max(self.annual_returns)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, points)
        
        # Calculate portfolio for each target return
        results = []
        for target_return in target_returns:
            self._mean_variance(target_return=target_return)
            metrics = self._calculate_portfolio_metrics()
            
            results.append({
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio']
            })
        
        return pd.DataFrame(results)


class PortfolioRebalancer:
    """Portfolio rebalancing strategies"""
    
    def __init__(self, target_weights: Dict[str, float], tolerance: float = 0.05):
        """
        Initialize rebalancer
        
        Args:
            target_weights: Target weights for each asset
            tolerance: Rebalancing tolerance (percentage deviation)
        """
        self.target_weights = target_weights
        self.tolerance = tolerance
        
    def check_rebalance_needed(self, current_values: Dict[str, float]) -> bool:
        """
        Check if rebalancing is needed
        
        Args:
            current_values: Current values for each asset
            
        Returns:
            True if rebalancing is needed
        """
        # Calculate current weights
        total_value = sum(current_values.values())
        if total_value == 0:
            return False
            
        current_weights = {symbol: value / total_value for symbol, value in current_values.items()}
        
        # Check if any weight deviates from target by more than tolerance
        for symbol, target_weight in self.target_weights.items():
            if symbol in current_weights:
                deviation = abs(current_weights[symbol] - target_weight)
                if deviation > self.tolerance:
                    return True
                    
        return False
        
    def calculate_rebalance_trades(self, 
                                 current_values: Dict[str, float], 
                                 prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate trades to rebalance portfolio
        
        Args:
            current_values: Current values for each asset
            prices: Current prices for each asset
            
        Returns:
            Dictionary of trades (symbol -> units to buy/sell)
        """
        total_value = sum(current_values.values())
        if total_value == 0:
            return {}
            
        # Calculate target values
        target_values = {symbol: total_value * weight for symbol, weight in self.target_weights.items()}
        
        # Calculate differences
        diffs = {symbol: target_values.get(symbol, 0) - current_values.get(symbol, 0) 
                for symbol in set(target_values.keys()) | set(current_values.keys())}
        
        # Convert value differences to units
        trades = {}
        for symbol, diff in diffs.items():
            if symbol in prices and prices[symbol] > 0:
                trades[symbol] = diff / prices[symbol]
            else:
                trades[symbol] = 0
                
        return trades


class FactorAnalyzer:
    """Analyze portfolio factor exposures"""
    
    def __init__(self, factors_data: Optional[pd.DataFrame] = None):
        """
        Initialize factor analyzer
        
        Args:
            factors_data: DataFrame of factor returns
        """
        self.factors_data = factors_data
        self.factor_exposures = None
        self.asset_returns = None
        
    def set_factors_data(self, factors_data: pd.DataFrame):
        """
        Set factor returns data
        
        Args:
            factors_data: DataFrame of factor returns
        """
        self.factors_data = factors_data
        
    def set_asset_returns(self, asset_returns: pd.DataFrame):
        """
        Set asset returns data
        
        Args:
            asset_returns: DataFrame of asset returns
        """
        self.asset_returns = asset_returns
        
    def analyze_factor_exposures(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze factor exposures for each asset
        
        Returns:
            Dictionary of factor exposures for each asset
        """
        if self.factors_data is None or self.asset_returns is None:
            logger.error("Factors data or asset returns not set")
            return {}
            
        # Align data
        common_index = self.factors_data.index.intersection(self.asset_returns.index)
        factors = self.factors_data.loc[common_index]
        assets = self.asset_returns.loc[common_index]
        
        if len(common_index) < 30:
            logger.warning(f"Insufficient data for factor analysis: {len(common_index)} points")
            return {}
            
        # Add constant for intercept
        factors_with_const = pd.concat([factors, pd.Series(1, index=factors.index, name='Alpha')], axis=1)
        
        # Calculate factor exposures for each asset
        exposures = {}
        for asset in assets.columns:
            # Linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(factors_with_const, assets[asset])
            
            # Store exposures
            asset_exposures = {}
            for i, factor in enumerate(factors_with_const.columns):
                if factor == 'Alpha':
                    asset_exposures[factor] = model.intercept_
                else:
                    asset_exposures[factor] = model.coef_[i]
                    
            exposures[asset] = asset_exposures
            
        self.factor_exposures = exposures
        return exposures
        
    def calculate_portfolio_exposures(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio factor exposures
        
        Args:
            weights: Asset weights
            
        Returns:
            Dictionary of portfolio factor exposures
        """
        if self.factor_exposures is None:
            logger.warning("Factor exposures not calculated yet")
            return {}
            
        # Initialize with zeros
        portfolio_exposures = {factor: 0.0 for factor in next(iter(self.factor_exposures.values())).keys()}
        
        # Weight factor exposures by asset weights
        for asset, asset_weight in weights.items():
            if asset in self.factor_exposures:
                for factor, exposure in self.factor_exposures[asset].items():
                    portfolio_exposures[factor] += exposure * asset_weight
                    
        return portfolio_exposures
        
    def generate_report(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate factor analysis report
        
        Args:
            weights: Asset weights
            
        Returns:
            Dictionary with factor analysis report
        """
        if self.factor_exposures is None:
            self.analyze_factor_exposures()
            
        if not self.factor_exposures:
            return {'error': 'Factor analysis failed'}
            
        # Calculate portfolio exposures
        portfolio_exposures = self.calculate_portfolio_exposures(weights)
        
        # Create report
        report = {
            'portfolio_exposures': portfolio_exposures,
            'asset_exposures': self.factor_exposures,
            'factor_correlation': self.factors_data.corr().to_dict() if self.factors_data is not None else {},
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return report
