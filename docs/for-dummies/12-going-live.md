# Chapter 12: Going Live

Welcome to the final chapter of "Mercurio AI for Dummies"! You've learned how to create, test, optimize, manage, and monitor trading strategies. Now it's time for the ultimate step: transitioning from paper trading to live trading with real money.

## The Transition to Live Trading

Moving from paper trading to live trading is a significant step that requires careful planning:

- **Psychological Readiness**: Are you emotionally prepared for the ups and downs?
- **Strategy Confidence**: Has your strategy proven itself through rigorous testing?
- **Risk Management**: Are your risk controls properly implemented?
- **Technical Setup**: Is your trading infrastructure reliable and secure?

## Prerequisites for Live Trading

Before going live, ensure you have:

1. **Thoroughly Tested Strategy**: At least 3 months of successful paper trading
2. **Clear Trading Plan**: Documented rules and guidelines
3. **Risk Management Framework**: Position sizing, stop-loss rules, etc.
4. **Emergency Procedures**: What to do if something goes wrong
5. **Proper Broker Integration**: API keys and authentication set up

## Connecting to Brokers

Mercurio AI can connect to various brokers through their APIs:

```python
from app.services.trading import TradingService

# Initialize live trading with a broker
trading_service = TradingService(
    mode="live",
    broker="alpaca",  # Replace with your broker
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET",
    base_url="https://api.alpaca.markets"  # URL depends on broker
)

# Verify connection
account_info = await trading_service.get_account_info()
print(f"Connected to {account_info['broker']} account: {account_info['account_id']}")
print(f"Account Value: ${account_info['equity']}")
print(f"Buying Power: ${account_info['buying_power']}")
```

### Supported Brokers

Mercurio AI supports several brokers:

- **Alpaca**: US stocks and ETFs
- **Interactive Brokers**: Global markets
- **Binance**: Cryptocurrencies
- **Others**: Via custom adapters

## Implementing a Live Trading System

Here's a basic implementation of a live trading system:

```python
import asyncio
import logging
from datetime import datetime, time
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.strategies.moving_average import MovingAverageStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_trading")

async def run_live_trading():
    """Run the live trading system."""
    
    # Initialize services
    market_data = MarketDataService()
    
    # IMPORTANT: In live mode, real trades will be placed
    trading = TradingService(
        mode="live",
        broker="alpaca",
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET"
    )
    
    # Create strategy
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        use_ml=True
    )
    
    # Trading parameters
    symbol = "AAPL"
    position_size_pct = 0.1  # 10% of portfolio per position
    max_positions = 5        # Maximum 5 simultaneous positions
    
    logger.info("Starting live trading system")
    
    while True:
        try:
            # Check if market is open
            if not is_market_open():
                logger.info("Market is closed. Waiting...")
                await asyncio.sleep(3600)  # Sleep for an hour
                continue
            
            # Get latest market data
            data = await market_data.get_recent_data(
                symbol=symbol,
                bars=100  # Get last 100 bars
            )
            
            # Check if we have enough data
            if len(data) < 50:
                logger.warning(f"Not enough data for {symbol}: {len(data)} bars")
                await asyncio.sleep(300)  # Wait 5 minutes
                continue
            
            # Process data with strategy
            processed_data = await strategy.preprocess_data(data)
            
            # Get trading signal
            signal, confidence = await strategy.predict(processed_data)
            
            logger.info(f"Signal for {symbol}: {signal} (confidence: {confidence:.2f})")
            
            # Get current position for this symbol
            current_position = trading.get_position(symbol)
            
            # Execute trades based on signal
            if signal == "BUY" and (current_position is None or current_position.quantity == 0):
                # Check if we have capacity for a new position
                current_positions = trading.get_positions()
                if len(current_positions) >= max_positions:
                    logger.info(f"Maximum positions reached ({max_positions}). Skipping buy signal.")
                    continue
                
                # Calculate position size
                account = await trading.get_account_info()
                buying_power = float(account['buying_power'])
                position_value = buying_power * position_size_pct
                
                # Get current price
                current_price = data['close'].iloc[-1]
                
                # Calculate quantity
                quantity = int(position_value / current_price)
                
                if quantity > 0:
                    # Place buy order
                    order = await trading.place_order(
                        symbol=symbol,
                        quantity=quantity,
                        side="buy",
                        order_type="market"
                    )
                    logger.info(f"BUY order placed for {quantity} shares of {symbol} at ~${current_price:.2f}")
            
            elif signal == "SELL" and current_position is not None and current_position.quantity > 0:
                # Place sell order for entire position
                order = await trading.place_order(
                    symbol=symbol,
                    quantity=current_position.quantity,
                    side="sell",
                    order_type="market"
                )
                logger.info(f"SELL order placed for {current_position.quantity} shares of {symbol}")
            
            else:
                logger.info(f"No action taken for {symbol}")
            
            # Wait before next check
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
            await asyncio.sleep(60)  # Wait a minute before retrying

def is_market_open():
    """Check if the market is currently open."""
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check market hours (9:30 AM to 4:00 PM Eastern Time)
    # Note: This is a simplified check. In a real system, you'd want to
    # account for holidays and use a proper market calendar.
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    current_time = now.time()
    
    return market_open <= current_time <= market_close

if __name__ == "__main__":
    asyncio.run(run_live_trading())
```

## Risk Management in Live Trading

Risk management is even more critical in live trading than in paper trading:

```python
class RiskManager:
    """Risk management system for live trading."""
    
    def __init__(self, max_portfolio_risk=0.02, max_position_risk=0.005, max_drawdown=0.1):
        """
        Initialize the risk manager.
        
        Args:
            max_portfolio_risk: Maximum daily risk for the entire portfolio (2%)
            max_position_risk: Maximum daily risk per position (0.5%)
            max_drawdown: Maximum allowable drawdown before trading halt (10%)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_drawdown = max_drawdown
        
        self.starting_equity = None
        self.equity_history = []
    
    async def check_risk_limits(self, trading_service):
        """Check if any risk limits have been exceeded."""
        # Get account information
        account = await trading_service.get_account_info()
        current_equity = float(account['equity'])
        
        # Initialize starting equity if not set
        if self.starting_equity is None:
            self.starting_equity = current_equity
        
        # Record equity
        self.equity_history.append(current_equity)
        
        # Check for drawdown limit
        peak_equity = max(self.equity_history)
        current_drawdown = (peak_equity - current_equity) / peak_equity
        
        if current_drawdown > self.max_drawdown:
            return {
                "limit_exceeded": True,
                "reason": f"Max drawdown exceeded: {current_drawdown:.2%} > {self.max_drawdown:.2%}",
                "action": "halt_trading"
            }
        
        # Check daily loss limit
        if len(self.equity_history) > 1:
            daily_change = (current_equity / self.equity_history[-2]) - 1
            
            if daily_change < -self.max_portfolio_risk:
                return {
                    "limit_exceeded": True,
                    "reason": f"Max daily portfolio loss exceeded: {-daily_change:.2%} > {self.max_portfolio_risk:.2%}",
                    "action": "reduce_exposure"
                }
        
        return {"limit_exceeded": False}
    
    def calculate_position_size(self, account_equity, current_price, stop_loss_price):
        """Calculate position size based on risk parameters."""
        # Risk amount in dollars
        risk_amount = account_equity * self.max_position_risk
        
        # Risk per share
        risk_per_share = abs(current_price - stop_loss_price)
        
        # Calculate position size
        position_size = int(risk_amount / risk_per_share)
        
        return position_size
```

## Monitoring Live Performance

Continuous monitoring is essential for live trading:

```python
async def monitor_live_trading(trading_service, strategies, alert_threshold=0.05):
    """Monitor live trading performance and send alerts."""
    
    # Get account info
    account = await trading_service.get_account_info()
    starting_equity = float(account['equity'])
    
    while True:
        try:
            # Get updated account info
            updated_account = await trading_service.get_account_info()
            current_equity = float(updated_account['equity'])
            
            # Calculate daily P&L
            daily_pnl = current_equity - starting_equity
            daily_pnl_pct = daily_pnl / starting_equity
            
            logger.info(f"Current equity: ${current_equity:.2f}")
            logger.info(f"Daily P&L: ${daily_pnl:.2f} ({daily_pnl_pct:.2%})")
            
            # Check for alert threshold
            if abs(daily_pnl_pct) > alert_threshold:
                send_alert(
                    f"Trading Alert: Daily P&L of {daily_pnl_pct:.2%} exceeds threshold of {alert_threshold:.2%}",
                    daily_pnl=daily_pnl,
                    daily_pnl_pct=daily_pnl_pct,
                    current_equity=current_equity
                )
            
            # Get open positions
            positions = trading_service.get_positions()
            logger.info(f"Open positions: {len(positions)}")
            
            for symbol, position in positions.items():
                logger.info(f"{symbol}: {position.quantity} shares, Current value: ${position.market_value:.2f}")
            
            # Wait before next check
            await asyncio.sleep(900)  # Check every 15 minutes
            
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
            await asyncio.sleep(60)  # Wait before retrying

def send_alert(message, **kwargs):
    """Send an alert when conditions are met."""
    logger.warning(f"ALERT: {message}")
    
    # In a real system, you might send an email, SMS, or push notification
    # For example:
    # send_email("trading_alert@example.com", "Trading Alert", message)
    # send_sms("+1234567890", message)
```

## Automating Your Trading System

For 24/7 operation, set up your trading system as a service:

### Linux (systemd)

Create a service file `/etc/systemd/system/mercurio-trading.service`:

```
[Unit]
Description=Mercurio AI Trading System
After=network.target

[Service]
User=trading
WorkingDirectory=/path/to/mercurio
ExecStart=/path/to/python /path/to/mercurio/live_trading.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:

```bash
sudo systemctl enable mercurio-trading.service
sudo systemctl start mercurio-trading.service
```

### Windows

Use Task Scheduler to create a task that runs at system startup.

## Security Considerations

When trading with real money, security is paramount:

1. **API Keys**: Store API keys securely using environment variables or a secrets manager
2. **Access Control**: Limit access to your trading system
3. **Network Security**: Use secure connections (HTTPS, VPN)
4. **Monitoring**: Set up alerts for unusual activity
5. **Backups**: Regularly back up your system and data

## Dealing with Technical Issues

Have a plan for handling technical problems:

```python
class TradingFailsafe:
    """Failsafe system for handling technical issues in live trading."""
    
    def __init__(self, trading_service):
        self.trading_service = trading_service
        self.last_connectivity_check = datetime.now()
        self.consecutive_failures = 0
        self.max_failures = 3
    
    async def check_connectivity(self):
        """Check if trading services are operational."""
        try:
            # Try to get account info
            await self.trading_service.get_account_info()
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            self.last_connectivity_check = datetime.now()
            
            return True
        except Exception as e:
            logger.error(f"Connectivity check failed: {e}")
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.max_failures:
                await self.trigger_emergency_shutdown()
                return False
            
            return False
    
    async def trigger_emergency_shutdown(self):
        """Execute emergency shutdown procedures."""
        logger.critical("EMERGENCY SHUTDOWN TRIGGERED")
        
        try:
            # Get all open positions
            positions = self.trading_service.get_positions()
            
            # Close all positions
            for symbol, position in positions.items():
                logger.info(f"Emergency closure: Closing position in {symbol}")
                
                try:
                    await self.trading_service.place_order(
                        symbol=symbol,
                        quantity=position.quantity,
                        side="sell" if position.quantity > 0 else "buy",
                        order_type="market"
                    )
                except Exception as close_error:
                    logger.error(f"Failed to close position in {symbol}: {close_error}")
            
            logger.info("Emergency procedures completed")
            
            # Send emergency notification
            send_alert("TRADING SYSTEM EMERGENCY SHUTDOWN", 
                      reason="Consecutive connectivity failures",
                      positions_closed=len(positions))
            
        except Exception as shutdown_error:
            logger.critical(f"Emergency shutdown failed: {shutdown_error}")
```

## Legal and Regulatory Considerations

Be aware of legal requirements:

- **Tax Reporting**: Keep records for tax purposes
- **Regulatory Compliance**: Understand regulations in your jurisdiction
- **Account Types**: Different accounts have different rules (retirement vs. standard)

## Scaling Your Trading Operation

As you gain experience, consider scaling up:

1. **More Strategies**: Add diverse strategies to your portfolio
2. **More Assets**: Expand to different markets and asset classes
3. **Increased Capital**: Gradually increase your trading capital
4. **Infrastructure**: Upgrade to more robust hardware and connectivity

## Conclusion

Congratulations! You've completed the "Mercurio AI for Dummies" guide. You now have a comprehensive understanding of algorithmic trading with Mercurio AI, from basic concepts to live trading.

Remember that trading involves risk, and no strategy guarantees profits. Always:

1. Start small and scale gradually
2. Never risk money you can't afford to lose
3. Continue learning and improving your strategies
4. Maintain proper risk management
5. Monitor your systems diligently

With the right approach, algorithmic trading can be a rewarding endeavor, both intellectually and financially. Good luck on your trading journey!

---

**Key Takeaways:**
- Transitioning to live trading requires careful planning and preparation
- Connecting to brokers enables real-money trading through APIs
- Robust risk management is essential for protecting capital
- Continuous monitoring helps detect and address issues quickly
- Security is paramount when trading with real money
- Having procedures for technical issues can prevent catastrophic losses
- Scaling your operation should be done gradually as you gain experience
