# MercurioAI Live Trading Guide

This document provides detailed instructions for transitioning from backtesting and paper trading to live trading with real money using the MercurioAI platform.

## Prerequisites

Before starting with live trading, ensure you have:

1. Successfully run the demo and tested your strategies with historical data
2. Completed extensive paper trading to validate strategy performance
3. Obtained and configured all necessary API keys
4. Understood the risks associated with algorithmic trading

## Setting Up Live Trading

### 1. Broker Account Setup

MercurioAI currently supports live trading through Alpaca. To set up your account:

1. Create a live trading account at [Alpaca](https://alpaca.markets/)
2. Complete account verification and deposit funds
3. Generate API keys for your live account
   - Go to your Alpaca dashboard
   - Navigate to "Paper/Live Trading API Keys"
   - Select "Live" keys (not Paper)
   - Generate and save your API Key ID and Secret Key

### 2. Configure MercurioAI for Live Trading

1. Update your `.env` file with live trading credentials:

```
# Replace paper trading keys with live trading keys
ALPACA_KEY=your_live_alpaca_key_here
ALPACA_SECRET=your_live_alpaca_secret_here

# Set trading mode to live
TRADING_MODE=live
```

2. Modify `app/services/trading.py` if necessary:

The `TradingService` class has a parameter `is_paper` which defaults to `True`. When transitioning to live trading, you'll need to initialize it with `is_paper=False`:

```python
trading_service = TradingService(is_paper=False)
```

### 3. Risk Management Configuration

Before trading with real money, configure risk management parameters:

1. Open `app/config/risk_management.py`
2. Adjust the following parameters according to your risk tolerance:
   - `MAX_POSITION_SIZE`: Maximum percentage of portfolio in any single position
   - `MAX_DAILY_DRAWDOWN`: Maximum allowed daily drawdown before halting trading
   - `STOP_LOSS_PERCENTAGE`: Default stop loss percentage for trades
   - `TAKE_PROFIT_PERCENTAGE`: Default take profit percentage for trades

### 4. Testing Progression

Follow this recommended progression before committing significant capital:

1. **Backtest**: Thoroughly test strategies against historical data
2. **Paper Trading**: Test in real-time market conditions without real money
3. **Minimal Live Trading**: Start with very small position sizes (1-2% of portfolio)
4. **Scaled Live Trading**: Gradually increase position sizes as confidence grows

## Monitoring Live Trading

### Dashboard and Notifications

1. Access the trading dashboard at `http://localhost:8000/dashboard` when running locally
2. Configure alerts in `app/services/notification.py` to receive:
   - Trade execution confirmations
   - Error notifications
   - Performance summaries
   - Risk threshold alerts

### Performance Tracking

Track your live trading performance through:

1. The built-in performance dashboard
2. Daily and weekly automated reports
3. Portfolio analytics tools

## Emergency Procedures

### Manual Intervention

In case of unexpected market events or system issues:

1. **Emergency Stop**: Use the `/api/trading/emergency-stop` endpoint to halt all trading activities
2. **Position Liquidation**: Use `/api/trading/liquidate-all` to close all open positions
3. **API Disconnection**: Revoke your API keys from the Alpaca dashboard to prevent further trading

### Recovery Procedures

After resolving issues:

1. Run diagnostics using `python diagnostic_tools.py`
2. Review system logs in the `logs/` directory
3. Gradually resume trading with reduced position sizes

## Compliance and Tax Considerations

1. **Trading Records**: MercurioAI maintains detailed records in the PostgreSQL database
2. **Tax Reporting**: Export trading activity reports for tax compliance:
   ```
   python tools/generate_tax_report.py --year 2023
   ```
3. **API Restrictions**: Be aware of API rate limits and trading restrictions

## Best Practices

1. **Start Small**: Begin with small amounts of capital
2. **Monitor Continuously**: Especially during initial live trading phases
3. **Regular Backups**: Backup your database regularly
4. **Update Strategies**: Periodically review and update strategies based on performance
5. **Keep Records**: Maintain detailed notes about system changes and trading decisions

## Support and Troubleshooting

If you encounter issues during live trading:

1. Check the log files in the `logs/` directory
2. Review the troubleshooting section in the main documentation
3. Ensure all API connections are functioning properly
4. Verify that risk management parameters are appropriate

Remember that all algorithmic trading comes with risks, and past performance does not guarantee future results. Always trade with capital you can afford to lose.
