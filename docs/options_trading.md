# Trading d'Options avec Mercurio AI

> [!NOTE]
> **Navigation Rapide:**
> - [🔍 Index de tous les guides](./GUIDES_INDEX.md)
> - [📈 Guide de Day Trading](./day_trading_guide.md)
> - [🧠 Guide d'Entraînement des Modèles](./model_training_guide.md)
> - [📔 Documentation Principale](./README.md)

## Introduction

Ce document décrit les fonctionnalités avancées de trading d'options implémentées dans la plateforme Mercurio AI. Le module de trading d'options permet d'exploiter l'abonnement Alpaca AlgoTrader Plus avec Options Trading Levels 1-3 pour exécuter diverses stratégies de trading d'options, des plus simples aux plus complexes. Ces stratégies peuvent être utilisées seules ou en combinaison avec les modèles de machine learning intégrés pour optimiser les performances.

## Architecture

Le système de trading d'options s'intègre parfaitement à l'architecture existante de Mercurio AI et se compose de deux composants principaux :

1. **Options Service** - Un service qui interagit avec l'API Alpaca pour les opérations liées aux options
2. **Options Strategy** - Une stratégie qui transforme les signaux des stratégies existantes en opportunités de trading d'options

### Diagramme de flux

```
┌───────────────┐    ┌────────────────┐    ┌──────────────────┐
│ Stratégies ML │───▶│ Options Strategy│───▶│ Options Service  │
│ existantes    │    │                │    │                  │
└───────────────┘    └────────────────┘    └──────────────────┘
                             │                      │
                             ▼                      ▼
                     ┌────────────────┐    ┌──────────────────┐
                     │ Market Data    │    │ Execution via    │
                     │ Service        │    │ Alpaca API       │
                     └────────────────┘    └──────────────────┘
```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```
┌───────────────┐    ┌────────────────┐    ┌──────────────────┐
│ Stratégies ML │───▶│ Options Strategy│───▶│ Options Service  │
│ existantes    │    │                │    │                  │
└───────────────┘    └────────────────┘    └──────────────────┘
                            │                      │
                            ▼                      ▼
                     ┌────────────────┐    ┌──────────────────┐
                     │ Gestionnaire de│◀───│ API Alpaca       │
                     │ risque         │    │ (Options Level 1)│
                     └────────────────┘    └──────────────────┘
```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```

## Configuration

Les paramètres de trading d'options sont configurables via le fichier `config/daytrader_config.json` dans la section `stock.options_trading` :

```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```json
"options_trading": {
  "enabled": true,
  "max_options_allocation_pct": 0.20,
  "max_options_per_symbol": 3,
  "min_confidence_for_options": 0.70,
  "risk_profile": "moderate",
  "max_days_to_expiry": 45,
  "preferred_strategies": [
    "Long Call",
    "Long Put",
    "Cash-Secured Put",
    "Covered Call"
  ],
  "base_strategies": [
    "TransformerStrategy",
    "LSTMPredictorStrategy",
    "MSIStrategy"
  ],
  "require_confirmation": true,
  "max_loss_per_trade_pct": 1.0,
  "strict_position_sizing": true
}
```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```

### Paramètres de configuration

| Paramètre | Description | Valeurs possibles |
|-----------|-------------|-------------------|
| `enabled` | Active ou désactive le trading d'options | `true`, `false` |
| `max_options_allocation_pct` | Pourcentage maximum du capital alloué aux options | `0.0` à `1.0` |
| `max_options_per_symbol` | Nombre maximum de contrats d'options par symbole | Entier positif |
| `min_confidence_for_options` | Seuil de confiance minimum pour exécuter un trading d'options | `0.0` à `1.0` |
| `risk_profile` | Profil de risque pour les stratégies d'options | `"conservative"`, `"moderate"`, `"aggressive"` |
| `max_days_to_expiry` | Nombre maximum de jours jusqu'à l'expiration | Entier positif |
| `preferred_strategies` | Liste des stratégies d'options préférées | Tableau de noms de stratégies |
| `base_strategies` | Liste des stratégies de base à utiliser pour les signaux | Tableau de noms de stratégies |
| `require_confirmation` | Exiger une confirmation avant d'exécuter un trade d'options | `true`, `false` |
| `max_loss_per_trade_pct` | Pourcentage maximum de perte par trade | `0.0` à `1.0` |
| `strict_position_sizing` | Activer le dimensionnement strict des positions | `true`, `false` |

## Stratégies d'options disponibles

Mercurio AI prend désormais en charge un large éventail de stratégies d'options, des stratégies simples de niveau 1 aux stratégies avancées de niveau 3 :

### Stratégies de Niveau 1

#### Long Call

**Description** : Achat d'une option d'achat, donnant le droit d'acheter l'actif sous-jacent à un prix déterminé.

**Utilisation** : Lorsque vous anticipez une hausse significative du prix de l'actif sous-jacent.

**Risque** : Limité au montant de la prime payée.

**Gain potentiel** : Théoriquement illimité à mesure que le prix de l'actif sous-jacent augmente.

#### Long Put

**Description** : Achat d'une option de vente, donnant le droit de vendre l'actif sous-jacent à un prix déterminé.

**Utilisation** : Lorsque vous anticipez une baisse significative du prix de l'actif sous-jacent.

**Risque** : Limité au montant de la prime payée.

**Gain potentiel** : Limité au prix d'exercice moins la prime payée (si le prix tombe à zéro).

#### Cash-Secured Put

**Description** : Vente d'une option de vente avec suffisamment de liquidités pour acheter l'actif sous-jacent si l'option est exercée.

**Utilisation** : Lorsque vous êtes prêt à acheter l'actif sous-jacent à un prix inférieur au prix actuel et que vous souhaitez générer un revenu en attendant.

**Risque** : Limité à la différence entre le prix d'exercice et zéro, moins la prime reçue.

**Gain potentiel** : Limité au montant de la prime reçue.

#### Covered Call

**Description** : Vente d'une option d'achat tout en détenant l'actif sous-jacent.

**Utilisation** : Lorsque vous détenez déjà l'actif sous-jacent et souhaitez générer un revenu supplémentaire, et êtes prêt à vendre l'actif à un prix supérieur au prix actuel.

**Risque** : Limité au coût d'opportunité si le prix de l'actif augmente au-dessus du prix d'exercice.

**Gain potentiel** : Limité au montant de la prime reçue plus l'appréciation potentielle jusqu'au prix d'exercice.

### Stratégies de Niveau 2

#### Iron Condor

**Description** : Combinaison de quatre options différentes (vente d'un spread call et vente d'un spread put) pour créer une fourchette de prix où le trader peut réaliser un profit.

**Utilisation** : Lorsque vous anticipez une faible volatilité et un marché stagnant dans une fourchette définie.

**Risque** : Limité à la différence entre les prix d'exercice des options achetées et vendues, moins la prime nette reçue.

**Gain potentiel** : Limité au montant de la prime nette reçue.

#### Butterfly Spread

**Description** : Combinaison de trois prix d'exercice différents avec quatre contrats d'options pour créer une position qui profite lorsque le prix de l'actif sous-jacent reste proche du prix d'exercice central.

**Utilisation** : Lorsque vous anticipez que le prix de l'actif sous-jacent restera stable près d'un niveau cible.

**Risque** : Limité au coût initial de la stratégie (primes nettes payées).

**Gain potentiel** : Maximal lorsque le prix de l'actif est exactement au prix d'exercice central à l'expiration.

### Stratégies Avancées (Niveau 3)

#### Straddle/Strangle

**Description** : Achat simultané d'options d'achat et de vente au même prix d'exercice (straddle) ou à des prix d'exercice différents (strangle).

**Utilisation** : Lorsque vous anticipez une forte volatilité mais êtes incertain de la direction du mouvement.

**Risque** : Limité aux primes totales payées pour les deux options.

**Gain potentiel** : Théoriquement illimité si le prix du sous-jacent bouge significativement dans l'une ou l'autre direction.

#### Calendar Spread

**Description** : Combinaison d'options avec le même prix d'exercice mais des dates d'expiration différentes.

**Utilisation** : Pour profiter de la différence de décroissance temporelle entre les options à court et à long terme.

**Risque** : Limité au coût initial de la stratégie.

**Gain potentiel** : Maximal lorsque le prix du sous-jacent est proche du prix d'exercice à l'expiration de l'option à court terme.

## Nouveaux Scripts pour le Trading d'Options

Mercurio AI inclut désormais plusieurs scripts spécialisés pour différentes approches du trading d'options :

### 1. Trading d'Options Quotidien

```bash
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --capital 100000
```

Ce script exécute des stratégies d'options sur une base quotidienne, en surveillant les opportunités pendant les heures de marché et en ajustant les positions selon les conditions du marché.

### 2. Trading d'Options Basé sur le ML

```bash
python -m scripts.options.run_ml_options_trader --ml-strategy LSTM --options-strategy COVERED_CALL --symbols AAPL MSFT --capital 100000
```

Ce script combine les capacités de prédiction des modèles ML (LSTM, Transformer, LLM, MSI) avec des stratégies d'options pour des décisions de trading plus précises.

### 3. Trading d'Options à Haut Volume

```bash
python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --max-symbols 50 --use-threads --use-custom-symbols
```

Optimisé pour trader jusqu'à 50 symboles simultanément avec une exécution parallèle pour une performance maximale.

### 4. Trading d'Options sur Crypto

```bash
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000
```

Spécialement conçu pour les spécificités du trading d'options sur cryptomonnaies, avec des paramètres adaptés à leur volatilité plus élevée.

### 5. Test des Stratégies d'Options

```bash
python -m scripts.options.test_options_strategies --test-all
```

Outil complet pour tester toutes les stratégies d'options, validant leur initialisation, conditions d'entrée/sortie, exécution, et gestion des risques.

## API des services d'options

### OptionsService

```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```python
class OptionsService:
    def __init__(self, trading_service: TradingService, market_data_service: MarketDataService):
        # Initialise le service d'options
        
    async def get_available_options(self, symbol: str, expiration_date: Optional[str] = None) -> List[Dict[str, Any]]:
        # Récupère les options disponibles pour un symbole donné
        
    async def execute_option_trade(self, option_symbol: str, action: TradeAction, quantity: int, order_type: str = "market", limit_price: Optional[float] = None, time_in_force: str = "day", strategy_name: str = "unknown") -> Dict[str, Any]:
        # Exécute un trade d'options
        
    async def get_option_position(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        # Récupère les détails d'une position d'options spécifique
        
    async def get_all_option_positions(self) -> List[Dict[str, Any]]:
        # Récupère toutes les positions d'options actuelles
        
    async def calculate_option_metrics(self, option_data: Dict[str, Any]) -> Dict[str, Any]:
        # Calcule les métriques importantes des options (Grecs)
        
    async def suggest_option_strategies(self, symbol: str, price_prediction: Dict[str, Any], risk_profile: str = "moderate") -> List[Dict[str, Any]]:
        # Suggère des stratégies d'options basées sur les prédictions de prix
```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```

### OptionsStrategy

```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```python
class OptionsStrategy(Strategy):
    def __init__(self, options_service: OptionsService, base_strategy_name: str, risk_profile: str = "moderate", max_days_to_expiry: int = 45, preferred_option_types: List[str] = None):
        # Initialise la stratégie d'options
        
    async def generate_signal(self, symbol: str, data: Dict[str, Any], timeframe: TimeFrame = TimeFrame.DAY) -> Dict[str, Any]:
        # Génère un signal de trading d'options basé sur le signal de la stratégie sous-jacente
        
    async def backtest(self, symbol: str, historical_data: List[Dict[str, Any]], timeframe: TimeFrame = TimeFrame.DAY) -> Dict[str, Any]:
        # Backteste la stratégie d'options (simplifié)
        
    async def optimize(self, symbol: str, historical_data: List[Dict[str, Any]], timeframe: TimeFrame = TimeFrame.DAY) -> Dict[str, Any]:
        # Optimise les paramètres de la stratégie d'options
```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```

## Outils Mathématiques pour les Options

Mercurio AI inclut désormais un module d'utilitaires mathématiques complet pour la tarification des options et le calcul des sensibilités (Grecs) :

```python
from app.utils.math_utils import (
    black_scholes_call, black_scholes_put,
    calculate_implied_volatility,
    calculate_delta, calculate_gamma, calculate_theta, calculate_vega
)

# Exemple de tarification d'option
prix_call = black_scholes_call(
    S=100,         # Prix actuel du sous-jacent
    K=105,         # Prix d'exercice
    t=30/365,      # Temps jusqu'à l'expiration (en années)
    r=0.03,        # Taux d'intérêt sans risque
    sigma=0.2      # Volatilité implicite
)

# Calcul des Grecs
delta = calculate_delta(S=100, K=105, t=30/365, r=0.03, sigma=0.2, option_type='call')
vega = calculate_vega(S=100, K=105, t=30/365, r=0.03, sigma=0.2)
```

Ces fonctions permettent une analyse sophistiquée des options et facilitent l'évaluation précise des opportunités de trading.

## Exemples d'utilisation

### Initialisation du service d'options

```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```python
from app.services.trading import TradingService
from app.services.market_data import MarketDataService
from app.services.options_service import OptionsService

# Initialiser les services requis
trading_service = TradingService(is_paper=True)
market_data_service = MarketDataService()

# Créer le service d'options
options_service = OptionsService(
    trading_service=trading_service,
    market_data_service=market_data_service
)
```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```

### Création d'une stratégie d'options basée sur une stratégie existante

```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```python
from app.strategies.options_strategy import OptionsStrategy

# Créer une stratégie d'options basée sur la stratégie TransformerStrategy
options_strategy = OptionsStrategy(
    options_service=options_service,
    base_strategy_name="TransformerStrategy",
    risk_profile="moderate",
    max_days_to_expiry=30,
    preferred_option_types=["Long Call", "Long Put"]
)

# Générer un signal d'options
signal = await options_strategy.generate_signal("AAPL", market_data)

# Exécuter un trade d'options basé sur le signal
if signal.get("action") != TradeAction.HOLD:
    result = await options_service.execute_option_trade(
        option_symbol=f"{signal['symbol']}_{signal['expiration']}_{signal['option_type'][0].upper()}_{int(signal['strike']*1000):08d}",
        action=signal["action"],
        quantity=1,
        strategy_name=options_strategy.name
    )
```

## Trading d'Options sur Crypto-monnaies

Mercurio AI comprend une fonctionnalité spécifique pour le trading d'options sur crypto-monnaies via l'API Alpaca. Le script `scripts/options/run_crypto_options_trader.py` permet d'exécuter diverses stratégies d'options sur un large éventail de crypto-monnaies.

### Fonctionnalités principales

- Trading d'options sur les principales crypto-monnaies (BTC, ETH, SOL, etc.)
- Utilisation des vraies données Alpaca (pas de simulation)
- Stratégies multiples : Long Call, Long Put, Iron Condor, Butterfly Spread
- Mode "MIXED" combinant plusieurs stratégies pour la diversification
- Trading en mode paper ou live
- Utilisation d'une liste personnalisée de crypto disponibles dans le fichier `.env`
- Exécution parallèle avec l'option `--use-threads`

### Exemples d'utilisation

```bash
# Stratégie unique avec des symboles spécifiques
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading --duration 2h

# Utilisation de la liste personnalisée de crypto en mode MIXED
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads

# Ajustement du seuil de volatilité pour augmenter les opportunités de trading
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
```

### Configuration des crypto-monnaies

Pour définir votre liste personnalisée de crypto-monnaies, ajoutez-les dans votre fichier `.env` :

```
# Liste personnalisée des crypto-monnaies disponibles sur Alpaca
PERSONALIZED_CRYPTO_LIST=BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,XRP/USD,DOGE/USD,LINK/USD,LTC/USD,AAVE/USD,BCH/USD,UNI/USD,BAT/USD,CRV/USD,SHIB/USD,BTC/USDT,ETH/USDT,BCH/USDT,AAVE/USDT
```

**Note importante** : Assurez-vous d'utiliser le format correct avec slashs (`BTC/USD`) et non `BTCUSD`, car l'API Alpaca nécessite ce format pour les crypto-monnaies.

### Paramètres de configuration

- `--strategy` : La stratégie à utiliser (LONG_CALL, LONG_PUT, IRON_CONDOR, BUTTERFLY, MIXED)
- `--symbols` : Liste des symboles crypto à trader (non requis si `--use-custom-symbols` est utilisé)
- `--use-custom-symbols` : Utilise la liste personnalisée dans le fichier `.env`
- `--capital` : Montant de capital à utiliser pour le trading
- `--duration` : Durée d'exécution du script (format : 1h, 30m, 1d)
- `--paper-trading` : Utilise le mode paper trading (pas de vrais ordres)
- `--use-threads` : Exécute le trading avec plusieurs threads en parallèle
- `--volatility-threshold` : Seuil de volatilité minimum pour entrer dans une position (par défaut : 0.05)
- `--days-to-expiry` : Nombre de jours avant l'expiration des options
- `--delta-target` : Delta cible pour la sélection des options (par défaut : 0.4)

## Backtesting des Stratégies d'Options

Mercurio AI propose un service de backtesting spécifique pour les stratégies d'options :

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    # Initialiser le backtester
    backtester = OptionsBacktester()
    
    # Configurer les paramètres de stratégie
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    # Exécuter le backtest
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Rendement total: {results['total_return']:.2f}%")
    print(f"Ratio de Sharpe: {results['sharpe_ratio']:.2f}")

# Exécuter le backtest
asyncio.run(backtest_covered_call())
```

## Trading d'Options Multi-Stratégies

Pour des approches plus sophistiquées, Mercurio AI permet d'exécuter plusieurs stratégies d'options simultanément :

```python
from app.services.options_service import OptionsService
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
import asyncio

async def run_multi_strategy():
    # Initialiser les services
    broker = AlpacaAdapter(is_paper=True)
    await broker.connect()
    
    trading_service = TradingService(broker, is_paper=True)
    options_service = OptionsService(broker)
    
    # Créer les stratégies
    iron_condor = IronCondorStrategy(
        underlying_symbol="SPY",
        max_position_size=0.05,
        days_to_expiration=45
    )
    iron_condor.broker_adapter = broker
    iron_condor.options_service = options_service
    
    butterfly = ButterflySpreadStrategy(
        underlying_symbol="QQQ",
        max_position_size=0.03,
        days_to_expiration=30
    )
    butterfly.broker_adapter = broker
    butterfly.options_service = options_service
    
    # Exécuter les stratégies
    strategies = [iron_condor, butterfly]
    
    for strategy in strategies:
        should_enter = await strategy.should_enter(None)  # Normalement, vous passeriez des données de marché ici
        
        if should_enter:
            result = await strategy.execute_entry()
            print(f"Entrée pour {strategy.__class__.__name__}: {result}")

# Exécuter les stratégies
asyncio.run(run_multi_strategy())
```

## Intégration avec l'Analyse de Sentiment

Mercurio AI peut maintenant intégrer l'analyse de sentiment pour améliorer les décisions de trading d'options :

```python
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.options.long_call import LongCallStrategy
import asyncio

async def sentiment_based_options():
    # Initialiser la stratégie LLM pour l'analyse de sentiment
    llm_strategy = LLMStrategy()
    
    # Analyser le sentiment pour un symbole
    sentiment_data = await llm_strategy.analyze_sentiment("AAPL")
    
    # Déterminer la stratégie d'options basée sur le sentiment
    if sentiment_data['sentiment_score'] > 0.7:  # Sentiment très positif
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            max_position_size=0.05,
            days_to_expiration=30
        )
        print("Sentiment très positif - Utilisation de Long Call Strategy")
    elif sentiment_data['sentiment_score'] < 0.3:  # Sentiment négatif
        # Utiliser une stratégie adaptée au sentiment négatif
        print("Sentiment négatif - Utilisation de Long Put Strategy")
    else:
        # Sentiment neutre
        print("Sentiment neutre - Utilisation de Iron Condor Strategy")

# Exécuter l'analyse
asyncio.run(sentiment_based_options())
```

## Bonnes pratiques et considérations de risque

### Gestion du risque

- **Limitez l'allocation** : Maintenez une allocation limitée pour le trading d'options (typiquement 10-20% du portefeuille).
- **Diversifiez les expirations** : Évitez de concentrer toutes vos positions sur une seule date d'expiration.
- **Surveillez les métriques** : Faites attention aux Greeks, en particulier le Theta (décroissance temporelle) qui érode la valeur des options au fil du temps.

### Bonnes pratiques

- **Commencez petit** : Démarrez avec un petit nombre de contrats pour comprendre le comportement des options.
- **Préférez les options liquides** : Choisissez des options avec un volume et un intérêt ouvert élevés pour minimiser les spreads.
- **Limitez les stratégies complexes** : Au niveau 1, restez concentré sur les stratégies simples comme les calls et puts longs.
- **Prenez en compte l'expiration** : Les options à court terme sont plus risquées mais moins chères, tandis que les options à long terme sont plus coûteuses mais offrent plus de temps pour que votre thèse se développe.

## Dépannage

### Problèmes courants

| Problème | Causes possibles | Solutions |
|----------|------------------|-----------|
| Erreur "Option non disponible" | L'option spécifiée n'existe pas ou l'expiration est incorrecte | Vérifiez que vous utilisez un format correct pour le symbole d'option et une date d'expiration valide |
| Position trop petite | Les restrictions de dimensionnement de position sont trop strictes | Ajustez `max_options_allocation_pct` dans la configuration |
| Aucun signal d'options généré | Confiance de la stratégie de base trop faible | Vérifiez que la stratégie de base génère des signaux avec une confiance supérieure à `min_confidence_for_options` |
| Erreur d'exécution du trade | Problèmes d'API avec Alpaca | Vérifiez vos clés API et assurez-vous que votre compte a un accès au trading d'options de niveau 1 |

## Conclusion

Le module de trading d'options pour Mercurio AI fournit une extension puissante mais contrôlée des capacités de trading existantes. En combinant les signaux générés par vos stratégies ML existantes avec des stratégies d'options soigneusement sélectionnées, vous pouvez potentiellement améliorer les rendements et gérer les risques de manière plus efficace.

Souvenez-vous toujours que le trading d'options comporte des risques intrinsèques différents du trading d'actions standard, et nécessite donc une surveillance et une gestion attentives.
