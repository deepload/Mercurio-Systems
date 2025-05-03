# Guide du Système de Day Trading - Mercurio AI

> [!NOTE]
> **Navigation Rapide:**
> - [🔍 Index de tous les guides](./GUIDES_INDEX.md)
> - [📊 Guide des Options](./options_trading.md)
> - [🧠 Guide d'Entraînement des Modèles](./model_training_guide.md)
> - [📔 Documentation Principale](./README.md)

Ce guide explique comment utiliser le système de day trading automatisé de Mercurio AI, qui permet d'exécuter un trading algorithmique sur les actions et les crypto-monnaies via l'API Alpaca.

## Vue d'ensemble

Le système de day trading de Mercurio AI comprend :

1. **Collecte de symboles** (`get_all_symbols.py`) - Récupère tous les symboles d'actions et crypto-monnaies disponibles
2. **Day trading sur actions** (`run_stock_daytrader_all.py`) - Exécute des stratégies de trading sur les actions
3. **Day trading sur crypto** (`run_overnight_crypto_trader.py`) - Exécute des stratégies de trading sur les crypto-monnaies
4. **Trading crypto avec stratégies avancées** (`run_strategy_crypto_trader.py`) - Permet de choisir différentes stratégies pour le trading de crypto
5. **Intégration avec les stratégies Mercurio AI** - Utilise toutes les stratégies disponibles (MovingAverage, LSTM, Momentum, etc.)

## Prérequis

- Python 3.8+
- Compte Alpaca (paper ou live)
- Clés API Alpaca configurées dans un fichier `.env`
- Modules Python requis (voir requirements.txt)

## Configuration

Créez un fichier `.env` à la racine du projet avec les informations suivantes :

```
ALPACA_PAPER_KEY=votre_clé_paper
ALPACA_PAPER_SECRET=votre_secret_paper
ALPACA_LIVE_KEY=votre_clé_live
ALPACA_LIVE_SECRET=votre_secret_live
ALPACA_MODE=paper  # ou "live" pour le trading réel
ALPACA_SUBSCRIPTION_LEVEL=1  # 1 pour Basic, 2 pour Pro
```

## Workflow de Trading

### Étape 1 : Récupération des symboles

```bash
python scripts/get_all_symbols.py
```

Ce script :
- Récupère tous les symboles d'actions disponibles via l'API Alpaca
- Récupère tous les symboles de crypto-monnaies disponibles
- Sauvegarde les listes dans des fichiers CSV dans le dossier `data/`
- Génère des métadonnées sur les symboles récupérés

### Étape 2 : Day Trading sur Crypto avec Stratégies Avancées

```bash
python scripts/run_strategy_crypto_trader.py --strategy momentum --use-custom-symbols
```

Ce nouveau script permet de trader des cryptomonnaies avec différentes stratégies avancées et configurations personnalisables.

Options principales :
- `--strategy` : Stratégie à utiliser (`moving_average`, `momentum`, `mean_reversion`, `breakout`, `statistical_arbitrage`, `transformer`)
- `--duration` : Durée de la session (`1h`, `4h`, `8h`, `night` pour 9h)
- `--position-size` : Taille de position en % du portefeuille (ex: `0.02` pour 2%)
- `--stop-loss` : Stop loss en pourcentage (ex: `0.03` pour 3%)
- `--take-profit` : Take profit en pourcentage (ex: `0.06` pour 6%)
- `--use-custom-symbols` : Utiliser la liste personnalisée de symboles

Options spécifiques par stratégie :
- `--fast-ma` / `--slow-ma` : Périodes pour la stratégie moyenne mobile
- `--momentum-lookback` : Période pour la stratégie momentum
- `--mean-reversion-lookback` : Période pour la stratégie mean reversion
- `--breakout-lookback` : Période pour la stratégie breakout
- `--volatility-lookback` : Période pour le calcul de volatilité

Options spécifiques pour la stratégie Transformer (deep learning) :
- `--sequence-length` : Longueur de la séquence d'entrée (défaut: 60)
- `--prediction-horizon` : Horizon de prédiction (défaut: 1)
- `--d-model` : Dimension du modèle (défaut: 64)
- `--nhead` : Nombre de têtes d'attention (défaut: 4)
- `--num-layers` : Nombre de couches (défaut: 2)
- `--dropout` : Taux de dropout (défaut: 0.1)
- `--signal-threshold` : Seuil de signal (défaut: 0.6)
- `--use-gpu` : Utiliser le GPU si disponible
- `--retrain` : Forcer le réentraînement du modèle

### Étape 3 : Day Trading sur Actions

```bash
python scripts/run_stock_daytrader_all.py --strategy all --filter active_assets --max-symbols 20 --duration continuous --use-custom-symbols
```

Options principales :
- `--strategy` : Stratégie à utiliser (`moving_average`, `lstm_predictor`, `transformer`, `msi`, `all`)
- `--filter` : Filtre pour les actions (`active_assets`, `top_volume`, `top_gainers`, etc.)
- `--max-symbols` : Nombre maximum de symboles à trader
- `--position-size` : Taille de position en % du portefeuille (ex: `0.02` pour 2%)
- `--duration` : Type de session (`market_hours`, `extended_hours`, `full_day`, `continuous`)
- `--market-check-interval` : Intervalle en minutes pour vérifier l'état du marché
- `--use-threads` : Utiliser le multithreading pour accélérer le traitement
- `--use-custom-symbols` : Utiliser les symboles des fichiers CSV générés par `get_all_symbols.py`
- `--refresh-symbols` : Exécuter `get_all_symbols.py` avant de démarrer le trading

### Intégration entre les scripts

La nouvelle intégration permet deux flux de travail principaux :

#### Workflow 1 : Exécution en deux étapes

1. Exécuter `get_all_symbols.py` pour récupérer et sauvegarder les symboles
2. Exécuter `run_stock_daytrader_all.py` avec l'option `--use-custom-symbols` pour utiliser les fichiers CSV générés

#### Workflow 2 : Exécution en une étape

Exécuter `run_stock_daytrader_all.py` avec l'option `--refresh-symbols` pour récupérer automatiquement les symboles avant de démarrer le trading.

## Mode Continu (Fonctionnement 24/7)

Pour lancer le système en mode continu (idéal pour un fonctionnement sur une semaine ou plus) :

```bash
python scripts/run_stock_daytrader_all.py --strategy all --filter active_assets --duration continuous --market-check-interval 30 --refresh-symbols
```

En mode continu, le script :
1. Vérifie l'état du marché toutes les 30 minutes (personnalisable avec `--market-check-interval`)
2. Si le marché est ouvert, exécute les stratégies de trading
3. Si le marché est fermé, attend jusqu'à la prochaine ouverture
4. Continue ce cycle indéfiniment jusqu'à ce qu'il soit arrêté manuellement
5. Génère des rapports de performance après chaque session

Pour arrêter proprement le script, utilisez `Ctrl+C`.

## Rapports et Monitoring

Le système génère :

1. **Logs détaillés** - Enregistrés dans des fichiers `stock_daytrader_log_YYYYMMDD_HHMMSS.txt`
2. **Rapports de performance** - Générés à la fin de chaque session dans `stock_trading_report_YYYYMMDD_HHMMSS.txt`
3. **Symboles disponibles** - Sauvegardés dans `data/all_stocks_YYYYMMDD.csv` et `data/all_crypto_YYYYMMDD.csv`

## Stratégies de Trading

Le système supporte plusieurs stratégies :

1. **MovingAverageStrategy** - Stratégie basée sur le croisement de moyennes mobiles
2. **MovingAverageMLStrategy** - Version améliorée avec ML pour les paramètres
3. **LSTMPredictorStrategy** - Prédictions basées sur des réseaux LSTM
4. **TransformerStrategy** - Utilise des modèles transformers pour les prédictions
5. **MSIStrategy** - Market Strength Index, indicateur propriétaire

Vous pouvez utiliser toutes les stratégies simultanément avec `--strategy all`.

## Dépannage

Si vous rencontrez des erreurs :

1. **Vérifiez les clés API** - Assurez-vous que vos clés API sont correctes dans le fichier `.env`
2. **Vérifiez l'état du marché** - Certaines erreurs peuvent survenir si le marché est fermé
3. **Vérifiez les symboles** - Certains symboles peuvent ne pas être tradables
4. **Consultez les logs** - Les fichiers de log contiennent des informations détaillées sur les erreurs

## Exemples d'utilisation

### Trading sur les actions les plus actives pendant les heures de marché
```bash
python scripts/run_stock_daytrader_all.py --strategy moving_average --filter active_assets --duration market_hours
```

### Trading continu sur une liste personnalisée
```bash
python scripts/run_stock_daytrader_all.py --strategy all --use-custom-symbols --duration continuous
```

### Trading sur un grand nombre d'actions avec multithreading
```bash
python scripts/run_stock_daytrader_all.py --strategy all --filter top_volume --max-symbols 50 --use-threads
```

### Mise à jour régulière des symboles (pour un fonctionnement prolongé)
```bash
python scripts/run_stock_daytrader_all.py --strategy all --duration continuous --refresh-symbols --market-check-interval 60
```

### Trading de cryptomonnaies avec la stratégie mean reversion
```bash
python scripts/run_strategy_crypto_trader.py --strategy mean_reversion --use-custom-symbols --position-size 0.01 --stop-loss 0.02 --take-profit 0.05
```

### Trading nocturne de cryptomonnaies avec la stratégie momentum
```bash
python scripts/run_strategy_crypto_trader.py --strategy momentum --duration night --momentum-lookback 15
```

### Trading de cryptomonnaies sur session courte avec stratégie breakout
```bash
python scripts/run_strategy_crypto_trader.py --strategy breakout --duration 1h --breakout-lookback 10 --stop-loss 0.015 --take-profit 0.04
```

### Trading de cryptomonnaies avec modèle Transformer pendant 8 heures
```bash
python scripts/run_strategy_crypto_trader.py --strategy transformer --duration 8h --use-custom-symbols --position-size 0.01
```

### Trading optimisé avec Transformer personnalisé pour marchés volatils (risque modéré)
```bash
python scripts/run_strategy_crypto_trader.py --strategy transformer --duration night --sequence-length 120 --d-model 128 --nhead 8 --num-layers 3 --signal-threshold 0.7 --position-size 0.005 --stop-loss 0.01 --take-profit 0.03 --use-gpu
```

### Trading optimisé avec Transformer personnalisé pour marchés volatils (risque accru)
```bash
python scripts/run_strategy_crypto_trader.py --strategy transformer --duration night --sequence-length 120 --d-model 128 --nhead 8 --num-layers 3 --signal-threshold 0.6 --position-size 0.015 --stop-loss 0.02 --take-profit 0.05 --use-gpu
```
