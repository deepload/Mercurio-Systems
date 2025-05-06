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
5. **Intégration avec les stratégies Mercurio AI** - Utilise toutes les stratégies disponibles (MovingAverage, LSTM, Transformer, MSI, LLM, etc.)

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
ALPACA_SUBSCRIPTION_LEVEL=1  # 1=Standard, 2=Standard+, 3=Premium (optionnel, détecté automatiquement si non spécifié)
```

## Niveaux d'API Alpaca

Mercurio AI prend en charge les différents niveaux d'abonnement Alpaca :

| Niveau | Nom | Description | Fonctionnalités principales |
|--------|-----|-------------|------------------------|
| 1 | Standard | Niveau de base | Données de marché en temps réel, barres de 1 minute |
| 2 | Standard+ | Niveau intermédiaire | Données historiques étendues, barres de 15 secondes |
| 3 | Premium | Niveau avancé | Book d'ordres L2, bars au tick, flux de données amélioré |

Le système détecte automatiquement votre niveau d'API et s'adapte en fonction des fonctionnalités disponibles. Vous pouvez également forcer l'utilisation d'un niveau spécifique avec l'option `--api-level`.

Si vous avez un niveau supérieur, le système utilisera les fonctionnalités avancées disponibles. Si vous forcez l'utilisation d'un niveau supérieur à celui de votre abonnement, le système reviendra automatiquement au niveau disponible.

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
python scripts/run_strategy_crypto_trader.py --strategy transformer --duration 1h --use-custom-symbols --api-level 0
```

Ce script permet de trader des cryptomonnaies avec différentes stratégies avancées et configurations personnalisables.

#### Options principales pour le trading de crypto :

| Option | Description | Valeurs possibles | Défaut |
|--------|-------------|-------------------|--------|
| `--strategy` | Stratégie à utiliser | `moving_average`, `momentum`, `mean_reversion`, `breakout`, `statistical_arbitrage`, `lstm_predictor`, `transformer`, `llm` | `moving_average` |
| `--duration` | Durée de la session | `1h`, `4h`, `8h`, `night` (9h), `continuous` | `1h` |
| `--position-size` | Taille de position en % du portefeuille | 0.01 - 1.0 | 0.02 (2%) |
| `--stop-loss` | Stop loss en pourcentage | 0.01 - 0.20 | 0.03 (3%) |
| `--take-profit` | Take profit en pourcentage | 0.01 - 0.50 | 0.06 (6%) |
| `--use-custom-symbols` | Utiliser la liste personnalisée de symboles | flag (pas de valeur) | non activé |
| `--api-level` | Niveau d'API Alpaca à utiliser | 0 (auto), 1, 2, 3 | 0 (auto-détection) |
| `--max-symbols` | Nombre maximum de cryptos à trader | 1-50 | 5 |
| `--refresh-symbols` | Rafraîchir les symboles disponibles avant de démarrer | flag (pas de valeur) | non activé |

#### Options spécifiques par stratégie :

**Stratégie Moving Average :**
- `--fast-ma` : Période de la moyenne mobile rapide (défaut: 9)
- `--slow-ma` : Période de la moyenne mobile lente (défaut: 21)

**Stratégie Momentum :**
- `--momentum-lookback` : Période pour le calcul du momentum (défaut: 14)
- `--momentum-threshold` : Seuil de déclenchement (défaut: 0.5)

**Stratégie Mean Reversion :**
- `--mean-reversion-lookback` : Période pour le calcul de la moyenne (défaut: 20)
- `--mean-reversion-std` : Nombre d'écarts-types pour déclencher un signal (défaut: 2.0)

**Stratégie Breakout :**
- `--breakout-lookback` : Période pour calculer les niveaux de support/résistance (défaut: 20)
- `--breakout-threshold` : Pourcentage de dépassement pour déclencher un signal (défaut: 0.02)

**Stratégie LSTM :**
- `--lookback-window` : Nombre de périodes historiques à utiliser (défaut: 60)
- `--retrain` : Force le réentraînement du modèle (flag)
- `--model-confidence` : Seuil de confiance pour déclencher un signal (défaut: 0.7)

**Stratégie Transformer :**
- `--sequence-length` : Longueur de la séquence d'entrée (défaut: 60)
- `--prediction-horizon` : Horizon de prédiction (défaut: 1)
- `--d-model` : Dimension du modèle (défaut: 64)
- `--nhead` : Nombre de têtes d'attention (défaut: 4)
- `--num-layers` : Nombre de couches (défaut: 2)
- `--dropout` : Taux de dropout (défaut: 0.1)
- `--signal-threshold` : Seuil de signal (défaut: 0.6)
- `--use-gpu` : Utiliser le GPU si disponible (flag)
- `--retrain` : Forcer le réentraînement du modèle (flag)

**Stratégie LLM :**
- `--model-path` : Chemin vers le modèle LLM local (défaut: models/llm/crypto_sentiment)
- `--use-api` : Utiliser une API externe plutôt qu'un modèle local (flag)
- `--api-key` : Clé API pour le service externe (si --use-api)
- `--sentiment-threshold` : Seuil de sentiment pour déclencher un signal (défaut: 0.6)

### Étape 3 : Day Trading sur Actions

```bash
python scripts/run_stock_daytrader_all.py --strategy all --filter active_assets --max-symbols 20 --duration continuous --use-custom-symbols --api-level 0
```

#### Options principales pour le trading d'actions :

| Option | Description | Valeurs possibles | Défaut |
|--------|-------------|-------------------|--------|
| `--strategy` | Stratégie à utiliser | `moving_average`, `moving_average_ml`, `lstm_predictor`, `transformer`, `msi`, `llm`, `all` | `moving_average` |
| `--filter` | Filtre pour les actions | `active_assets`, `top_volume`, `top_gainers`, `tech_stocks`, `finance_stocks`, `health_stocks`, `sp500`, `nasdaq100` | `active_assets` |
| `--max-symbols` | Nombre maximum de symboles à trader | 1-100 | 10 |
| `--position-size` | Taille de position en % du portefeuille | 0.01 - 1.0 | 0.02 (2%) |
| `--stop-loss` | Stop loss en pourcentage | 0.01 - 0.20 | 0.02 (2%) |
| `--take-profit` | Take profit en pourcentage | 0.01 - 0.50 | 0.04 (4%) |
| `--duration` | Type de session | `market_hours`, `extended_hours`, `full_day`, `continuous` | `market_hours` |
| `--market-check-interval` | Intervalle en minutes pour vérifier l'état du marché | 5-120 | 30 |
| `--use-threads` | Utiliser le multithreading | flag (pas de valeur) | non activé |
| `--use-custom-symbols` | Utiliser les symboles des fichiers CSV | flag (pas de valeur) | non activé |
| `--refresh-symbols` | Exécuter `get_all_symbols.py` avant de démarrer | flag (pas de valeur) | non activé |
| `--api-level` | Niveau d'API Alpaca à utiliser | 0 (auto), 1, 2, 3 | 0 (auto-détection) |
| `--auto-retrain` | Réentraîner automatiquement les modèles | flag (pas de valeur) | non activé |
| `--retrain-interval` | Intervalle en heures entre les réentraînements | 1-24 | 6 |
| `--retrain-symbols` | Nombre de symboles pour le réentraînement | 1-50 | 10 |

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

## Impact des Niveaux d'API Alpaca sur les Performances

Le niveau d'API Alpaca que vous utilisez peut avoir un impact significatif sur les performances de vos stratégies de trading :

### Niveau 1 (Standard) :
- **Données de marché :** Données en temps réel avec barres de 1 minute
- **Stratégies recommandées :** Moving Average, Mean Reversion, Momentum
- **Limites :** Pas idéal pour les stratégies HFT ou scalping
- **Performances attendues :** Bonnes pour le day trading classique et les stratégies à moyen terme

### Niveau 2 (Standard+) :
- **Données de marché :** Barres de 15 secondes, données historiques étendues
- **Stratégies recommandées :** Toutes les stratégies du niveau 1 + LSTM, Transformer avec délai réduit
- **Améliorations :** Signaux plus précis, meilleure réactivité aux mouvements de marché
- **Performances attendues :** Amélioration de 10-20% par rapport au niveau 1 sur la plupart des stratégies

### Niveau 3 (Premium) :
- **Données de marché :** Book d'ordres L2, barres au tick, données de haute fréquence
- **Stratégies recommandées :** Toutes les stratégies + analyses avancées de microstructure
- **Améliorations :** Signaux de trading de haute précision, détection des anomalies de marché
- **Performances attendues :** Potentiel d'amélioration de 20-40% sur les stratégies avancées

### Recommandations par stratégie :

| Stratégie | Niveau minimum recommandé | Remarques |
|-------------|--------------------------|----------|
| Moving Average | 1 | Fonctionne bien avec tous les niveaux |
| Momentum | 1 | Améliorations marginales aux niveaux supérieurs |
| Mean Reversion | 2 | Meilleure précision pour les points d'inversion |
| Breakout | 2 | Détection plus rapide des ruptures de niveaux |
| LSTM | 2 | Entraînement amélioré avec données plus granulaires |
| Transformer | 2 | Meilleure performance avec données plus fréquentes |
| LLM | 1 | Peu dépendant de la granularité des données |
| MSI | 3 | Optimisé pour utiliser toutes les données L2 |

> **Note :** Si vous spécifiez un niveau d'API mais que votre abonnement Alpaca n'y donne pas accès, le système reviendra automatiquement au niveau le plus élevé disponible. Par exemple, si vous spécifiez `--api-level 3` mais que vous n'avez qu'un abonnement Standard, le système utilisera le niveau 1.

## Dépannage

Si vous rencontrez des erreurs :

1. **Vérifiez les clés API** - Assurez-vous que vos clés API sont correctes dans le fichier `.env`
2. **Vérifiez l'état du marché** - Certaines erreurs peuvent survenir si le marché est fermé
3. **Vérifiez les symboles** - Certains symboles peuvent ne pas être tradables
4. **Consultez les logs** - Les fichiers de log contiennent des informations détaillées sur les erreurs
5. **Problèmes de niveau d'API** - Si vous rencontrez des erreurs liées aux fonctionnalités premium :
   - Vérifiez votre niveau d'abonnement Alpaca actuel
   - Essayez avec `--api-level 0` pour activer la détection automatique
   - Consultez les logs pour voir quel niveau a été effectivement détecté

## Exemples d'utilisation

### Trading d'actions avec détection automatique du niveau d'API
```bash
python scripts/run_stock_daytrader_all.py --strategy moving_average --filter active_assets --duration market_hours --api-level 0
```

### Trading d'actions avec niveau d'API spécifique
```bash
python scripts/run_stock_daytrader_all.py --strategy all --filter active_assets --api-level 3
```

### Trading continu sur une liste personnalisée
```bash
python scripts/run_stock_daytrader_all.py --strategy all --use-custom-symbols --duration continuous
```

### Trading sur un grand nombre d'actions avec multithreading
```bash
python scripts/run_stock_daytrader_all.py --strategy all --filter top_volume --max-symbols 50 --use-threads
```

### Mise à jour régulière des symboles avec réentraînement automatique des modèles
```bash
python scripts/run_stock_daytrader_all.py --strategy all --duration continuous --refresh-symbols --market-check-interval 60 --auto-retrain --retrain-interval 4
```

### Trading d'actions avec stratégie LSTM et niveau d'API 2
```bash
python scripts/run_stock_daytrader_all.py --strategy lstm_predictor --filter top_gainers --api-level 2 --max-symbols 15
```

### Trading d'actions avec stratégie Transformer et auto-retrain
```bash
python scripts/run_stock_daytrader_all.py --strategy transformer --auto-retrain --retrain-interval 8 --api-level 0
```

### Trading de cryptomonnaies avec la stratégie Transformer et détection automatique du niveau d'API
```bash
python scripts/run_strategy_crypto_trader.py --strategy transformer --duration 4h --use-custom-symbols --api-level 0
```

### Trading de cryptomonnaies avec LLM en mode nuit
```bash
python scripts/run_strategy_crypto_trader.py --strategy llm --duration night --sentiment-threshold 0.65 --api-level 2
```

### Trading nocturne de cryptomonnaies avec la stratégie momentum
```bash
python scripts/run_strategy_crypto_trader.py --strategy momentum --duration night --momentum-lookback 15 --api-level 1
```

### Trading de cryptomonnaies sur session courte avec stratégie breakout
```bash
python scripts/run_strategy_crypto_trader.py --strategy breakout --duration 1h --breakout-lookback 10 --stop-loss 0.015 --take-profit 0.04
```

### Trading de cryptomonnaies avec LSTM et réentraînement forcé
```bash
python scripts/run_strategy_crypto_trader.py --strategy lstm_predictor --retrain --model-confidence 0.75 --api-level 3
```

### Trading de cryptomonnaies sur des paires spécifiques
```bash
python scripts/run_strategy_crypto_trader.py --strategy moving_average --symbols BTC/USD,ETH/USD,SOL/USD --api-level 0
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
