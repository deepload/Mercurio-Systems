# Guide de Trading Haute Fréquence (HFT) avec Mercurio AI

## Introduction

Le Trading Haute Fréquence (HFT) est une méthode de trading algorithmique qui exécute un grand nombre d'ordres à des vitesses extrêmement élevées. Le script `run_hft_trader.py` de Mercurio AI vous permet d'implémenter des stratégies HFT pour le trading d'actions et de crypto-monnaies.

Ce guide vous expliquera comment utiliser efficacement ce script, configurer différentes stratégies et optimiser vos résultats de trading.

## Table des matières

1. [Prérequis](#prérequis)
2. [Modes d'exécution](#modes-dexécution)
3. [Types d'actifs supportés](#types-dactifs-supportés)
4. [Stratégies disponibles](#stratégies-disponibles)
5. [Options de configuration](#options-de-configuration)
6. [Exemples d'utilisation](#exemples-dutilisation)
7. [Configuration avancée](#configuration-avancée)
8. [Dépannage](#dépannage)
9. [Bonnes pratiques](#bonnes-pratiques)

## Prérequis

Avant d'utiliser le script de trading haute fréquence, assurez-vous d'avoir :

- Python 3.7+ installé
- Un compte Alpaca (Paper ou Live)
- Les variables d'environnement configurées :
  - `ALPACA_PAPER_KEY` et `ALPACA_PAPER_SECRET` pour le paper trading
  - `ALPACA_LIVE_KEY` et `ALPACA_LIVE_SECRET` pour le trading réel
  - `ALPACA_MODE` (réglé sur "paper" ou "live")

## Modes d'exécution

Le script `run_hft_trader.py` peut être exécuté en deux modes principaux :

### Mode Paper Trading (simulation)

C'est le mode par défaut et recommandé pour les tests. Aucun ordre réel n'est exécuté, mais tout se comporte comme si vous tradiez sur le marché réel.

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto
```

### Mode Live Trading (réel)

Dans ce mode, des ordres réels sont passés avec de l'argent réel. À utiliser avec prudence !

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --live
```

## Types d'actifs supportés

Le script prend en charge deux types d'actifs :

### Actions (stocks)

```bash
python scripts/run_hft_trader.py --asset-type stock --strategy moving_average
```

Par défaut, le script sélectionnera les actions les plus liquides du marché américain.

### Crypto-monnaies

```bash
python scripts/run_hft_trader.py --asset-type crypto --strategy moving_average
```

Par défaut, le script utilisera les principales paires crypto/USD disponibles sur Alpaca.

## Stratégies disponibles

Le script offre plusieurs stratégies de trading préintégrées :

### 1. Moyenne Mobile (Moving Average)

Stratégie basée sur le croisement de moyennes mobiles à court et long terme.

```bash
python scripts/run_hft_trader.py --strategy moving_average --fast-ma 3 --slow-ma 10
```

### 2. Mean Reversion

Stratégie qui anticipe un retour à la moyenne après une déviation significative.

```bash
python scripts/run_hft_trader.py --strategy mean_reversion --mean-reversion-zscore 2.0
```

### 3. Momentum

Stratégie qui suit la tendance en se basant sur la dynamique récente des prix.

```bash
python scripts/run_hft_trader.py --strategy momentum --momentum-lookback 5
```

### 4. LSTM Predictor

Stratégie utilisant des réseaux de neurones récurrents (LSTM) pour prédire les mouvements de prix.

```bash
python scripts/run_hft_trader.py --strategy lstm_predictor
```

### 5. Transformer Strategy

Stratégie basée sur l'architecture Transformer pour une analyse séquentielle avancée.

```bash
python scripts/run_hft_trader.py --strategy transformer
```

### 6. Multi-Source Intelligence (MSI)

Stratégie avancée combinant plusieurs sources de données et d'analyses.

```bash
python scripts/run_hft_trader.py --strategy msi
```

## Options de configuration

Le script offre de nombreuses options pour personnaliser votre stratégie de trading :

### Options générales

- `--live` : Active le mode live trading (utilise de l'argent réel)
- `--debug` : Affiche des informations de débogage détaillées
- `--verbose` : Affiche des messages détaillés sur l'exécution
- `--duration` : Durée maximale de la session (1h, 4h, 8h, night=9h, continuous)
- `--log-file` : Chemin vers un fichier de log spécifique
- `--no-stream` : Utilise des requêtes régulières au lieu de WebSockets
- `--backtest-mode` : Simule les ordres sans les passer réellement

### Sélection des actifs

- `--symbols` : Liste spécifique de symboles à trader
- `--use-custom-symbols` : Utilise une liste prédéfinie de symboles
- `--max-positions` : Nombre maximum de positions simultanées

### Paramètres de risque

- `--position-size` : Taille de position en pourcentage du portefeuille (default: 0.01 = 1%)
- `--stop-loss` : Stop loss en pourcentage (default: 0.002 = 0.2%)
- `--take-profit` : Take profit en pourcentage (default: 0.005 = 0.5%)

### Paramètres de stratégie

- `--fast-ma` : Période de la moyenne mobile rapide (pour stratégie MA)
- `--slow-ma` : Période de la moyenne mobile lente (pour stratégie MA)
- `--momentum-lookback` : Période de lookback pour la stratégie Momentum
- `--mean-reversion-zscore` : Score Z pour la stratégie Mean Reversion

### Options API Alpaca

- `--api-level` : Niveau d'API Alpaca à utiliser (1=basique, 2=standard+, 3=premium)
- `--market-check-interval` : Intervalle en secondes pour vérifier l'état du marché

## Exemples d'utilisation

Voici 20 exemples complets couvrant différents cas d'utilisation du script HFT :

### 1. Trading de base avec moyenne mobile (Crypto)

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --fast-ma 3 --slow-ma 10 --position-size 0.02 --stop-loss 0.003 --take-profit 0.008
```

Cette commande :
- Utilise la stratégie classique de moyenne mobile
- Cible les crypto-monnaies populaires
- Configure une moyenne mobile rapide sur 3 périodes et lente sur 10 périodes
- Alloue 2% du portefeuille par position
- Place un stop loss à 0.3% et un take profit à 0.8%

### 2. Trading d'actions avec stratégie momentum

```bash
python scripts/run_hft_trader.py --strategy momentum --asset-type stock --momentum-lookback 5 --position-size 0.01 --max-positions 3 --market-check-interval 5
```

Cette commande :
- Utilise la stratégie momentum sur les actions
- Configure une période de lookback de 5 unités
- Alloue prudemment 1% du portefeuille par position
- Limite à 3 positions simultanées
- Vérifie le marché toutes les 5 secondes

### 3. Trading de crypto personnalisé avec symboles spécifiques

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --use-custom-symbols --symbols BTCUSD ETHUSD SOLUSD AVAXUSD MATICUSD --no-stream --market-check-interval 3
```

Cette commande :
- Utilise une liste personnalisée de crypto-monnaies populaires
- Désactive les WebSockets et utilise le mode polling
- Vérifie l'état du marché toutes les 3 secondes

### 4. Backtesting avec stratégie LSTM Predictor

```bash
python scripts/run_hft_trader.py --strategy lstm_predictor --asset-type crypto --backtest-mode --duration 4h --symbols BTCUSD ETHUSD
```

Cette commande :
- Utilise la stratégie avancée LSTM Predictor sur BTC et ETH
- Simule les décisions sans passer d'ordres réels
- Fonctionne pendant 4 heures puis s'arrête

### 5. Trading avec Multi-Source Intelligence (MSI)

```bash
python scripts/run_hft_trader.py --strategy msi --asset-type crypto --position-size 0.03 --max-positions 10 --market-check-interval 1 --api-level 3 --debug
```

Cette commande :
- Utilise la stratégie MSI qui combine plusieurs sources de données
- Active le mode de débogage pour un suivi détaillé
- Utilise l'API Alpaca de niveau premium (niveau 3)

### 6. Trading de crypto à très haute fréquence

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --fast-ma 2 --slow-ma 5 --position-size 0.01 --stop-loss 0.001 --take-profit 0.003 --market-check-interval 1 --max-positions 5 --api-level 3
```

Cette commande :
- Configure un système ultra-rapide avec des moyennes mobiles courtes
- Utilise des stops et takes très serrés (0.1% et 0.3%)
- Vérifie le marché chaque seconde
- Nécessite l'API premium pour les données en temps réel

### 7. Trading d'actions à forte capitalisation uniquement

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type stock --use-custom-symbols --symbols AAPL MSFT GOOGL AMZN TSLA --position-size 0.05 --stop-loss 0.005 --take-profit 0.01
```

Cette commande :
- Se concentre sur les 5 plus grandes entreprises technologiques
- Utilise des positions plus importantes (5%) sur ces actions stables
- Définit des seuils de stop/take plus larges adaptés aux actions

### 8. Trading de crypto avec stratégie Transformer

```bash
python scripts/run_hft_trader.py --strategy transformer --asset-type crypto --position-size 0.02 --max-positions 8 --api-level 3 --use-custom-symbols --symbols BTCUSD ETHUSD SOLUSD AVAXUSD
```

Cette commande :
- Utilise la stratégie basée sur l'architecture Transformer
- Cible 4 crypto-monnaies principales
- Utilise l'API niveau 3 pour des données optimales

### 9. Trading en mode conservation d'énergie

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --fast-ma 5 --slow-ma 15 --position-size 0.01 --stop-loss 0.005 --take-profit 0.015 --market-check-interval 10 --no-stream
```

Cette commande :
- Utilise un intervalle de vérification plus long (10 secondes)
- Désactive les WebSockets pour économiser de la bande passante
- Définit des paramètres plus conservateurs

### 10. Trading de crypto avec LSTM et configuration d'analyse

```bash
python scripts/run_hft_trader.py --strategy lstm_predictor --asset-type crypto --position-size 0.02 --stop-loss 0.003 --take-profit 0.009 --max-positions 7 --debug --verbose --log-file lstm_trading_log.txt
```

Cette commande :
- Utilise le modèle LSTM préentraîné pour les prédictions
- Active les modes debug et verbose pour une analyse approfondie
- Enregistre toutes les opérations dans un fichier de log spécifique

### 11. Trading d'actions pendant une journée spécifique

```bash
python scripts/run_hft_trader.py --strategy momentum --asset-type stock --duration 8h --position-size 0.02 --stop-loss 0.004 --take-profit 0.01 --market-open-hour 9 --market-close-hour 17
```

Cette commande :
- Fonctionne pendant exactement 8 heures
- Spécifie explicitement les heures de marché
- Adapte les paramètres pour le trading d'actions

### 12. Trading avec modèle LLM pour analyse de sentiment

```bash
python scripts/run_hft_trader.py --strategy llm --asset-type crypto --use-custom-symbols --symbols BTCUSD --position-size 0.05 --sentiment-threshold 0.7 --api-level 3 --market-check-interval 60
```

Cette commande :
- Utilise un modèle de langage pour analyser le sentiment du marché
- Se concentre uniquement sur Bitcoin
- Nécessite un score de sentiment élevé (0.7+) pour trader
- Vérifie les données de sentiment toutes les minutes

### 13. Trading d'urgence pendant volatilité

```bash
python scripts/run_hft_trader.py --strategy mean_reversion --asset-type crypto --mean-reversion-zscore 3.0 --position-size 0.005 --stop-loss 0.01 --take-profit 0.03 --max-positions 2 --volatility-filter-threshold 2.0
```

Cette commande :
- Utilise une stratégie mean reversion avec un Z-score élevé
- Prend des positions très petites (0.5%) mais avec des ratios risque/récompense élevés
- Limite à 2 positions maximum pendant la volatilité
- Utilise un filtre de volatilité exigeant

### 14. Trading combiné crypto et actions

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type mixed --use-custom-symbols --symbols BTCUSD ETHUSD AAPL MSFT --position-size 0.02 --api-level 3 --max-positions 4
```

Cette commande :
- Utilise un type d'actif mixte pour trader à la fois crypto et actions
- Permet de diversifier les actifs dans un seul script
- Limite à une position par actif spécifié

### 15. Trading nocturne automatisé

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --duration night --fast-ma 5 --slow-ma 20 --position-size 0.01 --night-mode-risk-reduction 0.5
```

Cette commande :
- Configure le script pour fonctionner pendant la nuit (9 heures)
- Utilise des paramètres plus conservateurs pour le trading de nuit
- Réduit automatiquement le risque de 50% pendant cette période

### 16. Trading avec Transformer et données externes

```bash
python scripts/run_hft_trader.py --strategy transformer --asset-type crypto --use-custom-symbols --symbols BTCUSD ETHUSD --position-size 0.03 --use-external-data --external-data-path data/market_sentiment.csv --api-level 3
```

Cette commande :
- Intègre des données externes de sentiment du marché
- Limite l'analyse aux deux principales crypto-monnaies
- Augmente légèrement la taille des positions (3%)

### 17. Mode débogage complet pour analyse technique

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --debug --verbose --log-file debug_analysis.log --generate-charts --save-trades-json --market-check-interval 10 --duration 2h
```

Cette commande :
- Active tous les modes de débogage disponibles
- Génère des graphiques pour l'analyse visuelle
- Sauvegarde toutes les transactions au format JSON
- Fonctionne pendant 2 heures pour recueillir des données d'analyse

### 18. Trading de précision pour les altcoins

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --use-custom-symbols --symbols DOGEUSD SHIBUSD NEARUSD --position-size 0.02 --crypto-precision-fix --round-quantities --market-check-interval 3
```

Cette commande :
- Cible des altcoins à plus petite capitalisation
- Active les corrections de précision pour gérer les petites valeurs
- Arrondit automatiquement les quantités pour éviter les erreurs

### 19. Trading intégré avec entraînement de modèle

```bash
python scripts/run_integrated_trader.py --strategy lstm_predictor --asset-type crypto --duration continuous --refresh-symbols --auto-training --training-interval 4h
```

Cette commande :
- Utilise le script intégré qui alterne entre trading et entraînement
- Réentraîne le modèle LSTM toutes les 4 heures
- Rafraîchit automatiquement la liste des symboles les plus pertinents
- Fonctionne en continu jusqu'à interruption manuelle

### 20. Configuration super-sécurisée pour débutants

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --use-custom-symbols --symbols BTCUSD --position-size 0.005 --stop-loss 0.005 --take-profit 0.015 --single-trade-mode --paper --max-daily-loss 2.0 --simulation-mode
```

Cette commande :
- Limite au trading d'une seule crypto (BTC)
- Utilise des positions minuscules (0.5%)
- Ratio risque/récompense favorable (1:3)
- Active le mode trading unique (une position à la fois)
- Force le mode paper trading
- Définit une limite de perte quotidienne maximale de 2%
- Active un mode de simulation complet

## Configuration avancée

### Fichiers de configuration JSON

Pour une configuration plus avancée, vous pouvez créer un fichier JSON et le passer au script :

```json
{
  "strategy": "moving_average",
  "asset_type": "crypto",
  "symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],
  "params": {
    "fast_ma": 3,
    "slow_ma": 10
  },
  "risk_management": {
    "position_size": 0.02,
    "stop_loss": 0.003,
    "take_profit": 0.008,
    "max_positions": 5
  },
  "execution": {
    "api_level": 3,
    "market_check_interval": 2,
    "use_websockets": true
  }
}
```

Utilisez-le avec la commande :

```bash
python scripts/run_hft_trader.py --config path/to/your/config.json
```

### Créer une liste de symboles personnalisée

Vous pouvez créer un fichier CSV avec vos symboles préférés :

```
BTCUSD
ETHUSD
SOLUSD
AVAXUSD
NEARUSD
```

Et utiliser l'option `--custom-symbols-file` :

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --use-custom-symbols --custom-symbols-file path/to/your/symbols.csv
```

## Dépannage

### Problèmes de connexion WebSocket

Si vous rencontrez des erreurs de WebSocket, utilisez l'option `--no-stream` :

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --no-stream
```

### Problèmes avec les données historiques

Si le chargement des données historiques échoue, vous pouvez spécifier une période plus courte :

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --max-historical-bars 100
```

### Erreurs d'autorisation API

Vérifiez que vos variables d'environnement sont correctement configurées :

```bash
export ALPACA_PAPER_KEY="votre-clé-api"
export ALPACA_PAPER_SECRET="votre-secret-api"
export ALPACA_MODE="paper"
```

### Problèmes de précision sur les crypto-monnaies

Pour les crypto-monnaies à petite valeur ou avec des problèmes de précision, utilisez :

```bash
python scripts/run_hft_trader.py --strategy moving_average --asset-type crypto --use-custom-symbols --symbols BTCUSD ETHUSD --crypto-precision-fix
```

## Bonnes pratiques

1. **Commencez toujours en mode paper trading** pour tester vos stratégies.
2. **Utilisez des tailles de position conservatrices** (1-2% maximum).
3. **Surveillez régulièrement les performances** via les logs et rapports générés.
4. **Testez différentes combinaisons de paramètres** pour trouver la stratégie optimale.
5. **Créez des sauvegardes** de vos configurations qui fonctionnent bien.
6. **Démarrez avec des actifs liquides** pour minimiser les problèmes d'exécution d'ordres.
7. **Utilisez le mode `--debug`** pour comprendre les décisions de trading.
8. **Consultez les fichiers de rapport** générés dans le dossier racine.

## Conclusion

Le script `run_hft_trader.py` offre une plateforme puissante et flexible pour le trading haute fréquence, adaptable à différents marchés et stratégies. En utilisant les options et configurations décrites dans ce guide, vous pouvez développer et affiner votre propre approche de trading algorithmique.

Pour plus d'informations sur les autres scripts et fonctionnalités de Mercurio AI, consultez les documents associés dans le dossier `/docs/`.
