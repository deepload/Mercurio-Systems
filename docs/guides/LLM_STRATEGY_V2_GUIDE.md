# Guide de la Stratégie LLMStrategyV2

## Introduction

LLMStrategyV2 est une stratégie de trading avancée qui combine l'analyse technique traditionnelle avec l'analyse de sentiment basée sur des modèles de langage (LLM). Cette stratégie est conçue pour tirer parti des dernières avancées en intelligence artificielle pour améliorer les décisions de trading.

## Caractéristiques Principales

- **Analyse Hybride** : Combine des indicateurs techniques classiques avec l'analyse de sentiment générée par LLM
- **Pondération Ajustable** : Permet de définir l'importance relative de l'analyse technique vs. l'analyse de sentiment
- **Données Web Réelles** : Utilise l'EnhancedWebSentimentAgent pour garantir des données web réelles même en mode démo
- **Support Multimodèle** : Compatible avec divers modèles LLM (Mistral, Llama, OpenAI, etc.)
- **Backtesting Intégré** : Fonctionnalités d'évaluation des performances sur données historiques
- **Mode Démo Amélioré** : Fonctionne sans clé API réelle mais collecte quand même des données web réelles

## Prérequis

- Python 3.8+
- Mercurio AI installé et configuré
- Connexion Internet (sauf en mode local ou démo)
- Clé API pour les modèles LLM (optionnelle en mode démo)

## Installation et Configuration

### Configuration du fichier .env

Ajoutez les variables suivantes à votre fichier `.env` dans le répertoire racine:

```
LLM_API_KEY=votre_clé_api_ici
# Ou utilisez "demo_mode" pour les tests
LLM_API_KEY=demo_mode
```

### Modèles Supportés

LLMStrategyV2 prend en charge plusieurs types de modèles:

1. **Modèles Hugging Face** (par défaut): "mistralai/Mixtral-8x7B-Instruct-v0.1", etc.
2. **Modèles locaux**: Llama, Mistral, etc. (nécessite plus de ressources)
3. **APIs OpenAI**: GPT-3.5-Turbo, GPT-4, etc.

## Utilisation

### Utilisation de Base

```bash
python scripts/run_strategy_crypto_trader.py --strategy llm_v2
```

Cette commande lance la stratégie avec les paramètres par défaut en mode démo, mais avec l'EnhancedWebSentimentAgent qui collecte des données web réelles.

### Options Avancées

```bash
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 \
  --model-name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
  --api-key votre_clé_api \
  --sentiment-weight 0.8 \
  --min-confidence 0.7 \
  --news-lookback 48 \
  --position-size 0.02 \
  --stop-loss 0.03 \
  --take-profit 0.06 \
  --duration 24h \
  --use-env-symbols
```

Cette commande utilise le modèle Mixtral avec un poids élevé pour l'analyse de sentiment (0.8), un seuil de confiance de 0.7, et analyse les actualités des 48 dernières heures. La stratégie s'exécutera pendant 24 heures et utilisera les symboles définis dans le fichier .env. Notez que le paramètre `--duration` définit uniquement la durée d'exécution du script et n'affecte pas les calculs de probabilité ou la confiance dans les décisions de trading.

### Support pour Trading d'Actions

La stratégie est également compatible avec le trading d'actions:

```bash
python scripts/run_stock_daytrader_all.py --strategy llm_v2 \
  --symbols AAPL MSFT GOOGL \
  --sentiment-weight 0.5
```

## Paramètres Personnalisables

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `model_name` | Nom du modèle LLM principal | "mistralai/Mixtral-8x7B-Instruct-v0.1" |
| `sentiment_model_name` | Nom du modèle pour l'analyse de sentiment | Identique à `model_name` |
| `use_local_model` | Utiliser un modèle local | `False` |
| `local_model_path` | Chemin vers le modèle local | `None` |
| `api_key` | Clé API pour le service LLM | Valeur de `LLM_API_KEY` |
| `use_web_sentiment` | Activer l'analyse de sentiment web | `True` |
| `sentiment_weight` | Poids de l'analyse de sentiment (0.0 à 1.0) | `0.5` |
| `min_confidence` | Seuil minimal de confiance pour les signaux | `0.65` |
| `technical_indicators` | Liste d'indicateurs techniques | `["macd", "rsi", "ema", "bollinger"]` |
| `news_lookback_hours` | Heures d'actualités à analyser | `24` |
| `duration` | Durée d'exécution du script (ex: 1h, 4h, 8h, 24h, continuous) | `"continuous"` |

## EnhancedWebSentimentAgent

La stratégie LLMStrategyV2 intègre désormais l'EnhancedWebSentimentAgent, une amélioration majeure qui permet de garantir la collecte de données web réelles même en mode démo.

### Fonctionnalités principales

- **Collecte de données réelles garantie** : Ignore le réglage demo_mode pour les récupérations web
- **Sources multiples** : Analyse des données de X (Twitter), Reddit, LinkedIn, Coindesk, etc.
- **Analyse de sentiment détaillée** : Évalue le sentiment global du marché avec une plus grande précision
- **Mécanisme de secours** : Retour automatique à l'agent standard si nécessaire

### Utilisation

L'agent est automatiquement utilisé par LLMStrategyV2. Les paramètres `sentiment_weight` et `min_confidence` vous permettent de contrôler son influence sur les décisions de trading.

## Mode Démo Amélioré

Le mode démo est activé automatiquement lorsque `api_key` est défini sur `"demo_mode"`. Dans ce mode:

- Pas besoin de clé API réelle pour les modèles LLM
- L'EnhancedWebSentimentAgent collecte tout de même des données web réelles
- L'analyse technique fonctionne normalement sur des données de marché réelles
- Idéal pour tester la stratégie sans frais d'API tout en maintenant une bonne qualité d'analyse

## Backtesting

Pour exécuter un backtest de la stratégie:

```bash
python scripts/backtest_strategy.py --strategy llm_v2 \
  --symbol BTC-USD \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --api-key demo_mode
```

## Conseils d'Optimisation

1. **Ajustement du poids de sentiment**: 
   - Marché volatil/incertain: augmentez `sentiment_weight` (0.6-0.8)
   - Marché stable/prévisible: diminuez `sentiment_weight` (0.3-0.5)

2. **Selection des modèles**:
   - Mistral Mixtral-8x7B offre un bon équilibre performance/coût
   - Pour une performance maximale: GPT-4 ou Claude 3 Opus
   - Pour des tests rapides: modèles plus petits ou mode démo

3. **Paramètres de risque**:
   - Crypto volatiles: `position_size` plus faible (0.01-0.02), `stop_loss` plus serré
   - Actions plus stables: `position_size` plus élevé (0.03-0.05), `stop_loss` plus large

## Exemples de Cas d'Utilisation

1. **Trading de Crypto Intraday avec Haute Sensibilité au Sentiment**:
```bash
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --sentiment-weight 0.8 --min-confidence 0.7 --news-lookback 12 --position-size 0.01 --use-env-symbols
```

2. **Trading de Crypto sur 24h avec Données Web Augmentées**:
```bash
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --sentiment-weight 0.7 --min-confidence 0.6 --news-lookback 48 --position-size 0.02 --stop-loss 0.03 --take-profit 0.06 --duration 24h --use-env-symbols
```

3. **Trading d'Actions à Moyen Terme**:
```bash
python scripts/run_stock_daytrader_all.py --strategy llm_v2 --sentiment-weight 0.5 --min-confidence 0.7 --duration continuous
```

4. **Conversion USD en USDT pour Trading de Paires USDT**:
```bash
python scripts/convert_usd_to_usdt.py --amount 100
```

5. **Backtesting avec Configuration Personnalisée**:
```bash
python scripts/backtest_strategy.py --strategy llm_v2 --sentiment-weight 0.6 --technical-indicators macd,rsi,ema --api-key demo_mode
```

## Dépannage

1. **"Module not found" Error**:
   - Vérifiez que tous les modules requis sont installés: `pip install -r requirements.txt`

2. **Erreurs d'API**:
   - Vérifiez que votre clé API est valide et correctement configurée
   - Utilisez `--api-key demo_mode` pour les tests
   - Pour les erreurs Alpaca liées à "unexpected keyword argument 'data_url'", utilisez la dernière version de l'API Alpaca

3. **Erreur "insufficient balance for USDT"**:
   - Vous n'avez pas de USDT dans votre compte
   - Utilisez le script `scripts/convert_usd_to_usdt.py --amount 100` pour convertir des USD en USDT
   - Alternativement, concentrez-vous sur les paires USD (BTC/USD, ETH/USD, etc.)

4. **Performances sous-optimales**:
   - Ajustez `sentiment_weight` en fonction des conditions de marché
   - Augmentez `min_confidence` pour des signaux plus conservateurs
   - Augmentez `news_lookback_hours` pour inclure plus de données d'actualités
   - Analysez les logs pour identifier les sources potentielles d'erreur

## Remarques importantes

- Lors de l'exécution en mode live, assurez-vous d'avoir suffisamment de fonds disponibles dans votre compte
- L'option `--use-env-symbols` permet d'utiliser les symboles définis dans le fichier .env
- Pour des mises à jour régulières sur les performances de trading, vérifiez les logs et le terminal
- Le paramètre `--duration` contrôle uniquement la durée d'exécution du script de trading (combien de temps le programme tournera avant de s'arrêter automatiquement) et n'affecte pas les calculs de probabilité ou les décisions de trading de l'algorithme
