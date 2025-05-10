# Guide de la Stratégie LLMStrategyV2

## Introduction

LLMStrategyV2 est une stratégie de trading avancée qui combine l'analyse technique traditionnelle avec l'analyse de sentiment basée sur des modèles de langage (LLM). Cette stratégie est conçue pour tirer parti des dernières avancées en intelligence artificielle pour améliorer les décisions de trading.

## Caractéristiques Principales

- **Analyse Hybride** : Combine des indicateurs techniques classiques avec l'analyse de sentiment générée par LLM
- **Pondération Ajustable** : Permet de définir l'importance relative de l'analyse technique vs. l'analyse de sentiment
- **Mode Démo** : Fonctionne sans clé API réelle pour les tests et la démonstration
- **Support Multimodèle** : Compatible avec divers modèles LLM (Mistral, Llama, OpenAI, etc.)
- **Backtesting Intégré** : Fonctionnalités d'évaluation des performances sur données historiques

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

Cette commande lance la stratégie avec les paramètres par défaut en mode démo.

### Options Avancées

```bash
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 \
  --model-name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
  --api-key votre_clé_api \
  --sentiment-weight 0.6 \
  --min-confidence 0.7 \
  --news-lookback 48 \
  --position-size 0.02 \
  --stop-loss 0.03 \
  --take-profit 0.06
```

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
| `sentiment_weight` | Poids de l'analyse de sentiment | `0.5` |
| `min_confidence` | Seuil minimal de confiance | `0.65` |
| `technical_indicators` | Liste d'indicateurs techniques | `["macd", "rsi", "ema", "bollinger"]` |
| `news_lookback_hours` | Heures d'actualités à analyser | `24` |

## Mode Démo

Le mode démo est activé automatiquement lorsque `api_key` est défini sur `"demo_mode"`. Dans ce mode:

- Pas besoin de clé API réelle
- Des réponses prédéfinies sont utilisées pour simuler l'analyse de sentiment
- Idéal pour tester la stratégie sans frais d'API

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

1. **Trading de Crypto Intraday**:
```bash
python scripts/run_strategy_crypto_trader.py --strategy llm_v2 --sentiment-weight 0.7 --news-lookback 12 --position-size 0.01
```

2. **Trading d'Actions à Moyen Terme**:
```bash
python scripts/run_stock_daytrader_all.py --strategy llm_v2 --sentiment-weight 0.5 --min-confidence 0.7 --duration continuous
```

3. **Backtesting avec Configuration Personnalisée**:
```bash
python scripts/backtest_strategy.py --strategy llm_v2 --sentiment-weight 0.6 --technical-indicators macd,rsi,ema --api-key demo_mode
```

## Dépannage

1. **"Module not found" Error**:
   - Vérifiez que tous les modules requis sont installés: `pip install -r requirements.txt`

2. **Erreurs d'API**:
   - Vérifiez que votre clé API est valide et correctement configurée
   - Utilisez `--api-key demo_mode` pour les tests

3. **Performances sous-optimales**:
   - Ajustez `sentiment_weight` en fonction des conditions de marché
   - Augmentez `min_confidence` pour des signaux plus conservateurs
   - Analysez les logs pour identifier les sources potentielles d'erreur
