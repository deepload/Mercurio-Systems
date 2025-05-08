# Exemples d'utilisation des stratégies d'options dans Mercurio AI

Voici plus de 50 exemples détaillés pour exploiter toutes les fonctionnalités d'options de Mercurio AI.

## 1. Trading quotidien d'options (`run_daily_options_trader.py`)

### Stratégie COVERED_CALL

```bash
# 1. Stratégie de base avec symboles spécifiques
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --capital 100000

# 2. Covered Call avec faible delta (moins risqué)
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --delta-target 0.25 --capital 100000

# 3. Covered Call avec expiration plus courte (15 jours)
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --days-to-expiry 15 --capital 100000

# 4. Covered Call pour actions à forte volatilité
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols TSLA NVDA --delta-target 0.20 --capital 100000

# 5. Covered Call sur ETFs (moins volatile)
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols SPY QQQ IWM --delta-target 0.30 --days-to-expiry 45 --capital 100000

# 6. Covered Call en mode papier avec objectif de profit défini
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL GOOGL --profit-target 0.4 --stop-loss 0.5 --paper-trading --capital 100000
```

### Stratégie CASH_SECURED_PUT

```bash
# 7. CSP de base sur indices
python -m scripts.options.run_daily_options_trader --strategy CASH_SECURED_PUT --symbols SPY QQQ --capital 100000

# 8. CSP avec petit budget par trade (2%)
python -m scripts.options.run_daily_options_trader --strategy CASH_SECURED_PUT --symbols AAPL MSFT --allocation-per-trade 0.02 --capital 100000

# 9. CSP avec delta plus faible pour viser prix d'entrée plus bas
python -m scripts.options.run_daily_options_trader --strategy CASH_SECURED_PUT --symbols AAPL AMZN --delta-target 0.20 --capital 100000

# 10. CSP pour actions technologiques avec expiration courte
python -m scripts.options.run_daily_options_trader --strategy CASH_SECURED_PUT --symbols NVDA AMD INTC --days-to-expiry 14 --delta-target 0.30 --capital 100000

# 11. CSP sur actions financières avec stop-loss serré
python -m scripts.options.run_daily_options_trader --strategy CASH_SECURED_PUT --symbols JPM GS MS BAC --stop-loss 0.3 --capital 100000
```

### Stratégie LONG_CALL (Directionnelle Haussière)

```bash
# 12. Long Call standard
python -m scripts.options.run_daily_options_trader --strategy LONG_CALL --symbols AAPL MSFT --capital 100000

# 13. Long Call avec delta plus élevé (plus directionnelle)
python -m scripts.options.run_daily_options_trader --strategy LONG_CALL --symbols AAPL MSFT --delta-target 0.60 --capital 100000

# 14. Long Call avec expiration plus longue (stratégie LEAP)
python -m scripts.options.run_daily_options_trader --strategy LONG_CALL --symbols AAPL MSFT --days-to-expiry 90 --capital 100000

# 15. Long Call avec objectif de profit important et stop-loss
python -m scripts.options.run_daily_options_trader --strategy LONG_CALL --symbols AAPL MSFT --profit-target 1.0 --stop-loss 0.5 --capital 100000
```

### Stratégie LONG_PUT (Directionnelle Baissière)

```bash
# 16. Long Put standard
python -m scripts.options.run_daily_options_trader --strategy LONG_PUT --symbols AAPL MSFT --capital 100000

# 17. Long Put avec delta plus élevé (plus directionnelle)
python -m scripts.options.run_daily_options_trader --strategy LONG_PUT --symbols AAPL MSFT --delta-target 0.60 --capital 100000

# 18. Long Put comme couverture (hedge) sur les indices
python -m scripts.options.run_daily_options_trader --strategy LONG_PUT --symbols SPY QQQ --days-to-expiry 60 --allocation-per-trade 0.01 --capital 100000
```

### Stratégie IRON_CONDOR (Neutre)

```bash
# 19. Iron Condor standard
python -m scripts.options.run_daily_options_trader --strategy IRON_CONDOR --symbols SPY --capital 100000

# 20. Iron Condor avec ailes étroites (plus risqué, plus de prime)
python -m scripts.options.run_daily_options_trader --strategy IRON_CONDOR --symbols SPY --wing-width 0.05 --capital 100000

# 21. Iron Condor avec objectif de profit prudent
python -m scripts.options.run_daily_options_trader --strategy IRON_CONDOR --symbols SPY QQQ --profit-target 0.25 --stop-loss 0.5 --capital 100000

# 22. Iron Condor sur actions individuelles à forte volatilité
python -m scripts.options.run_daily_options_trader --strategy IRON_CONDOR --symbols TSLA NVDA --capital 100000
```

### Stratégie BUTTERFLY (Neutre Précise)

```bash
# 23. Butterfly standard
python -m scripts.options.run_daily_options_trader --strategy BUTTERFLY --symbols SPY --capital 100000

# 24. Butterfly avec ailes plus étroites
python -m scripts.options.run_daily_options_trader --strategy BUTTERFLY --symbols SPY --wing-width-pct 0.03 --capital 100000

# 25. Butterfly Call sur actions tech
python -m scripts.options.run_daily_options_trader --strategy BUTTERFLY --symbols AAPL MSFT --option-type call --capital 100000

# 26. Butterfly Put sur indices
python -m scripts.options.run_daily_options_trader --strategy BUTTERFLY --symbols SPY QQQ --option-type put --capital 100000

# 27. Butterfly avec bandes plus larges pour indices volatils
python -m scripts.options.run_daily_options_trader --strategy BUTTERFLY --symbols VIX --wing-width-pct 0.08 --capital 100000
```

### Stratégie MIXED (Multi-stratégies)

```bash
# 28. Mixed standard (combinaison automatique)
python -m scripts.options.run_daily_options_trader --strategy MIXED --symbols SPY AAPL MSFT --capital 100000

# 29. Mixed avec moins d'allocations aux stratégies risquées
python -m scripts.options.run_daily_options_trader --strategy MIXED --symbols SPY AAPL MSFT --allocation-per-trade 0.02 --capital 100000

# 30. Mixed avec objectifs de profits définis
python -m scripts.options.run_daily_options_trader --strategy MIXED --symbols SPY AAPL MSFT --profit-target 0.35 --stop-loss 0.5 --capital 100000
```

## 2. Trading d'options à haut volume (`run_high_volume_options_trader.py`)

```bash
# 31. Trading COVERED_CALL sur 50 actions en mode multi-thread
python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --use-custom-symbols --max-symbols 50 --use-threads --capital 100000 --paper-trading

# 32. Trading CASH_SECURED_PUT sur actions à forte volatilité
python -m scripts.options.run_high_volume_options_trader --strategy CASH_SECURED_PUT --filter most_volatile --max-symbols 20 --use-threads --capital 100000

# 33. Trading IRON_CONDOR sur les actions les plus actives
python -m scripts.options.run_high_volume_options_trader --strategy IRON_CONDOR --filter top_volume --max-symbols 15 --use-threads --capital 100000

# 34. Trading COVERED_CALL avec filtrage technique
python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --max-symbols 30 --use-threads --technical-filter --capital 100000

# 35. Trading CASH_SECURED_PUT sur actions à tendance haussière
python -m scripts.options.run_high_volume_options_trader --strategy CASH_SECURED_PUT --filter top_gainers --max-symbols 20 --use-threads --capital 100000

# 36. Trading COVERED_CALL avec allocation réduite par action
python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --use-custom-symbols --allocation-per-trade 0.01 --use-threads --capital 100000

# 37. Trading CASH_SECURED_PUT avec delta plus faible
python -m scripts.options.run_high_volume_options_trader --strategy CASH_SECURED_PUT --delta-target 0.20 --max-symbols 30 --use-threads --capital 100000

# 38. Trading IRON_CONDOR sur grands ETFs uniquement
python -m scripts.options.run_high_volume_options_trader --strategy IRON_CONDOR --symbols SPY QQQ IWM EEM EFA XLF XLE XLU XLK --use-threads --capital 100000
```

## 3. Trading d'options basé sur le ML (`run_ml_options_trader.py`)

```bash
# 39. Prédictions LSTM combinées à COVERED_CALL
python -m scripts.options.run_ml_options_trader --ml-strategy LSTM --options-strategy COVERED_CALL --symbols AAPL MSFT GOOG --capital 100000

# 40. Prédictions Transformer combinées à CASH_SECURED_PUT
python -m scripts.options.run_ml_options_trader --ml-strategy TRANSFORMER --options-strategy CASH_SECURED_PUT --symbols AAPL MSFT GOOG --capital 100000

# 41. Prédictions LLM avec sélection automatique de stratégie
python -m scripts.options.run_ml_options_trader --ml-strategy LLM --options-strategy AUTO --symbols AAPL MSFT GOOG --capital 100000

# 42. Prédictions MSI avec sélection automatique et seuil de confiance élevé
python -m scripts.options.run_ml_options_trader --ml-strategy MSI --options-strategy AUTO --symbols AAPL MSFT GOOG --confidence-threshold 0.8 --capital 100000

# 43. Combinaison LSTM + LONG_CALL pour actions très haussières
python -m scripts.options.run_ml_options_trader --ml-strategy LSTM --options-strategy LONG_CALL --symbols NVDA AMD TSLA --min-prediction 0.05 --capital 100000

# 44. Combinaison LLM + LONG_PUT pour actions avec sentiment négatif
python -m scripts.options.run_ml_options_trader --ml-strategy LLM --options-strategy LONG_PUT --symbols AAPL META NFLX --capital 100000

# 45. Analyse multi-source (MSI) pour stratégies directionnelles
python -m scripts.options.run_ml_options_trader --ml-strategy MSI --options-strategy AUTO --symbols AAPL MSFT GOOG AMZN --directional-only --capital 100000

# 46. Prédictions Transformer avec délai plus long
python -m scripts.options.run_ml_options_trader --ml-strategy TRANSFORMER --options-strategy AUTO --symbols AAPL MSFT GOOG --prediction-horizon 5 --capital 100000

# 47. Prédiction LSTM avec fenêtre large d'historique
python -m scripts.options.run_ml_options_trader --ml-strategy LSTM --options-strategy AUTO --symbols AAPL MSFT GOOG --lookback-window 120 --capital 100000

# 48. Stratégie LLM avec préférence pour stratégies neutres
python -m scripts.options.run_ml_options_trader --ml-strategy LLM --options-strategy AUTO --symbols AAPL MSFT GOOG --neutral-bias --capital 100000
```

## 4. Trading d'options sur crypto (`run_crypto_options_trader.py`)

```bash
# 49. Trading LONG_CALL sur BTC et ETH
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000 --paper-trading

# 50. Trading LONG_PUT sur crypto à forte volatilité
python -m scripts.options.run_crypto_options_trader --strategy LONG_PUT --symbols BTC ETH SOL AVAX --capital 50000 --paper-trading

# 51. Trading IRON_CONDOR sur BTC pour marché stable
python -m scripts.options.run_crypto_options_trader --strategy IRON_CONDOR --symbols BTC --capital 50000 --paper-trading

# 52. Trading BUTTERFLY sur ETH
python -m scripts.options.run_crypto_options_trader --strategy BUTTERFLY --symbols ETH --capital 50000 --paper-trading

# 53. Trading MIXED avec stratégies multiples
python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading

# 54. Trading limité aux principales cryptomonnaies
python -m scripts.options.run_crypto_options_trader --strategy MIXED --symbols BTC ETH --capital 50000 --paper-trading --premium-crypto-only

# 55. Trading sur options courtes (hebdomadaires)
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --days-to-expiry 7 --capital 50000 --paper-trading

# 56. Trading LONG_PUT pour protection de portefeuille crypto
python -m scripts.options.run_crypto_options_trader --strategy LONG_PUT --use-custom-symbols --allocation-per-trade 0.01 --capital 50000 --paper-trading
```

## 5. Test et validation de stratégies d'options

```bash
# 57. Test complet de toutes les stratégies
python -m scripts.options.test_options_strategies --test-all

# 58. Test spécifique de COVERED_CALL
python -m scripts.options.test_options_strategies --strategy COVERED_CALL --test-all

# 59. Test de l'entrée et sortie de CASH_SECURED_PUT
python -m scripts.options.test_options_strategies --strategy CASH_SECURED_PUT --test-entry-exit

# 60. Test de l'exécution de LONG_CALL
python -m scripts.options.test_options_strategies --strategy LONG_CALL --test-execution

# 61. Test des mécanismes de gestion des risques d'IRON_CONDOR
python -m scripts.options.test_options_strategies --strategy IRON_CONDOR --test-risk-management
```

## Personnalisation avancée

Toutes les stratégies d'options sont hautement personnalisables via les arguments de ligne de commande:

### Profils de risque

```bash
# Profil de risque conservateur
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --allocation-per-trade 0.02 --delta-target 0.2 --profit-target 0.3 --stop-loss 0.3 --capital 100000

# Profil de risque modéré
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --allocation-per-trade 0.05 --delta-target 0.3 --profit-target 0.5 --stop-loss 0.5 --capital 100000

# Profil de risque agressif
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --allocation-per-trade 0.1 --delta-target 0.4 --profit-target 0.7 --stop-loss 0.7 --capital 100000
```

### Paramètres des Grecques

```bash
# Focus sur le Delta (directionalité)
python -m scripts.options.run_daily_options_trader --strategy LONG_CALL --symbols AAPL MSFT --delta-target 0.7 --capital 100000

# Focus sur le Gamma (effet de levier)
python -m scripts.options.run_daily_options_trader --strategy LONG_CALL --symbols AAPL MSFT --gamma-focus --capital 100000

# Focus sur le Theta (optimisation du decay)
python -m scripts.options.run_daily_options_trader --strategy IRON_CONDOR --symbols SPY --theta-focus --capital 100000
```

### Exemples combinés avec stratégies avancées

```bash
# Gestion de portfolio complète avec diversification de stratégies
python -m scripts.options.run_daily_options_trader --strategy MIXED --symbols SPY AAPL MSFT GOOG AMZN --allocation-per-trade 0.03 --days-to-expiry 30 --profit-target 0.4 --stop-loss 0.5 --capital 100000 --paper-trading
```

## Utilisation des différentes stratégies selon les conditions de marché

### Marché haussier
- COVERED_CALL sur actions avec tendance modérément haussière
- CASH_SECURED_PUT sur actions souhaitées à prix réduit
- LONG_CALL pour profiter directement des hausses

### Marché baissier
- LONG_PUT pour profiter des baisses ou protéger un portefeuille
- CASH_SECURED_PUT avec delta très faible pour construire des positions à prix réduit

### Marché neutre (trading range)
- IRON_CONDOR pour profiter des marchés qui évoluent dans une fourchette
- BUTTERFLY pour cibler un prix spécifique
