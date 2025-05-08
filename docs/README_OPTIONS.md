# Support du Trading d'Options dans Mercurio AI

## Résumé

Ce document explique comment le support du trading d'options a été intégré dans Mercurio AI pour exploiter votre abonnement Alpaca AlgoTrader Plus avec Options Trading Levels 1-3, incluant toutes les fonctionnalités avancées et les nouvelles stratégies d'options.

## Composants ajoutés

1. **Service de trading d'options**
   - Fichier: `app/services/options_service.py`
   - Fonctionnalités: Gestion des requêtes API d'options, exécution de trades, suggestions de stratégies.

2. **Stratégies d'options spécifiques**
   - Dossier: `app/strategies/options/`
   - Fichiers: `base_options_strategy.py`, `butterfly_spread.py`, `cash_secured_put.py`, `covered_call.py`, `iron_condor.py`, `long_call.py`, `long_put.py`
   - Fonctionnalités: Implémentation de diverses stratégies d'options, de niveau 1 à niveau 3.

3. **Utilitaires mathématiques pour options**
   - Fichier: `app/utils/math_utils.py`
   - Fonctionnalités: Calcul des prix d'options (Black-Scholes), volatilité implicite, et les Grecs (Delta, Gamma, Theta, Vega).

4. **Backtester d'options**
   - Fichier: `app/services/options_backtester.py`
   - Fonctionnalités: Backtesting de stratégies d'options sur des données historiques.

5. **Scripts de trading d'options**
   - Dossier: `scripts/options/`
   - Fichiers: `run_daily_options_trader.py`, `run_ml_options_trader.py`, `run_high_volume_options_trader.py`, `run_crypto_options_trader.py`
   - Fonctionnalités: Scripts spécialisés pour différentes approches du trading d'options.

6. **Tests compréhensifs**
   - Fichier: `scripts/options/test_options_strategies.py`
   - Fonctionnalités: Test complet de toutes les stratégies d'options, validant leur fonctionnement correct.

7. **Documentation**
   - Fichiers: `docs/options_trading.md`, documentation mise à jour dans `docs/for-dummies/`
   - Fonctionnalités: Guide complet pour comprendre et utiliser les fonctionnalités de trading d'options.

## Stratégies d'options supportées

Le système supporte désormais un large éventail de stratégies d'options:

### Niveau 1 (Options Trading Level 1)
- **Long Call** - Achat d'option d'achat (bullish)
- **Long Put** - Achat d'option de vente (bearish)
- **Cash-Secured Put** - Vente d'option de vente couverte par du cash (neutral to bullish)
- **Covered Call** - Vente d'option d'achat couverte par des actions (neutral to bearish)

### Niveau 2 (Options Trading Level 2)
- **Iron Condor** - Combinaison de quatre options pour profiter d'un marché stagnant (neutral)
- **Butterfly Spread** - Stratégie à trois prix d'exercice qui profite quand le prix du sous-jacent reste stable (neutral)

### Niveau 3 (Options Trading Level 3)
- **Straddles/Strangles** - Achat d'options d'achat et de vente pour profiter de la volatilité (volatility play)
- **Calendar Spreads** - Combinaison d'options avec différentes dates d'expiration (time decay play)

## Intégration avec les stratégies existantes

Le système réutilise intelligemment vos stratégies ML existantes :

1. Les stratégies comme TransformerStrategy, LSTM, LLM et MSI génèrent des signaux directionnels.
2. De nouveaux scripts spécialisés comme `run_ml_options_trader.py` combinent ces signaux ML avec des stratégies d'options.
3. L'analyse de sentiment des grands modèles de langage (LLM) peut être utilisée pour améliorer la sélection des stratégies d'options.
4. Le système supporte maintenant le trading haute fréquence (HFT) avec options via le script `run_high_volume_options_trader.py`.

## Comment tester

Pour tester les nouvelles fonctionnalités, utilisez notre script de test complet :

```bash
python -m scripts.options.test_options_strategies --test-all
```

Ce script teste toutes les stratégies d'options pour :
- L'initialisation correcte
- Les conditions d'entrée/sortie
- L'exécution des trades
- Les paramètres de gestion des risques
- La gestion des cas limites

## Utilisation

Plusieurs scripts spécialisés sont maintenant disponibles pour le trading d'options :

1. **Trading quotidien d'options**

   Le script `run_daily_options_trader.py` permet d'exécuter des stratégies d'options quotidiennes sur les actions.
   
   **Exemples d'utilisation :**
   
   ```bash
   # Stratégie de base avec des symboles spécifiques
   python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --capital 100000
   
   # Personnalisation des paramètres de stratégie
   python -m scripts.options.run_daily_options_trader --strategy LONG_CALL --symbols AAPL MSFT GOOG --capital 100000 --allocation-per-trade 0.03 --days-to-expiry 45 --paper-trading
   
   # Définir des objectifs de profit et stop-loss personnalisés
   python -m scripts.options.run_daily_options_trader --strategy IRON_CONDOR --symbols SPY QQQ --capital 100000 --profit-target 0.3 --stop-loss 0.7 --days-to-expiry 14
   ```

2. **Trading d'options basé sur le ML**

   Le script `run_ml_options_trader.py` permet de combiner les modèles de machine learning avec des stratégies d'options.
   
   **Exemples d'utilisation :**
   
   ```bash
   # Utilisation du modèle LSTM pour COVERED_CALL
   python -m scripts.options.run_ml_options_trader --ml-strategy LSTM --options-strategy COVERED_CALL --symbols AAPL MSFT --capital 100000
   
   # Mode AUTO pour sélection automatique de la stratégie basée sur les signaux ML
   python -m scripts.options.run_ml_options_trader --ml-strategy TRANSFORMER --options-strategy AUTO --symbols AAPL MSFT GOOG --capital 100000 --confidence-threshold 0.7
   
   # Utilisation de l'analyse de sentiment LLM pour les décisions d'options
   python -m scripts.options.run_ml_options_trader --ml-strategy LLM --options-strategy AUTO --symbols TSLA AAPL MSFT --capital 100000 --paper-trading
   
   # Analyse multi-source pour plus de précision
   python -m scripts.options.run_ml_options_trader --ml-strategy MSI --options-strategy AUTO --symbols AAPL MSFT GOOG --capital 100000 --confidence-threshold 0.75
   ```

3. **Trading d'options à haut volume**

   Le script `run_high_volume_options_trader.py` permet de trader des options sur un grand nombre de symboles simultanément (jusqu'à 50).
   
   **Exemples d'utilisation :**
   
   ```bash
   # Trading sur les actions les plus volumineuses (jusqu'à 50)
   python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --max-symbols 50 --use-threads --capital 100000
   
   # Utilisation d'une liste personnalisée de symboles
   python -m scripts.options.run_high_volume_options_trader --strategy CASH_SECURED_PUT --use-custom-symbols --capital 100000 --use-threads --paper-trading
   
   # Sélection du top 20 des actions les plus volatiles
   python -m scripts.options.run_high_volume_options_trader --strategy IRON_CONDOR --filter most_volatile --max-symbols 20 --capital 100000 --technical-filter --use-threads
   
   # Utilisation du filtrage technique et limitation des allocations
   python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --filter top_gainers --max-symbols 30 --allocation-per-trade 0.02 --technical-filter --use-threads
   ```

4. **Trading d'options sur crypto**

   Le script `run_crypto_options_trader.py` permet de trader des options sur crypto-monnaies en utilisant les stratégies suivantes :
   
   - Stratégie unique: LONG_CALL, LONG_PUT, IRON_CONDOR, ou BUTTERFLY 
   - Stratégie combinée: MIXED (combine automatiquement plusieurs stratégies)
   
   **Exemples d'utilisation :**
   
   ```bash
   # Utilisation de base avec des symbôles spécifiques
   python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH SOL --capital 50000 --paper-trading
   
   # Utilisation de la liste personnalisée des crypto-monnaies dans .env
   python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --use-custom-symbols --capital 50000 --paper-trading
   
   # Combinaison de plusieurs stratégies avec MIXED
   python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --capital 50000 --paper-trading --use-threads
   
   # Ajustement du seuil de volatilité pour permettre plus d'entrées
   python -m scripts.options.run_crypto_options_trader --strategy MIXED --use-custom-symbols --volatility-threshold 0.02 --capital 50000 --paper-trading
   
   # Trading en mode live (attention: vérifiez votre compte d'abord)
   python -m scripts.options.run_crypto_options_trader --strategy MIXED --symbols BTC ETH --capital 50000
   
   # Durée personnalisée (format: 1h, 30m, 1d)
   python -m scripts.options.run_crypto_options_trader --strategy LONG_PUT --symbols BTC ETH --capital 50000 --duration 2h
   ```
   
   **Important** : Ce script utilise maintenant les vraies données d'Alpaca pour les crypto disponibles.

5. **Test des stratégies d'options**

   Le script `test_options_strategies.py` permet de tester toutes les stratégies d'options pour s'assurer de leur bon fonctionnement.
   
   **Exemples d'utilisation :**
   
   ```bash
   # Tester toutes les stratégies d'options
   python -m scripts.options.test_options_strategies --test-all
   
   # Tester une stratégie spécifique
   python -m scripts.options.test_options_strategies --strategy COVERED_CALL
   
   # Test ciblé sur les conditions d'entrée/sortie
   python -m scripts.options.test_options_strategies --strategy LONG_CALL --test-entry-exit
   
   # Test complet d'une seule stratégie
   python -m scripts.options.test_options_strategies --strategy IRON_CONDOR --test-initialization --test-entry-exit --test-execution --test-risk-management --test-edge-cases
   ```

## Personnalisation

Toutes les stratégies d'options sont hautement personnalisables via les arguments de ligne de commande. Voici quelques exemples de paramètres importants :

- **Profil de risque** : Contrôlez l'exposition au risque via les paramètres de taille de position et de stop-loss
- **Allocation maximale** : Définissez le pourcentage maximum de capital à allouer aux options
- **Stratégies préférées** : Sélectionnez les stratégies spécifiques à utiliser
- **Expiration** : Contrôlez la durée des options avec le paramètre `--days-to-expiry`
- **Delta cible** : Ajustez le delta cible pour les options avec `--delta-target`
- **Utilisation du ML** : Combinez les stratégies d'options avec différents modèles ML

## Documentation complète

Consultez `docs/options_trading.md` pour une documentation détaillée sur toutes les fonctionnalités de trading d'options, y compris :

- Description détaillée de toutes les stratégies (Niveau 1-3)
- Exemples d'utilisation des nouveaux scripts
- Intégration avec l'analyse de sentiment et les modèles ML
- Utilisation des utilitaires mathématiques pour le pricing des options
- Backtesting des stratégies d'options
- Bonnes pratiques et considérations de risque

Pour les débutants, consultez également le chapitre dédié aux options dans notre livre "Mercurio AI for Dummies" dans le dossier `docs/for-dummies/`.
