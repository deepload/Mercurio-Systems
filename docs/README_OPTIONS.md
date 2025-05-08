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
   ```bash
   python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT --capital 100000
   ```

2. **Trading d'options basé sur le ML**
   ```bash
   python -m scripts.options.run_ml_options_trader --ml-strategy LSTM --options-strategy COVERED_CALL --symbols AAPL MSFT
   ```

3. **Trading d'options à haut volume**
   ```bash
   python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --max-symbols 50 --use-threads --use-custom-symbols
   ```

4. **Trading d'options sur crypto**
   ```bash
   python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000
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
