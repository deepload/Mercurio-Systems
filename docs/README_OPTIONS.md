# Support du Trading d'Options dans Mercurio AI

## Résumé

Ce document explique comment le support du trading d'options a été intégré dans Mercurio AI pour exploiter votre abonnement Alpaca AlgoTrader Plus avec Options Trading Level 1.

## Composants ajoutés

1. **Service de trading d'options**
   - Fichier: `app/services/options_service.py`
   - Fonctionnalités: Gestion des requêtes API d'options, exécution de trades, suggestions de stratégies.

2. **Stratégie d'options**
   - Fichier: `app/strategies/options_strategy.py`
   - Fonctionnalités: Conversion des signaux des stratégies ML en opportunités d'options.

3. **Configuration des options**
   - Fichier: `config/daytrader_config.json`
   - Fonctionnalités: Paramètres pour personnaliser le comportement du trading d'options.

4. **Tests unitaires**
   - Fichier: `tests/test_options_service.py`
   - Fonctionnalités: Validation du fonctionnement des composants de trading d'options.

5. **Documentation**
   - Fichier: `docs/options_trading.md`
   - Fonctionnalités: Guide complet pour comprendre et utiliser les fonctionnalités de trading d'options.

## Stratégies d'options supportées

Le système supporte les stratégies suivantes, compatibles avec Options Trading Level 1 :

- **Long Call** - Achat d'option d'achat (bullish)
- **Long Put** - Achat d'option de vente (bearish)
- **Cash-Secured Put** - Vente d'option de vente couverte par du cash (neutral to bullish)
- **Covered Call** - Vente d'option d'achat couverte par des actions (neutral to bearish)

## Intégration avec les stratégies existantes

Le système réutilise intelligemment vos stratégies ML existantes :

1. Les stratégies comme TransformerStrategy, LSTM et MSI génèrent des signaux directionnels.
2. La OptionsStrategy utilise ces signaux pour identifier des opportunités d'options.
3. Le OptionsService exécute les trades d'options via Alpaca.

## Comment tester

Pour tester les nouvelles fonctionnalités, exécutez:

```bash
python -m pytest tests/test_options_service.py -v
```

## Utilisation

Le trading d'options est activé par défaut dans la configuration. Pour l'utiliser:

1. Assurez-vous que vos clés Alpaca API sont configurées dans le fichier `.env`
2. Exécutez le système normalement avec:
   ```bash
   python run.py stock --duration 4
   ```
3. Le système utilisera automatiquement le trading d'options lorsque c'est pertinent

## Personnalisation

Modifiez la section `options_trading` dans `config/daytrader_config.json` pour ajuster:
- Le profil de risque (`risk_profile`)
- L'allocation maximale aux options (`max_options_allocation_pct`)
- Les stratégies préférées (`preferred_strategies`)
- Les expirations maximales (`max_days_to_expiry`)

## Documentation complète

Consultez `docs/options_trading.md` pour une documentation détaillée sur toutes les fonctionnalités de trading d'options.
