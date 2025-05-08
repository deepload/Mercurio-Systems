# Mercurio AI - Guide du Day Trading

## Introduction

Mercurio AI fournit un système complet de day trading algorithmique qui permet d'exécuter diverses stratégies de trading sur des actions et des cryptomonnaies. Ce guide se concentre sur l'utilisation du script de day trading pour les actions (`run_stock_daytrader_all.py`) et explique comment gérer vos positions, y compris comment quitter proprement et liquider toutes vos positions en fin de journée.

## Prérequis

Avant d'utiliser le script de day trading, assurez-vous d'avoir :

1. Un compte Alpaca (paper trading ou live)
2. Configuré vos clés API dans le fichier `.env`
3. Installé toutes les dépendances du projet

## Démarrage du Day Trading

### Script principal : `run_stock_daytrader_all.py`

Le script principal pour le day trading d'actions est situé dans `scripts/run_stock_daytrader_all.py`. Ce script :
- Récupère des actions selon différents filtres
- Applique les stratégies de trading configurées
- Exécute des cycles de trading pendant les heures de marché
- Met à jour automatiquement l'état du portefeuille

### Options de ligne de commande

```bash
python scripts/run_stock_daytrader_all.py [options]
```

Options principales :

| Option | Description |
|--------|-------------|
| `--strategy` | Stratégie à utiliser (moving_average, moving_average_ml, lstm_predictor, transformer, msi, llm, all) |
| `--filter` | Filtre pour la sélection des actions (all, active_assets, top_volume, top_gainers, tech_stocks, finance_stocks, health_stocks, sp500, nasdaq100, custom) |
| `--max-symbols` | Nombre maximum de symboles à traiter |
| `--position-size` | Pourcentage du capital à allouer par position (0.01 à 1.0) |
| `--stop-loss` | Pourcentage de stop loss (0.01 à 0.5) |
| `--take-profit` | Pourcentage de take profit (0.01 à 0.5) |
| `--duration` | Durée de la session (market_hours, extended_hours, full_day, continuous) |
| `--use-threads` | Utiliser le multithreading pour un traitement plus rapide |
| `--api-level` | Forcer un niveau d'API Alpaca spécifique (1 à 3) |

### Exemple d'utilisation

```bash
python scripts/run_stock_daytrader_all.py --strategy moving_average_ml --filter active_assets --max-symbols 20 --api-level 3 --use-threads --duration market_hours
```

## Gestion des positions

Le script de day trading gère automatiquement les positions selon les stratégies configurées. Il :
- Ouvrira des positions lorsque les signaux d'achat sont générés
- Fermera des positions sur les signaux de vente
- Appliquera les stop loss et take profit configurés

## Arrêt du Day Trading et Liquidation des Positions

### Arrêt normal

Pour arrêter le script de day trading, appuyez simplement sur `Ctrl+C`. Le script intercepte ce signal et s'arrête proprement. Cependant, cela **ne liquidera pas automatiquement vos positions ouvertes**.

### Liquidation de toutes les positions

Pour liquider toutes vos positions après avoir arrêté le script de day trading, utilisez le script dédié `liquidate_all_positions.py` :

```bash
python scripts/liquidate_all_positions.py
```

Ce script :
1. Se connecte à votre compte Alpaca (paper ou live, selon la configuration)
2. Affiche toutes vos positions ouvertes actuelles
3. Demande confirmation avant de procéder
4. Liquide toutes les positions ouvertes
5. Fournit un rapport détaillé sur les positions liquidées
6. Affiche la valeur finale du portefeuille et le cash disponible

### Workflow complet pour une journée de trading

1. **Démarrer le day trading** :
   ```bash
   python scripts/run_stock_daytrader_all.py --strategy moving_average_ml --filter active_assets --max-symbols 20
   ```

2. **Surveiller l'activité de trading** pendant les heures de marché

3. **Arrêter le trading** lorsque vous êtes prêt à terminer la session :
   - Appuyez sur `Ctrl+C` pour arrêter proprement le script

4. **Liquider toutes les positions** :
   ```bash
   python scripts/liquidate_all_positions.py
   ```
   
5. **Confirmer** la liquidation lorsque demandé (entrez `y`)

## Paramètres Avancés

### Stratégies disponibles

Mercurio propose plusieurs stratégies de trading, chacune avec ses propres caractéristiques :

- **Moving Average** : Stratégie de moyenne mobile classique
- **Moving Average ML** : Moyenne mobile augmentée par machine learning
- **LSTM Predictor** : Utilise des réseaux LSTM pour prédire les mouvements
- **Transformer** : Utilise des modèles Transformer pour l'analyse
- **MSI (Multi-Source Intelligence)** : Combine des données de plusieurs sources
- **LLM (Large Language Model)** : Utilise des modèles de langage pour l'analyse de sentiment

### Entraînement automatique

Le script de day trading peut automatiquement réentraîner les modèles ML pendant les périodes d'inactivité du marché avec les options :

```bash
--auto-retrain --retrain-interval 6 --retrain-symbols 10
```

## Dépannage

### Problèmes courants

1. **"Erreur lors de l'initialisation des services Mercurio"** :
   - Vérifiez que vous avez correctement initialisé l'environnement
   - Assurez-vous que toutes les dépendances sont installées

2. **"Alpaca client not initialized"** :
   - Vérifiez vos clés API dans le fichier `.env`
   - Assurez-vous que votre compte Alpaca est actif

3. **Script s'arrête sans liquider les positions** :
   - C'est le comportement normal. Utilisez `liquidate_all_positions.py` pour fermer les positions.

## Bonnes Pratiques

1. **Toujours tester en paper trading** avant de passer en live
2. **Commencer avec peu de symboles** pour comprendre le comportement
3. **Liquider vos positions à la fin de chaque journée** de trading si vous ne souhaitez pas de positions ouvertes pendant la nuit
4. **Surveiller régulièrement** la performance de vos stratégies
5. **Ajuster les paramètres** de position size, stop loss et take profit selon votre tolérance au risque

## Conclusion

Le système de day trading de Mercurio offre une plateforme flexible et puissante pour exécuter des stratégies de trading algorithmique sur les marchés financiers. La combinaison du script principal de day trading avec l'outil de liquidation des positions vous donne un contrôle complet sur votre activité de trading.
