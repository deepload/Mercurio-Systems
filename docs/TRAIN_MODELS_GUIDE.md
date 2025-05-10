# Guide d'utilisation de train_all_models.py

Ce document décrit en détail comment utiliser le script `train_all_models.py` pour entraîner les modèles d'intelligence artificielle utilisés par MercurioAI, notamment les modèles LSTM et Transformer pour le trading algorithmique.

## Aperçu

Le script `train_all_models.py` permet d'entraîner en une seule commande tous les modèles d'IA utilisés par la plateforme MercurioAI. Il offre désormais la possibilité d'utiliser automatiquement les symboles générés par `get_all_symbols.py`, permettant ainsi d'entraîner les modèles sur l'ensemble des actifs disponibles (plus de 11 000 actions et cryptomonnaies).

## Prérequis

1. Environnement Python configuré avec toutes les dépendances de MercurioAI
2. Clés API Alpaca et Polygon configurées dans le fichier `.env`
3. (Optionnel mais recommandé) Exécution préalable de `get_all_symbols.py` pour générer les listes complètes d'actifs

## Options disponibles

### Options de base

| Option | Description | Valeur par défaut |
|--------|-------------|------------------|
| `--days` | Nombre de jours d'historique à utiliser | 180 |
| `--epochs` | Nombre d'époques d'entraînement | 50 |
| `--top_assets` | Nombre d'actifs populaires à inclure | 10 |
| `--symbols` | Liste des symboles spécifiés directement (séparés par virgules) | |
| `--use_gpu` | Utiliser le GPU si disponible | False |

### Options de sélection des symboles

| Option | Description | 
|--------|-------------|
| `--include_stocks` | Inclure les actions populaires |
| `--include_crypto` | Inclure les cryptomonnaies populaires |
| `--custom_stocks_file` | Fichier CSV personnalisé pour les actions |
| `--custom_crypto_file` | Fichier CSV personnalisé pour les cryptomonnaies |
| `--all_symbols` | Utiliser tous les symboles disponibles dans les fichiers générés par `get_all_symbols.py` |
| `--random_select` | Sélectionner aléatoirement les symboles plutôt que les premiers de la liste |
| `--max_symbols` | Limiter le nombre total de symboles à entraîner (0 = pas de limite) |

### Options de traitement par lots

| Option | Description | Valeur par défaut |
|--------|-------------|------------------|
| `--batch_mode` | Activer le mode batch pour les grandes listes | False |
| `--auto_batch` | Activer automatiquement le mode batch quand nécessaire | False |
| `--batch_size` | Taille des lots en mode batch | 20 |

## Exemples d'utilisation

### Utilisation basique (modèles par défaut)

```bash
python scripts/train_all_models.py
```

Cette commande entraîne les modèles sur les 10 actions et 10 cryptomonnaies par défaut les plus populaires.

### Spécifier une liste personnalisée de symboles

```bash
python scripts/train_all_models.py --symbols AAPL,MSFT,GOOGL,AMZN,TSLA,BTC-USD,ETH-USD
```

### Utiliser tous les symboles disponibles

```bash
python scripts/train_all_models.py --all_symbols
```

Cette commande utilisera tous les symboles trouvés dans les fichiers CSV les plus récents générés par `get_all_symbols.py`.

### Limiter le nombre de symboles (pour des tests rapides)

```bash
python scripts/train_all_models.py --all_symbols --max_symbols 50
```

### Sélectionner aléatoirement un sous-ensemble de symboles

```bash
python scripts/train_all_models.py --all_symbols --max_symbols 200 --random_select
```

### Traitement par lots pour les grandes listes

```bash
python scripts/train_all_models.py --all_symbols --batch_mode --batch_size 50
```

Cette commande traite les symboles par lots de 50, ce qui est utile pour les grandes listes (comme les 11 000+ symboles d'actions).

### Mode automatique optimisé pour les grandes listes

```bash
python scripts/train_all_models.py --all_symbols --auto_batch --max_symbols 500 --random_select
```

Cette commande sélectionne aléatoirement 500 symboles et active automatiquement le mode batch si nécessaire.

## Conseils pratiques

1. **Pour les tests initiaux** : Utilisez `--max_symbols 20` et `--epochs 5` pour des tests rapides
2. **Pour un entraînement efficace** : Utilisez `--auto_batch` et `--random_select` pour une meilleure représentativité
3. **Pour un entraînement complet** : Prévoyez beaucoup de temps de calcul si vous utilisez tous les symboles sans limitation
4. **Pour les machines avec peu de RAM** : Réduisez `--batch_size` à 10 pour minimiser l'utilisation de la mémoire

## Intégration avec les autres scripts

Ce script s'intègre parfaitement avec `run_integrated.py` pour permettre l'entraînement automatique des modèles pendant les périodes d'inactivité du marché :

```bash
python scripts/run_integrated_trader.py --strategy ALL --duration continuous --refresh-symbols --auto-training
```

## Sorties et résultats

Le script génère plusieurs sorties :

1. Les modèles entraînés dans les dossiers `models/lstm/` et `models/transformer/`
2. Un rapport de formation dans `reports/training_report_{date}_{heure}.csv`
3. Des logs détaillés dans `logs/train_all_models.log`
