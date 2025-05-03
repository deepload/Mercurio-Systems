# Guide d'Entraînement des Modèles MercurioAI

> [!NOTE]
> **Navigation Rapide:**
> - [🔍 Index de tous les guides](./GUIDES_INDEX.md)
> - [📈 Guide de Day Trading](./day_trading_guide.md)
> - [📊 Guide des Options](./options_trading.md)
> - [📔 Documentation Principale](./README.md)

Ce guide explique en détail comment entraîner, optimiser et dépanner les modèles d'intelligence artificielle intégrés dans MercurioAI.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Modèles disponibles](#modèles-disponibles)
3. [Scripts d'entraînement](#scripts-dentraînement)
4. [Exemples d'utilisation](#exemples-dutilisation)
5. [Optimisation des modèles](#optimisation-des-modèles)
6. [Résolution des problèmes](#résolution-des-problèmes)
7. [Questions fréquentes](#questions-fréquentes)

## Vue d'ensemble

MercurioAI utilise deux types principaux de modèles d'apprentissage profond pour la prédiction des marchés financiers :

1. **Modèles LSTM** (Long Short-Term Memory) : Ces réseaux de neurones récurrents sont spécialisés dans l'apprentissage des dépendances à long terme dans les séries temporelles. Chaque actif dispose de son propre modèle LSTM spécifiquement entraîné sur ses données historiques.

2. **Modèles Transformer** : Ces architectures avancées, basées sur le mécanisme d'attention, peuvent analyser simultanément plusieurs actifs et capturer les relations entre eux. Un seul modèle Transformer est entraîné sur l'ensemble des actifs.

## Modèles disponibles

### LSTM Predictor

**Force** : Excellente capacité à modéliser les tendances et motifs spécifiques à un actif particulier.

**Structure** :
- Couches LSTM bidirectionnelles
- Couches de dropout pour éviter le surapprentissage
- Sorties de classification (direction du prix) et de régression (magnitude du mouvement)

**Hyperparamètres clés** :
- `sequence_length` : Nombre de périodes d'historique utilisées (défaut : 60)
- `lstm_units` : Nombre d'unités LSTM (défaut : 50)
- `dropout_rate` : Taux de dropout (défaut : 0.2)
- `prediction_horizon` : Nombre de périodes futures à prédire (défaut : 5)

### Transformer Strategy

**Force** : Capacité à capturer les relations complexes entre différents actifs et à intégrer des contextes plus larges.

**Structure** :
- Architecture basée sur l'encodeur Transformer
- Mécanisme d'attention multi-têtes
- Capacité à traiter des données multi-actifs en parallèle

**Hyperparamètres clés** :
- `d_model` : Dimension du modèle (défaut : 64)
- `nhead` : Nombre de têtes d'attention (défaut : 4)
- `num_layers` : Nombre de couches d'encodeur (défaut : 2)
- `sequence_length` : Nombre de périodes d'historique (défaut : 60)

## Scripts d'entraînement

### `train_lstm_model.py`

Entraîne un modèle LSTM pour un actif spécifique.

```bash
python scripts/train_lstm_model.py --symbol BTC-USD --lookback 180 --epochs 100
```

Options principales :
- `--symbol` : Symbole de l'actif (obligatoire)
- `--lookback` : Nombre de jours d'historique (défaut : 180)
- `--sequence_length` : Longueur des séquences (défaut : 60)
- `--epochs` : Nombre d'époques d'entraînement (défaut : 50)

### `train_transformer_model.py`

Entraîne un modèle Transformer sur plusieurs actifs.

```bash
python scripts/train_transformer_model.py --symbols BTC-USD,ETH-USD,AAPL --epochs 100
```

Options principales :
- `--symbols` : Liste des symboles séparés par des virgules (obligatoire)
- `--lookback` : Nombre de jours d'historique (défaut : 180)
- `--epochs` : Nombre d'époques d'entraînement (défaut : 50)
- `--use_gpu` : Utiliser le GPU si disponible (facultatif)

### `train_all_models.py`

Entraîne tous les modèles en une seule commande.

```bash
python scripts/train_all_models.py --days 90 --top_assets an
```

Options principales :
- `--days` : Nombre de jours d'historique (défaut : 180)
- `--top_assets` : Nombre d'actifs populaires à inclure (défaut : 10)
- `--include_stocks` : Inclure les actions populaires
- `--include_crypto` : Inclure les crypto-monnaies populaires
- `--epochs` : Nombre d'époques d'entraînement (défaut : 50)
- `--custom_stocks_file` : Chemin vers un CSV contenant des symboles d'actions personnalisés
- `--custom_crypto_file` : Chemin vers un CSV contenant des symboles de crypto-monnaies personnalisés
- `--batch_mode` : Traiter les symboles par lots (pour les grandes listes)
- `--batch_size` : Taille des lots en mode batch (défaut : 20)
- `--max_symbols` : Limite le nombre total de symboles (0 = pas de limite)

### `list_trained_models.py`

Affiche les détails des modèles entraînés disponibles.

```bash
python scripts/list_trained_models.py
```

### `get_all_symbols.py`

Récupère tous les symboles disponibles via l'API Alpaca.

```bash
python scripts/get_all_symbols.py
```

## Exemples d'utilisation

### Scénario 1 : Entraînement rapide pour démarrer

Pour entraîner rapidement des modèles sur les actifs les plus populaires :

```bash
python scripts/train_all_models.py --days 60 --top_assets 10
```

Cette commande entraînera des modèles LSTM pour les 10 actions et 10 crypto-monnaies les plus populaires, ainsi qu'un modèle Transformer qui les inclut tous, en utilisant 60 jours d'historique.

### Scénario 2 : Entraînement pour un actif spécifique

Pour entraîner un modèle LSTM de haute qualité pour un actif spécifique :

```bash
python scripts/train_lstm_model.py --symbol BTC-USD --lookback 365 --epochs 200 --sequence_length 90
```

Cette commande entraînera un modèle LSTM pour Bitcoin avec un an d'historique, 200 époques d'entraînement et une séquence plus longue pour capturer les tendances à plus long terme.

### Scénario 3 : Entraînement sur tous les actifs disponibles

Pour entraîner des modèles sur un grand nombre d'actifs :

```bash
# Étape 1 : Récupérer tous les symboles disponibles
python scripts/get_all_symbols.py

# Étape 2 : Entraîner les modèles avec le mode batch
python scripts/train_all_models.py --custom_crypto_file data/all_crypto_YYYYMMDD.csv --batch_mode --batch_size 20 --days 60
```

## Optimisation des modèles

### Hyperparamètres clés pour LSTM

| Paramètre | Description | Valeur par défaut | Pour volatilité élevée | Pour tendances longues |
|-----------|-------------|-------------------|------------------------|------------------------|
| sequence_length | Périodes d'historique | 60 | 30-40 | 90-120 |
| lstm_units | Complexité du modèle | 50 | 80-100 | 40-60 |
| dropout_rate | Régularisation | 0.2 | 0.3-0.4 | 0.1-0.2 |
| epochs | Cycles d'entraînement | 50 | 80-100 | 50-70 |

### Hyperparamètres clés pour Transformer

| Paramètre | Description | Valeur par défaut | Pour multi-actifs | Pour précision |
|-----------|-------------|-------------------|-------------------|----------------|
| d_model | Dimension du modèle | 64 | 128 | 96 |
| nhead | Têtes d'attention | 4 | 8 | 6 |
| num_layers | Profondeur | 2 | 3-4 | 2-3 |
| dropout_rate | Régularisation | 0.1 | 0.2 | 0.15 |

## Résolution des problèmes

### Problèmes d'accès aux données

#### Erreur 403 pour les actions

Si vous rencontrez l'erreur `subscription does not permit querying recent SIP data` :

1. **Cause** : Votre abonnement Alpaca ne permet pas d'accéder aux données de marché SIP récentes.
2. **Solution** :
   - Vérifiez votre niveau d'abonnement Alpaca
   - Utilisez des données historiques plus anciennes avec `--days 365` (les données anciennes sont souvent accessibles)
   - Configurez une source de données alternative dans le fichier `.env`

#### Erreur pour certaines crypto-monnaies

Si vous rencontrez l'erreur `Could not get crypto data for ADA-USD from Alpaca` :

1. **Cause** : Certaines crypto-monnaies spécifiques peuvent ne pas être disponibles via l'API que vous utilisez.
2. **Solution** :
   - Exécutez `python scripts/get_all_symbols.py` pour obtenir une liste des crypto-monnaies réellement disponibles
   - Utilisez uniquement les symboles confirmés dans cette liste
   - Si vous avez besoin de ces crypto-monnaies spécifiques, envisagez d'ajouter une source de données alternative

### Problèmes d'entraînement

#### Valeurs NaN dans les fonctions de perte (modèle Transformer)

Si l'entraînement du Transformer affiche des valeurs `nan` dans la fonction de perte :

1. **Cause** : Problèmes d'explosion de gradient ou de normalisation des données.
2. **Solution** :
   - Réduisez le taux d'apprentissage avec `--learning_rate 0.0001`
   - Augmentez le taux de dropout avec `--dropout_rate 0.2`
   - Réduisez la dimension du modèle avec `--d_model 32`

#### Arrêt précoce avec une précision faible

Si l'entraînement s'arrête prématurément avec une faible précision :

1. **Cause** : Données insuffisantes ou hyperparamètres inadaptés.
2. **Solution** :
   - Augmentez la quantité de données avec `--lookback 365`
   - Réduisez la complexité du modèle (moins d'unités, moins de couches)
   - Ajustez la patience de l'arrêt précoce en modifiant `early_stopping_patience` dans le code

### Problèmes de mémoire

Si vous rencontrez des erreurs de mémoire insuffisante :

1. **Solution pour GPU** :
   - Réduisez la taille du lot avec `--batch_size 16` ou `--batch_size 8`
   - Réduisez la dimension du modèle avec `--d_model 32`
   - Utilisez le mode CPU avec `--use_gpu false`

2. **Solution pour CPU** :
   - Utilisez le mode batch avec `--batch_mode --batch_size 10`
   - Limitez le nombre de symboles avec `--max_symbols 50`
   - Réduisez les tailles de séquence avec `--sequence_length 30`

## Questions fréquentes

### Comment savoir si mes modèles sont correctement entraînés ?

Après l'entraînement, vérifiez les métriques suivantes :
- **Précision de validation** : Idéalement supérieure à 0.55 (55%)
- **Perte de validation** : Devrait diminuer progressivement puis se stabiliser

Exécutez `python scripts/list_trained_models.py` pour voir les détails de tous vos modèles.

### Combien de temps faut-il pour entraîner les modèles ?

Le temps d'entraînement dépend de plusieurs facteurs :
- **LSTM** : ~1-2 minutes par actif sur CPU
- **Transformer** : ~5-15 minutes pour 20 actifs sur CPU
- Avec GPU, ces temps peuvent être réduits de 50-80%

### Quelle est la fréquence recommandée pour réentraîner les modèles ?

- Pour les marchés volatils (crypto) : Hebdomadaire
- Pour les marchés plus stables (actions) : Bi-mensuel ou mensuel
- Après des événements de marché significatifs : Immédiatement

### Comment choisir entre LSTM et Transformer ?

- **LSTM** : Meilleur pour les prédictions spécifiques à un actif unique
- **Transformer** : Meilleur pour capturer les relations entre actifs et les influences de marché plus larges
- Dans la pratique, le screener d'actifs utilise les deux pour obtenir une perspective complète

---

Pour plus d'informations sur l'utilisation des modèles entraînés, consultez le [Guide du Screener d'Actifs](./ASSET_SCREENER_GUIDE.md).
