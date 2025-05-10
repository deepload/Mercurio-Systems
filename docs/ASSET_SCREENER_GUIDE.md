# Guide du Screener d'Actifs MercurioAI

> [!NOTE]
> **Navigation Rapide:**
> - [🔍 Index de tous les guides](./GUIDES_INDEX.md)
> - [📈 Guide de Day Trading](./day_trading_guide.md)
> - [📊 Guide des Options](./options_trading.md)
> - [🧠 Guide d'Entraînement des Modèles](./MODEL_TRAINING_GUIDE.md)
> - [📔 Documentation Principale](./README.md)

Ce guide explique comment utiliser le système de screening d'actifs de MercurioAI pour identifier les meilleures opportunités d'investissement parmi les actions et les cryptomonnaies.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Configuration initiale](#configuration-initiale)
3. [Entraînement des modèles](#entraînement-des-modèles)
4. [Utilisation du screener](#utilisation-du-screener)
5. [Interprétation des résultats](#interprétation-des-résultats)
6. [Flux de travail recommandé](#flux-de-travail-recommandé)
7. [Personnalisation](#personnalisation)
8. [Résolution des problèmes](#résolution-des-problèmes)

## Vue d'ensemble

Le Screener d'Actifs MercurioAI analyse une large liste d'actions et de cryptomonnaies pour identifier les meilleures opportunités d'investissement à moyen terme. Il combine plusieurs stratégies d'analyse et d'intelligence artificielle pour générer un score composite pour chaque actif.

**Caractéristiques principales :**
- Analyse multi-stratégies (moyenne mobile, LSTM, MSI, Transformer)
- Calcul de métriques techniques avancées (RSI, tendances récentes, volume)
- Génération de rapports visuels interactifs (HTML, CSV, graphiques)
- Support pour les actions et les cryptomonnaies
- Intégration avec les modèles d'IA entraînés

## Configuration initiale

Avant d'utiliser le screener d'actifs, assurez-vous que votre environnement MercurioAI est correctement configuré :

1. Vérifiez que votre fichier `.env` contient les clés API nécessaires :
   ```
   ALPACA_MODE=paper
   ALPACA_PAPER_KEY=votre_clé_paper
   ALPACA_PAPER_SECRET=votre_secret_paper
   ALPACA_LIVE_KEY=votre_clé_live
   ALPACA_LIVE_SECRET=votre_secret_live
   ```

2. Assurez-vous que les dépendances Python requises sont installées :
   ```bash
   pip install -r requirements.txt
   ```

3. Créez les répertoires nécessaires s'ils n'existent pas déjà :
   ```bash
   mkdir -p models/lstm models/transformer logs reports/best_assets
   ```

## Entraînement des modèles

Pour obtenir les meilleurs résultats du screener d'actifs, vous devez d'abord entraîner les modèles d'intelligence artificielle utilisés pour l'analyse.

### Option 1 : Entraînement de tous les modèles (recommandé)

Cette option entraîne tous les modèles en une seule commande, en utilisant une liste prédéfinie d'actifs populaires :

```bash
python scripts/train_all_models.py --days 90 --top_assets 20
```

Options principales :
- `--days 90` : Utilise 90 jours d'historique pour l'entraînement
- `--top_assets 20` : Inclut les 20 actions et 20 cryptomonnaies les plus populaires
- `--epochs 100` : Facultatif, pour un entraînement plus long et potentiellement plus précis

### Option 2 : Entraînement de modèles spécifiques

Si vous souhaitez plus de contrôle, vous pouvez entraîner les modèles individuellement :

**Entraîner un modèle LSTM pour un actif spécifique :**
```bash
python scripts/train_lstm_model.py --symbol BTC-USD --lookback 180 --epochs 100
```

**Entraîner un modèle Transformer pour plusieurs actifs :**
```bash
python scripts/train_transformer_model.py --symbols BTC-USD,ETH-USD,AAPL,MSFT --epochs 100
```

### Vérification des modèles entraînés

Vous pouvez vérifier quels modèles sont déjà entraînés et disponibles :

```bash
python scripts/list_trained_models.py
```

## Utilisation du screener

Une fois les modèles entraînés, vous pouvez utiliser le screener d'actifs pour identifier les meilleures opportunités d'investissement.

### Utilisation de base

Pour exécuter le screener avec les paramètres par défaut :

```bash
python scripts/best_assets_screener.py
```

Cela analysera les 50 meilleures actions et 100 meilleures cryptomonnaies (listes prédéfinies) sur les 30 derniers jours.

### Utilisation avancée

Vous pouvez personnaliser l'analyse en spécifiant vos propres listes d'actifs et paramètres :

```bash
python scripts/best_assets_screener.py --stocks AAPL,MSFT,GOOGL,AMZN,TSLA --crypto BTC-USD,ETH-USD,SOL-USD --lookback 60
```

Options principales :
- `--stocks` : Liste personnalisée d'actions à analyser
- `--crypto` : Liste personnalisée de cryptomonnaies à analyser
- `--lookback` : Nombre de jours d'historique à analyser
- `--top_stocks` : Nombre d'actions à inclure dans le rapport (défaut : 50)
- `--top_crypto` : Nombre de cryptomonnaies à inclure dans le rapport (défaut : 100)

## Interprétation des résultats

Le screener génère plusieurs fichiers de sortie dans le répertoire `reports/best_assets/[date]/` :

1. **report.html** : Rapport HTML interactif avec mise en forme conditionnelle
2. **top_stocks.csv** et **top_crypto.csv** : Fichiers CSV contenant les données brutes
3. **top_stocks_chart.png** et **top_crypto_chart.png** : Graphiques comparatifs

### Comprendre le score composite

Le score composite (0-100) est calculé en combinant :
- **40%** : Signaux des stratégies (BUY, SELL, HOLD) pondérés par leur confiance
- **30%** : Performance en backtest (rendement simulé)
- **30%** : Métriques techniques (tendance récente, RSI, volume, volatilité)

Interprétation des scores :
- **70-100** : Signal d'achat fort, excellente opportunité
- **50-70** : Signal d'achat modéré, à surveiller
- **0-50** : Signal faible ou négatif, peu recommandé pour un investissement à moyen terme

### Métriques techniques

En plus du score composite, le rapport fournit plusieurs métriques techniques utiles :
- **RSI** : Indice de force relative (>70 = potentiellement suracheté, <30 = potentiellement survendu)
- **Tendance récente** : Variation de prix récente en pourcentage
- **Volatilité** : Mesure de la volatilité annualisée
- **Volume** : Tendance du volume de transactions récent

## Flux de travail recommandé

Pour obtenir les meilleurs résultats avec le Screener d'Actifs MercurioAI, nous recommandons le flux de travail suivant :

1. **Hebdomadaire : Mise à jour des modèles**
   ```bash
   python scripts/train_all_models.py --days 90 --top_assets 20
   ```

2. **Quotidien : Exécution du screener**
   ```bash
   python scripts/best_assets_screener.py
   ```

3. **Analyse des résultats**
   - Ouvrez le rapport HTML généré
   - Identifiez les actifs avec les scores les plus élevés
   - Examinez les signaux des différentes stratégies
   - Vérifiez les métriques techniques

4. **Prise de décision**
   - Sélectionnez les 5-10 actifs les plus prometteurs
   - Effectuez une analyse plus approfondie si nécessaire
   - Intégrez ces actifs dans votre portefeuille ou liste de surveillance

## Personnalisation

Le Screener d'Actifs MercurioAI est hautement personnalisable. Voici quelques points que vous pouvez modifier :

### Personnalisation des listes d'actifs

Vous pouvez modifier les listes d'actifs par défaut en éditant les variables `DEFAULT_STOCKS` et `DEFAULT_CRYPTO` dans le fichier `scripts/best_assets_screener.py`.

### Ajustement de la formule de score

La méthode `_calculate_composite_score` dans la classe `AssetEvaluator` peut être modifiée pour ajuster la pondération des différents facteurs dans le calcul du score.

### Personnalisation du rapport

Le format du rapport HTML peut être modifié en éditant la méthode `generate_report` dans la classe `AssetScreener`.

## Résolution des problèmes

Voici quelques problèmes courants et leurs solutions :

### Erreurs d'API

Si vous rencontrez des erreurs 403 (Forbidden) lors de l'accès aux données, vérifiez :
- Que vos clés API sont correctes et actives
- Que votre compte a accès aux données demandées
- Que vous n'avez pas dépassé les limites de requêtes

### Modèles non entraînés

Si le screener utilise des valeurs par défaut au lieu des prédictions des modèles :
- Vérifiez que les modèles sont correctement entraînés (`python scripts/list_trained_models.py`)
- Réentraînez les modèles si nécessaire
- Assurez-vous que les symboles analysés correspondent aux symboles pour lesquels vous avez entraîné des modèles

### Données insuffisantes

Si certains actifs sont ignorés avec le message "Données insuffisantes" :
- Augmentez le nombre d'actifs analysés
- Réduisez la période d'analyse (`--lookback`)
- Vérifiez que les symboles sont correctement formatés (par exemple, "BTC-USD" au lieu de "BTC/USD")

---

Pour plus d'informations sur les scripts individuels, consultez le [Guide des Scripts](./SCRIPTS_GUIDE.md).
