# Guide des Scripts Utilitaires de MercurioAI

Ce document décrit les scripts utilitaires disponibles dans le dossier `/scripts` de MercurioAI. Ces scripts sont conçus pour faciliter l'utilisation, le test et la démonstration des fonctionnalités de la plateforme.

Chaque script est documenté avec ses paramètres disponibles et des exemples de cas d'utilisation pour vous aider à tirer le meilleur parti de la plateforme Mercurio AI.

## Table des matières

1. [Scripts de test](#scripts-de-test)
2. [Scripts de trading](#scripts-de-trading)
3. [Scripts d'analyse](#scripts-d'analyse)
4. [Scripts de démonstration](#scripts-de-démonstration)
5. [Scripts d'optimisation](#scripts-d'optimisation)
6. [Scripts de visualisation](#scripts-de-visualisation)

## Scripts de test

Ces scripts permettent de vérifier la configuration et le bon fonctionnement des différentes parties du système.

### `test_alpaca.py`

**Fonction** : Teste la connexion à l'API Alpaca en utilisant les informations d'identification configurées dans le fichier `.env`.

**Paramètres** : Ce script n'a pas de paramètres en ligne de commande, mais il est configuré via les variables d'environnement suivantes dans le fichier `.env` :

- `ALPACA_MODE` : Détermine le mode de trading ("paper" ou "live")
- `ALPACA_PAPER_KEY` et `ALPACA_PAPER_SECRET` : Clés API pour le paper trading
- `ALPACA_LIVE_KEY` et `ALPACA_LIVE_SECRET` : Clés API pour le live trading
- `ALPACA_PAPER_URL` et `ALPACA_LIVE_URL` : URLs de base pour les environnements paper et live

**Utilisation** :
```bash
python scripts/test_alpaca.py
```

**Description** : Ce script vérifie que les clés API d'Alpaca sont correctement configurées et que la connexion fonctionne. Il utilise automatiquement les clés appropriées selon le mode configuré dans `ALPACA_MODE` (paper ou live).

**Cas d'utilisation** :
1. **Validation initiale** : Après avoir configuré vos clés API pour la première fois
2. **Dépannage de connexion** : Si vous rencontrez des erreurs avec les API Alpaca
3. **Vérification avant trading live** : Avant de passer du paper trading au live trading

### `test_alpaca_connection.py`

**Fonction** : Version détaillée du test de connexion à Alpaca.

**Utilisation** :
```bash
python scripts/test_alpaca_connection.py
```

**Description** : Ce script fournit des informations plus détaillées sur le compte, y compris le solde, la valeur du portefeuille et le statut du marché.

### `test_api_access.py`

**Fonction** : Teste l'accès à diverses API de données de marché.

**Utilisation** :
```bash
python scripts/test_api_access.py
```

**Description** : Vérifie les connexions à toutes les sources de données configurées (Alpaca, Yahoo Finance, etc.) et affiche des informations sur chaque connexion.

### `test_market_data.py`

**Fonction** : Teste le service de données de marché de MercurioAI.

**Utilisation** :
```bash
python scripts/test_market_data.py
```

**Description** : Vérifie que le service MarketDataService fonctionne correctement et peut récupérer des données historiques et en temps réel.

### `test_stocks_and_crypto.py`

**Fonction** : Teste spécifiquement la récupération de données pour les actions et les cryptomonnaies.

**Utilisation** :
```bash
python scripts/test_stocks_and_crypto.py
```

**Description** : Vérifie que MercurioAI peut récupérer des données pour les symboles d'actions (comme AAPL, MSFT) et de cryptomonnaies (comme BTC-USD, ETH-USD).

## Scripts de trading

Ces scripts permettent d'exécuter différentes stratégies de trading.

### `auto_trader.py`

**Fonction** : Agent de trading automatisé avancé.

**Paramètres** :
- `--config <chemin>` : Chemin vers le fichier de configuration JSON (défaut: "config/agent_config.json")

**Format du fichier de configuration** :
```json
{
  "symbols": ["AAPL", "MSFT", "BTC-USD", "ETH-USD"],
  "strategies": ["MovingAverageStrategy", "LSTMPredictorStrategy"],
  "strategy_params": {
    "MovingAverageStrategy": {
      "short_window": 10,
      "long_window": 30,
      "use_ml": true
    },
    "LSTMPredictorStrategy": {
      "lookback_periods": 30,
      "prediction_periods": 5
    }
  },
  "risk_level": "moderate",  // "conservative", "moderate", "aggressive"
  "check_interval": 300,     // Intervalle de vérification en secondes
  "max_positions": 5,        // Nombre maximum de positions simultanées
  "position_size_pct": 0.1   // Taille de position en % du portefeuille
}
```

**Utilisation** :
```bash
python scripts/auto_trader.py --config configs/my_trading_config.json
```

**Description** : Un agent de trading complet qui peut sélectionner automatiquement entre différentes stratégies, optimiser les paramètres et prendre des décisions basées sur l'analyse de marché.

**Cas d'utilisation** :
1. **Trading automatisé multi-stratégies** : Pour exécuter simultanément plusieurs stratégies et basculer automatiquement entre elles en fonction des conditions du marché
2. **Trading adapté aux régimes de marché** : L'agent analyse en continu le régime du marché (haussier, baissier, latéral, volatil) et adapte ses stratégies en conséquence
3. **Trading avec gestion des risques avancée** : Intègre des fonctionnalités de gestion des risques comme les stop-loss dynamiques et l'ajustement des positions en fonction de la volatilité

### `enhanced_trader.py`

**Fonction** : Version améliorée du trader standard avec fonctionnalités supplémentaires.

**Paramètres** :
- `--strategy <nom>` : Nom de la stratégie à utiliser (défaut: "MovingAverageStrategy")
- `--symbols <liste>` : Liste de symboles séparés par des virgules (défaut: "AAPL,MSFT,GOOGL")
- `--interval <secondes>` : Intervalle de vérification en secondes (défaut: 300)
- `--risk_limit <ratio>` : Limite de risque par position en pourcentage (défaut: 0.02)
- `--config <chemin>` : Chemin vers un fichier de configuration JSON optionnel

**Utilisation** :
```bash
python scripts/enhanced_trader.py --strategy LSTMPredictorStrategy --symbols AAPL,MSFT,BTC-USD --interval 600 --risk_limit 0.03
```

**Description** : Ajoute des fonctionnalités comme la gestion avancée des risques, les rapports détaillés et une gestion améliorée des erreurs.

**Cas d'utilisation** :
1. **Trading avec paramètres personnalisés** : Pour exécuter une stratégie spécifique avec des symboles et des paramètres personnalisés
2. **Trading avec risque contrôlé** : Pour limiter le risque par position tout en permettant un trading autonome
3. **Surveillance à différents intervalles** : Pour adapter la fréquence de vérification des signaux selon le style de trading (court, moyen ou long terme)

### `trading_agent.py` et `trading_agent_fixed.py`

**Fonction** : Agents de trading qui simulent une session de trading complète.

**Paramètres** :
- `--mode <mode>` : Mode de trading ("paper" ou "backtest", défaut: "paper")
- `--strategy <nom>` : Nom de la stratégie à utiliser (défaut: "MovingAverageStrategy")
- `--symbols <liste>` : Liste de symboles séparés par des virgules (défaut: "AAPL,MSFT,GOOGL")
- `--capital <montant>` : Capital initial en USD (défaut: 10000)
- `--duration <heures>` : Durée de la simulation en heures (défaut: 24)
- `--log_level <niveau>` : Niveau de journalisation ("DEBUG", "INFO", "WARNING", "ERROR", défaut: "INFO")

**Utilisation** :
```bash
# Trading papier de base
python scripts/trading_agent_fixed.py --mode paper --strategy MovingAverageStrategy --symbols AAPL,MSFT,BTC-USD

# Backtest sur une durée spécifique
python scripts/trading_agent_fixed.py --mode backtest --strategy LSTMPredictorStrategy --symbols BTC-USD,ETH-USD --duration 48

# Trading papier avec capital personnalisé
python scripts/trading_agent_fixed.py --strategy MovingAverageStrategy --symbols TSLA --capital 50000
```

**Description** : Le script `trading_agent.py` est la version originale, tandis que `trading_agent_fixed.py` contient des corrections pour les problèmes connus. Les deux implémentent un agent de trading complet qui gère tout le cycle de trading, de l'acquisition des données à l'exécution des ordres.

**Cas d'utilisation** :
1. **Simulation de trading complète** : Pour exécuter une simulation de trading sur une période définie
2. **Comparaison de performances** : Pour tester différentes stratégies sur les mêmes symboles
3. **Tests sur différentes classes d'actifs** : Pour comparer les performances sur les actions vs. les cryptomonnaies

## Scripts d'analyse

Ces scripts permettent d'analyser le marché et d'évaluer les performances des stratégies.

### `market_analyzer.py`

**Fonction** : Analyse approfondie du marché pour divers instruments.

**Paramètres** :
- `--symbols <liste>` : Liste de symboles séparés par des virgules à analyser (défaut : "AAPL,MSFT,GOOGL,BTC-USD,ETH-USD")
- `--lookback <jours>` : Période d'analyse en jours (défaut : 90)
- `--output <format>` : Format de sortie ("console", "csv", "json", "html", défaut : "console")
- `--anomaly_scan` : Activer la détection d'anomalies (sans valeur)
- `--sentiment` : Inclure l'analyse de sentiment (sans valeur)
- `--save_path <chemin>` : Chemin où sauvegarder les résultats (défaut : "reports/market_analysis")

**Utilisation** :
```bash
# Analyse de base avec détection d'anomalies
python scripts/market_analyzer.py --symbols AAPL,MSFT,BTC-USD --lookback 90 --anomaly_scan

# Analyse complète avec sentiment et exportation JSON
python scripts/market_analyzer.py --symbols BTC-USD,ETH-USD --output json --sentiment --save_path reports/crypto_analysis

# Génération d'un rapport HTML
python scripts/market_analyzer.py --symbols AAPL,TSLA,AMZN --output html
```

**Description** : Fournit une analyse technique détaillée, des indicateurs de sentiment et des prévisions pour les symboles spécifiés. Le script identifie également les régimes de marché (haussier, baissier, latéral, volatil) et peut détecter des anomalies potentielles.

**Cas d'utilisation** :
1. **Analyse pré-trading** : Pour évaluer les conditions du marché avant de déployer des stratégies
2. **Détection d'anomalies** : Pour identifier des mouvements inhabituels ou des manipulations potentielles
3. **Analyse de régime** : Pour déterminer quel type de stratégie serait le plus efficace dans les conditions actuelles
4. **Analyse de sentiment** : Pour combiner l'analyse technique avec le sentiment du marché

### `run_all_strategies.py`

**Fonction** : Exécute toutes les stratégies disponibles sur un ensemble de symboles.

**Utilisation** :
```bash
python scripts/run_all_strategies.py --symbols AAPL,MSFT,BTC-USD
```

**Description** : Permet de comparer rapidement les performances de toutes les stratégies implémentées dans le système.

## Scripts de démonstration

Ces scripts montrent les fonctionnalités de MercurioAI à travers des exemples concrets.

### `first_script.py`

**Fonction** : Script d'introduction pour les nouveaux utilisateurs.

**Utilisation** :
```bash
python scripts/first_script.py
```

**Description** : Un exemple simple qui montre comment obtenir des données et exécuter une stratégie de base.

### `simplified_demo.py`

**Fonction** : Démo simplifiée de MercurioAI.

**Utilisation** :
```bash
python scripts/simplified_demo.py
```

**Description** : Une démonstration épurée qui présente les fonctionnalités essentielles du système.

### `demo_enhanced_architecture.py` et `demo_phase2_enhancements.py`

**Fonction** : Démontrent les améliorations architecturales et les nouvelles fonctionnalités.

**Utilisation** :
```bash
python scripts/demo_enhanced_architecture.py
```

**Description** : Ces scripts illustrent les améliorations apportées à l'architecture et les nouvelles fonctionnalités introduites dans les différentes phases de développement.

## Scripts d'optimisation

Ces scripts permettent d'optimiser les paramètres des stratégies de trading.

### `optimize_moving_average.py`

**Fonction** : Trouve les paramètres optimaux pour la stratégie de moyenne mobile.

**Utilisation** :
```bash
python scripts/optimize_moving_average.py --symbol AAPL --period 90
```

**Description** : Utilise diverses techniques pour trouver les meilleures fenêtres court terme et long terme pour la stratégie de moyenne mobile.

### `optimized_portfolio.py`

**Fonction** : Optimise l'allocation du portefeuille pour un ensemble de symboles.

**Utilisation** :
```bash
python scripts/optimized_portfolio.py --symbols AAPL,MSFT,GOOGL,BTC-USD
```

**Description** : Utilise des techniques d'optimisation de portefeuille pour maximiser le ratio de Sharpe ou minimiser le risque.

## Scripts de visualisation

Ces scripts génèrent des visualisations et des tableaux de bord pour suivre les performances de trading.

### `strategy_dashboard.py` et `trading_dashboard.py`

**Fonction** : Tableaux de bord interactifs pour évaluer les stratégies et suivre le trading.

**Utilisation** :
```bash
python scripts/strategy_dashboard.py
```

**Description** : Génèrent des interfaces utilisateur Streamlit qui permettent d'explorer interactivement les performances des stratégies (`strategy_dashboard.py`) ou des activités de trading (`trading_dashboard.py`).

### `comprehensive_dashboard.py`

**Fonction** : Tableau de bord complet qui combine toutes les métriques et visualisations.

**Utilisation** :
```bash
python scripts/comprehensive_dashboard.py
```

**Description** : Une interface utilisateur avancée qui intègre analyse de marché, suivi de portefeuille, performance des stratégies et journaux de trading.

### `generate_strategy_comparison_plot.py`

**Fonction** : Génère des graphiques comparatifs pour différentes stratégies.

**Utilisation** :
```bash
python scripts/generate_strategy_comparison_plot.py --output comparison.png
```

**Description** : Crée des visualisations qui comparent les performances de différentes stratégies sur un même graphique.

## Utilitaires auxiliaires

### `comprehensive_simulation.py`

**Fonction** : Simulation complète de trading sur des données historiques.

**Utilisation** :
```bash
python scripts/comprehensive_simulation.py --start 2022-01-01 --end 2022-12-31
```

**Description** : Effectue une simulation détaillée de toutes les stratégies sur une période historique définie, avec des rapports complets.

### `simulation_utils.py`

**Fonction** : Utilitaires pour les simulations de trading.

**Description** : Ce fichier n'est pas destiné à être exécuté directement, mais contient des fonctions utilisées par d'autres scripts de simulation.

---

## Comment utiliser ces scripts

1. Assurez-vous que votre fichier `.env` est correctement configuré avec les informations d'API nécessaires
2. Activez votre environnement virtuel Python
3. Exécutez les scripts depuis la racine du projet pour garantir que les chemins d'importation fonctionnent correctement

## Résolution des problèmes courants

- **Erreurs d'importation** : Assurez-vous d'exécuter les scripts depuis la racine du projet
- **Erreurs d'API** : Vérifiez vos clés API dans le fichier `.env` avec les scripts de test
- **Mode de trading** : Pour basculer entre paper trading et live trading, modifiez `ALPACA_MODE` dans `.env`

## Exemple de flux de travail

1. Utilisez `test_alpaca_connection.py` pour vérifier votre configuration
2. Exécutez `test_stocks_and_crypto.py` pour confirmer l'accès aux données
3. Utilisez `optimize_moving_average.py` pour optimiser votre stratégie
4. Lancez `strategy_dashboard.py` pour visualiser les performances
5. Démarrez le trading papier avec `auto_trader.py` pour tester en conditions réelles
