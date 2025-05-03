# Guide de Trading Crypto avec Alpaca sur Mercurio AI

Ce document détaille l'utilisation des scripts de trading de cryptomonnaies avec l'API Alpaca dans la plateforme Mercurio AI. Vous trouverez ici les informations concernant la configuration, l'utilisation des différents scripts et les fonctionnalités disponibles.

## Table des matières

1. [Prérequis](#prérequis)
2. [Configuration](#configuration)
3. [Scripts disponibles](#scripts-disponibles)
4. [Fonctionnalités](#fonctionnalités)
5. [Stratégies implémentées](#stratégies-implémentées)
6. [Rapports et journalisation](#rapports-et-journalisation)
7. [Dépannage](#dépannage)

## Prérequis

- Compte Alpaca avec API keys
- Niveau d'abonnement Alpaca approprié (Basic, Pro, ou Premium)
- Python 3.7+ avec les dépendances listées dans `requirements.txt`

## Configuration

### Fichier .env

Le système utilise un fichier `.env` pour stocker les informations sensibles. Voici les variables nécessaires pour le trading crypto :

```
# Mode de trading (paper ou live)
ALPACA_MODE=paper

# Clés API pour le paper trading
ALPACA_PAPER_KEY=votre_clé_paper
ALPACA_PAPER_SECRET=votre_secret_paper

# Clés API pour le live trading
ALPACA_LIVE_KEY=votre_clé_live
ALPACA_LIVE_SECRET=votre_secret_live

# URLs (optionnel - valeurs par défaut fournies par le système)
ALPACA_PAPER_URL=https://paper-api.alpaca.markets
ALPACA_LIVE_URL=https://api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets

# Niveau d'abonnement Alpaca (1=Basic, 2=Pro, 3=Premium)
ALPACA_SUBSCRIPTION_LEVEL=1
```

### Compatibilité avec les niveaux d'abonnement

Le système est conçu pour fonctionner avec différents niveaux d'abonnement Alpaca :

- **Niveau 1 (Basic)** : Fonctionnalités limitées, utilise des barres de prix au lieu de quotes en temps réel
- **Niveau 2 (Pro)** : Accès à davantage de fonctionnalités API
- **Niveau 3 (Premium)** : Accès complet à toutes les fonctionnalités API

Le code s'adapte automatiquement à votre niveau d'abonnement configuré dans `.env`.

## Scripts disponibles

### alpaca_crypto_trader.py

Script principal de trading crypto qui utilise directement l'API Alpaca sans passer par les services Mercurio.

**Utilisation** :
```bash
python alpaca_crypto_trader.py --duration 1h --log-level INFO
```

**Paramètres** :
- `--duration` : Durée de la session de trading (1h, 4h, 8h, ou custom)
- `--custom-seconds` : Durée personnalisée en secondes si --duration=custom
- `--log-level` : Niveau de journalisation (DEBUG, INFO, WARNING, ERROR)

### scripts/run_overnight_crypto_trader.py

Script utilitaire pour lancer un trader crypto pendant la nuit avec des paramètres optimisés.

**Utilisation** :
```bash
python scripts/run_overnight_crypto_trader.py
```

**Paramètres** :
- `--position-size` : Taille de position en pourcentage du portefeuille (défaut: 0.02 = 2%)
- `--stop-loss` : Stop loss en pourcentage (défaut: 0.03 = 3%)
- `--take-profit` : Take profit en pourcentage (défaut: 0.06 = 6%)
- `--fast-ma` : Période de la moyenne mobile rapide (défaut: 5 minutes)
- `--slow-ma` : Période de la moyenne mobile lente (défaut: 15 minutes)

### scripts/simple_crypto_trader.py

Version simplifiée pour les débutants qui veut tester le trading crypto.

**Utilisation** :
```bash
python scripts/simple_crypto_trader.py
```

### scripts/run_crypto_daytrader.py

Version qui utilise les services internes de Mercurio AI plutôt que d'accéder directement à l'API Alpaca.

**Utilisation** :
```bash
python scripts/run_crypto_daytrader.py
```

## Fonctionnalités

### Session de trading

Les scripts prennent en charge différentes durées de session :
- ONE_HOUR (3600 secondes)
- FOUR_HOURS (14400 secondes)
- EIGHT_HOURS (28800 secondes)
- NIGHT_RUN (32400 secondes / 9 heures)
- CUSTOM (durée personnalisée)

### Gestion du portefeuille

- Suivi en temps réel de la valeur du portefeuille
- Suivi des positions ouvertes
- Calcul des profits/pertes par position et global

### Gestion des risques

- Stop-loss configurables
- Take-profit configurables
- Limitation de la taille des positions

### Journal et rapports

- Journalisation détaillée des activités de trading
- Journalisation dans des fichiers horodatés
- Rapports de performance détaillés à la fin des sessions

## Stratégies implémentées

### Croisement de moyennes mobiles

La stratégie par défaut utilisée dans le trader crypto est basée sur le croisement de moyennes mobiles :

- Une moyenne mobile rapide (par défaut: 5 périodes)
- Une moyenne mobile lente (par défaut: 15 périodes)

**Signaux** :
- **Achat** : Quand la MA rapide croise au-dessus de la MA lente
- **Vente** : Quand la MA rapide croise en dessous de la MA lente

### Gestion des ordres

Tous les ordres sont placés en tant qu'ordres au marché avec une validité "Good Till Cancelled" (GTC).

## Rapports et journalisation

### Fichiers de log

Les fichiers de log sont générés avec un horodatage dans le nom de fichier :
```
crypto_trader_log_YYYYMMDD_HHMMSS.txt
```

### Rapports de performance

À la fin de chaque session, un rapport de performance détaillé est généré :
```
crypto_trading_report_YYYYMMDD_HHMMSS.txt
```

Ce rapport contient :
- Durée de la session
- Valeur initiale et finale du portefeuille
- Profit/perte global et en pourcentage
- Liste des positions ouvertes
- Historique des transactions importantes

## Dépannage

### Problèmes d'API

Si vous rencontrez des erreurs liées à l'API Alpaca :

1. Vérifiez que vos clés API sont correctes dans le fichier `.env`
2. Confirmez que votre niveau d'abonnement correspond à la valeur dans `ALPACA_SUBSCRIPTION_LEVEL`
3. Vérifiez que vous avez suffisamment de fonds dans votre compte

### Erreur "insufficient balance"

Si vous rencontrez une erreur du type :
```
insufficient balance for USDT (requested: 1990.05801181776, available: 0)
```

Cela signifie que vous essayez de trader une paire comme AVAX/USDT, mais que vous n'avez pas de USDT dans votre compte. Solutions :

1. **Utilisez le système par défaut** qui ne traite que les paires avec USD
2. **Ajoutez manuellement des USDT** à votre compte Alpaca Paper via leur interface
3. **Modifiez la configuration** pour seulement inclure les paires basées sur USD

### Compatibilité avec les niveaux d'abonnement

- **Niveau 1 (Basic)** : 
  - ✅ Trading basique fonctionnel
  - ❌ Pas d'accès aux quotes en temps réel, utilise les prix des barres

- **Niveau 2-3 (Pro/Premium)** : 
  - ✅ Toutes les fonctionnalités disponibles
  - ✅ Accès aux quotes en temps réel pour des prix plus précis

### Gestion des devises

Par défaut, les comptes Alpaca Paper Trading ont généralement des USD disponibles, mais pas forcément d'autres devises comme USDT ou USDC. Le système est configuré pour :

- **Filtrer automatiquement** les paires de trading en ne gardant que celles avec USD (par exemple ETH/USD, BTC/USD)
- **Éviter les paires** nécessitant USDT ou USDC sauf si vous avez explicitement ces devises dans votre compte
- **Afficher le solde disponible** en USD au démarrage du script

### Dépannage des rapports

Si les rapports ne sont pas générés correctement, vérifiez :
1. Les permissions d'écriture dans le répertoire courant
2. Que la session se termine normalement et n'est pas interrompue brutalement
