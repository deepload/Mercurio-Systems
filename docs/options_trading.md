# Trading d'Options avec Mercurio AI

> [!NOTE]
> **Navigation Rapide:**
> - [🔍 Index de tous les guides](./GUIDES_INDEX.md)
> - [📈 Guide de Day Trading](./day_trading_guide.md)
> - [🧠 Guide d'Entraînement des Modèles](./model_training_guide.md)
> - [📔 Documentation Principale](./README.md)

## Introduction

Ce document décrit les fonctionnalités de trading d'options implémentées dans la plateforme Mercurio AI. Le module de trading d'options permet d'exploiter l'abonnement Alpaca AlgoTrader Plus avec Options Trading Level 1 pour exécuter des stratégies de trading d'options basées sur les signaux générés par les autres stratégies du système.

## Architecture

Le système de trading d'options s'intègre parfaitement à l'architecture existante de Mercurio AI et se compose de deux composants principaux :

1. **Options Service** - Un service qui interagit avec l'API Alpaca pour les opérations liées aux options
2. **Options Strategy** - Une stratégie qui transforme les signaux des stratégies existantes en opportunités de trading d'options

### Diagramme de flux

```
┌───────────────┐    ┌────────────────┐    ┌──────────────────┐
│ Stratégies ML │───▶│ Options Strategy│───▶│ Options Service  │
│ existantes    │    │                │    │                  │
└───────────────┘    └────────────────┘    └──────────────────┘
                            │                      │
                            ▼                      ▼
                     ┌────────────────┐    ┌──────────────────┐
                     │ Gestionnaire de│◀───│ API Alpaca       │
                     │ risque         │    │ (Options Level 1)│
                     └────────────────┘    └──────────────────┘
```

## Configuration

Les paramètres de trading d'options sont configurables via le fichier `config/daytrader_config.json` dans la section `stock.options_trading` :

```json
"options_trading": {
  "enabled": true,
  "max_options_allocation_pct": 0.20,
  "max_options_per_symbol": 3,
  "min_confidence_for_options": 0.70,
  "risk_profile": "moderate",
  "max_days_to_expiry": 45,
  "preferred_strategies": [
    "Long Call",
    "Long Put",
    "Cash-Secured Put",
    "Covered Call"
  ],
  "base_strategies": [
    "TransformerStrategy",
    "LSTMPredictorStrategy",
    "MSIStrategy"
  ],
  "require_confirmation": true,
  "max_loss_per_trade_pct": 1.0,
  "strict_position_sizing": true
}
```

### Paramètres de configuration

| Paramètre | Description | Valeurs possibles |
|-----------|-------------|-------------------|
| `enabled` | Active ou désactive le trading d'options | `true`, `false` |
| `max_options_allocation_pct` | Pourcentage maximum du capital alloué aux options | `0.0` à `1.0` |
| `max_options_per_symbol` | Nombre maximum de contrats d'options par symbole | Entier positif |
| `min_confidence_for_options` | Seuil de confiance minimum pour exécuter un trading d'options | `0.0` à `1.0` |
| `risk_profile` | Profil de risque pour les stratégies d'options | `"conservative"`, `"moderate"`, `"aggressive"` |
| `max_days_to_expiry` | Nombre maximum de jours jusqu'à l'expiration | Entier positif |
| `preferred_strategies` | Liste des stratégies d'options préférées | Tableau de noms de stratégies |
| `base_strategies` | Liste des stratégies de base à utiliser pour les signaux | Tableau de noms de stratégies |
| `require_confirmation` | Exiger une confirmation avant d'exécuter un trade d'options | `true`, `false` |
| `max_loss_per_trade_pct` | Pourcentage maximum de perte par trade | `0.0` à `1.0` |
| `strict_position_sizing` | Activer le dimensionnement strict des positions | `true`, `false` |

## Stratégies d'options disponibles

Les stratégies suivantes sont disponibles pour le trading d'options de niveau 1 :

### Long Call

**Description** : Achat d'une option d'achat, donnant le droit d'acheter l'actif sous-jacent à un prix déterminé.

**Utilisation** : Lorsque vous anticipez une hausse significative du prix de l'actif sous-jacent.

**Risque** : Limité au montant de la prime payée.

**Gain potentiel** : Théoriquement illimité à mesure que le prix de l'actif sous-jacent augmente.

### Long Put

**Description** : Achat d'une option de vente, donnant le droit de vendre l'actif sous-jacent à un prix déterminé.

**Utilisation** : Lorsque vous anticipez une baisse significative du prix de l'actif sous-jacent.

**Risque** : Limité au montant de la prime payée.

**Gain potentiel** : Limité au prix d'exercice moins la prime payée (si le prix tombe à zéro).

### Cash-Secured Put

**Description** : Vente d'une option de vente avec suffisamment de liquidités pour acheter l'actif sous-jacent si l'option est exercée.

**Utilisation** : Lorsque vous êtes prêt à acheter l'actif sous-jacent à un prix inférieur au prix actuel et que vous souhaitez générer un revenu en attendant.

**Risque** : Limité à la différence entre le prix d'exercice et zéro, moins la prime reçue.

**Gain potentiel** : Limité au montant de la prime reçue.

### Covered Call

**Description** : Vente d'une option d'achat tout en détenant l'actif sous-jacent.

**Utilisation** : Lorsque vous détenez déjà l'actif sous-jacent et souhaitez générer un revenu supplémentaire, et êtes prêt à vendre l'actif à un prix supérieur au prix actuel.

**Risque** : Limité au coût d'opportunité si le prix de l'actif augmente au-dessus du prix d'exercice.

**Gain potentiel** : Limité au montant de la prime reçue plus l'appréciation potentielle jusqu'au prix d'exercice.

## API des services d'options

### OptionsService

```python
class OptionsService:
    def __init__(self, trading_service: TradingService, market_data_service: MarketDataService):
        # Initialise le service d'options
        
    async def get_available_options(self, symbol: str, expiration_date: Optional[str] = None) -> List[Dict[str, Any]]:
        # Récupère les options disponibles pour un symbole donné
        
    async def execute_option_trade(self, option_symbol: str, action: TradeAction, quantity: int, order_type: str = "market", limit_price: Optional[float] = None, time_in_force: str = "day", strategy_name: str = "unknown") -> Dict[str, Any]:
        # Exécute un trade d'options
        
    async def get_option_position(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        # Récupère les détails d'une position d'options spécifique
        
    async def get_all_option_positions(self) -> List[Dict[str, Any]]:
        # Récupère toutes les positions d'options actuelles
        
    async def calculate_option_metrics(self, option_data: Dict[str, Any]) -> Dict[str, Any]:
        # Calcule les métriques importantes des options (Grecs)
        
    async def suggest_option_strategies(self, symbol: str, price_prediction: Dict[str, Any], risk_profile: str = "moderate") -> List[Dict[str, Any]]:
        # Suggère des stratégies d'options basées sur les prédictions de prix
```

### OptionsStrategy

```python
class OptionsStrategy(Strategy):
    def __init__(self, options_service: OptionsService, base_strategy_name: str, risk_profile: str = "moderate", max_days_to_expiry: int = 45, preferred_option_types: List[str] = None):
        # Initialise la stratégie d'options
        
    async def generate_signal(self, symbol: str, data: Dict[str, Any], timeframe: TimeFrame = TimeFrame.DAY) -> Dict[str, Any]:
        # Génère un signal de trading d'options basé sur le signal de la stratégie sous-jacente
        
    async def backtest(self, symbol: str, historical_data: List[Dict[str, Any]], timeframe: TimeFrame = TimeFrame.DAY) -> Dict[str, Any]:
        # Backteste la stratégie d'options (simplifié)
        
    async def optimize(self, symbol: str, historical_data: List[Dict[str, Any]], timeframe: TimeFrame = TimeFrame.DAY) -> Dict[str, Any]:
        # Optimise les paramètres de la stratégie d'options
```

## Exemples d'utilisation

### Initialisation du service d'options

```python
from app.services.trading import TradingService
from app.services.market_data import MarketDataService
from app.services.options_service import OptionsService

# Initialiser les services requis
trading_service = TradingService(is_paper=True)
market_data_service = MarketDataService()

# Créer le service d'options
options_service = OptionsService(
    trading_service=trading_service,
    market_data_service=market_data_service
)
```

### Création d'une stratégie d'options basée sur une stratégie existante

```python
from app.strategies.options_strategy import OptionsStrategy

# Créer une stratégie d'options basée sur la stratégie TransformerStrategy
options_strategy = OptionsStrategy(
    options_service=options_service,
    base_strategy_name="TransformerStrategy",
    risk_profile="moderate",
    max_days_to_expiry=30,
    preferred_option_types=["Long Call", "Long Put"]
)

# Générer un signal d'options
signal = await options_strategy.generate_signal("AAPL", market_data)

# Exécuter un trade d'options basé sur le signal
if signal.get("action") != TradeAction.HOLD:
    result = await options_service.execute_option_trade(
        option_symbol=f"{signal['symbol']}_{signal['expiration']}_{signal['option_type'][0].upper()}_{int(signal['strike']*1000):08d}",
        action=signal["action"],
        quantity=1,
        strategy_name=options_strategy.name
    )
```

## Bonnes pratiques et considérations de risque

### Gestion du risque

- **Limitez l'allocation** : Maintenez une allocation limitée pour le trading d'options (typiquement 10-20% du portefeuille).
- **Diversifiez les expirations** : Évitez de concentrer toutes vos positions sur une seule date d'expiration.
- **Surveillez les métriques** : Faites attention aux Greeks, en particulier le Theta (décroissance temporelle) qui érode la valeur des options au fil du temps.

### Bonnes pratiques

- **Commencez petit** : Démarrez avec un petit nombre de contrats pour comprendre le comportement des options.
- **Préférez les options liquides** : Choisissez des options avec un volume et un intérêt ouvert élevés pour minimiser les spreads.
- **Limitez les stratégies complexes** : Au niveau 1, restez concentré sur les stratégies simples comme les calls et puts longs.
- **Prenez en compte l'expiration** : Les options à court terme sont plus risquées mais moins chères, tandis que les options à long terme sont plus coûteuses mais offrent plus de temps pour que votre thèse se développe.

## Dépannage

### Problèmes courants

| Problème | Causes possibles | Solutions |
|----------|------------------|-----------|
| Erreur "Option non disponible" | L'option spécifiée n'existe pas ou l'expiration est incorrecte | Vérifiez que vous utilisez un format correct pour le symbole d'option et une date d'expiration valide |
| Position trop petite | Les restrictions de dimensionnement de position sont trop strictes | Ajustez `max_options_allocation_pct` dans la configuration |
| Aucun signal d'options généré | Confiance de la stratégie de base trop faible | Vérifiez que la stratégie de base génère des signaux avec une confiance supérieure à `min_confidence_for_options` |
| Erreur d'exécution du trade | Problèmes d'API avec Alpaca | Vérifiez vos clés API et assurez-vous que votre compte a un accès au trading d'options de niveau 1 |

## Conclusion

Le module de trading d'options pour Mercurio AI fournit une extension puissante mais contrôlée des capacités de trading existantes. En combinant les signaux générés par vos stratégies ML existantes avec des stratégies d'options soigneusement sélectionnées, vous pouvez potentiellement améliorer les rendements et gérer les risques de manière plus efficace.

Souvenez-vous toujours que le trading d'options comporte des risques intrinsèques différents du trading d'actions standard, et nécessite donc une surveillance et une gestion attentives.
