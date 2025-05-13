#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Covered Call Strategy

Stratégie d'options qui:
1. Détient des actions sous-jacentes
2. Vend des calls OTM pour générer du revenu
3. Gère le roll-over et l'ajustement dynamique des strikes
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid

from app.strategies.options.base_options_strategy import BaseOptionsStrategy
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.services.options_service import OptionsService

logger = logging.getLogger(__name__)

class CoveredCallStrategy(BaseOptionsStrategy):
    """
    Stratégie Covered Call qui:
    1. Détient des actions sous-jacentes
    2. Vend des calls OTM pour générer du revenu
    3. Gère le roll-over et l'ajustement dynamique des strikes
    """
    
    def __init__(self,
                 underlying_symbol: str = None,
                 account_size: float = None,
                 max_position_size: float = 0.10,
                 profit_target_pct: float = 0.50,
                 stop_loss_pct: float = 0.20,
                 ticker: str = None,
                 market_data_service: 'MarketDataService' = None,
                 trading_service: 'TradingService' = None,
                 options_service: 'OptionsService' = None,
                 delta_target: float = 0.30,
                 dte_range: Tuple[int, int] = (30, 45),
                 iv_rank_min: float = 40.0,
                 profit_target: float = 0.50,
                 stop_loss_pct_old: float = 0.20,
                 sizing_pct: float = 0.05,
                 roll_threshold_days: int = 7,
                 max_active_positions: int = 5,
                 **kwargs):
        """
        Initialiser la stratégie Covered Call.
        
        Args:
            ticker: Symbole du sous-jacent
            market_data_service: Service de données de marché
            trading_service: Service d'exécution des trades
            options_service: Service de trading d'options
            delta_target: Delta cible pour selection du strike (0.20-0.40)
            dte_range: Plage de jours jusqu'à expiration (min, max)
            iv_rank_min: Rang IV minimum pour vendre des calls (%)
            profit_target: % du crédit initial pour prendre profit
            stop_loss_pct: % de perte maximale tolérée
            sizing_pct: % du portefeuille pour chaque position
            roll_threshold_days: Jours avant expiration pour roll
            max_active_positions: Nombre maximum de positions simultanées
        """
        super().__init__(name="CoveredCall", description="Covered Call Strategy")
        
        # For test compatibility
        self.underlying_symbol = underlying_symbol if underlying_symbol is not None else ticker
        self.account_size = account_size
        self.max_position_size = max_position_size
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        
        # Legacy attributes for backward compatibility
        self.ticker = ticker if ticker is not None else underlying_symbol
        self.market_data_service = market_data_service
        self.trading_service = trading_service
        self.options_service = options_service
        
        # Paramètres de stratégie
        self.delta_target = delta_target
        self.dte_min, self.dte_max = dte_range
        self.iv_rank_min = iv_rank_min
        self.profit_target = profit_target
        self.stop_loss_pct_old = stop_loss_pct_old
        self.sizing_pct = sizing_pct
        self.roll_threshold_days = roll_threshold_days
        self.max_active_positions = max_active_positions
        
        # État interne
        self.active_positions = {}  # {position_id: position_data}
        self.historical_trades = []
        
        # Journalisation et métriques
        self.metrics = {
            "total_premium_collected": 0.0,
            "total_assignments": 0,
            "total_rolls": 0,
            "win_rate": 0.0,
            "avg_hold_time_days": 0.0
        }
        self.current_call = None
        logger.info(f"Stratégie Covered Call initialisée pour {self.underlying_symbol}")
    
    async def should_enter(self, market_data: pd.DataFrame) -> bool:
        """
        Détermine si nous devons initier une nouvelle position Covered Call.
        
        Args:
            market_data: DataFrame avec les données de marché
            
        Returns:
            bool: True si on devrait entrer, False sinon
        """
        # Si nous avons atteint le nombre maximum de positions
        if len(self.active_positions) >= self.max_active_positions:
            return False
            
        # Vérifier si nous avons déjà une position pour ce ticker
        if any(pos["ticker"] == self.ticker for pos in self.active_positions.values()):
            return False
            
        # Vérifier les conditions de marché
        try:
            # 1. Obtenir le prix actuel du sous-jacent
            current_price = await self.market_data_service.get_latest_price(self.ticker)
            
            # 2. Obtenir le rang de volatilité implicite
            # Note: cette méthode doit être implémentée dans le MarketDataService
            iv_rank = await self._get_iv_rank(self.ticker)
            
            # 3. Vérifier si IV est assez élevée pour vendre des options
            if iv_rank < self.iv_rank_min:
                logger.info(f"{self.ticker} IV Rank ({iv_rank:.2f}%) trop bas, minimum requis: {self.iv_rank_min}%")
                return False
                
            # 4. Obtenir les chaînes d'options disponibles
            today = datetime.now().date()
            expiry_min = today + timedelta(days=self.dte_min)
            expiry_max = today + timedelta(days=self.dte_max)
            
            # Utiliser le service d'options pour obtenir les chaînes d'options
            option_chain = await self.options_service.get_available_options(
                symbol=self.ticker,
                option_type="call",
                expiry_range=(expiry_min.strftime('%Y-%m-%d'), expiry_max.strftime('%Y-%m-%d'))
            )
            
            if not option_chain or len(option_chain) == 0:
                logger.warning(f"Aucune option disponible pour {self.ticker} dans la plage d'expiration spécifiée")
                return False
                
            # 5. Sélectionner la meilleure option à vendre
            best_option = self._select_best_call_option(option_chain, current_price)
            
            if not best_option:
                return False
                
            # Stocker temporairement l'option sélectionnée pour l'exécution
            self._selected_option = best_option
            
            # Calculer le nombre d'actions à acheter (round down to nearest 100)
            account_value = await self.trading_service.get_account_value()
            position_value = account_value * self.sizing_pct
            shares_to_buy = int(position_value / current_price / 100) * 100
            
            if shares_to_buy < 100:
                logger.warning(f"Valeur du compte insuffisante pour acheter au moins 100 actions de {self.ticker}")
                return False
                
            self._shares_to_buy = shares_to_buy
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur dans should_enter: {str(e)}")
            return False
    
    async def execute_entry(self) -> bool:
        """
        Exécuter l'entrée en position: acheter le sous-jacent et vendre un call.
        
        Returns:
            bool: True si l'entrée est réussie, False sinon
        """
        if not hasattr(self, '_selected_option') or not hasattr(self, '_shares_to_buy'):
            logger.error("Tentative d'exécution sans option/actions sélectionnées")
            return False
            
        try:
            # 1. Acheter les actions
            buy_order = await self.trading_service.buy_shares(
                symbol=self.ticker,
                quantity=self._shares_to_buy,
                order_type="market"
            )
            
            if not buy_order or not buy_order.get("success"):
                logger.error(f"Échec de l'achat d'actions {self.ticker}")
                return False
                
            # 2. Vendre le call une fois que nous avons les actions
            option = self._selected_option
            contracts_to_sell = self._shares_to_buy // 100  # 1 contrat = 100 actions
            
            sell_order = await self.options_service.execute_option_trade(
                option_symbol=option.get("symbol"),
                action="SELL",
                quantity=contracts_to_sell,
                order_type="limit",
                limit_price=option.get("bid"),  # Vendre au prix bid
                strategy_name=self.name
            )
            
            if not sell_order or not sell_order.get("success"):
                logger.error(f"Échec de la vente de l'option call sur {self.ticker}")
                # Envisager de vendre les actions si l'option échoue?
                return False
                
            # 3. Enregistrer la position
            position_id = f"CC_{self.ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
            position_data = {
                "id": position_id,
                "ticker": self.ticker,
                "shares": self._shares_to_buy,
                "share_entry_price": buy_order.get("filled_avg_price"),
                "contracts": contracts_to_sell,
                "option": {
                    "symbol": option.get("symbol"),
                    "strike": option.get("strike"),
                    "expiry": option.get("expiry"),
                    "premium": option.get("bid") * 100 * contracts_to_sell  # Total premium
                },
                "entry_date": datetime.now().isoformat(),
                "status": "active"
            }
            
            self.active_positions[position_id] = position_data
            
            # Mise à jour des métriques
            self.metrics["total_premium_collected"] += position_data["option"]["premium"]
            
            logger.info(f"Position Covered Call initiée pour {self.ticker}:")
            logger.info(f"  - {self._shares_to_buy} actions achetées @ ${buy_order.get('filled_avg_price'):.2f}")
            logger.info(f"  - {contracts_to_sell} calls ${option.get('strike')} exp {option.get('expiry')} vendus @ ${option.get('bid'):.2f}")
            logger.info(f"  - Prime totale collectée: ${position_data['option']['premium']:.2f}")
            
            # Nettoyage des variables temporaires
            del self._selected_option
            del self._shares_to_buy
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur dans execute_entry: {str(e)}")
            return False
    
    async def should_exit(self, position_id: str, market_data: pd.DataFrame) -> bool:
        """
        Détermine si nous devons sortir d'une position Covered Call.
        
        Args:
            position_id: Identifiant de la position à évaluer
            market_data: Données de marché actuelles
            
        Returns:
            bool: True si on devrait sortir, False sinon
        """
        if position_id not in self.active_positions:
            return False
            
        position = self.active_positions[position_id]
        ticker = position["ticker"]
        
        try:
            # 1. Obtenir le prix actuel du sous-jacent
            current_price = await self.market_data_service.get_latest_price(ticker)
            
            # 2. Obtenir les détails actuels de l'option
            option_symbol = position["option"]["symbol"]
            option_details = await self.options_service.get_option_data(option_symbol)
            
            if not option_details:
                logger.warning(f"Impossible d'obtenir les détails actuels de l'option pour {ticker}")
                return False
                
            # 3. Calculer le profit actuel (en pourcentage de la prime initiale)
            initial_premium = position["option"]["premium"] / (position["contracts"] * 100)
            current_premium = option_details.get("ask", 0)  # Prix auquel on rachèterait l'option
            profit_pct = (initial_premium - current_premium) / initial_premium
            
            # 4. Vérifier les conditions de sortie
            
            # 4.1 Profit cible atteint?
            if profit_pct >= self.profit_target:
                logger.info(f"Profit cible atteint pour {ticker}: {profit_pct:.2%} > {self.profit_target:.2%}")
                return True
                
            # 4.2 Stop-loss atteint? (l'option est devenue beaucoup plus chère)
            if profit_pct <= -self.stop_loss_pct:
                logger.info(f"Stop-loss atteint pour {ticker}: {profit_pct:.2%} < -{self.stop_loss_pct:.2%}")
                return True
                
            # 4.3 Proche de l'expiration? (candidat pour roll)
            option_expiry = datetime.fromisoformat(position["option"]["expiry"].replace("Z", "+00:00"))
            days_to_expiry = (option_expiry.date() - datetime.now().date()).days
            
            if days_to_expiry <= self.roll_threshold_days:
                logger.info(f"Option proche de l'expiration: {days_to_expiry} jours restants")
                # Ici on pourrait implémenter une logique de roll
                # Pour cet exemple, nous sortons simplement
                return True
                
            # 4.4 L'option est-elle deep ITM? (risque d'assignation précoce)
            if current_price > position["option"]["strike"] * 1.05:  # 5% ITM
                delta = option_details.get("delta", 0.5)
                if delta > 0.85:  # Très forte probabilité d'être exercée
                    logger.info(f"Option deep ITM (delta {delta:.2f}), risque d'assignation élevé")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur dans should_exit: {str(e)}")
            return False
    
    async def execute_exit(self, position_id: str) -> bool:
        """
        Exécuter la sortie d'une position Covered Call.
        
        Args:
            position_id: Identifiant de la position à fermer
            
        Returns:
            bool: True si la sortie est réussie, False sinon
        """
        if position_id not in self.active_positions:
            logger.error(f"Position {position_id} non trouvée")
            return False
            
        position = self.active_positions[position_id]
        ticker = position["ticker"]
        
        try:
            # 1. Racheter l'option call vendue pour clôturer
            buy_option_order = await self.options_service.execute_option_trade(
                option_symbol=position["option"]["symbol"],
                action="BUY",
                quantity=position["contracts"],
                order_type="market",
                strategy_name=self.name
            )
            
            if not buy_option_order or not buy_option_order.get("success"):
                logger.error(f"Échec du rachat de l'option call sur {ticker}")
                return False
                
            # 2. Vendre les actions sous-jacentes
            sell_shares_order = await self.trading_service.sell_shares(
                symbol=ticker,
                quantity=position["shares"],
                order_type="market"
            )
            
            if not sell_shares_order or not sell_shares_order.get("success"):
                logger.error(f"Échec de la vente des actions {ticker}")
                return False
                
            # 3. Calculer le P&L
            entry_cost = position["share_entry_price"] * position["shares"]
            exit_proceeds = sell_shares_order.get("filled_avg_price") * position["shares"]
            
            initial_premium = position["option"]["premium"]
            closing_premium = buy_option_order.get("filled_avg_price") * 100 * position["contracts"]
            
            options_pnl = initial_premium - closing_premium
            shares_pnl = exit_proceeds - entry_cost
            total_pnl = options_pnl + shares_pnl
            
            # 4. Mettre à jour les métriques et l'historique
            position["exit_date"] = datetime.now().isoformat()
            position["exit_details"] = {
                "share_exit_price": sell_shares_order.get("filled_avg_price"),
                "option_close_price": buy_option_order.get("filled_avg_price"),
                "options_pnl": options_pnl,
                "shares_pnl": shares_pnl,
                "total_pnl": total_pnl
            }
            position["status"] = "closed"
            
            # Déplacer de active_positions vers historical_trades
            self.historical_trades.append(position)
            del self.active_positions[position_id]
            
            # Mise à jour des métriques
            self._update_metrics()
            
            logger.info(f"Position Covered Call clôturée pour {ticker}:")
            logger.info(f"  - Options P&L: ${options_pnl:.2f}")
            logger.info(f"  - Actions P&L: ${shares_pnl:.2f}")
            logger.info(f"  - P&L total: ${total_pnl:.2f} ({total_pnl/(entry_cost)*100:.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur dans execute_exit: {str(e)}")
            return False
    
    async def roll_position(self, position_id: str, new_expiry: str, new_strike: float) -> bool:
        """
        Rouler une position existante vers une nouvelle date d'expiration/strike.
        
        Args:
            position_id: ID de la position à rouler
            new_expiry: Nouvelle date d'expiration (format YYYY-MM-DD)
            new_strike: Nouveau prix d'exercice
            
        Returns:
            bool: True si le roll est réussi, False sinon
        """
        if position_id not in self.active_positions:
            logger.error(f"Position {position_id} non trouvée pour roll")
            return False
            
        position = self.active_positions[position_id]
        ticker = position["ticker"]
        
        try:
            # 1. Racheter l'option existante
            buy_option_order = await self.options_service.execute_option_trade(
                option_symbol=position["option"]["symbol"],
                action="BUY",
                quantity=position["contracts"],
                order_type="market",
                strategy_name=f"{self.name}_ROLL"
            )
            
            if not buy_option_order or not buy_option_order.get("success"):
                logger.error(f"Échec du rachat de l'option call pour roll sur {ticker}")
                return False
                
            # 2. Obtenir la chaîne d'options pour la nouvelle expiration
            option_chain = await self.options_service.get_available_options(
                symbol=ticker,
                option_type="call",
                expiration_date=new_expiry
            )
            
            # 3. Trouver l'option avec le strike demandé
            new_option = next((opt for opt in option_chain if opt.get("strike") == new_strike), None)
            
            if not new_option:
                logger.error(f"Option avec strike {new_strike} et expiration {new_expiry} non trouvée pour {ticker}")
                return False
                
            # 4. Vendre la nouvelle option
            sell_option_order = await self.options_service.execute_option_trade(
                option_symbol=new_option.get("symbol"),
                action="SELL",
                quantity=position["contracts"],
                order_type="limit",
                limit_price=new_option.get("bid"),
                strategy_name=f"{self.name}_ROLL"
            )
            
            if not sell_option_order or not sell_option_order.get("success"):
                logger.error(f"Échec de la vente de la nouvelle option call pour roll sur {ticker}")
                return False
                
            # 5. Mettre à jour la position
            # Garder trace de l'historique du roll
            if "roll_history" not in position:
                position["roll_history"] = []
                
            position["roll_history"].append({
                "old_strike": position["option"]["strike"],
                "old_expiry": position["option"]["expiry"],
                "old_premium": position["option"]["premium"],
                "buy_close_price": buy_option_order.get("filled_avg_price"),
                "roll_date": datetime.now().isoformat()
            })
            
            # Mettre à jour avec la nouvelle option
            position["option"] = {
                "symbol": new_option.get("symbol"),
                "strike": new_strike,
                "expiry": new_expiry,
                "premium": new_option.get("bid") * 100 * position["contracts"]
            }
            
            # Mettre à jour les métriques
            self.metrics["total_rolls"] += 1
            self.metrics["total_premium_collected"] += position["option"]["premium"]
            
            logger.info(f"Position Covered Call roulée pour {ticker}:")
            logger.info(f"  - Nouvelle expiration: {new_expiry}")
            logger.info(f"  - Nouveau strike: ${new_strike}")
            logger.info(f"  - Prime additionnelle collectée: ${position['option']['premium']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur dans roll_position: {str(e)}")
            return False
    
    def _select_best_call_option(self, option_chain: List[Dict[str, Any]], current_price: float) -> Optional[Dict[str, Any]]:
        """
        Sélectionne la meilleure option call à vendre selon nos critères.
        
        Args:
            option_chain: Liste d'options disponibles
            current_price: Prix actuel du sous-jacent
            
        Returns:
            Option: L'option sélectionnée, ou None si aucune ne convient
        """
        # Filtrer pour les options OTM
        otm_calls = [opt for opt in option_chain if opt.get("strike", 0) > current_price * 1.01]  # 1% OTM minimum
        
        if not otm_calls:
            logger.warning(f"Aucune option call OTM disponible pour {self.ticker}")
            return None
            
        # Trouver celle qui a le delta le plus proche de notre cible
        # Note: Delta devrait être disponible dans les données d'option
        otm_calls.sort(key=lambda x: abs(x.get("delta", 0.5) - self.delta_target))
        
        # Vérifier que le delta n'est pas trop éloigné de notre cible
        best_match = otm_calls[0]
        if abs(best_match.get("delta", 0.5) - self.delta_target) > 0.1:
            logger.warning(f"Pas d'option avec un delta proche de {self.delta_target} pour {self.ticker}")
            return None
            
        # Vérifier le ratio prime/risque
        premium_to_strike_ratio = (best_match.get("bid", 0) * 100) / (best_match.get("strike", 0) * 100)
        if premium_to_strike_ratio < 0.005:  # Moins de 0.5% de premium
            logger.warning(f"Prime trop faible ({premium_to_strike_ratio:.2%}) pour {self.ticker}")
            return None
            
        return best_match
    
    async def _get_iv_rank(self, symbol: str) -> float:
        """
        Obtenir le rang de volatilité implicite pour un symbole.
        À implémenter dans un service dédié dans une version future.
        
        Args:
            symbol: Symbole du sous-jacent
            
        Returns:
            float: Rang de volatilité implicite (0-100)
        """
        # Implémentation minimale - dans une vraie application, cela calculerait le rang IV
        # basé sur l'historique de IV sur 1 an
        try:
            # Essayer d'obtenir l'IV des options à partir du service d'options
            option_chain = await self.options_service.get_available_options(symbol)
            if option_chain and len(option_chain) > 0:
                # Moyenne d'IV sur les options ATM
                current_price = await self.market_data_service.get_latest_price(symbol)
                atm_options = [opt for opt in option_chain 
                              if 0.95 * current_price <= opt.get("strike", 0) <= 1.05 * current_price]
                
                if atm_options:
                    iv_values = [opt.get("implied_volatility", 0) * 100 for opt in atm_options]
                    current_iv = np.mean(iv_values)
                    
                    # Simuler un rang IV simple (généralement, cela nécessiterait des données historiques)
                    # Dans cet exemple, nous utilisons une valeur arbitraire
                    # Dans un cas réel, on calculerait le percentile de l'IV actuelle par rapport à l'historique
                    return min(current_iv * 1.5, 100)
            
            # Si aucune donnée n'est disponible, fournir une valeur par défaut basée sur la volatilité historique
            if "historic_volatility" in market_data.columns:
                hist_vol = market_data["historic_volatility"].iloc[-1] * 100
                return min(hist_vol * 1.2, 100)  # Approximation grossière
                
            # Valeur de fallback
            return 50.0
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de l'IV Rank pour {symbol}: {str(e)}")
            return 50.0  # Valeur par défaut
