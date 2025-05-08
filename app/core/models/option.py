#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Option Contract Models

Classes pour représenter les contrats d'options et positions dans Mercurio AI.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
from enum import Enum

logger = logging.getLogger(__name__)

class OptionType(str, Enum):
    """Type de contrat d'option"""
    CALL = "call"
    PUT = "put"

class OptionContract:
    """
    Représente un contrat d'option individuel.
    
    Contient toutes les informations relatives à un contrat d'option,
    y compris les prix, les Greek, et les caractéristiques du contrat.
    """
    
    def __init__(self, 
                 symbol: str,
                 underlying: str,
                 option_type: OptionType,
                 strike: float,
                 expiry_date: Union[str, date],
                 bid: float = 0.0,
                 ask: float = 0.0,
                 last: float = 0.0,
                 volume: int = 0,
                 open_interest: int = 0,
                 implied_volatility: float = 0.0,
                 delta: float = 0.0,
                 gamma: float = 0.0,
                 theta: float = 0.0,
                 vega: float = 0.0,
                 rho: float = 0.0):
        """
        Initialiser un contrat d'option.
        
        Args:
            symbol: Symbole complet du contrat d'option
            underlying: Symbole du sous-jacent
            option_type: Type d'option (call ou put)
            strike: Prix d'exercice
            expiry_date: Date d'expiration (YYYY-MM-DD ou objet date)
            bid: Prix d'achat actuel
            ask: Prix de vente actuel
            last: Dernier prix
            volume: Volume
            open_interest: Intérêt ouvert
            implied_volatility: Volatilité implicite
            delta: Greek delta
            gamma: Greek gamma
            theta: Greek theta
            vega: Greek vega
            rho: Greek rho
        """
        self.symbol = symbol
        self.underlying = underlying
        self.option_type = option_type
        self.strike = strike
        
        # Convertir la date d'expiration en objet date si nécessaire
        if isinstance(expiry_date, str):
            self.expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        else:
            self.expiry_date = expiry_date
            
        # Prix
        self.bid = bid
        self.ask = ask
        self.last = last
        self.mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        
        # Statistiques
        self.volume = volume
        self.open_interest = open_interest
        
        # Greeks
        self.implied_volatility = implied_volatility
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho
        
        # Attributs calculés
        self.days_to_expiry = (self.expiry_date - datetime.now().date()).days
        self.is_itm = False  # Sera défini par le service d'options
        
    def __str__(self):
        return f"{self.underlying} {self.expiry_date.strftime('%Y-%m-%d')} {self.option_type.value.upper()} ${self.strike}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour stockage/sérialisation"""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiry_date": self.expiry_date.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "days_to_expiry": self.days_to_expiry,
            "is_itm": self.is_itm
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionContract':
        """Créer une instance à partir d'un dictionnaire"""
        return cls(
            symbol=data.get("symbol", ""),
            underlying=data.get("underlying", ""),
            option_type=OptionType(data.get("option_type", "call")),
            strike=data.get("strike", 0.0),
            expiry_date=data.get("expiry_date", ""),
            bid=data.get("bid", 0.0),
            ask=data.get("ask", 0.0),
            last=data.get("last", 0.0),
            volume=data.get("volume", 0),
            open_interest=data.get("open_interest", 0),
            implied_volatility=data.get("implied_volatility", 0.0),
            delta=data.get("delta", 0.0),
            gamma=data.get("gamma", 0.0),
            theta=data.get("theta", 0.0),
            vega=data.get("vega", 0.0),
            rho=data.get("rho", 0.0)
        )

class OptionAction(str, Enum):
    """Action d'option: achat ou vente"""
    BUY = "buy"
    SELL = "sell"

class OptionPosition:
    """
    Représente une position ouverte sur un contrat d'option.
    
    Contient les informations sur la position, y compris la quantité,
    le prix d'entrée, et les données de performance.
    """
    
    def __init__(self,
                 contract: OptionContract,
                 quantity: int,
                 entry_price: float,
                 entry_date: Union[str, datetime],
                 action: OptionAction,
                 position_id: str = None):
        """
        Initialiser une position d'option.
        
        Args:
            contract: Le contrat d'option
            quantity: Nombre de contrats (positif pour long, négatif pour short)
            entry_price: Prix moyen d'entrée
            entry_date: Date d'entrée
            action: Action (achat ou vente)
            position_id: Identifiant unique de la position
        """
        self.contract = contract
        self.quantity = quantity
        self.entry_price = entry_price
        
        # Convertir la date d'entrée en objet datetime si nécessaire
        if isinstance(entry_date, str):
            self.entry_date = datetime.fromisoformat(entry_date)
        else:
            self.entry_date = entry_date
            
        self.action = action
        self.position_id = position_id or f"{contract.underlying}_{contract.symbol}_{entry_date.isoformat()}"
        
        # Métriques de performance
        self.current_price = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        
    def update_metrics(self, current_price: float):
        """
        Mettre à jour les métriques de performance avec le prix actuel.
        
        Args:
            current_price: Prix actuel du contrat
        """
        self.current_price = current_price
        
        # Calculer le P&L non réalisé (dépend de l'action: achat ou vente)
        contract_multiplier = 100  # 1 contrat = 100 actions
        
        if self.action == OptionAction.BUY:
            # Pour les options achetées (longues)
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity * contract_multiplier
        else:
            # Pour les options vendues (courtes)
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity * contract_multiplier
            
        # Calculer le pourcentage de P&L
        total_cost = abs(self.entry_price * self.quantity * contract_multiplier)
        if total_cost > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / total_cost) * 100
            
    def __str__(self):
        action_str = "LONG" if self.action == OptionAction.BUY else "SHORT"
        return f"{action_str} {self.quantity} {self.contract} @ ${self.entry_price:.2f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour stockage/sérialisation"""
        return {
            "position_id": self.position_id,
            "contract": self.contract.to_dict() if self.contract else {},
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.isoformat(),
            "action": self.action.value,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionPosition':
        """Créer une instance à partir d'un dictionnaire"""
        contract_data = data.get("contract", {})
        contract = OptionContract.from_dict(contract_data) if contract_data else None
        
        return cls(
            contract=contract,
            quantity=data.get("quantity", 0),
            entry_price=data.get("entry_price", 0.0),
            entry_date=data.get("entry_date", ""),
            action=OptionAction(data.get("action", "buy")),
            position_id=data.get("position_id", "")
        )

class OptionStrategy:
    """
    Représente une stratégie d'options multi-jambes.
    
    Contient les informations sur une stratégie d'options complète,
    potentiellement avec plusieurs jambes (ex: spread, condor, etc.)
    """
    
    def __init__(self,
                 name: str,
                 underlying: str,
                 legs: List[OptionPosition] = None,
                 strategy_id: str = None):
        """
        Initialiser une stratégie d'options.
        
        Args:
            name: Nom de la stratégie (ex: "Bull Call Spread")
            underlying: Symbole sous-jacent
            legs: Liste des positions formant la stratégie
            strategy_id: Identifiant unique de la stratégie
        """
        self.name = name
        self.underlying = underlying
        self.legs = legs or []
        self.strategy_id = strategy_id or f"{name}_{underlying}_{datetime.now().isoformat()}"
        
        # Métriques de la stratégie
        self.net_debit = 0.0
        self.net_credit = 0.0
        self.max_risk = 0.0
        self.max_reward = 0.0
        self.break_even_points = []
        
        # Calculer les métriques si des jambes sont fournies
        if legs:
            self._calculate_metrics()
            
    def add_leg(self, leg: OptionPosition):
        """
        Ajouter une jambe à la stratégie.
        
        Args:
            leg: Position d'option à ajouter
        """
        self.legs.append(leg)
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """Calculer les métriques de la stratégie basée sur les jambes"""
        # Réinitialiser les métriques
        self.net_debit = 0.0
        self.net_credit = 0.0
        
        # Calculer le débit/crédit net
        contract_multiplier = 100
        for leg in self.legs:
            if leg.action == OptionAction.BUY:
                self.net_debit += leg.entry_price * leg.quantity * contract_multiplier
            else:
                self.net_credit += leg.entry_price * leg.quantity * contract_multiplier
                
        # Calculer les métriques spécifiques à certaines stratégies connues
        # Note: Dans une implémentation complète, cette logique serait beaucoup
        # plus développée pour gérer différentes stratégies
        if self.name == "Bull Call Spread" and len(self.legs) == 2:
            self._calculate_bull_call_spread_metrics()
        elif self.name == "Bear Put Spread" and len(self.legs) == 2:
            self._calculate_bear_put_spread_metrics()
        elif self.name == "Iron Condor" and len(self.legs) == 4:
            self._calculate_iron_condor_metrics()
        # etc. pour d'autres stratégies
    
    def _calculate_bull_call_spread_metrics(self):
        """Calculer les métriques pour un Bull Call Spread"""
        # Pour un Bull Call Spread, on a:
        # - Achat d'un call à un strike plus bas
        # - Vente d'un call à un strike plus haut
        
        if len(self.legs) != 2:
            return
            
        long_call = None
        short_call = None
        
        for leg in self.legs:
            if leg.action == OptionAction.BUY and leg.contract.option_type == OptionType.CALL:
                long_call = leg
            elif leg.action == OptionAction.SELL and leg.contract.option_type == OptionType.CALL:
                short_call = leg
                
        if not long_call or not short_call:
            return
            
        # Calculer le risque maximum (net debit)
        self.max_risk = self.net_debit
        
        # Calculer le gain maximum (différence des strikes - net debit)
        width = (short_call.contract.strike - long_call.contract.strike) * 100
        self.max_reward = width - self.net_debit
        
        # Calculer le point d'équilibre
        self.break_even_points = [long_call.contract.strike + (self.net_debit / 100)]
    
    def _calculate_bear_put_spread_metrics(self):
        """Calculer les métriques pour un Bear Put Spread"""
        # Implémentation similaire au Bull Call Spread mais pour Bear Put
        pass
    
    def _calculate_iron_condor_metrics(self):
        """Calculer les métriques pour un Iron Condor"""
        # Implémentation pour Iron Condor
        pass
    
    def __str__(self):
        return f"{self.name} sur {self.underlying} - {len(self.legs)} jambes"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour stockage/sérialisation"""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "underlying": self.underlying,
            "legs": [leg.to_dict() for leg in self.legs],
            "net_debit": self.net_debit,
            "net_credit": self.net_credit,
            "max_risk": self.max_risk,
            "max_reward": self.max_reward,
            "break_even_points": self.break_even_points
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionStrategy':
        """Créer une instance à partir d'un dictionnaire"""
        legs_data = data.get("legs", [])
        legs = [OptionPosition.from_dict(leg) for leg in legs_data]
        
        return cls(
            name=data.get("name", ""),
            underlying=data.get("underlying", ""),
            legs=legs,
            strategy_id=data.get("strategy_id", "")
        )
