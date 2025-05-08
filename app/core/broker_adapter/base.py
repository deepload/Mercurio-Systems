#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Broker Adapter

Interfaces d'abstraction pour les différents brokers supportés par Mercurio AI.
Permet de changer facilement entre différents fournisseurs (Alpaca, IBKR, etc.).
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

# Import des modèles pour les contrats d'options
from app.core.models.option import OptionContract, OptionType, OptionPosition

logger = logging.getLogger(__name__)

class BrokerAdapter(ABC):
    """
    Interface de base pour tous les adaptateurs de broker.
    
    Cette classe abstraite définit les méthodes que tous les adaptateurs
    de broker doivent implémenter, permettant à Mercurio AI de changer
    facilement entre différents fournisseurs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'adaptateur de broker avec la configuration.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        self.name = "BaseBroker"
        self.description = "Base Broker Adapter"
        self.features = {
            "stocks": False,
            "options": False,
            "crypto": False,
            "futures": False,
            "forex": False,
            "l2_data": False,
            "hft": False
        }
        
        logger.info(f"Initialisation de l'adaptateur de broker: {self.name}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Établit la connexion avec le broker.
        
        Returns:
            bool: True si la connexion est établie avec succès, False sinon
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Ferme la connexion avec le broker.
        
        Returns:
            bool: True si la déconnexion est réussie, False sinon
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Récupère les informations du compte.
        
        Returns:
            Dict: Informations du compte (solde, valeur du portefeuille, etc.)
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Récupère les positions actuelles.
        
        Returns:
            List[Dict]: Liste des positions actuelles
        """
        pass
    
    @abstractmethod
    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres selon leur statut.
        
        Args:
            status: Statut des ordres à récupérer (open, closed, etc.)
            
        Returns:
            List[Dict]: Liste des ordres
        """
        pass
    
    # === Méthodes d'exécution d'ordres (actions) ===
    
    @abstractmethod
    async def place_stock_order(self,
                             symbol: str,
                             qty: Union[int, float],
                             side: str,
                             order_type: str = "market",
                             time_in_force: str = "day",
                             limit_price: Optional[float] = None,
                             stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place un ordre d'achat/vente d'actions.
        
        Args:
            symbol: Symbole de l'actif
            qty: Quantité à acheter/vendre
            side: Type d'ordre ("buy" ou "sell")
            order_type: Type d'ordre ("market", "limit", "stop", "stop_limit")
            time_in_force: Durée de validité de l'ordre ("day", "gtc", "ioc", "fok")
            limit_price: Prix limite (requis pour les ordres limit et stop_limit)
            stop_price: Prix stop (requis pour les ordres stop et stop_limit)
            
        Returns:
            Dict: Détails de l'ordre placé
        """
        pass
    
    # === Méthodes d'options ===
    
    @abstractmethod
    async def get_option_chain(self,
                             symbol: str,
                             expiration_date: Optional[str] = None,
                             option_type: Optional[OptionType] = None) -> List[OptionContract]:
        """
        Récupère la chaîne d'options pour un symbole.
        
        Args:
            symbol: Symbole du sous-jacent
            expiration_date: Date d'expiration optionnelle (YYYY-MM-DD)
            option_type: Type d'option optionnel (call ou put)
            
        Returns:
            List[OptionContract]: Liste des contrats d'options disponibles
        """
        pass
    
    @abstractmethod
    async def place_option_order(self,
                               option_symbol: str,
                               qty: int,
                               side: str,
                               order_type: str = "market",
                               time_in_force: str = "day",
                               limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place un ordre d'achat/vente d'options.
        
        Args:
            option_symbol: Symbole complet du contrat d'option
            qty: Quantité de contrats
            side: Type d'ordre ("buy" ou "sell")
            order_type: Type d'ordre ("market", "limit")
            time_in_force: Durée de validité de l'ordre
            limit_price: Prix limite (requis pour les ordres limit)
            
        Returns:
            Dict: Détails de l'ordre placé
        """
        pass
    
    @abstractmethod
    async def place_option_strategy(self,
                                  strategy_type: str,
                                  underlying: str,
                                  legs: List[Dict[str, Any]],
                                  qty: int) -> Dict[str, Any]:
        """
        Place un ordre pour une stratégie d'options multi-jambes.
        
        Args:
            strategy_type: Type de stratégie ("spread", "straddle", etc.)
            underlying: Symbole du sous-jacent
            legs: Liste des jambes de la stratégie
            qty: Quantité de stratégies
            
        Returns:
            Dict: Détails de l'ordre stratégie placé
        """
        pass
    
    # === Méthodes de données de marché ===
    
    @abstractmethod
    async def get_bars(self,
                     symbol: str,
                     timeframe: str,
                     start: Optional[Union[str, datetime]] = None,
                     end: Optional[Union[str, datetime]] = None,
                     limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Récupère les barres de prix historiques.
        
        Args:
            symbol: Symbole de l'actif
            timeframe: Intervalle de temps ("1m", "5m", "1h", "1d", etc.)
            start: Date/heure de début
            end: Date/heure de fin
            limit: Nombre maximum de barres à récupérer
            
        Returns:
            Dict: Données de barres de prix
        """
        pass
    
    @abstractmethod
    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère la dernière cotation pour un symbole.
        
        Args:
            symbol: Symbole de l'actif
            
        Returns:
            Dict: Dernière cotation
        """
        pass
    
    @abstractmethod
    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère la dernière transaction pour un symbole.
        
        Args:
            symbol: Symbole de l'actif
            
        Returns:
            Dict: Dernière transaction
        """
        pass
    
    # === Méthodes avancées pour HFT et L2 ===
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Récupère le carnet d'ordres (order book) pour un symbole.
        
        Args:
            symbol: Symbole de l'actif
            depth: Profondeur du carnet d'ordres
            
        Returns:
            Dict: Carnet d'ordres avec offres et demandes
        """
        pass
    
    @abstractmethod
    async def subscribe_to_quotes(self, symbols: List[str], callback) -> bool:
        """
        S'abonne aux cotations en temps réel pour une liste de symboles.
        
        Args:
            symbols: Liste des symboles
            callback: Fonction de rappel pour traiter les mises à jour
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon
        """
        pass
    
    @abstractmethod
    async def subscribe_to_trades(self, symbols: List[str], callback) -> bool:
        """
        S'abonne aux transactions en temps réel pour une liste de symboles.
        
        Args:
            symbols: Liste des symboles
            callback: Fonction de rappel pour traiter les mises à jour
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon
        """
        pass
    
    @abstractmethod
    async def subscribe_to_order_book(self, symbols: List[str], callback, depth: int = 10) -> bool:
        """
        S'abonne aux mises à jour du carnet d'ordres en temps réel.
        
        Args:
            symbols: Liste des symboles
            callback: Fonction de rappel pour traiter les mises à jour
            depth: Profondeur du carnet d'ordres
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon
        """
        pass
