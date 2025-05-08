#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Broker Adapter

Adaptateur pour Alpaca API qui implémente l'interface BrokerAdapter.
Supporte les fonctionnalités de niveau 1, 2 et 3 d'Alpaca, incluant les options.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import json
import websockets

# Import Alpaca API
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST
from alpaca_trade_api.stream import Stream

# Import des modèles
from app.core.models.option import OptionContract, OptionType, OptionPosition, OptionAction
from app.core.broker_adapter.base import BrokerAdapter

logger = logging.getLogger(__name__)

class AlpacaAdapter(BrokerAdapter):
    """
    Adaptateur pour Alpaca API V2+.
    
    Implémente l'interface BrokerAdapter pour fournir un accès aux
    fonctionnalités d'Alpaca, y compris le trading d'options et HFT.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'adaptateur Alpaca.
        
        Args:
            config: Dictionnaire de configuration contenant les clés API, etc.
        """
        super().__init__(config)
        
        self.name = "Alpaca"
        self.description = "Alpaca API V2+ Adapter"
        
        # Récupérer la configuration
        self.mode = config.get("mode", "paper")
        
        if self.mode == "live":
            self.api_key = config.get("live_key") or os.getenv("ALPACA_LIVE_KEY")
            self.api_secret = config.get("live_secret") or os.getenv("ALPACA_LIVE_SECRET")
            self.base_url = config.get("live_url") or os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
            logger.info("Adaptateur Alpaca configuré pour le trading LIVE (réel)")
        else:  # mode paper par défaut
            self.api_key = config.get("paper_key") or os.getenv("ALPACA_PAPER_KEY")
            self.api_secret = config.get("paper_secret") or os.getenv("ALPACA_PAPER_SECRET")
            self.base_url = config.get("paper_url") or os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
            logger.info("Adaptateur Alpaca configuré pour le trading PAPER (simulation)")
        
        # URL pour les données de marché
        self.data_url = config.get("data_url") or os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
        # Niveau d'abonnement
        self.subscription_level = int(config.get("subscription_level") or os.getenv("ALPACA_SUBSCRIPTION_LEVEL", "1"))
        logger.info(f"Utilisation du niveau d'abonnement Alpaca: {self.subscription_level}")
        
        # Activation des fonctionnalités
        self.enable_options = config.get("enable_options", True) if self.subscription_level >= 3 else False
        self.enable_hft = config.get("enable_hft", True) if self.subscription_level >= 3 else False
        self.enable_l2_data = config.get("enable_l2_data", True) if self.subscription_level >= 3 else False
        
        # Mise à jour des fonctionnalités disponibles
        self.features = {
            "stocks": True,
            "options": self.enable_options,
            "crypto": True,
            "futures": False,
            "forex": False,
            "l2_data": self.enable_l2_data,
            "hft": self.enable_hft
        }
        
        # Clients API
        self.api = None
        self.data_ws = None
        self.streaming_api = None
        logger.info(f"Adaptateur Alpaca initialisé avec les fonctionnalités: {self.features}")
    
    async def connect(self) -> bool:
        """
        Établit la connexion avec Alpaca API.
        
        Returns:
            bool: True si la connexion est établie avec succès, False sinon
        """
        try:
            # Initialiser le client REST API
            # Initialiser le client REST API sans data_url qui n'est pas supporté dans cette version
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Tester la connexion
            account = self.api.get_account()
            logger.info(f"Connecté à Alpaca - Compte ID: {account.id}, Status: {account.status}")
            
            # Initialiser le client WebSocket pour les données en temps réel si nécessaire
            if self.enable_hft or self.enable_l2_data:
                # Cette partie serait implémentée avec le client WebSocket Alpaca
                # pour les données en temps réel L1 et L2
                logger.info("Initialisation de la connexion WebSocket pour les données en temps réel...")
                # self.streaming_api = StreamConn(...)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur de connexion à Alpaca API: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Ferme la connexion avec Alpaca API.
        
        Returns:
            bool: True si la déconnexion est réussie, False sinon
        """
        try:
            # Fermer les connexions WebSocket si actives
            if self.streaming_api:
                # Logique pour fermer la connexion WebSocket
                self.streaming_api = None
                
            self.api = None
            logger.info("Déconnecté d'Alpaca API")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion d'Alpaca API: {str(e)}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Récupère les informations du compte Alpaca.
        
        Returns:
            Dict: Informations du compte
        """
        try:
            if not self.api:
                await self.connect()
                
            account = self.api.get_account()
            
            return {
                "id": account.id,
                "status": account.status,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "last_equity": float(account.last_equity),
                "daytrade_count": account.daytrade_count,
                "daytrading_buying_power": float(account.daytrading_buying_power)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations du compte: {str(e)}")
            return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Récupère les positions actuelles.
        
        Returns:
            List[Dict]: Liste des positions actuelles
        """
        try:
            if not self.api:
                await self.connect()
                
            positions = self.api.list_positions()
            result = []
            
            for pos in positions:
                position_data = {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "current_price": float(pos.current_price),
                    "lastday_price": float(pos.lastday_price),
                    "change_today": float(pos.change_today)
                }
                result.append(position_data)
                
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des positions: {str(e)}")
            return []
    
    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres selon leur statut.
        
        Args:
            status: Statut des ordres à récupérer (open, closed, all)
            
        Returns:
            List[Dict]: Liste des ordres
        """
        try:
            if not self.api:
                await self.connect()
                
            # Mapper le statut à celui attendu par Alpaca
            status_map = {
                "open": "open",
                "closed": "closed",
                "all": "all",
                None: "open"  # Par défaut, récupérer les ordres ouverts
            }
            alpaca_status = status_map.get(status, "open")
            
            orders = self.api.list_orders(status=alpaca_status)
            result = []
            
            for order in orders:
                order_data = {
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "filled_qty": float(order.filled_qty),
                    "side": order.side,
                    "type": order.type,
                    "time_in_force": order.time_in_force,
                    "limit_price": float(order.limit_price) if order.limit_price else None,
                    "stop_price": float(order.stop_price) if order.stop_price else None,
                    "status": order.status,
                    "created_at": order.created_at.isoformat() if hasattr(order.created_at, 'isoformat') else order.created_at,
                    "filled_at": order.filled_at.isoformat() if order.filled_at and hasattr(order.filled_at, 'isoformat') else order.filled_at,
                    "is_option": hasattr(order, "legs") and order.legs is not None
                }
                result.append(order_data)
                
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres: {str(e)}")
            return []
    
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
        try:
            if not self.api:
                await self.connect()
                
            # Vérification des paramètres
            if order_type in ["limit", "stop_limit"] and limit_price is None:
                raise ValueError(f"Le prix limite est requis pour les ordres de type {order_type}")
                
            if order_type in ["stop", "stop_limit"] and stop_price is None:
                raise ValueError(f"Le prix stop est requis pour les ordres de type {order_type}")
                
            # Préparation des paramètres pour Alpaca
            params = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force
            }
            
            if limit_price is not None:
                params["limit_price"] = str(limit_price)
                
            if stop_price is not None:
                params["stop_price"] = str(stop_price)
                
            # Placement de l'ordre
            order = self.api.submit_order(**params)
            
            # Formatage de la réponse
            response = {
                "success": True,
                "order_id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "status": order.status
            }
            
            if hasattr(order, "filled_avg_price") and order.filled_avg_price:
                response["filled_avg_price"] = float(order.filled_avg_price)
                
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors du placement de l'ordre d'action: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def get_option_positions(self) -> List[Dict[str, Any]]:
        """
        Récupère les positions d'options actuelles.
        
        Returns:
            List[Dict]: Liste des positions d'options actuelles
        """
        if not self.enable_options:
            logger.warning("Le trading d'options n'est pas activé dans cette configuration")
            return []
            
        try:
            if not self.api:
                await self.connect()
                
            # Dans la véritable API Alpaca, il y aurait une méthode spécifique pour récupérer les positions d'options
            option_positions = self.api.get_option_positions()
            result = []
            
            for pos in option_positions:
                # Traiter les données spécifiques aux options
                position_data = {
                    "symbol": pos.symbol,
                    "option_type": "call" if "C" in pos.symbol else "put",
                    "strike": float(pos.strike_price),
                    "expiry": pos.expiration_date,
                    "qty": int(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "underlying_symbol": pos.underlying_symbol,
                    "delta": float(pos.delta) if hasattr(pos, "delta") else None,
                    "gamma": float(pos.gamma) if hasattr(pos, "gamma") else None,
                    "theta": float(pos.theta) if hasattr(pos, "theta") else None,
                    "vega": float(pos.vega) if hasattr(pos, "vega") else None
                }
                result.append(position_data)
                
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des positions d'options: {str(e)}")
            return []
            
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
        if not self.enable_options:
            logger.warning("Le trading d'options n'est pas activé dans cette configuration")
            return []
            
        try:
            if not self.api:
                await self.connect()
                
            # Récupérer les dates d'expiration disponibles
            if not expiration_date:
                expiry_dates = self.api.get_option_expirations(symbol)
                if not expiry_dates:
                    logger.warning(f"Aucune date d'expiration disponible pour {symbol}")
                    return []
                expiration_date = expiry_dates[0]  # Utiliser la première date d'expiration
            
            # Récupérer les options pour cette date d'expiration
            options_params = {
                "symbol": symbol,
                "expiration_date": expiration_date
            }
            
            if option_type:
                options_params["option_type"] = option_type.value
                
            option_chain = self.api.get_option_chain(**options_params)
            
            result = []
            for opt in option_chain:
                contract_type = OptionType.CALL if opt.option_type.lower() == "call" else OptionType.PUT
                
                contract = OptionContract(
                    symbol=opt.symbol,
                    underlying=symbol,
                    option_type=contract_type,
                    strike=float(opt.strike_price),
                    expiry_date=opt.expiration_date,
                    bid=float(opt.bid_price) if hasattr(opt, "bid_price") else 0.0,
                    ask=float(opt.ask_price) if hasattr(opt, "ask_price") else 0.0,
                    last=float(opt.last_price) if hasattr(opt, "last_price") else 0.0,
                    volume=int(opt.volume) if hasattr(opt, "volume") else 0,
                    open_interest=int(opt.open_interest) if hasattr(opt, "open_interest") else 0,
                    implied_volatility=float(opt.implied_volatility) if hasattr(opt, "implied_volatility") else 0.0,
                    delta=float(opt.delta) if hasattr(opt, "delta") else 0.0,
                    gamma=float(opt.gamma) if hasattr(opt, "gamma") else 0.0,
                    theta=float(opt.theta) if hasattr(opt, "theta") else 0.0,
                    vega=float(opt.vega) if hasattr(opt, "vega") else 0.0,
                    rho=float(opt.rho) if hasattr(opt, "rho") else 0.0
                )
                result.append(contract)
                
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la chaîne d'options pour {symbol}: {str(e)}")
            return []
    
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
        if not self.enable_options:
            logger.warning("Le trading d'options n'est pas activé dans cette configuration")
            return {"success": False, "error": "Trading d'options non activé"}
            
        try:
            if not self.api:
                await self.connect()
                
            # Vérification des paramètres
            if order_type == "limit" and limit_price is None:
                raise ValueError("Le prix limite est requis pour les ordres limit")
                
            # Préparation des paramètres pour Alpaca
            params = {
                "symbol": option_symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force
            }
            
            if limit_price is not None:
                params["limit_price"] = str(limit_price)
                
            # Placement de l'ordre d'option
            order = self.api.submit_option_order(**params)
            
            # Formatage de la réponse
            response = {
                "success": True,
                "order_id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": int(order.qty),
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "status": order.status
            }
            
            if hasattr(order, "filled_avg_price") and order.filled_avg_price:
                response["filled_avg_price"] = float(order.filled_avg_price)
                
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors du placement de l'ordre d'option: {str(e)}")
            return {"success": False, "error": str(e)}
    
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
        if not self.enable_options:
            logger.warning("Le trading d'options n'est pas activé dans cette configuration")
            return {"success": False, "error": "Trading d'options non activé"}
            
        try:
            if not self.api:
                await self.connect()
                
            # Vérifier que nous avons au moins deux jambes pour un spread
            if strategy_type.lower() in ["spread", "iron_condor", "butterfly"] and len(legs) < 2:
                raise ValueError(f"Au moins deux jambes sont nécessaires pour un {strategy_type}")
                
            # Préparer les paramètres pour la stratégie d'options
            strategy_params = {
                "strategy_type": strategy_type,
                "underlying": underlying,
                "legs": legs,
                "qty": qty
            }
            
            # Placement de l'ordre de stratégie
            order = self.api.submit_option_strategy_order(**strategy_params)
            
            # Formatage de la réponse
            response = {
                "success": True,
                "order_id": order.id,
                "client_order_id": order.client_order_id,
                "underlying": underlying,
                "strategy_type": strategy_type,
                "qty": qty,
                "status": order.status,
                "legs": [{"symbol": leg.symbol, "side": leg.side} for leg in order.legs] if hasattr(order, "legs") else []
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors du placement de la stratégie d'options: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # === Méthodes de données de marché ===
    
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
        try:
            if not self.api:
                await self.connect()
                
            # Formatter les dates si nécessaire
            if isinstance(start, datetime):
                start = start.strftime("%Y-%m-%d")
                
            if isinstance(end, datetime):
                end = end.strftime("%Y-%m-%d")
                
            # Paramètres pour récupérer les barres
            params = {}
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            if limit:
                params["limit"] = limit
                
            # Mapper le timeframe à celui attendu par Alpaca
            timeframe_map = {
                "1m": "1Min",
                "5m": "5Min",
                "15m": "15Min",
                "30m": "30Min",
                "1h": "1Hour",
                "1d": "1Day",
                "1w": "1Week"
            }
            alpaca_timeframe = timeframe_map.get(timeframe, timeframe)
            
            # Récupérer les barres
            if self._is_crypto(symbol):
                # Pour les crypto
                formatted_symbol = self._format_crypto_symbol(symbol)
                bars = self.api.get_crypto_bars(formatted_symbol, alpaca_timeframe, **params)
            else:
                # Pour les actions
                bars = self.api.get_bars(symbol, alpaca_timeframe, **params)
            
            # Convertir en DataFrame
            if hasattr(bars, "df"):
                df = bars.df
            else:
                # Si ce n'est pas un DataFrame, essayer de le convertir
                records = []
                for bar in bars:
                    record = {
                        "timestamp": bar.t,
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": float(bar.v)
                    }
                    records.append(record)
                df = pd.DataFrame(records)
                if not df.empty and "timestamp" in df.columns:
                    df.set_index("timestamp", inplace=True)
            
            return {"success": True, "data": df.to_dict("records") if not df.empty else []}
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des barres pour {symbol}: {str(e)}")
            return {"success": False, "error": str(e), "data": []}
    
    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère la dernière cotation pour un symbole.
        
        Args:
            symbol: Symbole de l'actif
            
        Returns:
            Dict: Dernière cotation
        """
        try:
            if not self.api:
                await self.connect()
                
            if self._is_crypto(symbol):
                # Pour les crypto
                formatted_symbol = self._format_crypto_symbol(symbol)
                quote = self.api.get_latest_crypto_quote(formatted_symbol)
            else:
                # Pour les actions
                quote = self.api.get_latest_quote(symbol)
            
            return {
                "success": True,
                "symbol": symbol,
                "bid": float(quote.bp),
                "bid_size": float(quote.bs),
                "ask": float(quote.ap),
                "ask_size": float(quote.as_),
                "timestamp": quote.t.isoformat() if hasattr(quote.t, "isoformat") else quote.t
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la cotation pour {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère la dernière transaction pour un symbole.
        
        Args:
            symbol: Symbole de l'actif
            
        Returns:
            Dict: Dernière transaction
        """
        try:
            if not self.api:
                await self.connect()
                
            if self._is_crypto(symbol):
                # Pour les crypto
                formatted_symbol = self._format_crypto_symbol(symbol)
                trade = self.api.get_latest_crypto_trade(formatted_symbol)
            else:
                # Pour les actions
                trade = self.api.get_latest_trade(symbol)
            
            return {
                "success": True,
                "symbol": symbol,
                "price": float(trade.p),
                "size": float(trade.s),
                "timestamp": trade.t.isoformat() if hasattr(trade.t, "isoformat") else trade.t,
                "exchange": trade.x if hasattr(trade, "x") else None
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la transaction pour {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # === Méthodes avancées pour HFT et L2 ===
    
    async def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Récupère le carnet d'ordres (order book) pour un symbole.
        
        Args:
            symbol: Symbole de l'actif
            depth: Profondeur du carnet d'ordres
            
        Returns:
            Dict: Carnet d'ordres avec offres et demandes
        """
        if not self.enable_l2_data:
            logger.warning("Les données L2 ne sont pas activées dans cette configuration")
            return {"success": False, "error": "Données L2 non activées"}
            
        try:
            if not self.api:
                await self.connect()
                
            if self._is_crypto(symbol):
                # Format spécifique pour les crypto
                formatted_symbol = self._format_crypto_symbol(symbol)
                order_book = self.api.get_crypto_order_book(formatted_symbol, depth=depth)
            else:
                # Pour les actions
                order_book = self.api.get_order_book(symbol, depth=depth)
            
            # Formater le carnet d'ordres
            bids = [{"price": float(bid.p), "size": float(bid.s)} for bid in order_book.bids[:depth]]
            asks = [{"price": float(ask.p), "size": float(ask.s)} for ask in order_book.asks[:depth]]
            
            return {
                "success": True,
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "timestamp": order_book.timestamp.isoformat() if hasattr(order_book, "timestamp") and hasattr(order_book.timestamp, "isoformat") else None
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du carnet d'ordres pour {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def subscribe_to_quotes(self, symbols: List[str], callback) -> bool:
        """
        S'abonne aux cotations en temps réel pour une liste de symboles.
        
        Args:
            symbols: Liste des symboles
            callback: Fonction de rappel pour traiter les mises à jour
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon
        """
        if not self.enable_hft:
            logger.warning("HFT n'est pas activé dans cette configuration")
            return False
            
        try:
            if not self.streaming_api:
                # Initialiser l'API de streaming si ce n'est pas déjà fait
                self.streaming_api = Stream(
                    key_id=self.api_key,
                    secret_key=self.api_secret,
                    base_url=self.base_url,
                    data_feed="iex"  # Utiliser IEX pour les données temps réel (peut être modifié selon le niveau d'abonnement)
                )
            
            # Diviser les symboles en actions et crypto
            stock_symbols = [s for s in symbols if not self._is_crypto(s)]
            crypto_symbols = [self._format_crypto_symbol(s) for s in symbols if self._is_crypto(s)]
            
            # S'abonner aux cotations des actions
            if stock_symbols:
                for symbol in stock_symbols:
                    self.streaming_api.subscribe_quotes(callback, symbol)
                logger.info(f"Abonné aux cotations en temps réel pour {len(stock_symbols)} symboles actions")
            
            # S'abonner aux cotations des crypto
            if crypto_symbols:
                for symbol in crypto_symbols:
                    self.streaming_api.subscribe_crypto_quotes(callback, symbol)
                logger.info(f"Abonné aux cotations en temps réel pour {len(crypto_symbols)} symboles crypto")
            
            # Démarrer la connexion de streaming dans un thread séparé
            if not self.streaming_api._running:
                self.streaming_api.run_async()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'abonnement aux cotations: {str(e)}")
            return False
    
    async def subscribe_to_trades(self, symbols: List[str], callback) -> bool:
        """
        S'abonne aux transactions en temps réel pour une liste de symboles.
        
        Args:
            symbols: Liste des symboles
            callback: Fonction de rappel pour traiter les mises à jour
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon
        """
        if not self.enable_hft:
            logger.warning("HFT n'est pas activé dans cette configuration")
            return False
            
        try:
            if not self.streaming_api:
                # Initialiser l'API de streaming si ce n'est pas déjà fait
                self.streaming_api = Stream(
                    key_id=self.api_key,
                    secret_key=self.api_secret,
                    base_url=self.base_url,
                    data_feed="iex"  # Utiliser IEX pour les données temps réel (peut être modifié selon le niveau d'abonnement)
                )
            
            # Diviser les symboles en actions et crypto
            stock_symbols = [s for s in symbols if not self._is_crypto(s)]
            crypto_symbols = [self._format_crypto_symbol(s) for s in symbols if self._is_crypto(s)]
            
            # S'abonner aux transactions des actions
            if stock_symbols:
                for symbol in stock_symbols:
                    self.streaming_api.subscribe_trades(callback, symbol)
                logger.info(f"Abonné aux transactions en temps réel pour {len(stock_symbols)} symboles actions")
            
            # S'abonner aux transactions des crypto
            if crypto_symbols:
                for symbol in crypto_symbols:
                    self.streaming_api.subscribe_crypto_trades(callback, symbol)
                logger.info(f"Abonné aux transactions en temps réel pour {len(crypto_symbols)} symboles crypto")
            
            # Démarrer la connexion de streaming dans un thread séparé
            if not self.streaming_api._running:
                self.streaming_api.run_async()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'abonnement aux transactions: {str(e)}")
            return False
    
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
        if not self.enable_l2_data:
            logger.warning("Les données L2 ne sont pas activées dans cette configuration")
            return False
            
        try:
            if not self.streaming_api:
                # Initialiser l'API de streaming si ce n'est pas déjà fait
                self.streaming_api = Stream(
                    key_id=self.api_key,
                    secret_key=self.api_secret,
                    base_url=self.base_url,
                    data_feed="iex"  # Utiliser IEX pour les données temps réel (peut être modifié selon le niveau d'abonnement)
                )
            
            # Actuellement, Alpaca ne supporte les carnets d'ordres en temps réel que pour les crypto
            crypto_symbols = [self._format_crypto_symbol(s) for s in symbols if self._is_crypto(s)]
            
            if not crypto_symbols:
                logger.warning("L'abonnement aux carnets d'ordres n'est supporté que pour les crypto.")
                return False
            
            # S'abonner aux carnets d'ordres des crypto
            for symbol in crypto_symbols:
                self.streaming_api.subscribe_crypto_orderbooks(callback, symbol)
            logger.info(f"Abonné aux carnets d'ordres en temps réel pour {len(crypto_symbols)} symboles crypto")
            
            # Démarrer la connexion de streaming dans un thread séparé
            if not self.streaming_api._running:
                self.streaming_api.run_async()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'abonnement aux carnets d'ordres: {str(e)}")
            return False
            
    # === Méthodes utilitaires ===
    
    def _is_crypto(self, symbol: str) -> bool:
        """
        Détermine si un symbole est une cryptomonnaie.
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            bool: True si c'est une crypto, False sinon
        """
        # Vérification simple basée sur le format
        if "/" in symbol:
            return True  # Format BTC/USD
        if symbol.endswith("USD") or "BTC" in symbol or "ETH" in symbol:
            return True
        return False
    
    def _format_crypto_symbol(self, symbol: str) -> str:
        """
        Formate un symbole crypto pour Alpaca API v2.
        
        Args:
            symbol: Symbole crypto à formater
            
        Returns:
            str: Symbole formaté
        """
        # Si le symbole est déjà au format BTC/USD, le retourner tel quel
        if "/" in symbol:
            return symbol
            
        # Si c'est au format BTCUSD, convertir en BTC/USD
        if symbol.endswith("USD"):
            base = symbol[:-3]
            return f"{base}/USD"
            
        # Autre format, retourner tel quel
        return symbol
