#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de calendrier de marché pour MercurioAI
----------------------------------------------
Ce service gère les informations de calendrier de marché pour
déterminer quand les marchés sont ouverts ou fermés.
"""

import logging
import pandas as pd
from datetime import datetime, time, timedelta
import pytz
import os
import sys
from typing import Optional, Dict, List, Union, Tuple

# Configuration du logger
logger = logging.getLogger(__name__)

class MarketCalendarService:
    """
    Service de calendrier de marché qui fournit des informations
    sur les heures d'ouverture et de fermeture des marchés.
    """
    
    def __init__(self, market_type: str = 'stock'):
        """
        Initialise le service de calendrier de marché.
        
        Args:
            market_type: Type de marché ('stock' ou 'crypto')
        """
        self.market_type = market_type
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.holidays = self._load_market_holidays()
        logger.info(f"Service de calendrier de marché initialisé pour {market_type}")
    
    def _load_market_holidays(self) -> List[datetime]:
        """
        Charge les jours fériés du marché boursier américain pour l'année en cours.
        
        Returns:
            Liste des jours fériés
        """
        # Jours fériés communs pour les marchés américains en 2025
        current_year = datetime.now().year
        holidays = [
            # Jour de l'An
            datetime(current_year, 1, 1),
            # Martin Luther King Jr. Day (3ème lundi de janvier)
            datetime(current_year, 1, 20),
            # Presidents' Day (3ème lundi de février)
            datetime(current_year, 2, 17),
            # Good Friday (vendredi précédant le dimanche de Pâques)
            datetime(current_year, 4, 18),
            # Memorial Day (dernier lundi de mai)
            datetime(current_year, 5, 26),
            # Juneteenth National Independence Day
            datetime(current_year, 6, 19),
            # Independence Day
            datetime(current_year, 7, 4),
            # Labor Day (1er lundi de septembre)
            datetime(current_year, 9, 1),
            # Thanksgiving Day (4ème jeudi de novembre)
            datetime(current_year, 11, 27),
            # Christmas Day
            datetime(current_year, 12, 25)
        ]
        
        # Si le jour férié tombe un weekend, ajuster selon les règles
        adjusted_holidays = []
        for holiday in holidays:
            weekday = holiday.weekday()
            # Si le jour férié tombe un samedi, le jour de fermeture est le vendredi précédent
            if weekday == 5:  # samedi
                adjusted_holidays.append(holiday - timedelta(days=1))
            # Si le jour férié tombe un dimanche, le jour de fermeture est le lundi suivant
            elif weekday == 6:  # dimanche
                adjusted_holidays.append(holiday + timedelta(days=1))
            else:
                adjusted_holidays.append(holiday)
                
        return adjusted_holidays
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """
        Vérifie si le marché est ouvert à un moment donné.
        
        Args:
            check_time: Moment pour lequel vérifier (par défaut: maintenant)
            
        Returns:
            True si le marché est ouvert, False sinon
        """
        # Si aucun moment spécifié, utiliser le moment actuel
        if check_time is None:
            check_time = datetime.now(pytz.UTC).astimezone(self.eastern_tz)
        elif check_time.tzinfo is None:
            # Si le moment spécifié n'a pas de fuseau horaire, lui attribuer Eastern Time
            check_time = self.eastern_tz.localize(check_time)
        elif check_time.tzinfo != self.eastern_tz:
            # Si le fuseau horaire est différent, le convertir en Eastern Time
            check_time = check_time.astimezone(self.eastern_tz)
        
        # Pour les cryptomonnaies, le marché est toujours ouvert
        if self.market_type == 'crypto':
            return True
        
        # Pour les actions, vérifier les jours et heures d'ouverture
        weekday = check_time.weekday()
        current_time = check_time.time()
        current_date = check_time.date()
        
        # Le marché est fermé le weekend (samedi = 5, dimanche = 6)
        if weekday >= 5:
            logger.debug(f"Marché fermé: weekend ({['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][weekday]})")
            return False
        
        # Vérifier si c'est un jour férié
        for holiday in self.holidays:
            if current_date == holiday.date():
                logger.debug(f"Marché fermé: jour férié ({holiday.strftime('%Y-%m-%d')})")
                return False
        
        # Heures d'ouverture régulières: 9h30 à 16h00 Eastern Time
        market_open_time = time(9, 30)
        market_close_time = time(16, 0)
        
        # Vérifier si l'heure actuelle est dans la plage d'ouverture du marché
        if market_open_time <= current_time <= market_close_time:
            logger.debug(f"Marché ouvert: {current_time} est entre {market_open_time} et {market_close_time}")
            return True
        else:
            logger.debug(f"Marché fermé: {current_time} n'est pas entre {market_open_time} et {market_close_time}")
            return False
    
    def get_next_market_open(self, from_time: Optional[datetime] = None) -> datetime:
        """
        Détermine le prochain moment d'ouverture du marché.
        
        Args:
            from_time: Moment à partir duquel chercher (par défaut: maintenant)
            
        Returns:
            Datetime du prochain moment d'ouverture du marché
        """
        # Pour les cryptomonnaies, considérer que le marché est toujours ouvert
        if self.market_type == 'crypto':
            # Retourner le moment actuel
            if from_time is None:
                return datetime.now(self.eastern_tz)
            return from_time
        
        # Si aucun moment spécifié, utiliser le moment actuel
        if from_time is None:
            from_time = datetime.now(pytz.UTC).astimezone(self.eastern_tz)
        elif from_time.tzinfo is None:
            # Si le moment spécifié n'a pas de fuseau horaire, lui attribuer Eastern Time
            from_time = self.eastern_tz.localize(from_time)
        elif from_time.tzinfo != self.eastern_tz:
            # Si le fuseau horaire est différent, le convertir en Eastern Time
            from_time = from_time.astimezone(self.eastern_tz)
        
        # Heure d'ouverture du marché: 9h30 Eastern Time
        market_open_time = time(9, 30)
        
        # Commencer par le jour actuel
        check_date = from_time.date()
        
        # Si l'heure actuelle est après 9h30, passer au jour suivant
        if from_time.time() >= market_open_time:
            check_date = check_date + timedelta(days=1)
        
        # Vérifier chaque jour jusqu'à trouver un jour d'ouverture du marché
        max_days_to_check = 10  # Pour éviter une boucle infinie
        days_checked = 0
        
        while days_checked < max_days_to_check:
            # Construire le datetime pour 9h30 à la date vérifiée
            next_open = datetime.combine(check_date, market_open_time)
            next_open = self.eastern_tz.localize(next_open)
            
            # Vérifier si c'est un jour d'ouverture du marché
            weekday = check_date.weekday()
            
            # Le marché est fermé le weekend (samedi = 5, dimanche = 6)
            if weekday >= 5:
                check_date = check_date + timedelta(days=1)
                days_checked += 1
                continue
            
            # Vérifier si c'est un jour férié
            is_holiday = False
            for holiday in self.holidays:
                if check_date == holiday.date():
                    is_holiday = True
                    break
            
            if is_holiday:
                check_date = check_date + timedelta(days=1)
                days_checked += 1
                continue
            
            # Si on arrive ici, c'est un jour d'ouverture du marché
            return next_open
        
        # Si on n'a pas trouvé de jour d'ouverture dans les 10 jours,
        # retourner une estimation (prochain jour ouvrable)
        logger.warning("Impossible de trouver le prochain jour d'ouverture du marché dans les 10 prochains jours")
        return self.eastern_tz.localize(datetime.combine(check_date, market_open_time))
    
    def get_next_market_close(self, from_time: Optional[datetime] = None) -> datetime:
        """
        Détermine le prochain moment de fermeture du marché.
        
        Args:
            from_time: Moment à partir duquel chercher (par défaut: maintenant)
            
        Returns:
            Datetime du prochain moment de fermeture du marché
        """
        # Pour les cryptomonnaies, retourner un moment très lointain
        if self.market_type == 'crypto':
            # Retourner un moment un an dans le futur
            current_time = datetime.now(self.eastern_tz) if from_time is None else from_time
            return current_time + timedelta(days=365)
        
        # Si aucun moment spécifié, utiliser le moment actuel
        if from_time is None:
            from_time = datetime.now(pytz.UTC).astimezone(self.eastern_tz)
        elif from_time.tzinfo is None:
            # Si le moment spécifié n'a pas de fuseau horaire, lui attribuer Eastern Time
            from_time = self.eastern_tz.localize(from_time)
        elif from_time.tzinfo != self.eastern_tz:
            # Si le fuseau horaire est différent, le convertir en Eastern Time
            from_time = from_time.astimezone(self.eastern_tz)
        
        # Heure de fermeture du marché: 16h00 Eastern Time
        market_close_time = time(16, 0)
        
        # Si le marché est actuellement ouvert
        if self.is_market_open(from_time):
            # La fermeture est aujourd'hui à 16h00
            close_datetime = datetime.combine(from_time.date(), market_close_time)
            return self.eastern_tz.localize(close_datetime)
        
        # Sinon, trouver le prochain jour d'ouverture et retourner sa fermeture
        next_open = self.get_next_market_open(from_time)
        next_close = datetime.combine(next_open.date(), market_close_time)
        return self.eastern_tz.localize(next_close)
