#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour MarketCalendarService
-----------------------------------------
Ce module teste les fonctionnalités du service de calendrier de marché.
"""

import os
import sys
import unittest
from datetime import datetime, time, timedelta
import pytz
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au path pour importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Importer le service à tester
from app.services.market_calendar import MarketCalendarService


class TestMarketCalendarService(unittest.TestCase):
    """Classe de tests pour MarketCalendarService"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.stock_calendar = MarketCalendarService(market_type='stock')
        self.crypto_calendar = MarketCalendarService(market_type='crypto')
        self.eastern_tz = pytz.timezone('US/Eastern')
    
    def test_initialization(self):
        """Teste l'initialisation du service"""
        self.assertEqual(self.stock_calendar.market_type, 'stock')
        self.assertEqual(self.crypto_calendar.market_type, 'crypto')
        self.assertEqual(self.stock_calendar.eastern_tz, pytz.timezone('US/Eastern'))
        self.assertIsNotNone(self.stock_calendar.holidays)
        self.assertIsInstance(self.stock_calendar.holidays, list)
    
    def test_load_market_holidays(self):
        """Teste le chargement des jours fériés"""
        holidays = self.stock_calendar._load_market_holidays()
        
        # Vérifier que les jours fériés sont chargés
        self.assertIsInstance(holidays, list)
        self.assertGreater(len(holidays), 0)
        
        # Vérifier que tous les éléments sont des dates
        for holiday in holidays:
            self.assertIsInstance(holiday, datetime)
    
    def test_crypto_market_always_open(self):
        """Teste que le marché crypto est toujours considéré comme ouvert"""
        # Tester différents moments (weekend, jour férié, nuit)
        test_times = [
            datetime(2025, 1, 1, 12, 0),  # Jour de l'An (férié)
            datetime(2025, 2, 16, 12, 0),  # Dimanche
            datetime(2025, 3, 10, 3, 0)    # Lundi très tôt le matin
        ]
        
        for test_time in test_times:
            # Convertir en aware datetime
            test_time = self.eastern_tz.localize(test_time)
            self.assertTrue(self.crypto_calendar.is_market_open(test_time))
    
    def test_stock_market_closed_on_weekends(self):
        """Teste que le marché boursier est fermé le weekend"""
        # Samedi
        saturday = self.eastern_tz.localize(datetime(2025, 2, 15, 12, 0))
        self.assertFalse(self.stock_calendar.is_market_open(saturday))
        
        # Dimanche
        sunday = self.eastern_tz.localize(datetime(2025, 2, 16, 12, 0))
        self.assertFalse(self.stock_calendar.is_market_open(sunday))
    
    def test_stock_market_closed_on_holidays(self):
        """Teste que le marché boursier est fermé les jours fériés"""
        # Trouver un jour férié dans la liste
        if len(self.stock_calendar.holidays) > 0:
            holiday = self.stock_calendar.holidays[0]
            # Créer un datetime au milieu de la journée
            holiday_time = self.eastern_tz.localize(
                datetime.combine(holiday.date(), time(12, 0))
            )
            self.assertFalse(self.stock_calendar.is_market_open(holiday_time))
    
    def test_stock_market_open_during_regular_hours(self):
        """Teste que le marché boursier est ouvert pendant les heures normales"""
        # Mardi 4 Mars 2025 à 10h30 ET (jour ouvrable standard)
        tuesday_10_30 = self.eastern_tz.localize(datetime(2025, 3, 4, 10, 30))
        
        # Patcher la méthode _load_market_holidays pour s'assurer que ce n'est pas un jour férié
        with patch.object(MarketCalendarService, '_load_market_holidays', return_value=[]):
            calendar = MarketCalendarService(market_type='stock')
            self.assertTrue(calendar.is_market_open(tuesday_10_30))
    
    def test_stock_market_closed_outside_regular_hours(self):
        """Teste que le marché boursier est fermé en dehors des heures normales"""
        # Mardi 4 Mars 2025 à 9h00 ET (avant l'ouverture)
        tuesday_9_00 = self.eastern_tz.localize(datetime(2025, 3, 4, 9, 0))
        
        # Mardi 4 Mars 2025 à 16h30 ET (après la fermeture)
        tuesday_16_30 = self.eastern_tz.localize(datetime(2025, 3, 4, 16, 30))
        
        # Patcher la méthode _load_market_holidays pour s'assurer que ce n'est pas un jour férié
        with patch.object(MarketCalendarService, '_load_market_holidays', return_value=[]):
            calendar = MarketCalendarService(market_type='stock')
            self.assertFalse(calendar.is_market_open(tuesday_9_00))
            self.assertFalse(calendar.is_market_open(tuesday_16_30))
    
    def test_get_next_market_open_weekday(self):
        """Teste la fonction get_next_market_open pour un jour de semaine"""
        # Mardi 4 Mars 2025 à 8h00 ET (avant l'ouverture)
        tuesday_8_00 = self.eastern_tz.localize(datetime(2025, 3, 4, 8, 0))
        
        # Patcher la méthode _load_market_holidays pour s'assurer que ce n'est pas un jour férié
        with patch.object(MarketCalendarService, '_load_market_holidays', return_value=[]):
            calendar = MarketCalendarService(market_type='stock')
            next_open = calendar.get_next_market_open(tuesday_8_00)
            
            # Devrait être le même jour à 9h30
            self.assertEqual(next_open.date(), tuesday_8_00.date())
            self.assertEqual(next_open.hour, 9)
            self.assertEqual(next_open.minute, 30)
    
    def test_get_next_market_open_after_close(self):
        """Teste la fonction get_next_market_open après la fermeture du marché"""
        # Mardi 4 Mars 2025 à 16h30 ET (après la fermeture)
        tuesday_16_30 = self.eastern_tz.localize(datetime(2025, 3, 4, 16, 30))
        
        # Patcher la méthode _load_market_holidays pour s'assurer que ce n'est pas un jour férié
        with patch.object(MarketCalendarService, '_load_market_holidays', return_value=[]):
            calendar = MarketCalendarService(market_type='stock')
            next_open = calendar.get_next_market_open(tuesday_16_30)
            
            # Devrait être le lendemain à 9h30
            expected_next = tuesday_16_30 + timedelta(days=1)
            self.assertEqual(next_open.date(), expected_next.date())
            self.assertEqual(next_open.hour, 9)
            self.assertEqual(next_open.minute, 30)
    
    def test_get_next_market_open_weekend(self):
        """Teste la fonction get_next_market_open pour un weekend"""
        # Samedi 1er Mars 2025 à 12h00 ET
        saturday_12_00 = self.eastern_tz.localize(datetime(2025, 3, 1, 12, 0))
        
        # Patcher la méthode _load_market_holidays pour s'assurer que ce n'est pas un jour férié
        with patch.object(MarketCalendarService, '_load_market_holidays', return_value=[]):
            calendar = MarketCalendarService(market_type='stock')
            next_open = calendar.get_next_market_open(saturday_12_00)
            
            # Devrait être le lundi suivant à 9h30
            self.assertEqual(next_open.date(), datetime(2025, 3, 3).date())
            self.assertEqual(next_open.hour, 9)
            self.assertEqual(next_open.minute, 30)
    
    def test_get_next_market_close(self):
        """Teste la fonction get_next_market_close"""
        # Mardi 4 Mars 2025 à 10h30 ET (pendant les heures d'ouverture)
        tuesday_10_30 = self.eastern_tz.localize(datetime(2025, 3, 4, 10, 30))
        
        # Patcher la méthode _load_market_holidays et is_market_open
        with patch.object(MarketCalendarService, '_load_market_holidays', return_value=[]), \
             patch.object(MarketCalendarService, 'is_market_open', return_value=True):
            
            calendar = MarketCalendarService(market_type='stock')
            next_close = calendar.get_next_market_close(tuesday_10_30)
            
            # Devrait être le même jour à 16h00
            self.assertEqual(next_close.date(), tuesday_10_30.date())
            self.assertEqual(next_close.hour, 16)
            self.assertEqual(next_close.minute, 0)


class TestMarketCalendarServiceIntegration(unittest.TestCase):
    """Tests d'intégration pour MarketCalendarService"""
    
    def test_timezone_handling(self):
        """Teste la gestion des fuseaux horaires"""
        calendar = MarketCalendarService(market_type='stock')
        
        # Créer une date et heure en UTC
        utc_time = datetime(2025, 3, 4, 15, 0, tzinfo=pytz.UTC)  # 10h00 ET
        
        # Tester si le marché est ouvert (l'heure devrait être convertie)
        with patch.object(MarketCalendarService, '_load_market_holidays', return_value=[]):
            self.assertTrue(calendar.is_market_open(utc_time))
    
    def test_holiday_adjustment(self):
        """Teste l'ajustement des jours fériés tombant le weekend"""
        calendar = MarketCalendarService(market_type='stock')
        
        # Trouver les jours fériés qui tombent un samedi ou dimanche
        weekend_holidays = []
        for holiday in calendar.holidays:
            if holiday.weekday() >= 5:  # 5=Samedi, 6=Dimanche
                weekend_holidays.append(holiday)
        
        # Vérifier que les jours fériés tombant le weekend ont été ajustés
        holidays_set = set(h.date() for h in calendar.holidays)
        
        for holiday in weekend_holidays:
            original_date = holiday.date()
            
            # Si c'est un samedi, le jour férié devrait être observé le vendredi
            if original_date.weekday() == 5:
                friday = original_date - timedelta(days=1)
                self.assertIn(friday, holidays_set)
            
            # Si c'est un dimanche, le jour férié devrait être observé le lundi
            elif original_date.weekday() == 6:
                monday = original_date + timedelta(days=1)
                self.assertIn(monday, holidays_set)


if __name__ == '__main__':
    unittest.main()
