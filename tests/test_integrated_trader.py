#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour run_integrated_trader.py
--------------------------------------------
Ce module teste les fonctionnalités du script intégré qui combine trading et entraînement.
"""

import os
import sys
import unittest
import time
from datetime import datetime
import pytz
from unittest.mock import patch, MagicMock, call

# Ajouter le répertoire parent au path pour importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Importations du module à tester
# Nous importons les fonctions individuelles pour pouvoir les tester séparément
from scripts.run_integrated_trader import (
    is_market_open, 
    should_run_training, 
    run_training,
    run_trading,
)


class TestIntegratedTrader(unittest.TestCase):
    """Classe de tests pour le script integrated_trader"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Définir la variable globale pour les tests qui en ont besoin
        import scripts.run_integrated_trader
        scripts.run_integrated_trader.running = True
        scripts.run_integrated_trader.last_training_time = None
        scripts.run_integrated_trader.TRAINING_INTERVAL = 24 * 60 * 60  # 24 heures en secondes
    
    @patch('scripts.run_integrated_trader.MarketCalendarService')
    def test_is_market_open_with_service(self, mock_market_calendar):
        """Teste la fonction is_market_open avec MarketCalendarService disponible"""
        # Configuration du mock
        mock_instance = mock_market_calendar.return_value
        
        # Cas 1: Le marché est ouvert
        mock_instance.is_market_open.return_value = True
        self.assertTrue(is_market_open())
        
        # Cas 2: Le marché est fermé
        mock_instance.is_market_open.return_value = False
        self.assertFalse(is_market_open())
        
        # Vérifier que le service a été appelé correctement
        mock_market_calendar.assert_called_with()
        self.assertEqual(mock_instance.is_market_open.call_count, 2)
    
    @patch('scripts.run_integrated_trader.MarketCalendarService')
    def test_is_market_open_with_exception(self, mock_market_calendar):
        """Teste la fonction is_market_open quand une exception est levée"""
        # Le service lève une exception
        mock_instance = mock_market_calendar.return_value
        mock_instance.is_market_open.side_effect = Exception("Test exception")
        
        # Par défaut, on devrait avoir False en cas d'erreur
        self.assertFalse(is_market_open())
    
    @patch('scripts.run_integrated_trader.USE_MARKET_CALENDAR', False)
    @patch('scripts.run_integrated_trader.datetime')
    def test_is_market_open_alternative_method(self, mock_datetime):
        """Teste la méthode alternative de is_market_open quand MarketCalendarService n'est pas disponible"""
        # Configurer un mock pour datetime.now() qui renvoie une date/heure spécifique
        mock_now = MagicMock()
        mock_datetime.now.return_value = mock_now
        
        # Cas 1: Jour de semaine, heure d'ouverture du marché (mardi 10h30 ET)
        mock_now.weekday.return_value = 1  # Mardi
        mock_now.hour = 16  # Correspond à 10h00 ET avec le décalage de 6 heures
        mock_now.minute = 30
        self.assertTrue(is_market_open())
        
        # Cas 2: Jour de semaine, heure de fermeture du marché (mardi 8h00 ET)
        mock_now.hour = 14  # Correspond à 8h00 ET avec le décalage de 6 heures
        self.assertFalse(is_market_open())
        
        # Cas 3: Weekend (samedi)
        mock_now.weekday.return_value = 5  # Samedi
        mock_now.hour = 16
        self.assertFalse(is_market_open())
    
    def test_should_run_training(self):
        """Teste la fonction should_run_training"""
        import scripts.run_integrated_trader as trader
        
        # Cas 1: Auto-training désactivé
        self.assertFalse(should_run_training(auto_training=False))
        
        # Cas 2: Force training actif (devrait toujours retourner True)
        self.assertTrue(should_run_training(auto_training=False, force_training=True))
        self.assertTrue(should_run_training(auto_training=True, force_training=True))
        
        # Cas 3: Premier entraînement (last_training_time est None)
        self.assertTrue(should_run_training(auto_training=True))
        
        # Cas 4: Intervalle pas encore écoulé
        trader.last_training_time = time.time() - 1000  # 1000 secondes ago
        trader.TRAINING_INTERVAL = 3600  # 1 heure
        self.assertFalse(should_run_training(auto_training=True))
        
        # Cas 5: Intervalle écoulé
        trader.last_training_time = time.time() - 5000  # 5000 secondes ago
        trader.TRAINING_INTERVAL = 3600  # 1 heure
        self.assertTrue(should_run_training(auto_training=True))
    
    @patch('scripts.run_integrated_trader.subprocess.run')
    def test_run_training(self, mock_subprocess_run):
        """Teste la fonction run_training"""
        import scripts.run_integrated_trader as trader
        
        # Configurer le mock pour simuler une exécution réussie
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Entraînement terminé avec succès"
        mock_subprocess_run.return_value = mock_result
        
        # Cas 1: Appel avec les paramètres par défaut
        run_training()
        
        # Vérifier que subprocess.run a été appelé avec les bons arguments
        mock_subprocess_run.assert_called_once()
        args, kwargs = mock_subprocess_run.call_args
        
        # Vérifier que la commande contient les paramètres attendus
        cmd = args[0]
        self.assertIn('train_all_models.py', cmd[1])
        self.assertIn('--days', cmd)
        self.assertIn('90', cmd)
        self.assertIn('--include_stocks', cmd)
        self.assertIn('--include_crypto', cmd)
        
        # Vérifier que last_training_time a été mis à jour
        self.assertIsNotNone(trader.last_training_time)
        
        # Réinitialiser le mock
        mock_subprocess_run.reset_mock()
        
        # Cas 2: Appel avec des paramètres spécifiques
        symbols = ['BTC-USD', 'ETH-USD', 'AAPL']
        run_training(symbols=symbols, days=120, use_gpu=True)
        
        # Vérifier que subprocess.run a été appelé avec les bons arguments
        mock_subprocess_run.assert_called_once()
        args, kwargs = mock_subprocess_run.call_args
        
        # Vérifier que la commande contient les paramètres attendus
        cmd = args[0]
        self.assertIn('--days', cmd)
        self.assertIn('120', cmd)
        self.assertIn('--symbols', cmd)
        self.assertIn('BTC-USD,ETH-USD,AAPL', cmd)
        self.assertIn('--use_gpu', cmd)
    
    @patch('scripts.run_integrated_trader.subprocess.Popen')
    def test_run_trading(self, mock_subprocess_popen):
        """Teste la fonction run_trading"""
        from scripts.run_integrated_trader import TradingStrategy, SessionDuration
        
        # Configurer le mock pour simuler un processus
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["Démarrage du trading", ""]
        mock_process.wait.return_value = 0
        mock_subprocess_popen.return_value = mock_process
        
        # Appel avec des paramètres spécifiques
        run_trading(
            strategy=TradingStrategy.ALL, 
            duration=SessionDuration.CONTINUOUS, 
            refresh_symbols=True,
            auto_retrain=True,
            symbols=['BTC-USD', 'ETH-USD'],
            max_symbols=10
        )
        
        # Vérifier que subprocess.Popen a été appelé avec les bons arguments
        mock_subprocess_popen.assert_called_once()
        args, kwargs = mock_subprocess_popen.call_args
        
        # Vérifier que la commande contient les paramètres attendus
        cmd = args[0]
        self.assertIn('run_stock_daytrader_all.py', cmd[1])
        self.assertIn('--strategy', cmd)
        self.assertIn('ALL', cmd)
        self.assertIn('--duration', cmd)
        self.assertIn('continuous', cmd)
        self.assertIn('--refresh-symbols', cmd)
        self.assertIn('--auto-retrain', cmd)
        self.assertIn('--symbols', cmd)
        self.assertIn('BTC-USD,ETH-USD', cmd)
        self.assertIn('--max-symbols', cmd)
        self.assertIn('10', cmd)


class TestIntegratedTraderWithMocks(unittest.TestCase):
    """Tests avec des mocks plus complets pour le script integrated_trader"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Stocker les fonctions originales pour pouvoir les restaurer
        self.import_scripts_run_integrated_trader = __import__('scripts.run_integrated_trader')
        
        # Réinitialiser les variables globales
        import scripts.run_integrated_trader as trader
        trader.running = True
        trader.last_training_time = None
        trader.TRAINING_INTERVAL = 24 * 60 * 60
    
    @patch('scripts.run_integrated_trader.is_market_open')
    @patch('scripts.run_integrated_trader.run_trading')
    @patch('scripts.run_integrated_trader.should_run_training')
    @patch('scripts.run_integrated_trader.run_training')
    @patch('scripts.run_integrated_trader.time.sleep')
    def test_main_function_flow(self, mock_sleep, mock_run_training, 
                               mock_should_run_training, mock_run_trading, 
                               mock_is_market_open):
        """Teste le flux d'exécution principal de la fonction main"""
        # Importer main et parser pour les tests
        from scripts.run_integrated_trader import main
        import scripts.run_integrated_trader as trader
        
        # Configurer les mocks pour simuler différents scénarios
        
        # Scénario 1: Le marché est ouvert, on ne fait que du trading
        mock_is_market_open.return_value = True
        
        # Simuler une exécution de courte durée (2 itérations puis interruption)
        def stop_after_first_iteration(*args, **kwargs):
            trader.running = False
            return None
        
        # La fonction run_trading va arrêter l'exécution
        mock_run_trading.side_effect = stop_after_first_iteration
        
        # Exécuter la fonction main avec des arguments mock
        with patch('sys.argv', ['run_integrated_trader.py', 
                               '--strategy', 'ALL', 
                               '--duration', 'continuous', 
                               '--auto-training']):
            main()
        
        # Vérifier que le script a vérifié l'état du marché
        mock_is_market_open.assert_called_once()
        
        # Vérifier que run_trading a été appelé avec les bons arguments
        mock_run_trading.assert_called_once()
        
        # Vérifier que should_run_training et run_training n'ont pas été appelés
        mock_should_run_training.assert_not_called()
        mock_run_training.assert_not_called()
        
        # Réinitialiser les mocks pour le prochain scénario
        mock_is_market_open.reset_mock()
        mock_run_trading.reset_mock()
        trader.running = True
        
        # Scénario 2: Le marché est fermé, on fait de l'entraînement
        mock_is_market_open.return_value = False
        mock_should_run_training.return_value = True
        
        # La fonction run_training va arrêter l'exécution
        mock_run_training.side_effect = stop_after_first_iteration
        
        # Exécuter la fonction main avec des arguments mock
        with patch('sys.argv', ['run_integrated_trader.py', 
                               '--strategy', 'ALL', 
                               '--duration', 'continuous', 
                               '--auto-training']):
            main()
        
        # Vérifier que le script a vérifié l'état du marché
        mock_is_market_open.assert_called_once()
        
        # Vérifier que should_run_training a été appelé
        mock_should_run_training.assert_called_once()
        
        # Vérifier que run_training a été appelé
        mock_run_training.assert_called_once()
        
        # Vérifier que run_trading n'a pas été appelé
        mock_run_trading.assert_not_called()


if __name__ == '__main__':
    unittest.main()
