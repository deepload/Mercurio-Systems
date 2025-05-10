#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graceful Exit Handler
---------------------
Utilitaire pour gérer proprement l'arrêt des scripts Mercurio AI.
Permet d'arrêter tous les threads et processus en cours lors d'un Ctrl+C.
"""

import os
import sys
import signal
import threading
import logging
import atexit
import time
import subprocess
import threading
from threading import Timer
from typing import List, Callable, Optional, Dict, Any

logger = logging.getLogger("graceful_exit")

class GracefulExit:
    """
    Gestionnaire pour un arrêt propre des scripts de trading
    """
    
    def __init__(self, exit_timeout: int = 5, liquidation_timeout: int = 45):
        """
        Initialiser le gestionnaire d'arrêt propre
        
        Args:
            exit_timeout: Temps maximum (en secondes) à attendre pour un arrêt propre
            liquidation_timeout: Temps d'attente (en secondes) pour la décision de liquidation
        """
        self.running = True
        self.exit_timeout = exit_timeout
        self.liquidation_timeout = liquidation_timeout
        self.threads: List[threading.Thread] = []
        self.cleanup_callbacks: List[Callable] = []
        self.liquidation_callback: Optional[Callable] = None
        self.position_preservation_flag = False
        self.register_signal_handlers()
        
    def register_signal_handlers(self):
        """Enregistrer les gestionnaires de signaux pour gérer Ctrl+C proprement"""
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        atexit.register(self.cleanup)
        
    def handle_signal(self, sig, frame):
        """Gérer les signaux d'interruption (Ctrl+C)"""
        if not self.running:
            # Si déjà en train de s'arrêter, forcer l'arrêt immédiat
            logger.warning("Forçage de l'arrêt immédiat...")
            os._exit(1)
            
        logger.info("Signal d'arrêt reçu (Ctrl+C), arrêt en cours... Patientez s'il vous plaît.")
        self.running = False
        
        # Lancer le processus de nettoyage
        self.cleanup()
        
        # Donner un peu de temps pour le nettoyage avant de quitter
        timer = threading.Timer(self.exit_timeout, self.force_exit)
        timer.daemon = True
        timer.start()
    
    def force_exit(self):
        """Forcer l'arrêt si le nettoyage prend trop de temps"""
        logger.warning(f"Le nettoyage a pris plus de {self.exit_timeout} secondes, forçage de l'arrêt...")
        os._exit(1)
    
    def cleanup(self):
        """Exécuter toutes les fonctions de nettoyage enregistrées"""
        if not hasattr(self, 'cleanup_called'):
            self.cleanup_called = True
            logger.info("Exécution des opérations de nettoyage...")
            
            # Gérer la liquidation des positions si configuré
            if self.liquidation_callback:
                logger.info(f"\nATTENTION: Voulez-vous liquider toutes les positions ouvertes?")
                logger.info(f"Appuyez sur 'K' pour CONSERVER vos positions")
                logger.info(f"Appuyez sur 'L' pour LIQUIDER immédiatement vos positions")
                logger.info(f"Ou attendez {self.liquidation_timeout} secondes pour les LIQUIDER automatiquement\n")
                
                # Initialiser le drapeau
                self.position_preservation_flag = False
                
                # Créer et démarrer le thread d'attente d'input clavier
                keyboard_thread = threading.Thread(target=self.wait_for_keep_key, name="KeyboardWaitThread")
                keyboard_thread.daemon = True
                keyboard_thread.start()
                
                # Définir le timer pour la liquidation automatique
                timer = Timer(self.liquidation_timeout, self.liquidate_positions)
                timer.daemon = True
                timer.start()
                
                try:
                    # Attendre la fin du thread ou du timer
                    keyboard_thread.join(self.liquidation_timeout + 0.5)  # Ajouter un petit délai
                    if timer.is_alive():
                        timer.cancel()  # Annuler le timer si le thread s'est terminé
                except Exception as e:
                    logger.error(f"Erreur dans le processus de décision de liquidation: {e}")
            
            # Exécuter les callbacks de nettoyage
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Erreur lors du nettoyage: {e}")
            
            # Attendre la fin de tous les threads non-daemon
            for thread in self.threads:
                if thread.is_alive() and not thread.daemon:
                    logger.info(f"Attente de la fin du thread: {thread.name}")
                    thread.join(1.0)  # Attendre 1 seconde max par thread
                    
            logger.info("Nettoyage terminé")
    
    def wait_for_keep_key(self):
        """Attendre que l'utilisateur appuie sur une touche pour décider du sort des positions"""
        try:
            import msvcrt  # Module spécifique à Windows
            logger.info("Attente de l'appui sur 'K' pour CONSERVER les positions ou 'L' pour les LIQUIDER immédiatement...")
            start_time = time.time()
            
            # Initialiser un événement de liquidation immédiate
            self.immediate_liquidation = False
            
            while time.time() - start_time < self.liquidation_timeout:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').upper()
                    if key == 'K':
                        logger.info("Touche 'K' appuyée: les positions seront CONSERVÉES")
                        self.position_preservation_flag = True
                        break
                    elif key == 'L':
                        logger.info("Touche 'L' appuyée: les positions seront LIQUIDÉES immédiatement")
                        self.immediate_liquidation = True
                        self.position_preservation_flag = False
                        break
                time.sleep(0.1)
                
        except (ImportError, Exception) as e:
            logger.error(f"Erreur lors de l'attente de l'appui sur une touche: {e}")
    
    def liquidate_positions(self):
        """Liquider toutes les positions si l'utilisateur n'a pas choisi de les conserver"""
        if self.position_preservation_flag:
            logger.info("Positions CONSERVÉES comme demandé")
            return
        
        # Message approprié selon si c'est une liquidation immédiate ou par timeout
        if hasattr(self, 'immediate_liquidation') and self.immediate_liquidation:
            logger.info("\nLIQUIDATION IMMÉDIATE de toutes les positions...")
        else:
            logger.info("\nDélai écoulé: LIQUIDATION de toutes les positions...")
        
        if self.liquidation_callback:
            try:
                self.liquidation_callback()
                logger.info("Liquidation des positions terminée")
            except Exception as e:
                logger.error(f"Erreur lors de la liquidation des positions: {e}")
    
    def is_running(self) -> bool:
        """Vérifier si le programme doit continuer à s'exécuter"""
        return self.running
    
    def register_thread(self, thread: threading.Thread):
        """Enregistrer un thread pour le suivi et la terminaison propre"""
        self.threads.append(thread)
        
    def register_cleanup(self, callback: Callable):
        """Enregistrer une fonction de nettoyage à exécuter lors de l'arrêt"""
        self.cleanup_callbacks.append(callback)
    
    def register_liquidation_handler(self, callback: Callable):
        """Enregistrer une fonction pour gérer la liquidation des positions"""
        self.liquidation_callback = callback

# Instance singleton pour utilisation facile dans tout le code
graceful_exit = GracefulExit()

# Fonction utilitaire pour vérifier si le programme doit continuer à s'exécuter
def is_running() -> bool:
    """Vérifier si le programme doit continuer à s'exécuter"""
    return graceful_exit.is_running()

# Fonction utilitaire pour enregistrer un thread
def register_thread(thread: threading.Thread):
    """Enregistrer un thread pour la gestion d'arrêt propre"""
    graceful_exit.register_thread(thread)

# Fonction utilitaire pour enregistrer une fonction de nettoyage
def register_cleanup(callback: Callable):
    """Enregistrer une fonction de nettoyage à exécuter lors de l'arrêt"""
    graceful_exit.register_cleanup(callback)

# Fonction utilitaire pour enregistrer un gestionnaire de liquidation
def register_liquidation_handler(callback: Callable):
    """Enregistrer une fonction pour gérer la liquidation des positions"""
    graceful_exit.register_liquidation_handler(callback)

if __name__ == "__main__":
    # Test simple du mécanisme d'arrêt propre
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Fonction de test pour la liquidation
    def test_liquidation():
        logger.info("Simulation de liquidation des positions...")
        time.sleep(1)
        logger.info("Toutes les positions ont été liquidées!")
    
    def test_thread_function():
        """Fonction de test pour un thread qui tourne en continu"""
        count = 0
        while is_running():
            logger.info(f"Thread en cours d'exécution... {count}")
            count += 1
            time.sleep(1)
        logger.info("Thread arrêté proprement")
    
    # Fonction de nettoyage de test
    def test_cleanup():
        logger.info("Fonction de nettoyage exécutée!")
        time.sleep(0.5)  # Simuler une opération de nettoyage
    
    # Enregistrer la fonction de nettoyage
    register_cleanup(test_cleanup)
    
    # Enregistrer la fonction de liquidation
    register_liquidation_handler(test_liquidation)
    
    # Créer et démarrer un thread de test
    test_thread = threading.Thread(target=test_thread_function, name="TestThread")
    test_thread.daemon = False  # Thread non-daemon
    register_thread(test_thread)
    test_thread.start()
    
    logger.info("Programme principal en cours d'exécution. Appuyez sur Ctrl+C pour arrêter.")
    
    # Boucle principale
    try:
        while is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    
    logger.info("Programme terminé proprement")
