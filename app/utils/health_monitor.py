"""
Moniteur de santé MercurioAI

Fournit des mécanismes pour surveiller et rapporter la santé du système
de trading, y compris les performances, les erreurs et les métriques critiques.
"""

import os
import time
import json
import logging
import threading
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthMetrics:
    """Classe pour collecter et agréger les métriques de santé du système"""
    
    def __init__(self):
        # Performance de trading
        self.trading_metrics = {
            "signals_count": 0,
            "executed_trades": 0,
            "successful_trades": 0,
            "error_trades": 0,
            "strategies_success": {},
            "win_rate": 0.0,
            "avg_profit_loss": 0.0,
            "recent_trades": []
        }
        
        # Métriques système
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_space": 0.0,
            "process_uptime": 0,
            "last_error_time": None,
            "error_count_last_hour": 0,
            "api_latency_ms": 0,
            "data_fetches_count": 0,
            "data_fetch_errors": 0
        }
        
        # Métriques de stratégies
        self.strategy_metrics = {}
        
        # Statut global
        self.status = "healthy"  # healthy, degraded, critical
        self.status_reason = None
        
        # Horodatage de la dernière mise à jour
        self.last_updated = datetime.now()
    
    def reset_counters(self):
        """Réinitialise les compteurs (quotidiens/horaires)"""
        self.system_metrics["error_count_last_hour"] = 0
        self.trading_metrics["signals_count"] = 0
    
    def update_trading_metrics(self, metrics_update: Dict[str, Any]):
        """Met à jour les métriques de trading avec les nouvelles données"""
        self.trading_metrics.update(metrics_update)
        
        # Calculer le taux de réussite
        if self.trading_metrics["executed_trades"] > 0:
            self.trading_metrics["win_rate"] = (
                self.trading_metrics["successful_trades"] / 
                self.trading_metrics["executed_trades"] * 100
            )
    
    def update_system_metrics(self):
        """Met à jour les métriques système en temps réel"""
        # CPU et mémoire
        self.system_metrics["cpu_usage"] = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        self.system_metrics["memory_usage"] = memory.percent
        
        # Espace disque
        disk = psutil.disk_usage('/')
        self.system_metrics["disk_space"] = disk.percent
        
        # Temps d'exécution du processus
        process = psutil.Process(os.getpid())
        self.system_metrics["process_uptime"] = time.time() - process.create_time()
    
    def update_strategy_metric(self, strategy_name: str, metric_name: str, value: Any):
        """Met à jour une métrique spécifique pour une stratégie"""
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = {}
        
        self.strategy_metrics[strategy_name][metric_name] = value
    
    def assess_health(self) -> str:
        """Évalue l'état de santé global et retourne le statut"""
        self.last_updated = datetime.now()
        
        # Critères de santé critique
        critical_conditions = [
            self.system_metrics["cpu_usage"] > 90,
            self.system_metrics["memory_usage"] > 90,
            self.system_metrics["disk_space"] > 95,
            self.system_metrics["error_count_last_hour"] > 10,
            self.system_metrics["data_fetch_errors"] > 20
        ]
        
        # Critères de santé dégradée
        degraded_conditions = [
            self.system_metrics["cpu_usage"] > 70,
            self.system_metrics["memory_usage"] > 70,
            self.system_metrics["disk_space"] > 80,
            self.system_metrics["error_count_last_hour"] > 5,
            self.system_metrics["data_fetch_errors"] > 10,
            self.system_metrics["api_latency_ms"] > 1000
        ]
        
        if any(critical_conditions):
            self.status = "critical"
            self.status_reason = "Conditions critiques détectées"
        elif any(degraded_conditions):
            self.status = "degraded"
            self.status_reason = "Performance dégradée détectée"
        else:
            self.status = "healthy"
            self.status_reason = None
        
        return self.status
    
    def log_error(self, error_type: str = "general"):
        """Enregistre une erreur et met à jour les compteurs"""
        self.system_metrics["last_error_time"] = datetime.now().isoformat()
        self.system_metrics["error_count_last_hour"] += 1
        
        # Vérifier si l'erreur affecte la santé
        self.assess_health()
    
    def add_trade_result(self, trade_data: Dict[str, Any]):
        """Ajoute un résultat de trade aux statistiques"""
        is_successful = trade_data.get("profit", 0) > 0
        strategy = trade_data.get("strategy", "unknown")
        
        self.trading_metrics["executed_trades"] += 1
        
        if is_successful:
            self.trading_metrics["successful_trades"] += 1
        
        # Màj statistiques de stratégie
        if strategy not in self.trading_metrics["strategies_success"]:
            self.trading_metrics["strategies_success"][strategy] = {
                "success": 0, "total": 0, "rate": 0.0
            }
        
        self.trading_metrics["strategies_success"][strategy]["total"] += 1
        if is_successful:
            self.trading_metrics["strategies_success"][strategy]["success"] += 1
        
        self.trading_metrics["strategies_success"][strategy]["rate"] = (
            self.trading_metrics["strategies_success"][strategy]["success"] / 
            self.trading_metrics["strategies_success"][strategy]["total"] * 100
        )
        
        # Ajouter aux trades récents (garder 10 au max)
        self.trading_metrics["recent_trades"].append(trade_data)
        if len(self.trading_metrics["recent_trades"]) > 10:
            self.trading_metrics["recent_trades"].pop(0)
        
        # Calculer profit/perte moyenne
        profits = [t.get("profit", 0) for t in self.trading_metrics["recent_trades"]]
        if profits:
            self.trading_metrics["avg_profit_loss"] = sum(profits) / len(profits)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit toutes les métriques en dictionnaire"""
        return {
            "trading_metrics": self.trading_metrics,
            "system_metrics": self.system_metrics,
            "strategy_metrics": self.strategy_metrics,
            "status": self.status,
            "status_reason": self.status_reason,
            "last_updated": self.last_updated.isoformat()
        }
    
    def to_json(self) -> str:
        """Convertit toutes les métriques en JSON"""
        return json.dumps(self.to_dict(), default=str, indent=2)

class HealthMonitor:
    """
    Moniteur de santé pour le système de trading
    
    Surveille en continu la santé du système et génère des alertes
    si des problèmes sont détectés.
    """
    
    def __init__(self, report_directory: str = "reports/health", 
                 check_interval: int = 60):
        """
        Initialise le moniteur de santé
        
        Args:
            report_directory: Répertoire où stocker les rapports
            check_interval: Intervalle entre les vérifications (secondes)
        """
        self.report_dir = Path(report_directory)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.check_interval = check_interval
        self.metrics = HealthMetrics()
        self.alert_handlers = []
        self.periodic_tasks = []
        
        self.running = False
        self.monitor_thread = None
        
        # Informations sur l'environnement
        self.environment_info = {
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "cpu_cores": psutil.cpu_count(),
            "total_memory": psutil.virtual_memory().total
        }
        
        # Timestamp de démarrage
        self.start_time = datetime.now()
    
    def register_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """Enregistre un gestionnaire d'alertes"""
        self.alert_handlers.append(handler)
    
    def register_periodic_task(self, task: Callable, interval: int):
        """
        Enregistre une tâche à exécuter périodiquement
        
        Args:
            task: Fonction à exécuter
            interval: Intervalle en secondes
        """
        self.periodic_tasks.append({
            "task": task,
            "interval": interval,
            "last_run": 0  # Timestamp de dernière exécution
        })
    
    def start(self):
        """Démarre le moniteur de santé dans un thread distinct"""
        if self.running:
            logger.warning("Health monitor is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Health monitor started")
    
    def stop(self):
        """Arrête le moniteur de santé"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitor stopped")
    
    def _monitoring_loop(self):
        """Boucle principale du moniteur"""
        last_hourly_reset = time.time()
        last_report_time = time.time()
        
        while self.running:
            try:
                # Mettre à jour les métriques système
                self.metrics.update_system_metrics()
                
                # Exécuter les tâches périodiques
                current_time = time.time()
                for task_info in self.periodic_tasks:
                    if current_time - task_info["last_run"] >= task_info["interval"]:
                        try:
                            task_info["task"]()
                            task_info["last_run"] = current_time
                        except Exception as e:
                            logger.error(f"Error in periodic task: {e}")
                
                # Vérifier la santé
                health_status = self.metrics.assess_health()
                
                # Générer des alertes si nécessaire
                if health_status in ["degraded", "critical"]:
                    self._trigger_alerts(health_status)
                
                # Réinitialiser les compteurs horaires
                if time.time() - last_hourly_reset > 3600:
                    self.metrics.reset_counters()
                    last_hourly_reset = time.time()
                
                # Générer un rapport toutes les 15 minutes
                if time.time() - last_report_time > 900:  # 15 minutes
                    self.generate_report()
                    last_report_time = time.time()
                
                # Attendre l'intervalle configuré
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(10)  # Attendre un peu en cas d'erreur
    
    def _trigger_alerts(self, severity: str):
        """Déclenche les alertes pour tous les gestionnaires enregistrés"""
        alert_data = {
            "severity": severity,
            "reason": self.metrics.status_reason,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.to_dict()
        }
        
        for handler in self.alert_handlers:
            try:
                handler(severity, alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def log_api_call(self, endpoint: str, latency_ms: float, success: bool):
        """Enregistre une métrique pour un appel API"""
        self.metrics.system_metrics["api_latency_ms"] = latency_ms
        
        # Incrémenter les compteurs de données
        self.metrics.system_metrics["data_fetches_count"] += 1
        if not success:
            self.metrics.system_metrics["data_fetch_errors"] += 1
    
    def log_trade_signal(self, signal_data: Dict[str, Any]):
        """Enregistre un signal de trading émis par le système"""
        self.metrics.trading_metrics["signals_count"] += 1
    
    def log_trade_execution(self, trade_data: Dict[str, Any]):
        """Enregistre l'exécution d'un trade"""
        self.metrics.add_trade_result(trade_data)
    
    def log_error(self, error_type: str = "general"):
        """Enregistre une erreur dans le système"""
        self.metrics.log_error(error_type)
    
    def generate_report(self) -> str:
        """
        Génère un rapport de santé complet
        
        Returns:
            Chemin vers le fichier de rapport généré
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment_info,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "health_status": self.metrics.status,
            "metrics": self.metrics.to_dict()
        }
        
        # Générer un nom de fichier avec timestamp
        filename = f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.report_dir / filename
        
        # Écrire le rapport
        with open(report_path, 'w') as f:
            json.dump(report, f, default=str, indent=2)
        
        logger.info(f"Health report generated: {report_path}")
        return str(report_path)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques actuelles
        
        Returns:
            Dictionnaire des métriques actuelles
        """
        return self.metrics.to_dict()

# Gestionnaire par défaut pour les alertes de console
def console_alert_handler(severity: str, alert_data: Dict[str, Any]):
    """Gestionnaire d'alertes qui affiche les alertes dans la console"""
    logger.warning(f"[{severity.upper()} ALERT] {alert_data['reason']}")
    if severity == "critical":
        logger.error(f"Critical metrics: CPU={alert_data['metrics']['system_metrics']['cpu_usage']}%, "
                   f"Memory={alert_data['metrics']['system_metrics']['memory_usage']}%, "
                   f"Errors={alert_data['metrics']['system_metrics']['error_count_last_hour']}")

# Instance globale du moniteur de santé
health_monitor = HealthMonitor()
health_monitor.register_alert_handler(console_alert_handler)
