#!/usr/bin/env python
"""
MercurioAI - Best Assets Screener

Ce script analyse une large liste d'actions et de cryptomonnaies pour identifier
les meilleurs actifs pour le trading à moyen terme, en utilisant les stratégies
disponibles dans MercurioAI. Il génère un rapport classant les actifs selon
un score composite basé sur plusieurs indicateurs techniques et prédictions.

Exemple d'utilisation:
    python scripts/best_assets_screener.py --top_stocks 50 --top_crypto 100
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Assurez-vous que le script peut importer les modules MercurioAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assurez-vous que les répertoires nécessaires existent
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/best_assets", exist_ok=True)

# Importez les modules MercurioAI
from app.services.market_data import MarketDataService
from app.strategies.base import BaseStrategy
from app.strategies.moving_average import MovingAverageStrategy
from app.strategies.lstm_predictor import LSTMPredictorStrategy
from app.strategies.msi_strategy import MultiSourceIntelligenceStrategy
from app.strategies.transformer_strategy import TransformerStrategy

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/best_assets_screener.log")
    ]
)
logger = logging.getLogger(__name__)

# Listes par défaut d'actifs à évaluer
DEFAULT_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NVDA", "JPM", "V", "PG", 
    "UNH", "HD", "BAC", "ADBE", "CRM", "PFE", "NFLX", "AVGO", "CSCO", "VZ", 
    "COST", "ABT", "MRK", "DIS", "INTC", "WMT", "CMCSA", "XOM", "T", "ORCL", 
    "NKE", "QCOM", "AMD", "TXN", "IBM", "GS", "MMM", "CVX", "HON", "AMGN", 
    "MCD", "LIN", "LOW", "UNP", "NEE", "RTX", "SBUX", "MDT", "LLY", "BA",
    "GE", "GM", "F", "PYPL", "SQ", "SHOP", "ZM", "ABNB", "UBER", "COIN", 
    "PLTR", "SNOW", "DASH", "RBLX", "U", "NET", "TDOC", "ETSY", "PINS", "PTON"
]

DEFAULT_CRYPTO = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", "XRP-USD", "DOGE-USD", 
    "AVAX-USD", "LUNA-USD", "SHIB-USD", "MATIC-USD", "LINK-USD", "UNI-USD", "ATOM-USD", 
    "LTC-USD", "ALGO-USD", "BCH-USD", "XLM-USD", "VET-USD", "MANA-USD", "FTM-USD", 
    "SAND-USD", "HBAR-USD", "EGLD-USD", "EOS-USD", "XTZ-USD", "ONE-USD", "AXS-USD", 
    "GRT-USD", "ENJ-USD", "ZEC-USD", "SUSHI-USD", "CHZ-USD", "HOT-USD", "BAT-USD", 
    "COMP-USD", "MKR-USD", "YFI-USD", "REN-USD", "BAND-USD", "OGN-USD", "SKL-USD",
    "CRV-USD", "OMG-USD", "ANKR-USD", "1INCH-USD", "LRC-USD", "STORJ-USD", "NU-USD"
]

class AssetEvaluator:
    """Classe pour évaluer les actifs en utilisant différentes stratégies"""

    def __init__(self, lookback_days: int = 120, prediction_days: int = 15):
        """Initialise l'évaluateur d'actifs
        
        Args:
            lookback_days: Nombre de jours d'historique à analyser
            prediction_days: Nombre de jours à prédire pour le score
        """
        self.market_data = MarketDataService()
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.start_date = datetime.now() - timedelta(days=lookback_days)
        self.end_date = datetime.now()
        
        # Initialiser les stratégies que nous utiliserons pour l'évaluation
        self.strategies = {
            "MovingAverage": MovingAverageStrategy(short_window=10, long_window=30, use_ml=True),
            "LSTM": LSTMPredictorStrategy(lookback_periods=20, prediction_periods=5),
            "MSI": MultiSourceIntelligenceStrategy(symbol="BTC-USD"),
            "Transformer": TransformerStrategy()
        }
        
        self.results = {}
        
    async def evaluate_asset(self, symbol: str) -> Dict[str, Any]:
        """Évalue un actif en utilisant toutes les stratégies disponibles
        
        Args:
            symbol: Le symbole de l'actif à évaluer
            
        Returns:
            Dict avec les résultats de l'évaluation
        """
        try:
            logger.info(f"Évaluation de {symbol}...")
            
            # Récupérer les données historiques
            data = await self.market_data.get_historical_data(
                symbol=symbol, 
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if data is None or len(data) < 20:
                logger.warning(f"Données insuffisantes pour {symbol}, ignoré")
                return {
                    "symbol": symbol,
                    "score": 0,
                    "valid": False,
                    "error": "Données insuffisantes"
                }
            
            # Stocker les résultats pour chaque stratégie
            strategy_results = {}
            signals = {}
            confidences = {}
            returns = {}
            
            # Évaluer avec chaque stratégie
            for name, strategy in self.strategies.items():
                try:
                    # Vérification si le modèle est prêt (pour les stratégies basées sur ML)
                    is_model_ready = True
                    if hasattr(strategy, 'model') and (strategy.model is None or not hasattr(strategy.model, 'predict')):
                        logger.warning(f"Le modèle pour la stratégie {name} n'est pas encore entraîné. Utilisation du signal par défaut.")
                        is_model_ready = False
                    
                    # Prétraiter les données pour cette stratégie
                    processed_data = await strategy.preprocess_data(data)
                    
                    if is_model_ready:
                        # Obtenir le signal et la confiance
                        signal, confidence = await strategy.predict(processed_data)
                        
                        # Convertir TradeAction enum en chaîne
                        if hasattr(signal, 'value'):
                            signal_str = signal.value.upper()
                        else:
                            signal_str = str(signal).upper()
                        
                        signals[name] = signal_str
                        confidences[name] = confidence
                        
                        # Simuler un backtest pour cette stratégie
                        backtest_result = await strategy.backtest(processed_data, initial_capital=10000)
                        
                        # Calculer le rendement
                        if backtest_result and "final_capital" in backtest_result:
                            returns[name] = (backtest_result["final_capital"] / 10000) - 1
                        else:
                            returns[name] = 0
                    else:
                        # Stratégie non prête, utiliser valeurs par défaut
                        signal_str = "HOLD"
                        signals[name] = signal_str
                        confidences[name] = 0.5
                        returns[name] = 0
                        
                    strategy_results[name] = {
                        "signal": signal_str if 'signal_str' in locals() else "HOLD",
                        "confidence": confidences.get(name, 0),
                        "return": returns.get(name, 0),
                        "model_ready": is_model_ready
                    }
                        
                except Exception as e:
                    logger.error(f"Erreur lors de l'évaluation de {symbol} avec {name}: {e}")
                    signals[name] = "HOLD"
                    confidences[name] = 0
                    returns[name] = 0
                    strategy_results[name] = {
                        "signal": "HOLD",
                        "confidence": 0,
                        "return": 0,
                        "error": str(e),
                        "model_ready": False
                    }
            
            # Calculer des métriques techniques supplémentaires
            technical_metrics = self._calculate_technical_metrics(data)
            
            # Calculer le score composite
            composite_score = self._calculate_composite_score(signals, confidences, returns, technical_metrics)
            
            # Résultat final
            result = {
                "symbol": symbol,
                "score": composite_score,
                "valid": True,
                "strategy_results": strategy_results,
                "technical_metrics": technical_metrics,
                "price_current": data["close"].iloc[-1] if not data.empty else 0,
                "volume_avg": data["volume"].mean() if not data.empty and "volume" in data.columns else 0,
                "volatility": data["close"].pct_change().std() * np.sqrt(252) if not data.empty else 0  # Annualisée
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de {symbol}: {e}")
            return {
                "symbol": symbol,
                "score": 0,
                "valid": False,
                "error": str(e)
            }
    
    def _calculate_technical_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calcule des métriques techniques supplémentaires sur les données
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Dict avec les métriques techniques
        """
        metrics = {}
        
        if data is None or data.empty:
            return metrics
        
        # Tendance récente (2 semaines vs 1 mois)
        if len(data) >= 30:
            metrics["recent_trend"] = data["close"].iloc[-10:].mean() / data["close"].iloc[-30:-10].mean() - 1
        else:
            metrics["recent_trend"] = 0
        
        # RSI (14 jours)
        try:
            delta = data["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            metrics["rsi"] = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) and rs.iloc[-1] != np.inf else 50
        except:
            metrics["rsi"] = 50
        
        # Volatilité relative (comparée à la volatilité historique)
        try:
            recent_vol = data["close"].iloc[-10:].pct_change().std()
            historical_vol = data["close"].pct_change().std()
            metrics["relative_volatility"] = recent_vol / historical_vol if historical_vol > 0 else 1
        except:
            metrics["relative_volatility"] = 1
        
        # Volume trend (volume récent vs volume historique)
        if "volume" in data.columns:
            try:
                recent_volume = data["volume"].iloc[-5:].mean()
                historical_volume = data["volume"].mean()
                metrics["volume_trend"] = recent_volume / historical_volume if historical_volume > 0 else 1
            except:
                metrics["volume_trend"] = 1
        else:
            metrics["volume_trend"] = 1
            
        return metrics
    
    def _calculate_composite_score(self, signals: Dict[str, str], confidences: Dict[str, float], 
                                  returns: Dict[str, float], metrics: Dict[str, float]) -> float:
        """Calcule un score composite basé sur tous les résultats
        
        Args:
            signals: Dict de signaux par stratégie
            confidences: Dict de confiances par stratégie
            returns: Dict de rendements par stratégie
            metrics: Dict de métriques techniques
            
        Returns:
            Score composite entre 0 et 100
        """
        score = 0
        
        # 1. Score basé sur les signaux (40%)
        signal_score = 0
        for name, signal in signals.items():
            if signal == "BUY":
                signal_score += 1 * confidences.get(name, 0.5)
            elif signal == "SELL":
                signal_score -= 0.5 * confidences.get(name, 0.5)
        
        # Normaliser de -1 à 1, puis de 0 à 40
        max_signal_score = len(signals)  # Score maximum possible
        normalized_signal_score = (signal_score / max_signal_score if max_signal_score > 0 else 0) * 40
        score += max(0, normalized_signal_score)
        
        # 2. Score basé sur les rendements (30%)
        return_score = sum(returns.values()) / len(returns) if returns else 0
        # Normaliser de -0.1 à 0.1, puis de 0 à 30
        normalized_return_score = min(max(return_score * 10, -1), 1) * 15 + 15
        score += normalized_return_score
        
        # 3. Score basé sur les métriques techniques (30%)
        tech_score = 0
        
        # Tendance récente (positif = bon)
        tech_score += min(max(metrics.get("recent_trend", 0) * 10, -1), 1) * 10
        
        # RSI (entre 40 et 60 = neutre, < 30 = survendu, > 70 = suracheté)
        rsi = metrics.get("rsi", 50)
        if rsi < 30:  # Survendu = opportunité d'achat
            tech_score += (30 - rsi) / 30 * 5
        elif rsi > 70:  # Suracheté = moins intéressant
            tech_score -= (rsi - 70) / 30 * 5
            
        # Volume trend (volume en hausse = bon signe)
        vol_trend = metrics.get("volume_trend", 1)
        if vol_trend > 1:
            tech_score += min((vol_trend - 1) * 5, 5)
        
        # Normaliser de -10 à 20, puis de 0 à 30
        normalized_tech_score = (tech_score + 10) / 30 * 30
        score += max(0, min(normalized_tech_score, 30))
        
        return max(0, min(score, 100))


class AssetScreener:
    """Classe principale qui gère le screening des actifs"""
    
    def __init__(self, top_stocks: int = 50, top_crypto: int = 100, 
                 lookback_days: int = 30, custom_stocks: List[str] = None, 
                 custom_crypto: List[str] = None):
        """Initialise le screener d'actifs
        
        Args:
            top_stocks: Nombre d'actions à inclure dans le rapport final
            top_crypto: Nombre de cryptomonnaies à inclure dans le rapport final
            lookback_days: Nombre de jours d'historique à analyser
            custom_stocks: Liste personnalisée d'actions à évaluer (si None, utilise DEFAULT_STOCKS)
            custom_crypto: Liste personnalisée de cryptomonnaies à évaluer (si None, utilise DEFAULT_CRYPTO)
        """
        self.top_stocks = top_stocks
        self.top_crypto = top_crypto
        self.lookback_days = lookback_days
        
        self.stocks = custom_stocks if custom_stocks else DEFAULT_STOCKS
        self.crypto = custom_crypto if custom_crypto else DEFAULT_CRYPTO
        
        self.evaluator = AssetEvaluator(lookback_days=lookback_days)
        
        self.stock_results = []
        self.crypto_results = []
        
        self.report_path = f"reports/best_assets/{datetime.now().strftime('%Y-%m-%d')}"
        os.makedirs(self.report_path, exist_ok=True)
        
    async def run_screening(self):
        """Exécute le screening sur tous les actifs"""
        logger.info(f"Démarrage du screening des actifs...")
        logger.info(f"Actions à évaluer: {len(self.stocks)}")
        logger.info(f"Cryptomonnaies à évaluer: {len(self.crypto)}")
        
        # Évaluer les actions
        logger.info("Démarrage de l'évaluation des actions...")
        for symbol in self.stocks:
            result = await self.evaluator.evaluate_asset(symbol)
            if result["valid"]:
                self.stock_results.append(result)
        
        # Évaluer les cryptomonnaies
        logger.info("Démarrage de l'évaluation des cryptomonnaies...")
        for symbol in self.crypto:
            result = await self.evaluator.evaluate_asset(symbol)
            if result["valid"]:
                self.crypto_results.append(result)
        
        # Trier les résultats par score
        self.stock_results.sort(key=lambda x: x["score"], reverse=True)
        self.crypto_results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Screening terminé. {len(self.stock_results)} actions et {len(self.crypto_results)} cryptomonnaies évaluées.")
        
    def generate_report(self):
        """Génère un rapport des meilleurs actifs"""
        logger.info("Génération du rapport...")
        
        # Limiter le nombre d'actifs selon les paramètres
        top_stocks = self.stock_results[:self.top_stocks]
        top_crypto = self.crypto_results[:self.top_crypto]
        
        # 1. Générer des DataFrames pour l'analyse
        stock_df = pd.DataFrame([
            {
                "Symbol": r["symbol"],
                "Score": r["score"],
                "Price": r["price_current"],
                "Volatility": r["volatility"],
                "RSI": r["technical_metrics"].get("rsi", 50),
                "Recent Trend": r["technical_metrics"].get("recent_trend", 0) * 100,
                "Volume Trend": r["technical_metrics"].get("volume_trend", 1),
                "MovingAverage Signal": r["strategy_results"].get("MovingAverage", {}).get("signal", "HOLD"),
                "LSTM Signal": r["strategy_results"].get("LSTM", {}).get("signal", "HOLD"),
                "MSI Signal": r["strategy_results"].get("MSI", {}).get("signal", "HOLD"),
                "Transformer Signal": r["strategy_results"].get("Transformer", {}).get("signal", "HOLD")
            } for r in top_stocks
        ])
        
        crypto_df = pd.DataFrame([
            {
                "Symbol": r["symbol"],
                "Score": r["score"],
                "Price": r["price_current"],
                "Volatility": r["volatility"],
                "RSI": r["technical_metrics"].get("rsi", 50),
                "Recent Trend": r["technical_metrics"].get("recent_trend", 0) * 100,
                "Volume Trend": r["technical_metrics"].get("volume_trend", 1),
                "MovingAverage Signal": r["strategy_results"].get("MovingAverage", {}).get("signal", "HOLD"),
                "LSTM Signal": r["strategy_results"].get("LSTM", {}).get("signal", "HOLD"),
                "MSI Signal": r["strategy_results"].get("MSI", {}).get("signal", "HOLD"),
                "Transformer Signal": r["strategy_results"].get("Transformer", {}).get("signal", "HOLD")
            } for r in top_crypto
        ])
        
        # 2. Sauvegarder les rapports CSV
        stock_df.to_csv(f"{self.report_path}/top_stocks.csv", index=False)
        crypto_df.to_csv(f"{self.report_path}/top_crypto.csv", index=False)
        
        # 3. Générer un rapport HTML
        with open(f"{self.report_path}/report.html", "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>MercurioAI - Asset Screening Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .score-high {{ background-color: #dff0d8; }}
        .score-medium {{ background-color: #fcf8e3; }}
        .score-low {{ background-color: #f2dede; }}
        .signal-buy {{ color: green; font-weight: bold; }}
        .signal-sell {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>MercurioAI - Asset Screening Report</h1>
    <p>Date: {datetime.now().strftime('%Y-%m-%d')}</p>
    <p>Lookback period: {self.lookback_days} days</p>
    
    <h2>Top {len(top_stocks)} Stocks for Medium-Term Trading</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Symbol</th>
            <th>Score</th>
            <th>Price</th>
            <th>RSI</th>
            <th>Recent Trend (%)</th>
            <th>Volatility</th>
            <th>Moving Avg</th>
            <th>LSTM</th>
            <th>MSI</th>
            <th>Transformer</th>
        </tr>
""")
            
            # Add stock rows
            for i, row in stock_df.iterrows():
                score_class = "score-high" if row["Score"] >= 70 else "score-medium" if row["Score"] >= 50 else "score-low"
                
                # Sécuriser les signaux pour éviter les erreurs
                ma_signal = str(row['MovingAverage Signal']).lower() if pd.notna(row['MovingAverage Signal']) else "hold"
                lstm_signal = str(row['LSTM Signal']).lower() if pd.notna(row['LSTM Signal']) else "hold"
                msi_signal = str(row['MSI Signal']).lower() if pd.notna(row['MSI Signal']) else "hold"
                transformer_signal = str(row['Transformer Signal']).lower() if pd.notna(row['Transformer Signal']) else "hold"
                
                # Limiter les signaux valides à buy, sell, hold
                ma_signal = ma_signal if ma_signal in ["buy", "sell", "hold"] else "hold"
                lstm_signal = lstm_signal if lstm_signal in ["buy", "sell", "hold"] else "hold"
                msi_signal = msi_signal if msi_signal in ["buy", "sell", "hold"] else "hold"
                transformer_signal = transformer_signal if transformer_signal in ["buy", "sell", "hold"] else "hold"
                
                f.write(f"""        <tr class="{score_class}">
            <td>{i+1}</td>
            <td>{row['Symbol']}</td>
            <td>{row['Score']:.1f}</td>
            <td>${row['Price']:.2f}</td>
            <td>{row['RSI']:.1f}</td>
            <td>{row['Recent Trend']:.1f}%</td>
            <td>{row['Volatility']*100:.1f}%</td>
            <td class="signal-{ma_signal}">{row['MovingAverage Signal']}</td>
            <td class="signal-{lstm_signal}">{row['LSTM Signal']}</td>
            <td class="signal-{msi_signal}">{row['MSI Signal']}</td>
            <td class="signal-{transformer_signal}">{row['Transformer Signal']}</td>
        </tr>
""")
                
            f.write(f"""    </table>
    
    <h2>Top {len(top_crypto)} Cryptocurrencies for Medium-Term Trading</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Symbol</th>
            <th>Score</th>
            <th>Price</th>
            <th>RSI</th>
            <th>Recent Trend (%)</th>
            <th>Volatility</th>
            <th>Moving Avg</th>
            <th>LSTM</th>
            <th>MSI</th>
            <th>Transformer</th>
        </tr>
""")
            
            # Add crypto rows
            for i, row in crypto_df.iterrows():
                score_class = "score-high" if row["Score"] >= 70 else "score-medium" if row["Score"] >= 50 else "score-low"
                
                # Sécuriser les signaux pour éviter les erreurs
                ma_signal = str(row['MovingAverage Signal']).lower() if pd.notna(row['MovingAverage Signal']) else "hold"
                lstm_signal = str(row['LSTM Signal']).lower() if pd.notna(row['LSTM Signal']) else "hold"
                msi_signal = str(row['MSI Signal']).lower() if pd.notna(row['MSI Signal']) else "hold"
                transformer_signal = str(row['Transformer Signal']).lower() if pd.notna(row['Transformer Signal']) else "hold"
                
                # Limiter les signaux valides à buy, sell, hold
                ma_signal = ma_signal if ma_signal in ["buy", "sell", "hold"] else "hold"
                lstm_signal = lstm_signal if lstm_signal in ["buy", "sell", "hold"] else "hold"
                msi_signal = msi_signal if msi_signal in ["buy", "sell", "hold"] else "hold"
                transformer_signal = transformer_signal if transformer_signal in ["buy", "sell", "hold"] else "hold"
                
                f.write(f"""        <tr class="{score_class}">
            <td>{i+1}</td>
            <td>{row['Symbol']}</td>
            <td>{row['Score']:.1f}</td>
            <td>${row['Price']:.2f}</td>
            <td>{row['RSI']:.1f}</td>
            <td>{row['Recent Trend']:.1f}%</td>
            <td>{row['Volatility']*100:.1f}%</td>
            <td class="signal-{ma_signal}">{row['MovingAverage Signal']}</td>
            <td class="signal-{lstm_signal}">{row['LSTM Signal']}</td>
            <td class="signal-{msi_signal}">{row['MSI Signal']}</td>
            <td class="signal-{transformer_signal}">{row['Transformer Signal']}</td>
        </tr>
""")
                
            f.write(f"""    </table>
    
    <h3>Report Summary</h3>
    <p>This report was generated using MercurioAI's multi-strategy evaluation system which combines technical analysis, machine learning predictions, and backtesting results.</p>
    <p>Score interpretation:</p>
    <ul>
        <li><strong>70-100</strong>: Strong buy signals across multiple strategies</li>
        <li><strong>50-70</strong>: Moderate buy signals, worth monitoring</li>
        <li><strong>0-50</strong>: Weak or negative signals, not recommended for medium-term trading</li>
    </ul>
</body>
</html>
""")
            
        # 4. Générer des graphiques
        try:
            # Graphique des scores des meilleures actions
            plt.figure(figsize=(12, 8))
            plt.title("Top Stocks by Score")
            sns.barplot(x="Symbol", y="Score", data=stock_df.head(20))
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"{self.report_path}/top_stocks_chart.png")
            plt.close()
            
            # Graphique des scores des meilleures cryptomonnaies
            plt.figure(figsize=(12, 8))
            plt.title("Top Cryptocurrencies by Score")
            sns.barplot(x="Symbol", y="Score", data=crypto_df.head(20))
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"{self.report_path}/top_crypto_chart.png")
            plt.close()
        except Exception as e:
            logger.error(f"Erreur lors de la génération des graphiques: {e}")
        
        # 5. Générer un rapport texte pour la console
        logger.info("\n" + "=" * 80)
        logger.info(f"RAPPORT DE SCREENING - {datetime.now().strftime('%Y-%m-%d')}")
        logger.info("=" * 80)
        
        logger.info("\nTOP 10 DES ACTIONS:")
        logger.info(tabulate(stock_df.head(10)[["Symbol", "Score", "Price", "RSI", "Recent Trend", "MovingAverage Signal", "LSTM Signal"]], 
                             headers="keys", tablefmt="pretty", floatfmt=".2f"))
        
        logger.info("\nTOP 10 DES CRYPTOMONNAIES:")
        logger.info(tabulate(crypto_df.head(10)[["Symbol", "Score", "Price", "RSI", "Recent Trend", "MovingAverage Signal", "LSTM Signal"]], 
                             headers="keys", tablefmt="pretty", floatfmt=".2f"))
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Rapport sauvegardé dans {self.report_path}/")
        logger.info(f"Fichiers générés: report.html, top_stocks.csv, top_crypto.csv, top_stocks_chart.png, top_crypto_chart.png")
        
        return {
            "report_path": self.report_path,
            "top_stocks": top_stocks,
            "top_crypto": top_crypto
        }


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="MercurioAI - Best Assets Screener")
    
    parser.add_argument("--top_stocks", type=int, default=50,
                       help="Nombre d'actions à inclure dans le rapport final (défaut: 50)")
    parser.add_argument("--top_crypto", type=int, default=100,
                       help="Nombre de cryptomonnaies à inclure dans le rapport final (défaut: 100)")
    parser.add_argument("--lookback", type=int, default=120,
                       help="Nombre de jours d'historique à analyser (défaut: 120)")
    parser.add_argument("--stocks", type=str, default="",
                       help="Liste d'actions personnalisée séparée par des virgules (si vide, utilise la liste par défaut)")
    parser.add_argument("--crypto", type=str, default="",
                       help="Liste de cryptomonnaies personnalisée séparée par des virgules (si vide, utilise la liste par défaut)")
    
    args = parser.parse_args()
    
    # Convertir les listes personnalisées si fournies
    custom_stocks = args.stocks.split(",") if args.stocks.strip() else None
    custom_crypto = args.crypto.split(",") if args.crypto.strip() else None
    
    logger.info("MercurioAI - Best Assets Screener")
    logger.info(f"Paramètres: top_stocks={args.top_stocks}, top_crypto={args.top_crypto}, lookback={args.lookback}")
    
    try:
        # Initialiser et exécuter le screener
        screener = AssetScreener(
            top_stocks=args.top_stocks,
            top_crypto=args.top_crypto,
            lookback_days=args.lookback,
            custom_stocks=custom_stocks,
            custom_crypto=custom_crypto
        )
        
        await screener.run_screening()
        result = screener.generate_report()
        
        logger.info("Screening terminé avec succès!")
        logger.info(f"Rapport disponible dans: {result['report_path']}/report.html")
        
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur critique non gérée: {e}")
        sys.exit(1)
