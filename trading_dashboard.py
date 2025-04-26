#!/usr/bin/env python
"""
MercurioAI Trading Dashboard

Tableau de bord web pour surveiller en temps réel les performances de trading
et l'état des stratégies dans MercurioAI.
"""

import os
import json
import logging
import asyncio
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from flask import Flask

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Dossier pour les rapports et les données
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")
REFRESH_INTERVAL = 5  # secondes

# Initialisation de l'application Dash
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = "MercurioAI - Tableau de bord de trading"

# Mise en page du tableau de bord
app.layout = html.Div([
    html.Div([
        html.H1("MercurioAI - Tableau de bord de trading", className="dashboard-title"),
        html.Div(id="last-update", className="last-update"),
    ], className="header"),
    
    html.Div([
        html.Div([
            html.H3("Performance du portefeuille"),
            dcc.Graph(id="portfolio-chart", className="chart"),
            html.Div(id="portfolio-stats", className="stats-box"),
        ], className="panel"),
        
        html.Div([
            html.H3("Performance des stratégies"),
            dcc.Graph(id="strategy-performance", className="chart"),
            html.Div(id="strategy-weights", className="stats-box"),
        ], className="panel"),
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("Positions actuelles"),
            html.Div(id="positions-table", className="data-table"),
        ], className="panel"),
        
        html.Div([
            html.H3("Signaux récents"),
            html.Div(id="signals-table", className="data-table"),
        ], className="panel"),
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("État du marché"),
            html.Div(id="market-state", className="stats-box"),
        ], className="panel"),
        
        html.Div([
            html.H3("Anomalies détectées"),
            html.Div(id="anomalies", className="alerts-box"),
        ], className="panel"),
    ], className="row"),
    
    dcc.Interval(
        id="interval-component",
        interval=REFRESH_INTERVAL * 1000,  # en millisecondes
        n_intervals=0
    ),
    
    # CSS
    html.Style("""
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .dashboard-title {
            margin: 0;
            font-size: 24px;
        }
        .last-update {
            font-size: 14px;
        }
        .row {
            display: flex;
            margin: 10px;
            gap: 10px;
        }
        .panel {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            flex: 1;
        }
        .chart {
            height: 300px;
        }
        .stats-box {
            margin-top: 15px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .data-table {
            margin-top: 15px;
            overflow-x: auto;
        }
        .alerts-box {
            margin-top: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
            text-align: left;
        }
        th {
            background-color: #f0f0f0;
        }
        .positive { color: green; }
        .negative { color: red; }
    """)
], className="dashboard")

def get_latest_performance_report() -> Dict[str, Any]:
    """Récupère le dernier rapport de performance généré"""
    try:
        report_files = list(REPORTS_DIR.glob("performance_*.json"))
        if not report_files:
            return {}
            
        latest_report_file = max(report_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_report_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du rapport de performance: {e}")
        return {}

def get_performance_history() -> pd.DataFrame:
    """Récupère l'historique des performances à partir des rapports"""
    try:
        report_files = list(REPORTS_DIR.glob("performance_*.json"))
        if not report_files:
            return pd.DataFrame()
            
        data = []
        for report_file in sorted(report_files, key=lambda x: x.stat().st_mtime):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    
                data.append({
                    "timestamp": report.get("timestamp"),
                    "portfolio_value": report.get("portfolio_value", 0),
                    "cash": report.get("cash", 0),
                    "transaction_costs": report.get("transaction_costs", 0),
                    "net_value": report.get("net_value", 0)
                })
            except Exception:
                continue
                
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            
        return df
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {e}")
        return pd.DataFrame()

def get_positions_data() -> List[Dict[str, Any]]:
    """Récupère les données de positions actuelles"""
    latest_report = get_latest_performance_report()
    positions = latest_report.get("positions", {})
    
    positions_list = []
    for symbol, pos in positions.items():
        positions_list.append({
            "symbol": symbol,
            "quantity": pos.get("qty", 0),
            "value": pos.get("market_value", 0),
            "avg_price": pos.get("avg_entry_price", 0),
            "pl": pos.get("unrealized_pl", 0),
            "pl_percent": pos.get("unrealized_plpc", 0) * 100 if "unrealized_plpc" in pos else 0
        })
    
    return positions_list

def get_signals_data() -> List[Dict[str, Any]]:
    """Récupère les données des signaux récents"""
    try:
        signal_files = list((DATA_DIR / "signals").glob("signals_*.json"))
        if not signal_files:
            return []
            
        latest_signal_file = max(signal_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_signal_file, 'r') as f:
            signals = json.load(f)
            
        signals_list = []
        for signal in signals:
            signals_list.append({
                "timestamp": signal.get("timestamp"),
                "symbol": signal.get("symbol", ""),
                "strategy": signal.get("strategy", ""),
                "action": signal.get("action", ""),
                "confidence": signal.get("confidence", 0),
                "executed": signal.get("executed", False)
            })
            
        return signals_list
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des signaux: {e}")
        return []

def get_strategy_weights() -> Dict[str, float]:
    """Récupère les poids actuels des stratégies"""
    latest_report = get_latest_performance_report()
    return latest_report.get("strategy_weights", {})

def get_market_regimes() -> Dict[str, str]:
    """Récupère les régimes de marché actuels"""
    latest_report = get_latest_performance_report()
    return latest_report.get("market_regimes", {})

@app.callback(
    [Output("last-update", "children"),
     Output("portfolio-chart", "figure"),
     Output("portfolio-stats", "children"),
     Output("strategy-performance", "figure"),
     Output("strategy-weights", "children"),
     Output("positions-table", "children"),
     Output("signals-table", "children"),
     Output("market-state", "children"),
     Output("anomalies", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n):
    """Met à jour tous les composants du tableau de bord"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_update = html.P(f"Dernière mise à jour: {current_time}")
    
    # Récupérer les données
    performance_history = get_performance_history()
    latest_report = get_latest_performance_report()
    positions_data = get_positions_data()
    signals_data = get_signals_data()
    strategy_weights = get_strategy_weights()
    market_regimes = get_market_regimes()
    
    # Graphique de performance du portefeuille
    if not performance_history.empty and "net_value" in performance_history.columns:
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=performance_history.index,
            y=performance_history["net_value"],
            mode="lines",
            name="Valeur nette",
            line=dict(color="#2c3e50", width=2)
        ))
        portfolio_fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Valeur ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        portfolio_fig = go.Figure()
        portfolio_fig.add_annotation(
            text="Aucune donnée de performance disponible",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Statistiques du portefeuille
    portfolio_value = latest_report.get("portfolio_value", 0)
    cash = latest_report.get("cash", 0)
    transaction_costs = latest_report.get("transaction_costs", 0)
    
    # Calculer le changement depuis le début
    initial_value = performance_history["net_value"].iloc[0] if not performance_history.empty and "net_value" in performance_history.columns else 100000
    current_value = latest_report.get("net_value", portfolio_value)
    total_return = (current_value / initial_value - 1) * 100 if initial_value else 0
    
    portfolio_stats = [
        html.Div([
            html.P("Valeur du portefeuille"),
            html.H4(f"${portfolio_value:,.2f}")
        ]),
        html.Div([
            html.P("Liquidités"),
            html.H4(f"${cash:,.2f}")
        ]),
        html.Div([
            html.P("Coûts de transaction"),
            html.H4(f"${transaction_costs:,.2f}")
        ]),
        html.Div([
            html.P("Performance totale"),
            html.H4(f"{total_return:+.2f}%", className="positive" if total_return >= 0 else "negative")
        ])
    ]
    
    # Graphique de performance des stratégies
    strategy_fig = go.Figure()
    
    if strategy_weights:
        labels = list(strategy_weights.keys())
        values = list(strategy_weights.values())
        
        strategy_fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker=dict(colors=["#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#f39c12"])
        ))
        
        strategy_fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
    else:
        strategy_fig.add_annotation(
            text="Aucune donnée de stratégie disponible",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Affichage des poids des stratégies
    strategy_weights_display = []
    for strategy, weight in strategy_weights.items():
        strategy_weights_display.append(html.Div([
            html.P(strategy),
            html.H4(f"{weight*100:.1f}%")
        ]))
    
    # Tableau des positions
    if positions_data:
        positions_table = html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in ["Symbole", "Quantité", "Valeur", "Prix moyen", "P&L", "P&L %"]])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(pos["symbol"]),
                    html.Td(f"{float(pos['quantity']):.6f}"),
                    html.Td(f"${float(pos['value']):,.2f}"),
                    html.Td(f"${float(pos['avg_price']):,.2f}"),
                    html.Td(f"${float(pos['pl']):,.2f}", className="positive" if float(pos['pl']) >= 0 else "negative"),
                    html.Td(f"{float(pos['pl_percent']):+.2f}%", className="positive" if float(pos['pl_percent']) >= 0 else "negative")
                ]) for pos in positions_data
            ])
        ])
    else:
        positions_table = html.P("Aucune position active")
    
    # Tableau des signaux
    if signals_data:
        signals_table = html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in ["Horodatage", "Symbole", "Stratégie", "Action", "Confiance", "Exécuté"]])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(signal["timestamp"]),
                    html.Td(signal["symbol"]),
                    html.Td(signal["strategy"]),
                    html.Td(signal["action"]),
                    html.Td(f"{float(signal['confidence']):.2f}"),
                    html.Td("Oui" if signal["executed"] else "Non")
                ]) for signal in signals_data
            ])
        ])
    else:
        signals_table = html.P("Aucun signal récent")
    
    # État du marché
    market_state_display = []
    for symbol, regime in market_regimes.items():
        color_class = {
            "bullish": "positive",
            "bearish": "negative",
            "volatile": "negative",
            "sideways": ""
        }.get(regime, "")
        
        market_state_display.append(html.Div([
            html.P(symbol),
            html.H4(regime.capitalize(), className=color_class)
        ]))
    
    if not market_state_display:
        market_state_display = [html.P("Aucune donnée d'état de marché disponible")]
    
    # Anomalies
    anomalies_display = []
    if "anomalies" in latest_report:
        for symbol, anomaly in latest_report.get("anomalies", {}).items():
            if anomaly.get("detected", False):
                anomalies_display.append(html.Div([
                    html.H4(f"Anomalie sur {symbol}"),
                    html.P(f"Probabilité de manipulation: {anomaly.get('manipulation_probability', 0)*100:.1f}%"),
                    html.P(f"Type: {', '.join(k for k, v in anomaly.items() if v and k != 'detected' and k != 'manipulation_probability' and k != 'timestamp')}")
                ], className="alert"))
    
    if not anomalies_display:
        anomalies_display = [html.P("Aucune anomalie détectée")]
    
    return (
        last_update, 
        portfolio_fig, 
        portfolio_stats, 
        strategy_fig, 
        strategy_weights_display, 
        positions_table, 
        signals_table, 
        market_state_display, 
        anomalies_display
    )

if __name__ == "__main__":
    # Créer les répertoires si nécessaire
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR / "signals", exist_ok=True)
    
    logger.info("Démarrage du tableau de bord MercurioAI...")
    app.run_server(debug=True, host="0.0.0.0", port=8050)
