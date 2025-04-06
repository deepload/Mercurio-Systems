"""
Database models for Mercurio AI platform
"""
import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship

from app.db.database import Base

class TradeAction(enum.Enum):
    """Enum for trade actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class Trade(Base):
    """Model for storing trade records"""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    strategy = Column(String(50), index=True, nullable=False)
    action = Column(Enum(TradeAction), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional metadata
    confidence = Column(Float)
    model_id = Column(Integer, ForeignKey("ai_models.id"), nullable=True)
    trade_metadata = Column(JSON, nullable=True)
    
    # Relationships
    model = relationship("AIModel", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, action={self.action})>"

class BacktestResult(Base):
    """Model for storing backtest results"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy = Column(String(50), index=True, nullable=False)
    symbol = Column(String(10), index=True, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Performance metrics
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Run metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    parameters = Column(JSON, nullable=True)
    
    # Relationships
    model_id = Column(Integer, ForeignKey("ai_models.id"), nullable=True)
    model = relationship("AIModel", back_populates="backtest_results")
    
    def __repr__(self):
        return f"<BacktestResult(id={self.id}, strategy={self.strategy}, return={self.total_return})>"

class AIModel(Base):
    """Model for storing trained AI models metadata"""
    __tablename__ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    strategy = Column(String(50), index=True, nullable=False)
    model_type = Column(String(50), nullable=False)  # e.g., "RandomForest", "LSTM"
    symbols = Column(JSON, nullable=False)  # List of symbols this model was trained on
    
    # Training metadata
    train_start_date = Column(DateTime, nullable=False)
    train_end_date = Column(DateTime, nullable=False)
    
    # File paths and parameters
    model_path = Column(String(255), nullable=False)
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)  # Training and validation metrics
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    trades = relationship("Trade", back_populates="model")
    backtest_results = relationship("BacktestResult", back_populates="model")
    
    def __repr__(self):
        return f"<AIModel(id={self.id}, name={self.name}, strategy={self.strategy})>"
