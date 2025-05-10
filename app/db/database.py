"""
Database setup and connection handling for Mercurio AI
"""
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://mercurio_user:mercurio_password@db:5432/mercurio")
# Convert to asyncpg format if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False, poolclass=NullPool)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Base class for SQLAlchemy models
Base = declarative_base()

async def get_db() -> AsyncSession:
    """
    Dependency for getting async DB session
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """
    Initialize database with all models
    """
    async with engine.begin() as conn:
        # Import all models to ensure they are registered with Base
        from app.db.models import Trade, BacktestResult, AIModel
        
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
