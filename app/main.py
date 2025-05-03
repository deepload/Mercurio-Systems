"""
Mercurio AI - Trading Platform
Main application entry point
"""
import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import environment loader to ensure .env variables are loaded
from app.utils import env_loader

from app.db.database import init_db
from app.api.routes import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize database and other resources on startup
    """
    await init_db()
    yield
    # Cleanup code if needed

# Initialize FastAPI
app = FastAPI(
    title="Mercurio AI",
    description="AI-Powered Trading Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": app.version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
