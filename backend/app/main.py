from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.pipeline import router as pipeline_router, pipeline_services
from app.core.optimizer import LatencyOptimizer

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for Voice-Enabled RAG Model (Hacker House Goa 2026)",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Pipeline Router
app.include_router(pipeline_router)

import asyncio

@app.on_event("startup")
async def startup_event():
    print("[*] FastAPI Server Startup: Initializing services...")
    if pipeline_services.embedding_engine is None:
        pipeline_services.initialize()
    print("[+] FastAPI Server ready for requests!")

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Welcome to Voice-Enabled RAG Model API",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "sarvam_configured": bool(settings.SARVAM_API_KEY),
        "gemini_configured": bool(settings.GEMINI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
