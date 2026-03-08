from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "groq_key_loaded": bool(settings.groq_api_key),
        "groq_key_preview": settings.groq_api_key[:8] + "..." if settings.groq_api_key else "EMPTY",
        "groq_model": settings.groq_model
    }
