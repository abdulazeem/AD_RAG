# rag_app/generation/api/routers/admin.py

from fastapi import APIRouter
router = APIRouter()

@router.get("/health", tags=["admin"])
async def health_check():
    return {"status": "healthy"}

@router.get("/version", tags=["admin"])
async def version_check():
    return {"version": "0.1.0"}
