"""Health check endpoint — used by the desktop launcher to know the backend is up."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
