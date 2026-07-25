from fastapi import APIRouter

from siren.config import SIREN_VERSION
from siren.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=SIREN_VERSION)
