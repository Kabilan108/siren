from siren.api.align import router as align_router
from siren.api.diarize import router as diarize_router
from siren.api.health import router as health_router
from siren.api.jobs import router as jobs_router
from siren.api.models import router as models_router
from siren.api.transcriptions import router as transcriptions_router

__all__ = [
    "align_router",
    "diarize_router",
    "health_router",
    "jobs_router",
    "models_router",
    "transcriptions_router",
]
