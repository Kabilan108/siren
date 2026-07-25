import logging

from fastapi import APIRouter, Depends, HTTPException, status

from siren import models
from siren.api.auth import verify_token
from siren.logging_utils import log_event
from siren.schemas import ModelsResponse

router = APIRouter()


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_token)],
)
async def list_models() -> ModelsResponse:
    try:
        return ModelsResponse(data=models.get_available_models())
    except Exception as exc:
        log_event(logging.ERROR, "models_list_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(exc)}",
        )
