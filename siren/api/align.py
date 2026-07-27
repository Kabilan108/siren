from fastapi import APIRouter, Depends, HTTPException, status

from siren.alignment import align_words
from siren.api.auth import verify_token
from siren.concurrency import run_in_worker_thread
from siren.schemas import AlignmentRequest, AlignmentResponse

router = APIRouter()
_MAX_WORDS = 200_000
_MAX_TURNS = 50_000


@router.post(
    "/v1/audio/align",
    response_model=AlignmentResponse,
    dependencies=[Depends(verify_token)],
)
async def align_audio(request: AlignmentRequest) -> AlignmentResponse:
    if len(request.words) > _MAX_WORDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many words: maximum is {_MAX_WORDS}.",
        )
    if len(request.turns) > _MAX_TURNS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many turns: maximum is {_MAX_TURNS}.",
        )

    try:
        return await run_in_worker_thread(
            align_words,
            request.words,
            request.turns,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
