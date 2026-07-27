from typing import Literal

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class TranscriptionWord(BaseModel):
    start: float
    end: float
    word: str


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptionWord] | None = None


class VerboseTranscriptionResponse(BaseModel):
    task: Literal["transcribe"] = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[TranscriptionSegment]


class TranscriptionResult(BaseModel):
    text: str
    language: str
    duration: float
    segments: list[TranscriptionSegment]


class DiarizationTurn(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizationResponse(BaseModel):
    duration: float
    model: str
    speakers: list[str]
    turns: list[DiarizationTurn]


class AlignmentRequest(BaseModel):
    words: list[TranscriptionWord]
    turns: list[DiarizationTurn]


class AlignedWord(TranscriptionWord):
    speaker: str


class AlignedSegment(BaseModel):
    id: int
    start: float
    end: float
    speaker: str
    text: str
    words: list[AlignedWord]


class AlignmentResponse(BaseModel):
    speakers: list[str]
    segments: list[AlignedSegment]


class ModelInfo(BaseModel):
    id: str


class ModelsResponse(BaseModel):
    data: list[ModelInfo]


JobStatus = Literal["queued", "running", "completed", "failed"]
JobPhase = Literal["chunking", "transcribing", "diarizing", "aligning"]


class TranscriptJobAccepted(BaseModel):
    id: str
    status: Literal["queued"]
    position: int = Field(ge=1)


class TranscriptJobStatus(BaseModel):
    id: str
    status: JobStatus
    phase: JobPhase | None = None
    progress: float = Field(ge=0.0, le=1.0)
    position: int | None = Field(default=None, ge=1)
    error: str | None = None


class TranscriptJobSegment(BaseModel):
    id: int
    start: float
    end: float
    speaker: str
    text: str


class TranscriptJobResult(BaseModel):
    text: str
    language: str
    duration: float
    model: str
    diarization_model: str
    speakers: list[str]
    segments: list[TranscriptJobSegment]
