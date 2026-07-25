from typing import Literal

from pydantic import BaseModel


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


class ModelInfo(BaseModel):
    id: str


class ModelsResponse(BaseModel):
    data: list[ModelInfo]
