from typing import Literal

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


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


class ModelInfo(BaseModel):
    id: str


class ModelsResponse(BaseModel):
    data: list[ModelInfo]
