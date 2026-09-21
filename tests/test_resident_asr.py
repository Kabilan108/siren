import asyncio
import wave
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

from siren import models
from siren.api.transcriptions import router
from siren.jobs import worker
from siren.jobs.resident import ResidentBackend, validate_resident_url
from siren.schemas import TranscriptionResult, TranscriptionSegment, TranscriptionWord


@pytest.mark.parametrize("url", [
    "http://example.com:8301", "https://127.0.0.1:8301",
    "http://127.0.0.1:8301/path", "http://user:password@localhost:8301",
    "http://localhost:8301?target=elsewhere", "http://localhost:wrong",
])
def test_resident_asr_rejects_nonlocal_or_ambiguous_origins(url: str) -> None:
    with pytest.raises(ValueError):
        validate_resident_url(url)


@pytest.mark.asyncio
async def test_job_uses_resident_model_and_offsets_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIREN_JOB_ASR_URL", "http://127.0.0.1:8301")
    paths = [tmp_path/'first.wav', tmp_path/'second.wav']
    for path in paths:
        path.write_bytes(b"test wave")
    requests = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers['authorization'] == 'Bearer dev_token'
        body = (await request.aread()).decode()
        assert 'name="timestamp_granularities[]"' in body
        assert 'verbose_json' in body and 'turbo' in body
        assert 'name="language"' not in body
        return httpx.Response(200, json={
            'task':'transcribe', 'text':'hello', 'language':'en', 'duration':1.0,
            'segments':[{'id':0,'text':'hello','start':0.2,'end':0.7,
                         'words':[{'word':'hello','start':0.2,'end':0.7}]}],
        })

    actual_client = httpx.AsyncClient
    clients = []

    def client_factory(**kwargs: object) -> httpx.AsyncClient:
        assert kwargs['trust_env'] is False and kwargs['follow_redirects'] is False
        client = actual_client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(worker.httpx, 'AsyncClient', client_factory)
    load = MagicMock(side_effect=AssertionError('A job must not load another ASR copy'))
    monkeypatch.setattr(worker, 'load_backend', load)
    text, language, words = await worker._transcribe_chunks(
        [(paths[0],0.0),(paths[1],300.0)], model_name='turbo', language=None,
    )
    assert len(requests)==2 and not load.called
    assert clients[0].is_closed
    assert text=='hello hello' and language=='en'
    assert [w.start for w in words]==[0.2,300.2]


@pytest.mark.asyncio
async def test_resident_error_does_not_echo_server_body(tmp_path: Path) -> None:
    path=tmp_path/'audio.wav'
    path.write_bytes(b'audio')
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(500, text='sensitive error body')
    )) as client:
        backend=ResidentBackend(client,'http://127.0.0.1:8301','turbo')
        with pytest.raises(RuntimeError, match='Resident ASR returned HTTP 500') as caught:
            await backend.transcribe(str(path), language='en', word_timestamps=True)
        assert 'sensitive' not in str(caught.value)


@pytest.mark.asyncio
async def test_resident_adapter_real_api_contract_and_serialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path/'audio.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\x00\x00' * 16000)

    class Backend:
        active: int = 0
        peak: int = 0
        calls: int = 0

        async def transcribe(self, audio_path: str, **kwargs: object) -> TranscriptionResult:
            assert kwargs['word_timestamps'] is True
            assert kwargs['language']=='en'
            assert Path(audio_path).exists()
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls += 1
            await asyncio.sleep(0.01)
            self.active -= 1
            return TranscriptionResult(
                text='hello', language='en', duration=1.0,
                segments=[TranscriptionSegment(id=0, start=0.2, end=0.7,text='hello',
                    words=[TranscriptionWord(start=0.2,end=0.7,word='hello')])],
            )

    backend = Backend()
    monkeypatch.setattr(models, 'current_model_name', 'turbo')
    monkeypatch.setattr(models, 'current_backend', backend)
    load = MagicMock(side_effect=AssertionError('Already resident'))
    monkeypatch.setattr(models, 'load_backend', load)
    app=FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        resident=ResidentBackend(client, 'http://127.0.0.1:8301', 'turbo')
        results=await asyncio.gather(*[
            resident.transcribe(str(path),language='en',word_timestamps=True)
            for _ in range(2)
        ])
    assert backend.calls==2 and backend.peak==1
    assert not load.called
    assert all(r.segments[0].words[0].word=='hello' for r in results)
