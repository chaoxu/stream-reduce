"""Provider failure recovery must preserve completed transcription work."""

import httpx
import pytest

from app.config import Settings
from app.llm import LlmResult
from app.pipeline import transcribe


@pytest.fixture
def chunk_call(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, transcribe_max_retries=3, transcribe_rate_limit=0,
    )
    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"first audio bytes")
    delays = []
    monkeypatch.setattr(transcribe.time, "sleep", delays.append)

    def invoke():
        return transcribe._transcribe_gemini_chunk(
            chunk, settings, None, transcribe.RateLimiter(0),
            model="audio-model", prompt="transcribe", system="verbatim",
        )
    return invoke, chunk, delays


def error(code, body="", headers=None):
    response = httpx.Response(
        code, text=body, headers=headers,
        request=httpx.Request("POST", "https://provider.test/chat/completions"),
    )
    return httpx.HTTPStatusError("provider error", request=response.request, response=response)


def test_transient_limit_retries_then_reuses_success(monkeypatch, chunk_call):
    invoke, chunk, delays = chunk_call
    calls = []
    def generate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise error(429, headers={"retry-after": "3"})
        return LlmResult(text="测试转写", finish_reason="stop")

    monkeypatch.setattr("app.llm.generate_with_audio", generate)
    assert invoke() == "测试转写"
    assert delays == [3]
    assert len(calls) == 2
    assert invoke() == "测试转写"
    assert len(calls) == 2
    # A different audio chunk must not inherit the cached words.
    chunk.write_bytes(b"changed audio bytes")
    assert invoke() == "测试转写"
    assert len(calls) == 3


def test_daily_quota_is_reported_without_short_retries(monkeypatch, chunk_call):
    invoke, _, delays = chunk_call
    def generate(*args, **kwargs):
        raise error(429, "Generate content requests per day limit exceeded")
    monkeypatch.setattr("app.llm.generate_with_audio", generate)
    with pytest.raises(RuntimeError, match="quota exhausted"):
        invoke()
    assert delays == []


def test_truncated_output_is_not_cached(monkeypatch, chunk_call):
    invoke, _, _ = chunk_call
    monkeypatch.setattr(
        "app.llm.generate_with_audio",
        lambda *a, **kw: LlmResult(text="incomplete speech", finish_reason="length"),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        invoke()
    monkeypatch.setattr(
        "app.llm.generate_with_audio",
        lambda *a, **kw: LlmResult(text="complete speech", finish_reason="stop"),
    )
    assert invoke() == "complete speech"


def test_repeated_output_is_not_cached(monkeypatch, chunk_call):
    invoke, _, _ = chunk_call
    monkeypatch.setattr(
        "app.llm.generate_with_audio",
        lambda *a, **kw: LlmResult(text="这" * 1000, finish_reason="stop"),
    )
    with pytest.raises(RuntimeError, match="repetitive"):
        invoke()
    monkeypatch.setattr(
        "app.llm.generate_with_audio",
        lambda *a, **kw: LlmResult(text="正常的转写文本", finish_reason="stop"),
    )
    assert invoke() == "正常的转写文本"
