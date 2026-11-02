"""Chunked speech-to-text through OpenRouter or a Gemini audio route."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import get_settings
from app.pipeline.audio import probe_duration, split_audio
from app.pipeline.metrics import StageTracker

logger = logging.getLogger(__name__)


class TruncatedAudioError(RuntimeError):
    """Raised when the decodable audio is far shorter than the expected length.

    Signals a corrupt/incomplete download whose container header may still claim
    the full duration. The runner deletes the file so a retry re-downloads it.
    """


@dataclass
class TranscribeResult:
    language: str | None = None
    segments: list[dict] = field(default_factory=list)
    text: str = ""


class RateLimiter:
    """Simple minimum-interval limiter (requests per minute)."""

    def __init__(self, per_minute: int):
        self.min_interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def transcribe_audio(
    audio_path: str,
    tracker: StageTracker | None = None,
    expected_duration: float | None = None,
) -> TranscribeResult:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    limiter = RateLimiter(settings.transcribe_rate_limit)
    with tempfile.TemporaryDirectory(prefix="sr_chunks_") as tmp:
        workdir = Path(tmp)
        chunks = split_audio(audio_path, settings.transcribe_chunk_seconds, workdir)
        if not chunks:
            raise RuntimeError(f"no audio chunks produced from {audio_path}")

        # Chunks are re-encoded (i.e. fully decoded) by ffmpeg, so their summed
        # length is the REAL transcribable audio — unlike the container header,
        # which can claim the full duration even when the stream is truncated
        # (a common flaky-CDN download failure). Bail out loudly so the item is
        # retried instead of silently summarizing only the first minutes.
        chunk_durations = [probe_duration(c) for c in chunks]
        decodable = sum(chunk_durations)
        if expected_duration and decodable < expected_duration * 0.9:
            raise TruncatedAudioError(
                f"truncated audio: only {decodable:.0f}s decodable of expected "
                f"{expected_duration:.0f}s ({decodable / expected_duration:.0%}); "
                "the download is incomplete — will retry"
            )
        if tracker is not None:
            tracker.set_chunks(len(chunks))

        segments: list[dict] = []
        offset = 0.0
        language: str | None = None
        with httpx.Client(timeout=300) as client:
            for idx, chunk in enumerate(chunks):
                limiter.wait()
                duration = chunk_durations[idx]
                text, detected = _transcribe_chunk(client, chunk, settings, tracker)
                language = language or detected
                if text:
                    segments.append({
                        "start": round(offset, 2),
                        "end": round(offset + duration, 2),
                        "text": text.strip(),
                    })
                offset += duration
                if tracker is not None:
                    tracker.chunk_progress(idx + 1)

    full_text = "\n".join(s["text"] for s in segments)
    return TranscribeResult(language=language, segments=segments, text=full_text)


def transcribe_via_gemini_audio(
    audio_path: str,
    tracker: StageTracker | None = None,
    expected_duration: float | None = None,
) -> TranscribeResult:
    """Transcribe chunked audio through a direct Gemini multimodal route.

    This is the fallback for deployments without an OpenRouter STT key. It
    deliberately uses the direct Gemini LiteLLM route: the local ``agy-high``
    compatibility proxy accepts text only and drops ``input_audio`` content.
    """
    settings = get_settings()
    model = settings.gemini_audio_model
    limiter = RateLimiter(settings.transcribe_rate_limit)
    prompt = (
        "Transcribe this audio verbatim. Return only the spoken words, in the "
        "original language, with no summary, headings, timestamps, or commentary. "
        "Preserve names, technical terms, and speaker turns as spoken."
    )
    system = "You are a precise speech transcription engine."

    with tempfile.TemporaryDirectory(prefix="sr_gemini_chunks_") as tmp:
        workdir = Path(tmp)
        chunks = split_audio(audio_path, settings.transcribe_chunk_seconds, workdir)
        if not chunks:
            raise RuntimeError(f"no audio chunks produced from {audio_path}")

        chunk_durations = [probe_duration(c) for c in chunks]
        decodable = sum(chunk_durations)
        if expected_duration and decodable < expected_duration * 0.9:
            raise TruncatedAudioError(
                f"truncated audio: only {decodable:.0f}s decodable of expected "
                f"{expected_duration:.0f}s ({decodable / expected_duration:.0%}); "
                "the download is incomplete — will retry"
            )
        if tracker is not None:
            tracker.set_chunks(len(chunks))

        segments: list[dict] = []
        offset = 0.0
        for idx, chunk in enumerate(chunks):
            text = _transcribe_gemini_chunk(
                chunk, settings, tracker, limiter, model=model, prompt=prompt, system=system,
            )
            duration = chunk_durations[idx]
            segments.append({
                "start": round(offset, 2),
                "end": round(offset + duration, 2),
                "text": text,
            })
            offset += duration
            if tracker is not None:
                tracker.chunk_progress(idx + 1)

    return TranscribeResult(
        language=settings.default_language or None,
        segments=segments,
        text="\n".join(s["text"] for s in segments),
    )


def _transcribe_gemini_chunk(
    chunk: Path, settings, tracker: StageTracker | None, limiter: RateLimiter,
    *, model: str, prompt: str, system: str,
) -> str:
    """Keep completed chunks across retries, without exposing a partial transcript."""
    from app.llm import generate_with_audio

    audio = chunk.read_bytes()
    digest = hashlib.sha256()
    digest.update(json.dumps([model, prompt, system]).encode())
    digest.update(audio)
    cache_dir = settings.data_dir / "transcription-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{digest.hexdigest()}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if isinstance(cached.get("text"), str) and cached["text"].strip():
                return cached["text"]
        except (ValueError, AttributeError):
            logger.warning("ignoring invalid transcription cache %s", cache_path.name)

    backoff = 2.0
    for attempt in range(settings.transcribe_max_retries):
        limiter.wait()
        start = time.monotonic()
        try:
            result = generate_with_audio(
                prompt, base64.b64encode(audio).decode("ascii"), "mp3",
                system=system, model=model,
            )
        except httpx.HTTPError as exc:
            response = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
            code = response.status_code if response is not None else None
            if tracker is not None:
                tracker.record_call(
                    provider="litellm", model=model, endpoint="/chat/completions",
                    latency_ms=int((time.monotonic() - start) * 1000),
                    status_code=code, is_429=code == 429,
                )
            if code is not None and code != 429 and code < 500:
                raise
            # A daily quota cannot be fixed by a short retry. Keep completed
            # chunks for a later job instead of repeatedly spending them again.
            daily_quota = response is not None and (
                "requests per day" in response.text.lower()
                or "perday" in response.text.lower()
                or "daily quota" in response.text.lower()
            )
            retry_after = response.headers.get("retry-after", "") if response is not None else ""
            try:
                delay = max(backoff, float(retry_after))
            except ValueError:
                delay = backoff
            if daily_quota or delay > 300:
                raise RuntimeError(
                    f"Gemini audio quota exhausted for {model}; completed chunks are saved"
                ) from exc
            if attempt + 1 == settings.transcribe_max_retries:
                raise
            logger.warning(
                "Gemini audio HTTP %s on %s; retry in %.1fs", code, chunk.name, delay,
            )
            time.sleep(delay)
            backoff = min(backoff * 2, 60.0)
            continue

        if tracker is not None:
            tracker.record_call(
                provider="litellm", model=model, endpoint="/chat/completions",
                latency_ms=result.latency_ms, status_code=200,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens, tokens=result.total_tokens,
            )
        text = result.text.strip()
        if not text:
            raise RuntimeError(f"Gemini returned empty transcript for {chunk.name}")
        if result.finish_reason not in (None, "stop"):
            raise RuntimeError(
                f"Gemini transcript incomplete for {chunk.name}: {result.finish_reason}"
            )
        if re.search(r"(\S)\1{30,}", text):
            raise RuntimeError(f"Gemini returned repetitive transcript for {chunk.name}")
        # Write atomically so an interrupted worker cannot leave a partial cache.
        with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as out:
            json.dump({"text": text}, out, ensure_ascii=False)
            temporary = Path(out.name)
        temporary.replace(cache_path)
        return text
    raise RuntimeError("Gemini transcription requires at least one attempt")


def _transcribe_chunk(
    client: httpx.Client, chunk: Path, settings, tracker: StageTracker | None
) -> tuple[str, str | None]:
    from app.runtime_config import effective_stt_model

    model = effective_stt_model()
    audio_b64 = base64.b64encode(chunk.read_bytes()).decode("ascii")
    payload: dict = {
        "model": model,
        "input_audio": {"data": audio_b64, "format": "mp3"},
    }
    if settings.default_language:
        payload["language"] = settings.default_language

    url = f"{settings.openrouter_base_url}/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.openrouter_referer,
        "X-Title": settings.openrouter_title,
    }

    backoff = 2.0
    last_exc: Exception | None = None
    for _attempt in range(settings.transcribe_max_retries):
        start = time.monotonic()
        try:
            resp = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:  # network errors -> retry
            last_exc = exc
            time.sleep(backoff + random.uniform(0, 1))
            backoff *= 2
            continue
        latency_ms = int((time.monotonic() - start) * 1000)

        if resp.status_code == 429 or resp.status_code >= 500:
            if tracker is not None:
                tracker.record_call(
                    provider="openrouter", model=model,
                    endpoint="/audio/transcriptions", latency_ms=latency_ms,
                    status_code=resp.status_code, is_429=resp.status_code == 429,
                )
            retry_after = resp.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
            logger.warning(
                "STT %s on %s, retry in %.1fs", resp.status_code, chunk.name, delay
            )
            time.sleep(delay + random.uniform(0, 1))
            backoff *= 2
            continue

        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        if tracker is not None:
            tracker.record_call(
                provider="openrouter", model=model,
                endpoint="/audio/transcriptions", latency_ms=latency_ms,
                status_code=resp.status_code,
                tokens=int(usage.get("total_tokens", 0) or 0),
                cost_usd=float(usage.get("cost", 0) or 0),
            )
        text = data.get("text") or ""
        language = data.get("language")
        return text, language

    raise RuntimeError(
        f"transcription failed for {chunk.name} after {settings.transcribe_max_retries} attempts"
    ) from last_exc
