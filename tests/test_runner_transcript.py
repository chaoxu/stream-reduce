"""A successful audio fallback must persist its transcript before summarizing."""

from contextlib import contextmanager

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.adapters.base import ContentMeta
from app.config import Settings
from app.models import Item, ItemStatus, Platform, Summary, Transcript, TranscriptSource
from app.pipeline import metrics, runner, summarize, transcribe


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)

    @contextmanager
    def sessions():
        with Session(engine) as session:
            yield session
            session.commit()

    settings = Settings(
        _env_file=None, data_dir=tmp_path,
        openrouter_api_key="", enable_gemini_audio_fallback=True,
        gemini_audio_model="audio-route", enable_embeddings=False,
        track_api_calls=False,
    )
    monkeypatch.setattr(runner, "session_scope", sessions)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(metrics, "get_settings", lambda: settings)
    monkeypatch.setattr(runner, "effective_llm_model", lambda: "text-route")
    monkeypatch.setattr(runner, "effective_stt_model", lambda: "unused")
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"test audio")

    class Adapter:
        name = "test"
        def fetch_metadata(self, url):
            return ContentMeta(title="Test", duration_s=60)
        def get_danmaku(self, url):
            return []
        def get_native_transcript(self, url, language):
            return None

    monkeypatch.setattr("app.adapters.registry.get_adapter", lambda _: Adapter())
    monkeypatch.setattr(runner, "_download_decodable", lambda *a: (audio, 60.0))
    with sessions() as session:
        session.add(Item(id=1, platform=Platform.youtube, source_url="https://example.test/video"))
    yield sessions
    engine.dispose()


def test_audio_fallback_stores_transcript_and_summarizes_without_openrouter(monkeypatch, pipeline):
    def gemini(*args, **kwargs):
        return transcribe.TranscribeResult(
            language="zh", segments=[{"start": 0, "end": 60, "text": "测试转写"}],
        )

    def openrouter(*args, **kwargs):
        raise AssertionError("successful audio fallback must not call OpenRouter")

    def summary(session, item_id, tracker):
        transcript = session.exec(select(Transcript).where(Transcript.item_id == item_id)).one()
        assert transcript.source == TranscriptSource.gemini
        assert transcript.text == "测试转写"
        session.add(Summary(item_id=item_id, markdown=transcript.text))

    monkeypatch.setattr(transcribe, "transcribe_via_gemini_audio", gemini)
    monkeypatch.setattr(transcribe, "transcribe_audio", openrouter)
    monkeypatch.setattr(summarize, "summarize_item", summary)
    runner.process_item(1)
    with pipeline() as session:
        item = session.get(Item, 1)
        assert item.status == ItemStatus.done
        assert item.error is None
        assert session.exec(select(Transcript)).one().text == "测试转写"
        assert session.exec(select(Summary)).one().markdown == "测试转写"


def test_audio_failure_does_not_create_transcript_or_summary(monkeypatch, pipeline):
    def failed(*args, **kwargs):
        raise RuntimeError("audio provider unavailable")

    monkeypatch.setattr(transcribe, "transcribe_via_gemini_audio", failed)
    with pytest.raises(RuntimeError, match="audio provider unavailable"):
        runner.process_item(1)
    with pipeline() as session:
        assert session.get(Item, 1).status == ItemStatus.error
        assert session.exec(select(Transcript)).all() == []
        assert session.exec(select(Summary)).all() == []
