"""Converter <-> speech engine: chunk order, parallel requests, error handling, cache keys."""
import threading
import time

import numpy as np
import pytest

from epub2audiobook.converter import ConvertOptions, Converter
from epub2audiobook.epub_parser import Block, Book, Chapter
from epub2audiobook.tts import EngineConfig, EngineError, TTSEngine


class FakeEngine(TTSEngine):
    """Constant tone whose length encodes the text, so the assembled chapter reveals the chunk order.
    Odd-length texts answer slower, so parallel results come back out of order."""

    def __init__(self, max_parallel=1, fail=None):
        self.max_parallel = max_parallel
        self.fail = fail
        self.calls = []
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()

    def describe(self):
        return "fake"

    def synthesize(self, text, voice, lang, speed=1.0):
        with self._lock:
            self.calls.append(text)
            self._active += 1
            self.peak = max(self.peak, self._active)
        try:
            if self.fail:
                raise self.fail
            time.sleep(0.02 if len(text) % 2 else 0.002)
            return np.full(200 + len(text), 0.1, dtype=np.float32)
        finally:
            with self._lock:
                self._active -= 1


def make_book():
    blocks = [Block("heading", "Chapter One", 1)]
    blocks += [Block("para", " ".join(f"Sentence {i}.{j} is here." for j in range(25 + i))) for i in range(5)]
    blocks += [Block("break"), Block("para", "A short closing line.")]
    ch = Chapter(index=1, href="c1.xhtml", title="Chapter One", kind="chapter", blocks=blocks)
    book = Book(path="t.epub", title="T", authors=["A"], language_iso="en", kokoro_lang="a", year=None,
                cover_bytes=None, cover_ext="", chapters=[ch])
    return book, ch


def chapter_audio(engine):
    book, ch = make_book()
    conv = Converter(book, [ch], ConvertOptions(output_path="t.m4b"), engine=engine)
    return conv._synthesize_chapter(ch, "af_heart", 1, 1)


def test_parallel_requests_give_the_same_audio_as_sequential():
    seq, par = FakeEngine(1), FakeEngine(4)
    assert np.array_equal(chapter_audio(seq), chapter_audio(par))
    assert seq.peak == 1
    assert par.peak > 1
    assert sorted(seq.calls) == sorted(par.calls)


def test_engine_error_stops_the_conversion_without_splitting():
    engine = FakeEngine(1, fail=EngineError("HTTP 401: bad key"))
    with pytest.raises(EngineError):
        chapter_audio(engine)
    assert len(engine.calls) == 1


def test_engine_error_from_a_parallel_request_stops_the_conversion():
    with pytest.raises(EngineError):
        chapter_audio(FakeEngine(4, fail=EngineError("HTTP 401: bad key")))


def test_other_errors_are_still_retried_in_smaller_pieces():
    engine = FakeEngine(1, fail=RuntimeError("out of memory"))
    audio = chapter_audio(engine)  # every piece fails: silence, but no exception
    assert audio.size > 0
    assert any(len(t) < 40 for t in engine.calls)  # it split down to the minimum size


def test_cache_key_changes_only_with_settings_that_change_the_sound():
    key = lambda **kw: ConvertOptions(output_path="t.m4b", engine=EngineConfig(**kw)).cache_key()
    default = key()
    assert key(device="cpu", threads=4, max_vram_gb=2, concurrency=8) == default
    assert key(model="kokoro-v1.1-zh") != default
    server = key(provider="openai", base_url="http://localhost:8880/v1")
    assert server != default
    assert key(provider="openai", base_url="http://localhost:8880/v1/", concurrency=16) == server
    assert key(provider="openai", base_url="http://localhost:8880/v1", api_model="tts-1") != server
