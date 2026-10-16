"""Conversion orchestrator: book + selected chapters + options -> chaptered M4B.

Pipeline per chapter: synthesize (local GPU/CPU, or parallel requests to a TTS server) -> WAV -> AAC
(.m4a, background ffmpeg, overlapped with the next chapter's synthesis). The final step is a
stream-copy mux of the chapter files, which takes seconds. Runs synchronously in the calling thread;
the GUI runs it in a worker thread and passes callbacks + a cancel event.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout  # not the builtin TimeoutError on Python 3.10
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import __version__
from .audio import (SAMPLE_RATE, build_m4b, encode_chapter_aac, fade_edges, find_ffmpeg, fmt_hms, normalize_level,
                    prepare_cover, silence, trim_silence, write_wav)
from .epub_parser import Book, Chapter
from .textproc import chunk_paragraph, normalize_text
from .tts import EngineConfig, EngineError, TTSEngine, create_engine
from .voices import DEFAULT_MODEL, default_voice

log = logging.getLogger(__name__)


class CancelledError(Exception):
    pass


# Chapter-marker title for the opening credits, per Kokoro language
CREDITS_TITLE = {"a": "Opening credits", "b": "Opening credits", "e": "Créditos iniciales", "f": "Générique",
                 "h": "प्रारंभिक श्रेय", "i": "Titoli di testa", "p": "Créditos iniciais", "j": "オープニングクレジット",
                 "z": "片头字幕"}
_WRITTEN_BY = {"a": "Written by {}.", "b": "Written by {}.", "e": "Escrito por {}.", "f": "Écrit par {}.",
               "h": "लेखक: {}.", "i": "Scritto da {}.", "p": "Escrito por {}.", "j": "著者、{}。", "z": "作者：{}。"}
_AND = {"a": " and ", "b": " and ", "e": " y ", "f": " et ", "h": " और ", "i": " e ", "p": " e ", "j": "、", "z": "、"}


def credits_author_line(authors: list[str], lang: str) -> str:
    names = [a.strip() for a in authors if a.strip()]
    if not names:
        return ""
    if len(names) == 1:
        joined = names[0]
    else:
        joined = ", ".join(names[:-1]) + _AND.get(lang, " and ") + names[-1]
    return _WRITTEN_BY.get(lang, _WRITTEN_BY["a"]).format(joined)


@dataclass
class ConvertOptions:
    output_path: str
    voice: str = "af_heart"
    lang: str = "a"
    speed: float = 1.0
    engine: EngineConfig = field(default_factory=EngineConfig)
    bitrate: str = "64k"
    max_chars: int = 350
    read_titles: bool = True
    read_credits: bool = True      # opening credits: title, subtitle, "written by <authors>"
    speech_friendly: bool = True   # Roman numerals as digits, no markdown leftovers or symbols (textproc)
    keep_work_files: bool = False
    work_dir: Optional[str] = None
    encode_workers: int = 0        # 0 = auto (half the CPUs, 1..4)
    # pauses in seconds (what makes it feel like a human reading)
    pause_chunk: float = 0.15
    pause_para: float = 0.55
    pause_heading: float = 1.0
    pause_break: float = 1.3
    pause_chapter_start: float = 0.7
    pause_chapter_end: float = 1.2
    target_rms_db: float = -20.0

    def cache_key(self) -> str:
        d = asdict(self)
        for k in ("output_path", "engine", "bitrate", "keep_work_files", "work_dir", "encode_workers"):
            d.pop(k, None)
        if self.engine.audio_identity():
            d["engine"] = self.engine.audio_identity()
        d["v"] = __version__
        return repr(sorted(d.items()))


ProgressCb = Callable[[float, str], None]
LogCb = Callable[[str], None]
ChapterCb = Callable[[int, str], None]   # (chapter.index or 0 for credits, state) state: synth|encode|done|cached
Step = tuple[Optional[str], float, int]  # (text to speak or None for a pure pause, seconds of silence after, chars)


class Converter:
    def __init__(self, book: Book, chapters: list[Chapter], options: ConvertOptions,
                 on_progress: Optional[ProgressCb] = None, on_log: Optional[LogCb] = None,
                 cancel_event: Optional[threading.Event] = None, engine: Optional[TTSEngine] = None,
                 on_chapter: Optional[ChapterCb] = None):
        self.book = book
        self.chapters = chapters
        self.opt = options
        self._progress = on_progress or (lambda f, m: None)
        self._log = on_log or (lambda m: log.info(m))
        self._chapter_cb = on_chapter or (lambda i, s: None)
        self._cancel = cancel_event or threading.Event()
        self.engine = engine or create_engine(options.engine)
        self.total_chars = 0
        self.done_chars = 0
        self._t0 = 0.0
        self._encode_lock = threading.Lock()
        self._encoded = 0
        self._encode_total = 0

    # ------------------------------------------------------------------
    def _check_cancel(self):
        if self._cancel.is_set():
            raise CancelledError()

    def _report(self, msg: str):
        frac = (self.done_chars / self.total_chars) if self.total_chars else 0.0
        elapsed = time.time() - self._t0
        eta = ""
        if frac > 0.02 and elapsed > 5:
            remaining = elapsed / frac - elapsed
            eta = f"  (ETA {fmt_hms(remaining)})"
        self._progress(frac * 0.90, msg + eta)

    # ------------------------------------------------------------------
    def run(self) -> str:
        opt = self.opt
        if not self.chapters:
            raise ValueError("No chapters selected.")
        if not find_ffmpeg():
            raise RuntimeError("ffmpeg not found on PATH. Install it (winget install Gyan.FFmpeg) and restart.")
        out_path = Path(opt.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = Path(opt.work_dir) if opt.work_dir else out_path.parent / (out_path.stem + "_work")
        work_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

        self.total_chars = sum(ch.chars + len(ch.title) for ch in self.chapters)
        self.done_chars = 0
        voice = opt.voice or default_voice(opt.lang, DEFAULT_MODEL if opt.engine.is_server else opt.engine.model)
        workers = opt.encode_workers or max(1, min(4, (os.cpu_count() or 2) // 2))

        self._log(f"Book: {self.book.title} — {self.book.author or 'unknown author'}")
        self._log(f"Language: {opt.lang}  Voice: {voice}  Speed: {opt.speed}  Chapters: {len(self.chapters)}  "
                  f"AAC {opt.bitrate} ({workers} encoder thread{'s' if workers > 1 else ''})")
        self._progress(0.0, "Loading speech engine…")
        self.engine.warm_up(opt.lang, voice)
        self._log(f"Speech engine: {self.engine.describe()}")
        if opt.engine.is_server:
            self._log(f"About {self.total_chars:,} characters will be sent to the server (paid APIs bill per character)")

        # (index, title, m4a_path, duration) in playback order; futures fill in durations
        items: list[tuple[int, str, Path, float]] = []
        futures: list[Future] = []
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="aac")
        try:
            jobs: list[tuple[int, str, Chapter | None]] = []
            if opt.read_credits:
                jobs.append((0, CREDITS_TITLE.get(opt.lang, CREDITS_TITLE["a"]), None))
            jobs += [(ch.index, ch.title, ch) for ch in self.chapters]
            self._encode_total = len(jobs)

            for n, (idx, title, ch) in enumerate(jobs, 1):
                self._check_cancel()
                key = self._cache_key(voice, ch)
                base = work_dir / f"ch{idx:03d}_{key}"
                m4a = base.with_suffix(".m4a")
                wav = base.with_suffix(".wav")
                if m4a.exists() and m4a.stat().st_size > 500:
                    self._log(f"[{n}/{len(jobs)}] {title}: reusing cached audio")
                    self.done_chars += (ch.chars + len(ch.title)) if ch else 0
                    self._chapter_cb(idx, "cached")
                    with self._encode_lock:
                        self._encoded += 1
                    items.append((idx, title, m4a, _duration_of(m4a, wav)))
                    self._report(f"Chapter {n}/{len(jobs)} cached")
                    continue

                t_ch = time.time()
                self._chapter_cb(idx, "synth")
                if ch is None:
                    self._log(f"[{n}/{len(jobs)}] Opening credits: " + " / ".join(
                        x for x in (self.book.title, self.book.subtitle, self.book.author) if x))
                    self._progress(0.0, "Opening credits…")
                    audio = self._synthesize_credits(voice)
                else:
                    self._log(f"[{n}/{len(jobs)}] {title} ({ch.words} words)")
                    audio = self._synthesize_chapter(ch, voice, n, len(jobs))
                audio = normalize_level(audio, opt.target_rms_db)
                if not (wav.exists() and wav.stat().st_size > 1000):
                    tmp = wav.with_suffix(".part.wav")
                    write_wav(tmp, audio)
                    tmp.replace(wav)
                dur = audio.size / SAMPLE_RATE
                spent = time.time() - t_ch
                self._log(f"    {dur / 60:.1f} min of audio in {spent:.0f}s ({dur / max(spent, 0.01):.1f}x realtime)")
                items.append((idx, title, m4a, dur))
                self._chapter_cb(idx, "encode")
                futures.append(pool.submit(self._encode_job, idx, title, wav, m4a, dur))

            self._check_cancel()
            # wait for the background encoders
            pending = [f for f in futures if not f.done()]
            if pending:
                self._log(f"Waiting for {len(pending)} chapter encode(s) to finish…")
            for f in futures:
                while not f.done():
                    self._check_cancel()
                    with self._encode_lock:
                        k = self._encoded
                    self._progress(0.90 + 0.05 * k / max(1, self._encode_total),
                                   f"Encoding chapters to AAC: {k}/{self._encode_total} done")
                    try:
                        f.result(timeout=0.5)
                    except TimeoutError:
                        pass
                    except Exception:
                        break
                f.result()  # re-raise encode errors
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

        self._check_cancel()
        self._progress(0.95, "Assembling M4B…")
        cover = prepare_cover(self.book.cover_bytes, self.book.cover_ext, work_dir, find_ffmpeg()) if self.book.cover_bytes else None
        build_m4b([(str(p), t, d) for _, t, p, d in items], out_path, title=self.book.title, author=self.book.author,
                  year=self.book.year, cover_path=cover, bitrate=opt.bitrate, work_dir=work_dir,
                  on_progress=lambda f, m: self._progress(0.95 + 0.05 * f, m), on_log=self._log)
        total = time.time() - self._t0
        total_audio = sum(d for _, _, _, d in items)
        self._log(f"Done in {fmt_hms(total)}: {fmt_hms(total_audio)} of audio, {len(items)} chapters -> {out_path}")
        if not opt.keep_work_files:
            shutil.rmtree(work_dir, ignore_errors=True)
        self._progress(1.0, "Finished")
        return str(out_path)

    # ------------------------------------------------------------------
    def _cache_key(self, voice: str, ch: Chapter | None) -> str:
        opt = self.opt
        if ch is None:
            payload = "credits" + self.book.title + self.book.subtitle + self.book.author
        else:
            payload = ch.title + "\x1e".join(b.kind + ":" + b.text + (f"#{b.numerals}" if b.numerals else "")
                                              for b in ch.blocks)
        return hashlib.sha1((opt.cache_key() + voice + opt.bitrate + payload).encode("utf-8")).hexdigest()[:12]

    def _encode_job(self, idx: int, title: str, wav: Path, m4a: Path, dur: float) -> None:
        t0 = time.time()
        encode_chapter_aac(wav, m4a, self.opt.bitrate)
        spent = time.time() - t0
        with self._encode_lock:
            self._encoded += 1
            k, n = self._encoded, self._encode_total
        self._log(f"    encoded '{title[:40]}' to AAC in {spent:.1f}s ({dur / max(spent, 0.01):.0f}x)  [{k}/{n}]")
        self._chapter_cb(idx, "done")
        if not self.opt.keep_work_files:
            try:
                wav.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    def _synth(self, text: str, voice: str) -> np.ndarray:
        """Synthesize one chunk with graceful degradation on failure."""
        opt = self.opt
        try:
            a = self.engine.synthesize(text, voice, opt.lang, opt.speed)
        except EngineError:
            raise  # wrong URL/key/model/voice: smaller pieces would fail too, and the book would come out silent
        except Exception as e:  # e.g. OOM or G2P hiccup: split and retry
            self._log(f"    warning: synthesis failed ({type(e).__name__}: {str(e)[:80]}), retrying in halves")
            if len(text) < 40:
                return np.zeros(0, dtype=np.float32)
            mid = text.rfind(" ", 0, len(text) // 2 + 20)
            mid = mid if mid > 0 else len(text) // 2
            left = self._synth(text[:mid].strip(), voice)
            right = self._synth(text[mid:].strip(), voice)
            return np.concatenate([left, silence(opt.pause_chunk), right]) if left.size or right.size else left
        return fade_edges(trim_silence(a))

    def _speak(self, steps: list[Step], voice: str, label: str) -> list[Optional[np.ndarray]]:
        """Audio for each step, in order (None for pauses). Server engines get up to engine.max_parallel chunks
        at once; the local engine (max_parallel 1) runs them one at a time in this thread."""
        par = self.engine.max_parallel
        pool = ThreadPoolExecutor(max_workers=par, thread_name_prefix="tts") if par > 1 else None
        try:
            pending = [pool.submit(self._synth, text, voice) if pool and text is not None else None
                       for text, _, _ in steps]
            out: list[Optional[np.ndarray]] = []
            spoken = 0
            for (text, _, chars), fut in zip(steps, pending):
                self._check_cancel()
                if text is None:
                    out.append(None)
                else:
                    out.append(self._wait(fut) if fut else self._synth(text, voice))
                    spoken += 1
                self.done_chars += chars
                if text is not None and spoken % 4 == 0:
                    self._report(label)
            return out
        finally:
            if pool:
                pool.shutdown(wait=False, cancel_futures=True)

    def _wait(self, fut: Future) -> np.ndarray:
        while True:
            self._check_cancel()
            try:
                return fut.result(timeout=0.5)
            except FutureTimeout:
                pass

    def _assemble(self, steps: list[Step], audios: list[Optional[np.ndarray]], parts: list[np.ndarray]) -> None:
        for (text, pause, _), a in zip(steps, audios):
            if text is None:
                if pause:
                    parts.append(silence(pause))
            elif a.size:
                parts += [a, silence(pause)]

    def _synthesize_credits(self, voice: str) -> np.ndarray:
        """Opening credits like a professional audiobook: title, subtitle, 'written by' authors."""
        lang, sf = self.opt.lang, self.opt.speech_friendly
        lines = [normalize_text(self.book.title, lang, is_heading=True, speech_friendly=sf)]
        if self.book.subtitle:
            lines.append(normalize_text(self.book.subtitle, lang, is_heading=True, speech_friendly=sf))
        if self.book.authors:
            lines.append(credits_author_line(self.book.authors, lang))
        steps = [(line, 1.0 if i < len(lines) - 1 else 1.8, 0) for i, line in enumerate(lines)]
        parts: list[np.ndarray] = [silence(0.8)]
        self._assemble(steps, self._speak(steps, voice, "Opening credits"), parts)
        return np.concatenate(parts)

    def _synthesize_chapter(self, ch: Chapter, voice: str, n: int, total: int) -> np.ndarray:
        opt = self.opt
        blocks = ch.blocks
        label = f"Chapter {n}/{total}: {ch.title[:40]}"
        first_is_heading = bool(blocks) and blocks[0].kind == "heading"
        steps: list[Step] = []
        if opt.read_titles and not first_is_heading and ch.title:
            steps.append((normalize_text(ch.title, opt.lang, is_heading=True, speech_friendly=opt.speech_friendly),
                          opt.pause_heading, len(ch.title)))
        for b in blocks:
            if b.kind == "break":
                steps.append((None, opt.pause_break, 0))
                continue
            text = normalize_text(b.text, opt.lang, is_heading=(b.kind == "heading"),
                                  speech_friendly=opt.speech_friendly, numerals=b.numerals)
            if not text:
                steps.append((None, 0.0, len(b.text)))
            elif b.kind == "heading":
                steps.append((text, opt.pause_heading, len(b.text)))
            else:
                chunks = chunk_paragraph(text, opt.lang, opt.max_chars)
                steps += [(c, opt.pause_chunk if ci < len(chunks) - 1 else opt.pause_para, len(c))
                          for ci, c in enumerate(chunks)]
        parts: list[np.ndarray] = [silence(opt.pause_chapter_start)]
        self._assemble(steps, self._speak(steps, voice, label), parts)
        parts.append(silence(opt.pause_chapter_end))
        return np.concatenate(parts)


def _duration_of(m4a: Path, wav: Path) -> float:
    """Duration of a cached chapter: from its WAV if still there, else via ffprobe."""
    try:
        if wav.exists():
            from .audio import wav_duration
            return wav_duration(wav)
    except Exception:
        pass
    try:
        import subprocess
        ffprobe = str(Path(find_ffmpeg() or "ffmpeg").with_name("ffprobe" + (".exe" if os.name == "nt" else "")))
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(m4a)],
                             capture_output=True, text=True, creationflags=0x08000000 if os.name == "nt" else 0)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def preview_audio(book_lang: str, voice: str, text: str, speed: float = 1.0, engine: Optional[TTSEngine] = None,
                  max_chars: int = 350, speech_friendly: bool = True) -> np.ndarray:
    """Synthesize a short sample (used by the GUI preview)."""
    engine = engine or create_engine(EngineConfig())
    parts: list[np.ndarray] = []
    for chunk in chunk_paragraph(normalize_text(text, book_lang, speech_friendly=speech_friendly), book_lang,
                                 max_chars)[:3]:
        a = engine.synthesize(chunk, voice, book_lang, speed)
        if a.size:
            parts += [fade_edges(trim_silence(a)), silence(0.15)]
    if not parts:
        return silence(0.5)
    return normalize_level(np.concatenate(parts))
