"""Speech engines: the interface the converter uses, engine settings, and the local Kokoro-82M engine.

The OpenAI-compatible server engine lives in tts_openai.py; create_engine() picks one from an EngineConfig.
"""
from __future__ import annotations

import logging
import os
import threading
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .voices import DEFAULT_MODEL, LANGUAGES, MODELS

SAMPLE_RATE = 24000
REPO_ID = MODELS[DEFAULT_MODEL][0]

log = logging.getLogger(__name__)
_default_threads: Optional[int] = None  # PyTorch's own CPU thread count, before any engine changed it


class EngineError(RuntimeError):
    """A setup or connection problem (wrong URL, API key, model or voice). It stops the conversion, because
    retrying smaller pieces of text (what the converter does for other synthesis errors) cannot fix it."""


class TTSEngine:
    """What the converter needs from a speech engine."""

    max_parallel = 1  # chunks the converter may synthesize at the same time

    def describe(self) -> str:
        """One line for the log, e.g. 'Kokoro-82M on CPU'."""
        raise NotImplementedError

    def warm_up(self, lang: str, voice: str) -> None:
        pass

    def synthesize(self, text: str, voice: str, lang: str, speed: float = 1.0) -> np.ndarray:
        """Return float32 mono audio at SAMPLE_RATE for `text` (empty array if nothing was produced)."""
        raise NotImplementedError

    def release(self) -> None:
        pass


# Server providers and their defaults. All of them use the OpenAI speech API (tts_openai.py); a preset only
# fills in the settings that the user does not give.
SERVER_PRESETS = {
    "openai": {"base_url": "", "api_model": "kokoro", "api_key_env": "OPENAI_API_KEY", "api_format": "wav"},
    # OpenRouter sends only mp3 or pcm, and pcm does not tell the sample rate
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_model": "hexgrad/kokoro-82m",
                   "api_key_env": "OPENROUTER_API_KEY", "api_format": "mp3"},
}
API_FORMATS = ("wav", "mp3")


@dataclass(frozen=True)
class EngineConfig:
    """Which speech engine to use and how; the CLI and the GUI build one, create_engine() turns it into an engine."""
    provider: str = "local"              # "local": Kokoro in this process; a SERVER_PRESETS key: a TTS server
    model: str = DEFAULT_MODEL           # local: key of voices.MODELS
    device: str = "auto"                 # local: auto | cuda | cpu
    threads: int = 0                     # local: PyTorch CPU threads, 0 = PyTorch default (one per physical core)
    max_vram_gb: float = 0.0             # local, CUDA: cap on PyTorch's GPU memory, 0 = no cap
    base_url: str = ""                   # server: e.g. http://localhost:8880/v1
    api_model: str = "kokoro"            # server: model name the server expects
    api_key_env: str = "OPENAI_API_KEY"  # server: environment variable that holds the API key (never stored)
    api_format: str = "wav"              # server: audio format to ask for, one of API_FORMATS
    concurrency: int = 4                 # server: parallel requests

    @classmethod
    def for_provider(cls, provider: str, **settings) -> EngineConfig:
        """Config for `provider`: the settings that are not None, the provider's preset for the others."""
        given = {k: v for k, v in settings.items() if v is not None}
        return cls(provider=provider, **{**SERVER_PRESETS.get(provider, {}), **given})

    @property
    def is_server(self) -> bool:
        return self.provider in SERVER_PRESETS

    def audio_identity(self) -> str:
        """The settings that change the sound, for the chapter cache key ('' for the default local model, so
        caches made before engines were configurable stay valid)."""
        if self.is_server:
            return f"openai|{self.base_url.rstrip('/')}|{self.api_model}|{self.api_format}"
        return "" if self.model == DEFAULT_MODEL else self.model


def create_engine(cfg: EngineConfig) -> TTSEngine:
    if cfg.is_server:
        from .tts_openai import OpenAICompatibleEngine
        return OpenAICompatibleEngine(cfg.base_url, cfg.api_model, os.environ.get(cfg.api_key_env, ""), cfg.concurrency,
                                      audio_format=cfg.api_format, api_key_env=cfg.api_key_env)
    if cfg.provider != "local":
        raise EngineError(f"Unknown speech engine provider: {cfg.provider!r}")
    if cfg.model not in MODELS:
        raise EngineError(f"Unknown local model {cfg.model!r} (available: {', '.join(MODELS)})")
    return KokoroEngine(cfg.device, MODELS[cfg.model][0], threads=cfg.threads, max_vram_gb=cfg.max_vram_gb)


def resolve_device(device: Optional[str]) -> str:
    if device and device.lower() not in {"auto", ""}:
        return device.lower()
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def device_description() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "CPU"


class KokoroEngine(TTSEngine):
    """Thread-safe (serialised) text-to-speech using Kokoro-82M.

    The model (~330 MB) and the voices are downloaded from Hugging Face on first use and cached
    in the standard HF cache directory. espeak-ng (needed for non-English languages and for
    out-of-dictionary English words) is provided by the `espeakng-loader` wheel.
    """

    def __init__(self, device: Optional[str] = "auto", repo_id: str = REPO_ID, threads: int = 0,
                 max_vram_gb: float = 0.0):
        self.device = resolve_device(device)
        self.repo_id = repo_id
        self.threads = threads
        self.max_vram_gb = max_vram_gb
        self._model = None
        self._pipes: dict[str, object] = {}
        self._lock = threading.Lock()

    def describe(self) -> str:
        where = device_description() if self.device == "cuda" else self.device.upper()
        extras = [f"{self.threads} CPU threads" if self.threads > 0 else "",
                  f"max {self.max_vram_gb:g} GB VRAM" if self.max_vram_gb > 0 and self.device == "cuda" else ""]
        return f"{self.repo_id.split('/')[-1]} on {where}" + "".join(f", {e}" for e in extras if e)

    def _apply_limits(self) -> None:
        """CPU threads and the VRAM cap; called once, right before the model loads. Both are process-wide, so a
        0 ("default"/"no cap") restores the defaults an earlier engine in this process may have changed."""
        global _default_threads
        import torch
        if _default_threads is None:
            _default_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads if self.threads > 0 else _default_threads)
        if self.device == "cuda" and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            fraction = min(1.0, self.max_vram_gb * 2**30 / total) if self.max_vram_gb > 0 else 1.0
            torch.cuda.set_per_process_memory_fraction(fraction, 0)

    # ------------------------------------------------------------------
    def pipeline(self, lang: str):
        if lang not in LANGUAGES:
            raise ValueError(f"Unsupported Kokoro language code: {lang!r}")
        with self._lock:
            if lang in self._pipes:
                return self._pipes[lang]
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            # huggingface_hub nags about anonymous downloads; they work fine (set HF_TOKEN to silence/speed up)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            for name in ("huggingface_hub.utils._http", "huggingface_hub.file_download"):
                logging.getLogger(name).setLevel(logging.ERROR)
            from kokoro import KPipeline  # heavy import, deferred
            if self._model is None:
                self._apply_limits()
                pipe = KPipeline(lang_code=lang, repo_id=self.repo_id, device=self.device)
                self._model = pipe.model
            else:
                pipe = KPipeline(lang_code=lang, repo_id=self.repo_id, model=self._model, device=self.device)
            self._pipes[lang] = pipe
            return pipe

    def warm_up(self, lang: str, voice: str) -> None:
        pipe = self.pipeline(lang)
        try:
            pipe.load_voice(voice)
        except Exception as e:  # pragma: no cover
            log.warning("could not preload voice %s: %s", voice, e)

    # ------------------------------------------------------------------
    def synthesize(self, text: str, voice: str, lang: str, speed: float = 1.0) -> np.ndarray:
        """Return float32 mono audio at 24 kHz for `text` (empty array if nothing was produced)."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        pipe = self.pipeline(lang)
        parts: list[np.ndarray] = []
        with self._lock:
            for result in pipe(text, voice=voice, speed=speed, split_pattern=None):
                audio = getattr(result, "audio", None)
                if audio is None:
                    continue
                try:
                    arr = audio.detach().cpu().numpy()
                except AttributeError:
                    arr = np.asarray(audio)
                arr = arr.astype(np.float32).reshape(-1)
                if arr.size:
                    parts.append(arr)
        if not parts:
            return np.zeros(0, dtype=np.float32)
        if len(parts) == 1:
            return parts[0]
        gap = np.zeros(int(0.08 * SAMPLE_RATE), dtype=np.float32)
        joined: list[np.ndarray] = []
        for i, p in enumerate(parts):
            if i:
                joined.append(gap)
            joined.append(p)
        return np.concatenate(joined)

    def release(self) -> None:
        with self._lock:
            self._pipes.clear()
            self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
