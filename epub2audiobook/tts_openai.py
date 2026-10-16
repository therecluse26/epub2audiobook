"""Speech from any server that implements OpenAI's POST /v1/audio/speech: a local Kokoro-FastAPI, OpenAI itself,
OpenRouter, or a hosted Kokoro API. The converter sends up to `concurrency` chunks at once, so a fast server
narrates a book in minutes. Only the standard request fields are sent (model, input, voice, speed, response_format).
"""
from __future__ import annotations

import http.client
import io
import json
import random
import time
import urllib.error
import urllib.request
from typing import Optional

import numpy as np
import soundfile as sf

from .tts import SAMPLE_RATE, EngineError, TTSEngine

RETRY_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}  # busy or temporarily broken: wait and try again


class OpenAICompatibleEngine(TTSEngine):
    def __init__(self, base_url: str, model: str = "kokoro", api_key: str = "", concurrency: int = 4,
                 timeout: float = 120.0, retries: int = 5, backoff: float = 1.0, audio_format: str = "wav",
                 api_key_env: str = ""):
        if not base_url:
            raise EngineError("No server URL, e.g. http://localhost:8880/v1. Set it in the GUI (Server URL) "
                              "or on the command line (--base-url or $OPENAI_BASE_URL).")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_parallel = max(1, int(concurrency))
        self.audio_format = audio_format  # wav or mp3: both carry their sample rate (raw pcm does not)
        self._api_key = api_key
        self._api_key_env = api_key_env  # only to name it in error messages
        self._timeout = timeout
        self._retries = retries
        self._backoff = backoff  # seconds before the first retry; doubles each time

    def describe(self) -> str:
        return f"'{self.model}' at {self.base_url} ({self.audio_format}, {self.max_parallel} parallel requests)"

    def warm_up(self, lang: str, voice: str) -> None:
        # one tiny request, so a wrong URL, key, model or voice fails now and not at chapter 1
        self.synthesize("Ready.", voice, lang)

    def synthesize(self, text: str, voice: str, lang: str, speed: float = 1.0) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        request = {"model": self.model, "input": text, "voice": voice, "response_format": self.audio_format}
        if speed != 1.0:  # 1.0 is the API default; some models have no speed control and reject the field
            request["speed"] = speed
        body = json.dumps(request).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = self.base_url + "/audio/speech"
        for attempt in range(self._retries + 1):
            last = attempt == self._retries
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    raw = r.read()
                return _decode(raw)
            except urllib.error.HTTPError as e:
                detail = _error_detail(e)
                if e.code not in RETRY_HTTP:
                    raise EngineError(f"{url} answered HTTP {e.code}: {detail}{self._hint(e.code, detail)}") from None
                if last:  # the server keeps failing on this text: let the converter retry smaller pieces
                    raise RuntimeError(f"{url} answered HTTP {e.code} {self._retries + 1} times: {detail}") from None
                wait = _retry_after(e)
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
                if last:
                    reason = getattr(e, "reason", e)
                    raise EngineError(f"Cannot reach {url}: {reason}") from None
                wait = None
            except (sf.LibsndfileError, ValueError) as e:  # truncated or garbled audio
                if last:
                    raise RuntimeError(f"{url} returned audio that cannot be decoded: {e}") from None
                wait = None
            time.sleep(wait if wait is not None else self._backoff * 2 ** attempt * (0.5 + random.random()))
        raise AssertionError("unreachable")

    def _hint(self, code: int, detail: str) -> str:
        """What to change, for the setup errors whose server message does not say it."""
        if code in (401, 403) and not self._api_key:
            var = f"${self._api_key_env}" if self._api_key_env else "the API key variable"
            return (f". No API key was sent: {var} is not set in the environment that started this program, "
                    f"and no key is saved (API key field in the GUI).")
        if code in (400, 422) and "response_format" in detail:
            other = "mp3" if self.audio_format == "wav" else "wav"
            return (f". The server does not send {self.audio_format} audio: set the audio format to {other} "
                    f"(--api-format {other}, or Audio format in the GUI).")
        return ""


def _decode(raw: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    return resample(audio, sr) if sr != SAMPLE_RATE else audio.astype(np.float32)


def resample(audio: np.ndarray, sr: int) -> np.ndarray:
    """Band-limited (FFT) resampling to SAMPLE_RATE; chunks are short, so the whole-signal FFT is cheap."""
    n = int(round(audio.size * SAMPLE_RATE / sr))
    if audio.size == 0 or n == 0:
        return np.zeros(0, dtype=np.float32)
    spec = np.fft.rfft(audio)
    keep = n // 2 + 1
    spec = spec[:keep] if keep <= spec.size else np.pad(spec, (0, keep - spec.size))
    return (np.fft.irfft(spec, n) * (n / audio.size)).astype(np.float32)


def _error_detail(e: urllib.error.HTTPError) -> str:
    try:
        body = e.read().decode("utf-8", "replace")
    except Exception:
        return e.reason or ""
    try:
        err = json.loads(body)
    except ValueError:
        return body[:300]
    for key in ("error", "detail"):  # OpenAI: {"error": {"message": ...}}; FastAPI servers: {"detail": ...}
        if isinstance(err, dict) and key in err:
            err = err[key]
    msg = err.get("message", err) if isinstance(err, dict) else err
    return " ".join(str(msg).split())[:300]  # some servers send pretty-printed JSON as the message


def _retry_after(e: urllib.error.HTTPError) -> Optional[float]:
    try:
        return min(60.0, float(e.headers.get("Retry-After", "")))
    except (TypeError, ValueError):
        return None
