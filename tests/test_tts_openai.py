"""OpenAICompatibleEngine against a small fake /v1/audio/speech server."""
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from epub2audiobook.tts import EngineConfig, EngineError, create_engine
from epub2audiobook.tts_openai import OpenAICompatibleEngine, resample


def wav_bytes(sr, seconds=0.5, freq=220.0):
    t = np.arange(int(sr * seconds)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def server():
    """state["script"]: HTTP statuses to answer, in order (then 200); state["requests"]: what was received."""
    state = {"script": [], "requests": [], "sr": 24000}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, dict(self.headers), body))
            status = state["script"].pop(0) if state["script"] else 200
            if status == 200:
                data, ctype = wav_bytes(state["sr"]), "audio/wav"
            else:
                data, ctype = json.dumps({"error": {"message": f"fake error {status}"}}).encode(), "application/json"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if status == 429:
                self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}/v1"
    yield state
    srv.shutdown()


def engine(url, **kw):
    kw.setdefault("backoff", 0.0)
    return OpenAICompatibleEngine(url, **kw)


def test_sends_a_standard_openai_request_and_decodes_the_wav(server):
    audio = engine(server["url"], model="kokoro", api_key="sk-test").synthesize("Hello there.", "af_heart", "a", 1.1)
    assert audio.dtype == np.float32
    assert audio.size == 12000  # 0.5 s at 24 kHz
    path, headers, body = server["requests"][0]
    assert path == "/v1/audio/speech"
    assert body == {"model": "kokoro", "input": "Hello there.", "voice": "af_heart", "speed": 1.1,
                    "response_format": "wav"}
    assert headers["Authorization"] == "Bearer sk-test"


def test_no_api_key_means_no_authorization_header(server):
    engine(server["url"]).synthesize("Hi.", "af_heart", "a")
    assert "Authorization" not in server["requests"][0][1]


def test_empty_text_sends_nothing(server):
    assert engine(server["url"]).synthesize("   ", "af_heart", "a").size == 0
    assert server["requests"] == []


def test_busy_server_is_retried(server):
    server["script"] = [429, 503]
    assert engine(server["url"]).synthesize("Hi.", "af_heart", "a").size == 12000
    assert len(server["requests"]) == 3


def test_bad_key_is_fatal_and_not_retried(server):
    server["script"] = [401]
    with pytest.raises(EngineError, match="401.*fake error 401"):
        engine(server["url"]).synthesize("Hi.", "af_heart", "a")
    assert len(server["requests"]) == 1


def test_persistent_server_error_lets_the_converter_split_the_text(server):
    server["script"] = [500, 500, 500]
    with pytest.raises(RuntimeError) as err:
        engine(server["url"], retries=2).synthesize("Hi.", "af_heart", "a")
    assert not isinstance(err.value, EngineError)
    assert len(server["requests"]) == 3


def test_unreachable_server_is_fatal():
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(EngineError, match="Cannot reach"):
        engine(f"http://127.0.0.1:{port}/v1", retries=1).synthesize("Hi.", "af_heart", "a")


def test_other_sample_rates_are_resampled_to_24k(server):
    server["sr"] = 48000
    audio = engine(server["url"]).synthesize("Hi.", "af_heart", "a")
    assert audio.size == 12000
    peak_hz = np.argmax(np.abs(np.fft.rfft(audio))) * 24000 / audio.size
    assert abs(peak_hz - 220) < 3


def test_resample_upsamples_too():
    assert resample(np.zeros(16000, dtype=np.float32), 16000).size == 24000


def test_missing_url_is_reported_before_any_request():
    with pytest.raises(EngineError, match="--base-url"):
        create_engine(EngineConfig(provider="openai", base_url=""))


def test_api_key_comes_from_the_named_environment_variable(server, monkeypatch):
    monkeypatch.setenv("MY_TTS_KEY", "sk-env")
    create_engine(EngineConfig(provider="openai", base_url=server["url"], api_key_env="MY_TTS_KEY")) \
        .synthesize("Hi.", "af_heart", "a")
    assert server["requests"][0][1]["Authorization"] == "Bearer sk-env"
