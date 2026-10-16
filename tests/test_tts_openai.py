"""OpenAICompatibleEngine against a small fake /v1/audio/speech server."""
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from epub2audiobook.tts import SERVER_PRESETS, EngineConfig, EngineError, create_engine
from epub2audiobook.tts_openai import OpenAICompatibleEngine, resample


def audio_bytes(sr, fmt="wav", seconds=0.5, freq=220.0):
    t = np.arange(int(sr * seconds)) / sr
    buf = io.BytesIO()
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(buf, tone, sr, format="WAV", subtype="PCM_16") if fmt == "wav" else sf.write(buf, tone, sr, format="MP3")
    return buf.getvalue()


# OpenRouter's answer to response_format "wav" (2026-10)
ZOD_FORMAT_ERROR = {"success": False, "error": {"name": "ZodError", "message": json.dumps(
    [{"code": "invalid_value", "values": ["mp3", "pcm"], "path": ["response_format"],
      "message": 'Invalid option: expected one of "mp3"|"pcm"'}], indent=2)}}


@pytest.fixture
def server():
    """state["script"]: HTTP statuses to answer, in order (then 200); state["requests"]: what was received;
    state["formats"]: the response formats the server accepts."""
    state = {"script": [], "requests": [], "sr": 24000, "formats": {"wav", "mp3"}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, dict(self.headers), body))
            status = state["script"].pop(0) if state["script"] else 200
            fmt = body.get("response_format")
            if status == 200 and fmt not in state["formats"]:
                status, data, ctype = 400, json.dumps(ZOD_FORMAT_ERROR).encode(), "application/json"
            elif status == 200:
                data, ctype = audio_bytes(state["sr"], fmt), f"audio/{fmt}"
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


def test_default_speed_is_not_sent(server):
    engine(server["url"]).synthesize("Hi.", "af_heart", "a", 1.0)
    assert "speed" not in server["requests"][0][2]


def test_mp3_is_requested_and_decoded(server):
    audio = engine(server["url"], audio_format="mp3").synthesize("Hi.", "af_heart", "a")
    assert server["requests"][0][2]["response_format"] == "mp3"
    assert abs(audio.size - 12000) < 600  # mp3 frames may pad the end a little
    peak_hz = np.argmax(np.abs(np.fft.rfft(audio))) * 24000 / audio.size
    assert abs(peak_hz - 220) < 5


def test_rejected_format_names_the_setting_to_change(server):
    server["formats"] = {"mp3"}
    with pytest.raises(EngineError, match="does not send wav audio: set the audio format to mp3") as err:
        engine(server["url"]).synthesize("Hi.", "af_heart", "a")
    assert "\n" not in str(err.value)  # the pretty-printed JSON message is flattened
    assert len(server["requests"]) == 1


def test_missing_key_is_named_in_the_error(server):
    server["script"] = [401]
    with pytest.raises(EngineError, match=r"No API key was sent: \$OPENROUTER_API_KEY is not set"):
        engine(server["url"], api_key_env="OPENROUTER_API_KEY").synthesize("Hi.", "af_heart", "a")


def test_wrong_key_has_no_missing_key_hint(server):
    server["script"] = [401]
    with pytest.raises(EngineError) as err:
        engine(server["url"], api_key="sk-wrong", api_key_env="OPENROUTER_API_KEY").synthesize("Hi.", "af_heart", "a")
    assert "No API key was sent" not in str(err.value)


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


def test_openrouter_preset_fills_only_what_is_not_given():
    cfg = EngineConfig.for_provider("openrouter", base_url=None, api_model=None, api_key_env=None, api_format=None)
    assert (cfg.base_url, cfg.api_model, cfg.api_key_env, cfg.api_format) == (
        "https://openrouter.ai/api/v1", "hexgrad/kokoro-82m", "OPENROUTER_API_KEY", "mp3")
    cfg = EngineConfig.for_provider("openrouter", api_model="google/gemini-3.8-flash-tts", concurrency=16)
    assert cfg.api_model == "google/gemini-3.8-flash-tts" and cfg.concurrency == 16 and cfg.api_format == "mp3"
    assert EngineConfig.for_provider("local", base_url=None).is_server is False


def test_every_server_provider_creates_the_openai_engine(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    eng = create_engine(EngineConfig.for_provider("openrouter"))
    assert isinstance(eng, OpenAICompatibleEngine)
    assert (eng.base_url, eng.model, eng.audio_format) == ("https://openrouter.ai/api/v1", "hexgrad/kokoro-82m", "mp3")
    assert set(SERVER_PRESETS) == {"openai", "openrouter"}
