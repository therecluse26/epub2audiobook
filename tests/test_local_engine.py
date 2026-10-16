"""Local engine settings: model picker, CPU threads, VRAM cap (torch is faked; no GPU is touched)."""
import pytest
import torch

from epub2audiobook import tts
from epub2audiobook.tts import EngineConfig, EngineError, KokoroEngine, create_engine
from epub2audiobook.voices import DEFAULT_MODEL, MODELS, default_voice, voices_for


def test_model_picker_selects_the_kokoro_weights():
    assert create_engine(EngineConfig()).repo_id == "hexgrad/Kokoro-82M"
    assert create_engine(EngineConfig(model="kokoro-v1.1-zh")).repo_id == "hexgrad/Kokoro-82M-v1.1-zh"
    with pytest.raises(EngineError, match="Unknown local model"):
        create_engine(EngineConfig(model="piper-x"))


def test_each_model_has_its_own_languages_and_voices():
    assert set(voices_for(DEFAULT_MODEL)) == set("abefhipjz")
    assert set(voices_for("kokoro-v1.1-zh")) == {"a", "b", "z"}
    assert len(voices_for("kokoro-v1.1-zh")["z"]) == 100
    assert default_voice("a") == "af_heart"
    assert default_voice("a", "kokoro-v1.1-zh") == "af_maple"
    assert all(m in MODELS for m in ("kokoro-v1.0", "kokoro-v1.1-zh"))


@pytest.fixture
def fake_torch(monkeypatch):
    calls = {}
    monkeypatch.setattr(tts, "_default_threads", None)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.__setitem__("threads", n))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: type("Props", (), {"total_memory": 8 * 2**30}))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: calls.__setitem__("vram", (f, d)))
    return calls


def test_threads_and_vram_cap_are_applied(fake_torch):
    KokoroEngine("cuda", threads=4, max_vram_gb=2)._apply_limits()
    assert fake_torch == {"threads": 4, "vram": (0.25, 0)}


def test_defaults_undo_what_an_earlier_engine_changed(fake_torch):
    KokoroEngine("cuda", threads=4, max_vram_gb=2)._apply_limits()
    KokoroEngine("cuda")._apply_limits()
    assert fake_torch == {"threads": 8, "vram": (1.0, 0)}


def test_vram_cap_is_ignored_on_cpu(fake_torch):
    KokoroEngine("cpu", max_vram_gb=2)._apply_limits()
    assert "vram" not in fake_torch


def test_describe_names_the_model_and_limits():
    assert KokoroEngine("cpu", threads=6).describe() == "Kokoro-82M on CPU, 6 CPU threads"
    assert KokoroEngine("cpu", "hexgrad/Kokoro-82M-v1.1-zh").describe() == "Kokoro-82M-v1.1-zh on CPU"
