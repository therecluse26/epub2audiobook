"""API keys: their own file with permission 600, never in the settings or cache key; which key wins."""
import stat
import sys

import pytest

from epub2audiobook import keys
from epub2audiobook.cli import with_saved_key
from epub2audiobook.converter import ConvertOptions
from epub2audiobook.tts import EngineConfig, create_engine


@pytest.fixture
def keys_file(tmp_path, monkeypatch):
    path = tmp_path / ".epub2audiobook-keys.json"
    monkeypatch.setattr(keys, "KEYS_FILE", path)
    return path


def test_keys_round_trip(keys_file):
    keys.save_keys({"openrouter": "sk-or-1", "openai": ""})
    assert keys.load_keys() == {"openrouter": "sk-or-1"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_key_file_is_readable_only_by_the_owner(keys_file):
    keys_file.write_text("{}")
    keys_file.chmod(0o644)  # an old file with wider rights
    keys.save_keys({"openrouter": "sk-or-1"})
    assert stat.S_IMODE(keys_file.stat().st_mode) == 0o600
    assert not keys_file.with_name(keys_file.name + ".tmp").exists()


def test_no_key_left_removes_the_file(keys_file):
    keys.save_keys({"openrouter": "sk-or-1"})
    keys.save_keys({"openrouter": ""})
    assert not keys_file.exists()
    keys.save_keys({})  # and no empty file is created
    assert not keys_file.exists()


def test_unreadable_file_means_no_keys(keys_file):
    keys_file.write_text("not json")
    assert keys.load_keys() == {}
    keys_file.write_text('["sk-or-1"]')
    assert keys.load_keys() == {}


def test_key_is_not_in_the_config_repr_or_the_cache_key():
    cfg = EngineConfig.for_provider("openrouter", api_key="sk-or-secret")
    opts = ConvertOptions(output_path="t.m4b", engine=cfg)
    for text in (repr(cfg), repr(opts), cfg.audio_identity(), opts.cache_key()):
        assert "sk-or-secret" not in text


def test_a_given_key_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    assert create_engine(EngineConfig.for_provider("openrouter", api_key="sk-field"))._api_key == "sk-field"
    assert create_engine(EngineConfig.for_provider("openrouter"))._api_key == "sk-env"


def test_command_line_uses_the_environment_then_the_saved_key(keys_file, monkeypatch):
    keys.save_keys({"openrouter": "sk-saved"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    cfg = with_saved_key(EngineConfig.for_provider("openrouter"))
    assert create_engine(cfg)._api_key == "sk-env"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert with_saved_key(EngineConfig.for_provider("openrouter")).api_key == "sk-saved"
    assert with_saved_key(EngineConfig.for_provider("openai")).api_key == ""  # keys are per provider
    assert with_saved_key(EngineConfig()).api_key == ""  # the local engine needs no key
