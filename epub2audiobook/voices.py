"""Language and voice catalogue for the local models (Kokoro-82M v1.0 and v1.1-zh weights)."""
from __future__ import annotations

# Local models offered by the model picker: key -> (Hugging Face repo, label).
# Keys carry the model family, so other families (Piper, XTTS, ...) can be added next to Kokoro later.
MODELS = {
    "kokoro-v1.0": ("hexgrad/Kokoro-82M", "Kokoro-82M v1.0 (9 languages)"),
    "kokoro-v1.1-zh": ("hexgrad/Kokoro-82M-v1.1-zh", "Kokoro-82M v1.1-zh (Mandarin, English)"),
}
DEFAULT_MODEL = "kokoro-v1.0"

# Kokoro language codes -> human names
LANGUAGES = {
    "a": "English (US)",
    "b": "English (UK)",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "p": "Portuguese (Brazil)",
    "j": "Japanese",
    "z": "Chinese (Mandarin)",
}

# ISO-639 (2/3 letter, optional region) -> Kokoro code
_ISO_TO_KOKORO = {
    "en": "a", "eng": "a", "en-us": "a", "en-gb": "b", "en-uk": "b", "en-au": "b", "en-ie": "b", "en-nz": "b",
    "es": "e", "spa": "e",
    "fr": "f", "fra": "f", "fre": "f",
    "hi": "h", "hin": "h",
    "it": "i", "ita": "i",
    "pt": "p", "por": "p", "pt-br": "p", "pt-pt": "p",
    "ja": "j", "jpn": "j",
    "zh": "z", "zho": "z", "chi": "z", "zh-cn": "z", "zh-tw": "z", "cmn": "z",
}

# Voices per language. (id, label). Ordered best-first (Kokoro README grades).
VOICES = {
    "a": [
        ("af_heart", "Heart (female, grade A)"),
        ("af_bella", "Bella (female, A-)"),
        ("af_nicole", "Nicole (female, B-, whispery)"),
        ("af_aoede", "Aoede (female, C+)"),
        ("af_kore", "Kore (female, C+)"),
        ("af_sarah", "Sarah (female, C+)"),
        ("af_nova", "Nova (female, C)"),
        ("af_sky", "Sky (female, C-)"),
        ("af_alloy", "Alloy (female, C)"),
        ("af_jessica", "Jessica (female, D)"),
        ("af_river", "River (female, D)"),
        ("am_michael", "Michael (male, C+)"),
        ("am_fenrir", "Fenrir (male, C+)"),
        ("am_puck", "Puck (male, C+)"),
        ("am_echo", "Echo (male, D)"),
        ("am_eric", "Eric (male, D)"),
        ("am_liam", "Liam (male, D)"),
        ("am_onyx", "Onyx (male, D)"),
        ("am_adam", "Adam (male, F+)"),
        ("am_santa", "Santa (male, D-)"),
    ],
    "b": [
        ("bf_emma", "Emma (female, B-)"),
        ("bf_isabella", "Isabella (female, C)"),
        ("bf_alice", "Alice (female, D)"),
        ("bf_lily", "Lily (female, D)"),
        ("bm_george", "George (male, C)"),
        ("bm_fable", "Fable (male, C)"),
        ("bm_lewis", "Lewis (male, D+)"),
        ("bm_daniel", "Daniel (male, D)"),
    ],
    "e": [
        ("ef_dora", "Dora (female)"),
        ("em_alex", "Alex (male)"),
        ("em_santa", "Santa (male)"),
    ],
    "f": [
        ("ff_siwis", "Siwis (female, B-)"),
    ],
    "h": [
        ("hf_alpha", "Alpha (female, C)"),
        ("hf_beta", "Beta (female, C)"),
        ("hm_omega", "Omega (male, C)"),
        ("hm_psi", "Psi (male, C)"),
    ],
    "i": [
        ("if_sara", "Sara (female, C)"),
        ("im_nicola", "Nicola (male, C)"),
    ],
    "p": [
        ("pf_dora", "Dora (female)"),
        ("pm_alex", "Alex (male)"),
        ("pm_santa", "Santa (male)"),
    ],
    "j": [
        ("jf_alpha", "Alpha (female, C+)"),
        ("jf_gongitsune", "Gongitsune (female, C)"),
        ("jf_tebukuro", "Tebukuro (female, C)"),
        ("jf_nezumi", "Nezumi (female, C-)"),
        ("jm_kumo", "Kumo (male, C-)"),
    ],
    "z": [
        ("zf_xiaoxiao", "Xiaoxiao (female, D)"),
        ("zf_xiaobei", "Xiaobei (female, D)"),
        ("zf_xiaoni", "Xiaoni (female, D)"),
        ("zf_xiaoyi", "Xiaoyi (female, D)"),
        ("zm_yunyang", "Yunyang (male, D)"),
        ("zm_yunjian", "Yunjian (male, D)"),
        ("zm_yunxi", "Yunxi (male, D)"),
        ("zm_yunxia", "Yunxia (male, D)"),
    ],
}


# Kokoro-82M v1.1-zh: 100 numbered Mandarin voices plus three English ones (no quality grades published).
_V11_ZF = ("001 002 003 004 005 006 007 008 017 018 019 021 022 023 024 026 027 028 032 036 038 039 040 042 043 044 "
           "046 047 048 049 051 059 060 067 070 071 072 073 074 075 076 077 078 079 083 084 085 086 087 088 090 092 "
           "093 094 099").split()
_V11_ZM = ("009 010 011 012 013 014 015 016 020 025 029 030 031 033 034 035 037 041 045 050 052 053 054 055 056 057 "
           "058 061 062 063 064 065 066 068 069 080 081 082 089 091 095 096 097 098 100").split()
VOICES_V11_ZH = {
    "a": [("af_maple", "Maple (female)"), ("af_sol", "Sol (female)")],
    "b": [("bf_vale", "Vale (female)")],
    "z": [(f"zf_{n}", f"Female {n}") for n in _V11_ZF] + [(f"zm_{n}", f"Male {n}") for n in _V11_ZM],
}

_MODEL_VOICES = {"kokoro-v1.0": VOICES, "kokoro-v1.1-zh": VOICES_V11_ZH}


def voices_for(model: str) -> dict[str, list[tuple[str, str]]]:
    """Voices per language for a local model; its keys are the languages the model can narrate."""
    return _MODEL_VOICES.get(model, VOICES)


def default_voice(lang_code: str, model: str = DEFAULT_MODEL) -> str:
    voices = voices_for(model)
    return voices.get(lang_code, next(iter(voices.values())))[0][0]


def iso_to_kokoro(iso: str | None) -> str | None:
    """Map an ISO language tag (e.g. 'en-US', 'ita', 'pt-BR') to a Kokoro code, or None."""
    if not iso:
        return None
    tag = iso.strip().lower().replace("_", "-")
    if tag in _ISO_TO_KOKORO:
        return _ISO_TO_KOKORO[tag]
    base = tag.split("-")[0]
    return _ISO_TO_KOKORO.get(base)


def voice_lang(voice_id: str) -> str:
    """Kokoro language code implied by a voice id (first letter)."""
    return voice_id[0] if voice_id else "a"
