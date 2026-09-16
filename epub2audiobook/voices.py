"""Language and voice catalogue for Kokoro-82M (v1.0 weights)."""
from __future__ import annotations

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


def default_voice(lang_code: str) -> str:
    return VOICES.get(lang_code, VOICES["a"])[0][0]


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
