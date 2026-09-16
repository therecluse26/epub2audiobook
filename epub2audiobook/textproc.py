"""Text normalisation and sentence-aware chunking for TTS."""
from __future__ import annotations

import re
import unicodedata

CJK_LANGS = {"j", "z"}

# Common abbreviations that end with a period but don't end a sentence (multi-language, lower-case, no dot)
_ABBREVS = {
    # English
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "fig", "no", "vol", "ch", "pp",
    "ed", "eds", "cf", "inc", "ltd", "co", "corp", "gen", "col", "lt", "capt", "sgt", "rev", "hon", "pres",
    "u.s", "u.k", "u.s.a", "e.g", "i.e", "a.m", "p.m", "approx", "dept", "est", "misc", "govt", "ave", "blvd",
    # Italian
    "sig", "sig.ra", "sig.na", "dott", "dott.ssa", "ing", "avv", "prof.ssa", "ecc", "pag", "cap", "sec",
    "es", "n", "on", "gen", "geom", "rag",
    # Spanish / Portuguese
    "sra", "srta", "ud", "uds", "pág", "págs", "núm", "tel", "av", "ej", "excmo",
    # French
    "mme", "mlle", "mm", "m", "pp", "éd", "chap", "boul", "ex", "env",
}

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
              "Œ": "Oe", "œ": "oe" if False else "œ"}

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_RE = re.compile(r"^(?=[ivxlcdm]+$)m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.I)


def roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.lower()):
        v = _ROMAN[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def normalize_text(text: str, lang: str = "a", is_heading: bool = False) -> str:
    """Clean text for narration. Removes note markers, fixes typography, collapses whitespace."""
    t = unicodedata.normalize("NFC", text)
    for k, v in _LIGATURES.items():
        t = t.replace(k, v)
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    t = re.sub(r"[​‌‍⁠﻿­]", "", t)

    # bracketed note markers: [12] [note 3] [*] (12) after words
    t = re.sub(r"\s*\[\s*(?:note\s*)?\d{1,3}\s*\]", "", t, flags=re.I)
    t = re.sub(r"\s*\[\s*[\*†‡§¶]+\s*\]", "", t)
    t = re.sub(r"(?<=[A-Za-zÀ-ɏ.,;:!?\"”’)])\(\d{1,3}\)(?=\s|$)", "", t)
    t = re.sub(r"(?<=[.!?\"”’])\s\(\d{1,3}\)(?=\s|$)", "", t)
    # superscript digits glued to a word/punctuation: "recovered.12 The" -> "recovered. The"
    t = re.sub(r"(?<=[A-Za-zÀ-ɏ][.!?,;:\"”’)])\d{1,3}(?=\s|$)", "", t)
    t = re.sub(r"(?<=[A-Za-zÀ-ɏ])[¹²³⁰-⁹]+", "", t)
    t = re.sub(r"(?<=[A-Za-zÀ-ɏ\"”’])[*†‡§¶]{1,3}(?=[\s.,;:!?]|$)", "", t)

    # typography
    t = t.replace("…", "...")
    t = re.sub(r"\s*–\s*", " — ", t) if lang not in CJK_LANGS else t  # en-dash -> em-dash (spoken pause)
    t = re.sub(r"\s*—\s*", " — ", t)
    t = re.sub(r"(\d)\s*—\s*(\d)", r"\1 to \2", t) if lang in {"a", "b"} else t
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    t = t.replace("„", '"').replace("‚", "'")
    t = re.sub(r"\s*/\s*", " / ", t) if lang not in CJK_LANGS else t
    # URLs read more gracefully
    t = re.sub(r"https?://(www\.)?", "", t)
    t = re.sub(r"\bwww\.", "", t)

    if is_heading:
        t = _normalize_heading(t, lang)

    t = re.sub(r"\s+", " ", t).strip()
    # make sure a paragraph ends with terminal punctuation for a natural cadence
    if t and lang not in CJK_LANGS and not re.search(r"[.!?:;\"')\]…]$", t):
        t += "."
    return t


def _normalize_heading(t: str, lang: str) -> str:
    # Roman numerals as standalone tokens: "Part IV" -> "Part 4"; "IV. Title" -> "4. Title"
    def repl(m):
        s = m.group(0)
        try:
            n = roman_to_int(s)
        except Exception:
            return s
        return str(n) if 1 <= n <= 200 else s
    t = re.sub(r"(?<![A-Za-z])(?:M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))(?![A-Za-z])",
               lambda m: repl(m) if len(m.group(0)) >= 2 else m.group(0), t)
    # "CHAPTER ONE" -> "Chapter One" (all-caps headings are read letter by letter by some G2Ps)
    if len(t) > 3 and t.upper() == t and re.search(r"[A-Z]{3,}", t):
        t = t.title()
    return t


# --------------------------------------------------------------------------------------
# Sentence splitting & chunking
# --------------------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"([.!?]+[\"'”’»)\]]*)\s+(?=[\"'“‘«(\[]?[A-ZÀ-ÞЀ-Я0-9])")
_CJK_END_RE = re.compile(r"(?<=[。！？!?])")


def split_sentences(text: str, lang: str = "a") -> list[str]:
    if lang in CJK_LANGS:
        parts = [p for p in _CJK_END_RE.split(text) if p.strip()]
        return parts or [text]
    sents: list[str] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.end(1)
        before = text[start:end]
        # abbreviation check: last token before the period
        tok = re.findall(r"(\S+)[.!?]+[\"'”’»)\]]*$", before)
        if tok:
            w = tok[-1].lower().rstrip(".")
            if w in _ABBREVS or (len(w) == 1 and w.isalpha()) or re.fullmatch(r"[a-z]\.[a-z](\.[a-z])*", w):
                continue
            if re.fullmatch(r"\d+", w) and lang in {"e", "f", "i", "p"}:
                # "1." enumerations in romance languages (e.g. 'art. 5.') are ambiguous; keep splitting
                pass
        sents.append(before.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sents.append(tail)
    return sents or [text]


def _split_long(sentence: str, max_chars: int, lang: str) -> list[str]:
    """Split an over-long sentence on clause punctuation, then on spaces."""
    if len(sentence) <= max_chars:
        return [sentence]
    seps = ["; ", ": ", " — ", ", ", "，", "、", "；", "："] if lang in CJK_LANGS else ["; ", ": ", " — ", ", "]
    for sep in seps:
        if sep in sentence:
            parts = sentence.split(sep)
            out, cur = [], ""
            for i, p in enumerate(parts):
                piece = p + (sep.rstrip() if i < len(parts) - 1 else "")
                cand = (cur + " " + piece).strip() if cur else piece
                if len(cand) <= max_chars or not cur:
                    cur = cand
                else:
                    out.append(cur)
                    cur = piece
            if cur:
                out.append(cur)
            if all(len(x) <= max_chars for x in out):
                return out
            return [y for x in out for y in _split_long(x, max_chars, lang)]
    # hard split on spaces (or characters for CJK)
    if lang in CJK_LANGS:
        return [sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)]
    words = sentence.split(" ")
    out, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def chunk_paragraph(text: str, lang: str = "a", max_chars: int = 350) -> list[str]:
    """Split a paragraph into TTS-sized chunks at sentence boundaries."""
    if lang in CJK_LANGS:
        max_chars = min(max_chars, 120)
    sents = split_sentences(text, lang)
    pieces: list[str] = []
    for s in sents:
        pieces.extend(_split_long(s, max_chars, lang))
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        cand = (cur + " " + p).strip() if cur else p
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks
