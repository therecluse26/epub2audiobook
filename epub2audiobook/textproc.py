"""Text normalisation and sentence-aware chunking for TTS."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

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


def roman_value(s: str) -> Optional[int]:
    """Value of a well-formed Roman numeral (either case), else None."""
    return roman_to_int(s) if _ROMAN_RE.match(s) else None


def normalize_text(text: str, lang: str = "a", is_heading: bool = False, speech_friendly: bool = False,
                   numerals: Sequence[tuple[int, int]] = ()) -> str:
    """Clean text for narration. Removes note markers, fixes typography, collapses whitespace.

    speech_friendly adds the rewrites of ConvertOptions.speech_friendly: Roman numerals in references
    become digits, markdown leftovers and decorative symbols go. `numerals` are spans of `text` that the
    parser knows are Roman numerals (Block.numerals); they are only used with speech_friendly.
    """
    t = text
    if speech_friendly:
        for start, end in sorted(numerals, reverse=True):
            t = t[:start] + NUM_OPEN + t[start:end] + NUM_CLOSE + t[end:]
    t = unicodedata.normalize("NFC", t)
    for k, v in _LIGATURES.items():
        t = t.replace(k, v)
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    t = re.sub(r"[​‌‍⁠﻿­]", "", t)
    if speech_friendly:
        t = _clean_symbols(t, lang)  # before the note markers: "**very**" must not lose only its closing stars

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

    if speech_friendly:  # after the note markers, which would delete a new "(2)" as a footnote reference
        t = _speak_roman_numerals(t)
    t = t.replace(NUM_OPEN, "").replace(NUM_CLOSE, "")

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
# Speech-friendly rewrites (ConvertOptions.speech_friendly)
# --------------------------------------------------------------------------------------

# The parser wraps Roman numerals it is sure about (publisher tags, numeral-only blocks) in these private-use
# characters while it builds a block; normalize_text wraps Block.numerals in them again. They never reach an engine.
NUM_OPEN, NUM_CLOSE = "\ue000", "\ue001"
_MARKED_RE = re.compile(f"{NUM_OPEN}([^{NUM_OPEN}{NUM_CLOSE}]*){NUM_CLOSE}")

# Words that announce a numbered division ("Part IV", "chapitre ix", "Sect. V."); English, Italian, French,
# Spanish and Portuguese in one list. "letter" and "appendix" are left out: "the letter X", "Appendix C".
_SECTION_WORDS = ("chapter|part|book|volume|vol|act|scene|section|sect|canto|stave|psalm|lecture|article"
                  "|capitolo|parte|libro|atto|scena|sezione"
                  "|chapitre|partie|livre|tome|acte|scène|chant"
                  "|capítulo|tomo|volumen|acto|escena|sección"
                  "|livro|ato|cena|secção|seção")
_SECTION_NUM_RE = re.compile(rf"\b((?i:{_SECTION_WORDS})\.?)(\s+)([IVXLCDM]+|[ivxlcdm]+)\b(?!['’]\w)")

# "Luke xiii. 3", "1 Cor. xiii. 4", "St. Luke XI. 9": a book of the Bible, a chapter numeral, the verse.
# The book name keeps "(v. 23)" (verse 23), "See v. 23" and "Rex v. Bailey" (versus) out; without it the numeral
# needs two letters or more ("(xiv. 20, 21)", "in xx. 31").
_BIBLE_BOOKS = sorted(
    "Genesis Gen Exodus Exod Ex Leviticus Lev Numbers Num Deuteronomy Deut Joshua Josh Judges Judg Ruth Samuel "
    "Sam Kings Kgs Chronicles Chron Chr Ezra Nehemiah Neh Esther Esth Job Psalms Psalm Ps Psa Proverbs Prov "
    "Ecclesiastes Eccles Eccl Song Sol Solomon Canticles Cant Isaiah Isa Jeremiah Jer Lamentations Lam Ezekiel "
    "Ezek Daniel Dan Hosea Hos Joel Amos Obadiah Obad Jonah Micah Mic Nahum Nah Habakkuk Hab Zephaniah Zeph "
    "Haggai Hag Zechariah Zech Malachi Mal Matthew Matt Mat Mark Mk Luke Lk John Jn Acts Romans Rom Corinthians "
    "Cor Galatians Gal Ephesians Eph Philippians Phil Colossians Col Thessalonians Thess Timothy Tim Titus Tit "
    "Philemon Philem Hebrews Heb James Jas Peter Pet Jude Revelation Rev".split(), key=len, reverse=True)
_BIBLE_REF_RE = re.compile(rf"\b((?:{'|'.join(_BIBLE_BOOKS)})\.?\s+)([ivxlc]{{1,8}}|[IVXLC]{{1,8}})\.\s?(\d{{1,3}})\b")
_CHAPTER_VERSE_RE = re.compile(r"(?<![\w.])()([ivxlc]{2,8})\.\s?(\d{1,3})\b")
# "II. The second objection": a numbered paragraph. Two letters or more; "I." and "V." may be the pronoun or initials.
_LEADING_NUM_RE = re.compile(r"^([IVXLC]{2,8})(\.\s)")

# "(ii) Jeanne Baudin"; not "d(i)spatch", and upper-case "(I)" may be the pronoun
_LIST_LABEL_RE = re.compile(r"(?<!\w)\(([ivx]{1,6})\)(?!\w)")

_MD_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\([^)\s]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]\n]{1,200})\]\([^)\s]{1,500}\)")
_MD_CODE_RE = re.compile(r"`([^`\n]{1,200})`")
_EMPHASIS_RES = (  # *word* **word** _word_ __word__; not "f*ck", "2*3*4" or snake_case
    re.compile(r"(?<![\w*])(\*\*?)(?=[^\s*])([^*\n]{1,300}?)(?<=[^\s*])\1(?![\w*])"),
    re.compile(r"(?<![\w_])(__?)(?=[^\s_])([^_\n]{1,300}?)(?<=[^\s_])\1(?![\w_])"),
)
_STAR_RUN_RE = re.compile(r"\*(?:\s*\*){2,}")       # "* * * * * *" inside a paragraph
_DECOR_RUN_RE = re.compile(r"={2,}|_{2,}|~{2,}|\+{3,}")
_NUMBER_WORD = {"a": "number", "b": "number", "e": "número", "f": "numéro", "i": "numero", "p": "número"}


def _clean_symbols(t: str, lang: str) -> str:
    """Markdown leftovers and decorative symbols, which speech engines read out ("number", "asterisk")."""
    t = _MD_IMAGE_RE.sub("", t)
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _MD_CODE_RE.sub(r"\1", t)
    for rx in _EMPHASIS_RES:
        t = rx.sub(r"\2", t)
    t = _STAR_RUN_RE.sub(" ... ", t)
    t = _DECOR_RUN_RE.sub(" ", t)
    t = re.sub(r"^\s*(?:#{1,6}|[*•])\s+", "", t)  # markdown heading or bullet marker
    word = _NUMBER_WORD.get(lang)
    t = re.sub(r"(?<!\w)#\s?(?=\d)", f"{word} " if word else "", t)  # "#5" -> "number 5"
    t = re.sub(r"(?<!\w)#+", " ", t)  # any other loose "#"; "C#" stays
    return t


def _speak_roman_numerals(t: str) -> str:
    """Roman numerals that are certainly numbers become digits: marked by the parser, after a section word,
    in Bible references, at the start of a numbered paragraph and as list labels."""
    t = _MARKED_RE.sub(lambda m: str(roman_value(m.group(1).strip()) or m.group(1)), t)
    t = _SECTION_NUM_RE.sub(_section_numeral, t)
    t = _BIBLE_REF_RE.sub(_bible_ref, t)
    t = _CHAPTER_VERSE_RE.sub(_bible_ref, t)
    t = _LEADING_NUM_RE.sub(lambda m: f"{v}{m.group(2)}" if (v := roman_value(m.group(1))) and v <= 200 else m.group(0), t)
    t = _LIST_LABEL_RE.sub(lambda m: f"({roman_value(m.group(1))})" if roman_value(m.group(1)) else m.group(0), t)
    return t


def _section_numeral(m: re.Match) -> str:
    word, space, num = m.groups()
    value = roman_value(num)
    if value is None or value > 399 or (len(num) == 1 and num.upper() not in "IVX"):  # "Part C", "Section L"
        return m.group(0)
    if num == "I":
        # The pronoun ("the part I played") is safe unless the section word starts the heading or sentence
        # ("CHAPTER I Peril from the Sky") or the numeral ends the phrase ("Book I, chapter 2").
        before, after = m.string[:m.start()], m.string[m.end():]
        starts = not before.strip() or re.search(r"[.:;!?—]\s*$", before)
        ends = re.match(r"\s*(?:[.,:;!?)\]—]|$)", after)
        if not (starts or ends):
            return m.group(0)
    return f"{word}{space}{value}"


def _bible_ref(m: re.Match) -> str:
    value = roman_value(m.group(2))
    if value is None or value > 150:
        return m.group(0)
    return f"{m.group(1)}{value}:{m.group(3)}"


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
