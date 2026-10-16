"""EPUB parsing: extract narration-worthy text per chapter, classify sections, get metadata & cover.

The goal is clean narration text: no footnote/endnote markers, no page numbers, no running
headers/footers, no navigation / index / copyright boilerplate (those sections are detected and
unselected by default, but the user can override in the GUI).
"""
from __future__ import annotations

import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import warnings

from bs4 import BeautifulSoup, NavigableString, Tag

try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:  # older bs4
    pass
from ebooklib import epub, ITEM_DOCUMENT, ITEM_IMAGE, ITEM_COVER

from .textproc import NUM_CLOSE, NUM_OPEN, roman_value
from .voices import iso_to_kokoro

# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------


@dataclass
class Block:
    kind: str          # 'heading' | 'para' | 'break'
    text: str = ""
    level: int = 0     # heading level (1..6)
    numerals: list[tuple[int, int]] = field(default_factory=list)  # spans of text that are surely Roman numerals


@dataclass
class Chapter:
    index: int
    href: str
    title: str
    kind: str
    blocks: list[Block] = field(default_factory=list)
    include_default: bool = True

    @property
    def words(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks if b.kind != "break")

    @property
    def chars(self) -> int:
        return sum(len(b.text) for b in self.blocks if b.kind != "break")

    def text_preview(self, n: int = 300) -> str:
        out = " ".join(b.text for b in self.blocks if b.kind != "break")
        return out[:n]


@dataclass
class Book:
    path: str
    title: str
    authors: list[str]
    language_iso: Optional[str]
    kokoro_lang: Optional[str]
    year: Optional[str]
    cover_bytes: Optional[bytes]
    cover_ext: str
    chapters: list[Chapter]
    subtitle: str = ""

    @property
    def author(self) -> str:
        return ", ".join(self.authors)


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------

# kinds narrated by default
BODY_KINDS = {
    "body", "chapter", "part", "foreword", "preface", "introduction", "prologue",
    "epilogue", "afterword", "conclusion", "dedication", "epigraph", "appendix",
    "frontmatter", "backmatter",
}
# kinds skipped by default
SKIP_KINDS = {
    "cover", "titlepage", "halftitle", "copyright", "toc", "landmarks", "pagelist",
    "acknowledgments", "glossary", "index", "endnotes", "footnotes", "bibliography",
    "about-author", "about-publisher", "praise", "also-by", "ads", "resources", "colophon", "empty", "contributors",
    "imprint", "credits", "license",
}

# epub:type / guide type / role -> kind
_TYPE_MAP = {
    "cover": "cover", "coverpage": "cover", "cover-image": "cover",
    "titlepage": "titlepage", "title-page": "titlepage", "halftitlepage": "halftitle", "halftitle": "halftitle",
    "copyright-page": "copyright", "copyright": "copyright", "imprint": "imprint", "imprimatur": "imprint",
    "toc": "toc", "landmarks": "landmarks", "page-list": "pagelist", "loi": "toc", "lot": "toc",
    "dedication": "dedication", "epigraph": "epigraph",
    "foreword": "foreword", "preface": "preface", "introduction": "introduction", "prologue": "prologue",
    "epilogue": "epilogue", "afterword": "afterword", "conclusion": "conclusion",
    "part": "part", "chapter": "chapter", "bodymatter": "body", "text": "body",
    "acknowledgments": "acknowledgments", "acknowledgements": "acknowledgments",
    "glossary": "glossary", "index": "index", "bibliography": "bibliography", "endnotes": "endnotes",
    "footnotes": "footnotes", "rearnotes": "endnotes", "notes": "endnotes",
    "appendix": "appendix", "colophon": "colophon", "contributors": "contributors", "credits": "credits",
    "other-credits": "credits", "praise": "praise", "abouttheauthor": "about-author",
    "frontmatter": "frontmatter", "backmatter": "backmatter",
}

# keyword regexes applied to spine ids / file names (lower-cased); order matters
_KEYWORD_RULES: list[tuple[str, str]] = [
    (r"\bcover\b|_cvi_|^cvi", "cover"),
    (r"half[-_ ]?title|_htp_|halftitle", "halftitle"),
    (r"title[-_ ]?page|_tit_|^tit_", "titlepage"),
    (r"copyright|_crt_|^crt|\bimprint\b|colophon", "copyright"),
    (r"\btoc\b|contents|_toc_", "toc"),
    (r"dedicat|_ded_", "dedication"),
    (r"epigraph|_epi_", "epigraph"),
    (r"acknowledg|_ack_|^ack", "acknowledgments"),
    (r"glossar|_glo_", "glossary"),
    (r"\bindex\b|_idx_|^idx", "index"),
    (r"end[-_ ]?notes|footnotes|\bnotes\b|_end_|_not_|^notes", "endnotes"),
    (r"bibliograph|references|works[-_ ]cited|further[-_ ]reading|_bib_|_ref_", "bibliography"),
    (r"about[-_ ]the[-_ ]publisher|about[-_ ]publisher|_atp_", "about-publisher"),
    (r"about[-_ ]the[-_ ]author|about[-_ ]author|_ata_|^ata_", "about-author"),
    (r"praise|_pra_|reviews?[-_ ]for", "praise"),
    (r"also[-_ ]by|other[-_ ]books|_alb_|books[-_ ]by", "also-by"),
    (r"next[-_ ]?reads|newsletter|adcard|\bads?\b|advert|sign[-_ ]?up|_bm_ad|promo", "ads"),
    (r"resources|_rs_", "resources"),
    (r"foreword|_fw_|^fw", "foreword"),
    (r"preface|_pre_", "preface"),
    (r"introduction|_int_|_intro", "introduction"),
    (r"prologue|_pro_", "prologue"),
    (r"epilogue|_epl_", "epilogue"),
    (r"afterword|_aft_", "afterword"),
    (r"appendix|_app_", "appendix"),
    (r"\bpart\b|_p\d{2,3}_|^p\d{2,3}", "part"),
    (r"chapter|_c\d{2,3}_|^c\d{2,3}|_ch\d", "chapter"),
    (r"_fm_|frontmatter", "frontmatter"),
    (r"_bm_|backmatter", "backmatter"),
]

_HEADING_KEYWORDS: list[tuple[str, str]] = [
    (r"^(table of )?contents$", "toc"),
    (r"^copyright", "copyright"),
    (r"^dedication$", "dedication"),
    (r"^acknowledg", "acknowledgments"),
    (r"^glossary", "glossary"),
    (r"^index$", "index"),
    (r"^(end)?notes$", "endnotes"),
    (r"^(bibliography|references|works cited|further reading|sources)$", "bibliography"),
    (r"^about the author", "about-author"),
    (r"^about the publisher", "about-publisher"),
    (r"^resources$", "resources"),
    (r"^praise for", "praise"),
    (r"^also by", "also-by"),
    (r"^foreword", "foreword"),
    (r"^preface", "preface"),
    (r"^introduction", "introduction"),
    (r"^prologue", "prologue"),
    (r"^epilogue", "epilogue"),
    (r"^afterword", "afterword"),
    (r"^appendix", "appendix"),
    (r"^part\b", "part"),
    (r"^chapter\b", "chapter"),
]


def _match_rules(text: str, rules) -> Optional[str]:
    t = text.lower()
    for pat, kind in rules:
        if re.search(pat, t):
            return kind
    return None


# --------------------------------------------------------------------------------------
# HTML cleaning
# --------------------------------------------------------------------------------------

_REMOVE_TAGS = {"script", "style", "nav", "aside", "figure", "figcaption", "img", "svg", "math",
                "sup", "footer", "video", "audio", "iframe", "object", "button", "form", "input", "select",
                "title", "head"}
_REMOVE_EPUB_TYPES = {"noteref", "footnote", "endnote", "footnotes", "endnotes", "rearnote", "rearnotes",
                      "pagebreak", "page-list", "toc", "landmarks", "sidebar", "annotation", "note",
                      "backlink", "help", "warning"}
_REMOVE_ROLES = {"doc-noteref", "doc-footnote", "doc-endnote", "doc-endnotes", "doc-footnotes",
                 "doc-pagebreak", "doc-toc", "doc-index", "doc-backlink", "doc-notice", "note",
                 "doc-pagelist", "presentation", "img", "figure"}
_REMOVE_CLASS_RE = re.compile(
    r"(?:^|[\s_\-])(footnote|endnote|noteref|enref|fnref|fn-ref|note-ref|notemark|pagenum|page-?num|"
    r"page-?break|pagebreak|folio|running-?head|runninghead|running-?foot|hidden|sr-only|calibre-?hidden|"
    r"note_backlink|backlink|toc-?entry|caption)(?:$|[\s_\-\d])", re.I)

_BLOCK_TAGS = {"p", "div", "section", "article", "main", "body", "header", "blockquote", "ul", "ol",
               "li", "dl", "dt", "dd", "pre", "table", "tbody", "thead", "tfoot", "tr", "td", "th",
               "h1", "h2", "h3", "h4", "h5", "h6", "hr", "address", "center", "details", "summary"}
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

_GUTENBERG_IDS = {"pg-header", "pg-footer"}
_GUTENBERG_CLASSES = {"pg-boilerplate", "pgheader"}
# older Project Gutenberg files have no such markup, only these lines around the book
_PG_START_RE = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
_PG_END_RE = re.compile(r"^(?:\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG|End of (?:the )?Project Gutenberg)", re.I)

_NOTE_MARK_RE = re.compile(r"^\s*[\[\(]?\s*(\d{1,3}|[ivxlc]{1,5}|[*†‡§¶#a-z])\s*[\]\)]?\s*$", re.I)
_SCENE_BREAK_RE = re.compile(r"^\s*(?:[\*•✱✲⁂~#\-–—_=·]\s*){1,7}\s*$|^\s*⁂\s*$")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.I)

_HEADING_CLASS_RE = re.compile(
    r"(?:^|\s)(chapter[-_ ]?(title|number|name|num)|ch[-_]?title|cn|ct|cst|chaptitle|chapnum|chap[-_]?title|"
    r"part[-_]?title|pt|pn|title|h1|h2|h3|heading\d?)(?:$|\s)", re.I)


def _attr_tokens(tag: Tag, name: str) -> set[str]:
    v = tag.attrs.get(name)
    if not v:
        return set()
    if isinstance(v, list):
        return {x.lower() for x in v}
    return set(str(v).lower().split())


def _class_str(tag: Tag) -> str:
    cls = tag.attrs.get("class")
    if not cls:
        return ""
    return " ".join(cls) if isinstance(cls, list) else str(cls)


def _should_remove(tag: Tag) -> bool:
    if tag.name in _REMOVE_TAGS:
        return True
    if tag.attrs.get("hidden") is not None or tag.attrs.get("aria-hidden") == "true":
        return True
    style = tag.attrs.get("style", "")
    if isinstance(style, str) and re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", style, re.I):
        return True
    if _attr_tokens(tag, "epub:type") & _REMOVE_EPUB_TYPES:
        return True
    if _attr_tokens(tag, "role") & _REMOVE_ROLES:
        return True
    if _REMOVE_CLASS_RE.search(_class_str(tag)):
        return True
    ident = str(tag.attrs.get("id", ""))
    if ident in _GUTENBERG_IDS or _attr_tokens(tag, "class") & _GUTENBERG_CLASSES:
        return True  # Project Gutenberg's header ("Release date: … [eBook #485]") and licence
    if tag.name in {"a", "span", "sup"} and re.match(r"^(fn|footnote|endnote|note|noteref|nref|ftn|page_?\d|pg\d)", ident, re.I):
        return True
    if tag.name == "a":
        href = str(tag.attrs.get("href", ""))
        txt = tag.get_text(" ", strip=True)
        if "#" in href and txt and _NOTE_MARK_RE.match(txt):
            return True
        if re.search(r"note|fn|ref", href, re.I) and txt and len(txt) <= 4:
            return True
    return False


def _clean_soup(body: Tag) -> None:
    """Remove non-narrative elements in place."""
    for tag in list(body.find_all(True)):
        try:
            if tag.parent is None:
                continue  # already removed with an ancestor
            if _should_remove(tag):
                tag.decompose()
        except Exception:
            pass
    for br in body.find_all("br"):
        br.replace_with(" ")


def _mark_tagged_numerals(body: Tag) -> None:
    """Wrap the Roman numerals a publisher tagged (epub:type z3998:roman, used by Standard Ebooks) in the
    numeral markers, so they can be spoken as numbers without guessing. _postprocess_blocks turns the markers
    into Block.numerals."""
    for tag in body.find_all(True):
        if "z3998:roman" not in _attr_tokens(tag, "epub:type") or tag.find(True) is not None:
            continue
        text = tag.get_text()
        num = text.strip()
        if roman_value(num) is not None:
            lead, trail = text[:len(text) - len(text.lstrip())], text[len(text.rstrip()):]
            tag.string = f"{lead}{NUM_OPEN}{num}{NUM_CLOSE}{trail}"


def _norm_ws(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u200a", " ").replace("\u202f", " ")
    s = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", s)   # "Miller , British" -> "Miller, British"
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    return s.strip()


def _is_heading_like(tag: Tag) -> Optional[int]:
    if tag.name in _HEADING_TAGS:
        return _HEADING_TAGS[tag.name]
    roles = _attr_tokens(tag, "role")
    types = _attr_tokens(tag, "epub:type")
    if "doc-subtitle" in roles or "subtitle" in types or "title" in types:
        return 2
    if tag.name in {"p", "div"} and _HEADING_CLASS_RE.search(_class_str(tag)):
        # only treat as heading if short
        if len(tag.get_text(" ", strip=True)) <= 120:
            return 2
    return None


def _extract_blocks(body: Tag) -> list[Block]:
    blocks: list[Block] = []
    buf: list[str] = []

    def flush():
        if buf:
            text = _norm_ws("".join(buf))
            buf.clear()
            if text:
                blocks.append(Block("para", text))

    def walk(node: Tag):
        for child in list(node.children):
            if isinstance(child, NavigableString):
                if type(child).__name__ in {"Comment", "Doctype", "ProcessingInstruction", "Declaration"}:
                    continue
                buf.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            lvl = _is_heading_like(child)
            if lvl is not None:
                flush()
                text = _norm_ws(child.get_text("", strip=False))
                if text:
                    blocks.append(Block("heading", text, lvl))
                continue
            if child.name == "hr":
                flush()
                blocks.append(Block("break"))
                continue
            if child.name in _BLOCK_TAGS:
                flush()
                if child.find(_BLOCK_TAGS) is None:
                    text = _norm_ws(child.get_text("", strip=False))
                    if text:
                        blocks.append(Block("para", text))
                else:
                    walk(child)
                    flush()
            elif child.find(_BLOCK_TAGS) is not None:
                flush()
                walk(child)
                flush()
            else:
                buf.append(child.get_text("", strip=False))
        flush()

    walk(body)
    return blocks


_DROPCAP_RE = re.compile(r"^[\"\u201c\u2018']?[A-Z\u00C0-\u00DE]$")
_DROPCAP_SPLIT_RE = re.compile(r"^([\"\u201c\u2018']?[B-HJ-Z\u00C0-\u00DE])\s+([a-z\u00DF-\u00FF]+)\b")


def _merge_drop_caps(blocks: list[Block]) -> list[Block]:
    """Join decorative initials with the word they start.

    Publishers render drop caps as <span class="dropcap">T</span>he (handled by flattening without a
    separator) or as a separate block/paragraph holding just the letter, which is what this merges.
    """
    out: list[Block] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if (b.kind == "para" and _DROPCAP_RE.match(b.text) and i + 1 < len(blocks)
                and blocks[i + 1].kind == "para" and blocks[i + 1].text[:1].islower()):
            nxt = blocks[i + 1]
            out.append(Block("para", b.text + nxt.text, nxt.level))
            i += 2
            continue
        if b.kind == "para":
            # same thing inside one paragraph when the markup put whitespace after the initial
            # ("T he word"). A and I are excluded: "A dog", "I am" are real words.
            b.text = _DROPCAP_SPLIT_RE.sub(lambda m: m.group(1) + m.group(2), b.text, count=1)
        out.append(b)
        i += 1
    return out


_MARKED_RE = re.compile(f"{NUM_OPEN}([^{NUM_OPEN}{NUM_CLOSE}]*){NUM_CLOSE}")
_NUMERAL_ONLY_RE = re.compile(r"([IVXLC]+)([.:]?)")


def _plain(s: str) -> str:
    return s.replace(NUM_OPEN, "").replace(NUM_CLOSE, "")


def _mark_numeral_only(b: Block) -> None:
    """A heading or paragraph holding nothing but a Roman numeral ("V", "II.") is a chapter or part number.
    Upper case only; a bare "I." paragraph being the pronoun is rare enough to accept."""
    m = _NUMERAL_ONLY_RE.fullmatch(b.text)
    if m and (roman_value(m.group(1)) or 999) <= 200:
        b.text = f"{NUM_OPEN}{m.group(1)}{NUM_CLOSE}{m.group(2)}"


def _take_numeral_marks(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Remove the numeral markers; return the clean text and the spans they enclosed."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = n = 0
    for m in _MARKED_RE.finditer(text):
        before = _plain(text[pos:m.start()])
        parts += [before, m.group(1)]
        n += len(before)
        spans.append((n, n + len(m.group(1))))
        n += len(m.group(1))
        pos = m.end()
    parts.append(_plain(text[pos:]))
    return "".join(parts), spans


def _postprocess_blocks(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    blocks = _merge_drop_caps(blocks)
    for b in blocks:
        if b.kind == "break":
            if out and out[-1].kind != "break":
                out.append(b)
            continue
        t = _plain(b.text)
        if not t:
            continue
        if _SCENE_BREAK_RE.match(t):
            if out and out[-1].kind != "break":
                out.append(Block("break"))
            continue
        if b.kind == "para" and (_PAGE_NUMBER_RE.match(t) or _NOTE_MARK_RE.match(t)):
            continue  # stray page number / note marker
        _mark_numeral_only(b)
        out.append(b)
    # merge consecutive heading lines ("1." + "A Sea Change on Pain")
    merged: list[Block] = []
    for b in out:
        if b.kind == "heading" and merged and merged[-1].kind == "heading":
            prev = merged[-1]
            sep = " " if re.search(r"[.:!?]$", _plain(prev.text)) else ". "
            prev.text = f"{prev.text}{sep}{b.text}"
            continue
        merged.append(b)
    while merged and merged[-1].kind == "break":
        merged.pop()
    while merged and merged[0].kind == "break":
        merged.pop(0)
    for b in merged:
        b.text, b.numerals = _take_numeral_marks(b.text)
    return merged


def _blocks_from_body(body: Tag) -> list[Block]:
    """Narration blocks of one spine document (modifies `body`)."""
    _clean_soup(body)
    _mark_tagged_numerals(body)
    return _postprocess_blocks(_extract_blocks(body))


def _drop_gutenberg_text(blocks: list[Block]) -> tuple[list[Block], bool]:
    """Cut an older Project Gutenberg file at its START/END lines; True when the END line was in it."""
    for i, b in enumerate(blocks):
        if _PG_START_RE.match(b.text):
            blocks = blocks[i + 1:]
            break
    end = next((i for i, b in enumerate(blocks) if _PG_END_RE.match(b.text)), None)
    if end is not None:
        blocks = blocks[:end]
    while blocks and blocks[0].kind == "break":
        blocks.pop(0)
    while blocks and blocks[-1].kind == "break":
        blocks.pop()
    return blocks, end is not None


# --------------------------------------------------------------------------------------
# EPUB loading
# --------------------------------------------------------------------------------------


def _href_key(href: str) -> str:
    href = unquote(href.split("#", 1)[0])
    return posixpath.normpath(href).lower()


def _toc_map(book: epub.EpubBook) -> dict[str, str]:
    """href (no fragment) -> title, first occurrence wins."""
    m: dict[str, str] = {}

    def visit(items):
        for it in items:
            if isinstance(it, (tuple, list)):
                visit(it)
            elif isinstance(it, epub.Link):
                k = _href_key(it.href)
                if k not in m and it.title:
                    m[k] = _norm_ws(it.title)
            elif isinstance(it, epub.Section):
                if getattr(it, "href", None) and it.title:
                    k = _href_key(it.href)
                    if k not in m:
                        m[k] = _norm_ws(it.title)

    try:
        visit(book.toc)
    except Exception:
        pass
    return m


def _landmarks_map(book: epub.EpubBook) -> dict[str, str]:
    """href -> kind from the OPF guide + nav landmarks."""
    m: dict[str, str] = {}
    for g in getattr(book, "guide", []) or []:
        typ = str(g.get("type", "")).lower()
        href = g.get("href")
        if href and typ in _TYPE_MAP:
            m.setdefault(_href_key(href), _TYPE_MAP[typ])
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        props = getattr(item, "properties", []) or []
        if "nav" not in props:
            continue
        try:
            soup = BeautifulSoup(item.get_content(), "lxml")
        except Exception:
            continue
        base = posixpath.dirname(item.get_name())
        for nav in soup.find_all("nav"):
            if "landmarks" not in _attr_tokens(nav, "epub:type"):
                continue
            for a in nav.find_all("a"):
                href = a.get("href")
                for typ in _attr_tokens(a, "epub:type"):
                    if href and typ in _TYPE_MAP:
                        full = posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0])))
                        m.setdefault(_href_key(full), _TYPE_MAP[typ])
    return m


def _doc_epub_type(soup: BeautifulSoup) -> Optional[str]:
    if soup.body is None:
        return None
    tags = [soup.body] + soup.body.find_all(["section", "div", "article"], limit=6)
    for tag in tags:
        for t in _attr_tokens(tag, "epub:type"):
            if t in _TYPE_MAP:
                return _TYPE_MAP[t]
        for r in _attr_tokens(tag, "role"):
            r2 = r.replace("doc-", "")
            if r2 in _TYPE_MAP:
                return _TYPE_MAP[r2]
    return None


def _classify(item_id: str, href: str, landmark_kind: Optional[str], doc_kind: Optional[str],
              blocks: list[Block], raw_soup: BeautifulSoup) -> str:
    if landmark_kind and landmark_kind not in {"body", "text"}:
        return landmark_kind
    if doc_kind and doc_kind not in {"body", "text", "frontmatter", "backmatter"}:
        return doc_kind
    k = _match_rules(item_id, _KEYWORD_RULES) or _match_rules(posixpath.basename(href), _KEYWORD_RULES)
    if k:
        return k
    first_heading = next((b.text for b in blocks if b.kind == "heading"), None)
    if first_heading:
        k = _match_rules(first_heading, _HEADING_KEYWORDS)
        if k:
            return k
    if doc_kind:
        return doc_kind
    words = sum(len(b.text.split()) for b in blocks)
    if words == 0:
        return "empty"
    paras = [b.text for b in blocks if b.kind == "para"]
    if paras:
        numbered = sum(1 for p in paras if re.match(r"^\d{1,3}[.:)]\s", p))
        if numbered >= 10 and numbered / len(paras) > 0.6:
            return "endnotes"
    if words < 40 and raw_soup.find("img") is not None:
        return "cover"
    return "body"


def _cover_from_book(book: epub.EpubBook) -> tuple[Optional[bytes], str]:
    cover_id = None
    for meta in book.get_metadata("OPF", "cover"):
        try:
            cover_id = meta[1].get("content")
        except Exception:
            pass
    candidates = []
    if cover_id:
        it = book.get_item_with_id(cover_id)
        if it is not None:
            candidates.append(it)
    candidates += list(book.get_items_of_type(ITEM_COVER))
    for it in book.get_items_of_type(ITEM_IMAGE):
        props = getattr(it, "properties", []) or []
        name = it.get_name().lower()
        if "cover-image" in props or "cover" in name or "cover" in str(it.get_id()).lower():
            candidates.append(it)
    for it in candidates:
        data = it.get_content()
        if data and len(data) > 1000:
            ext = Path(it.get_name()).suffix.lower().lstrip(".") or "jpg"
            return data, ext
    best = None
    for it in book.get_items_of_type(ITEM_IMAGE):
        d = it.get_content()
        if d and (best is None or len(d) > len(best[0])):
            best = (d, Path(it.get_name()).suffix.lower().lstrip(".") or "jpg")
    return best if best else (None, "jpg")


def load_book(path: str | Path) -> Book:
    path = str(path)
    book = epub.read_epub(path, options={"ignore_ncx": False})

    def meta(ns, name):
        try:
            vals = book.get_metadata(ns, name)
            return [v[0] for v in vals if v and v[0]]
        except Exception:
            return []

    titles = [_norm_ws(str(t)) for t in (meta("DC", "title") or []) if _norm_ws(str(t))]
    title = titles[0] if titles else Path(path).stem
    subtitle = titles[1] if len(titles) > 1 and titles[1].lower() != title.lower() else ""
    if not subtitle and ": " in title:  # "Title: Subtitle" packed into one dc:title
        title, subtitle = (x.strip() for x in title.split(": ", 1))
    authors = []
    for a in meta("DC", "creator") or []:
        a = _norm_ws(re.sub(r"[;,\s]+$", "", str(a)))
        if a and a not in authors:
            authors.append(a)
    lang = (meta("DC", "language") or [None])[0]
    date = (meta("DC", "date") or [None])[0]
    m = re.search(r"\d{4}", date) if date else None
    year = m.group(0) if m else None
    cover_bytes, cover_ext = _cover_from_book(book)

    toc = _toc_map(book)
    landmarks = _landmarks_map(book)

    chapters: list[Chapter] = []
    idx = 0
    after_pg_end = False  # past an old-style Project Gutenberg END line: only the licence follows
    for spine_entry in book.spine:
        item_id = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        href = item.get_name()
        try:
            soup = BeautifulSoup(item.get_content(), "lxml")
        except Exception:
            continue
        body = soup.body or soup
        props = getattr(item, "properties", []) or []
        doc_kind = "toc" if "nav" in props else _doc_epub_type(soup)
        raw_copy = BeautifulSoup(str(body), "lxml")
        blocks, pg_end = _drop_gutenberg_text(_blocks_from_body(body))
        kind = _classify(str(item_id), href, landmarks.get(_href_key(href)), doc_kind, blocks, raw_copy)
        if kind in {"body", "frontmatter", "backmatter"} and not blocks:
            kind = "empty"
        if after_pg_end:
            kind = "license"
        after_pg_end = after_pg_end or pg_end

        ttl = toc.get(_href_key(href))
        if not ttl:
            heads = [b.text for b in blocks if b.kind == "heading"]
            if heads:
                ttl = heads[0]
        if not ttl:
            ttl = _title_from_id(str(item_id)) or f"Section {idx + 1}"
        idx += 1
        chapters.append(Chapter(index=idx, href=href, title=ttl[:120], kind=kind, blocks=blocks,
                                include_default=(kind in BODY_KINDS)))

    _strip_running_lines(chapters)
    for ch in chapters:
        if ch.include_default and ch.words == 0:
            ch.include_default = False
    return Book(path=path, title=_norm_ws(title), subtitle=subtitle, authors=authors, language_iso=lang,
                kokoro_lang=iso_to_kokoro(lang), year=year, cover_bytes=cover_bytes, cover_ext=cover_ext,
                chapters=chapters)


def _title_from_id(item_id: str) -> Optional[str]:
    s = re.sub(r"^[xX]?\d{2,4}_", "", item_id)
    s = re.sub(r"^[a-z]{1,4}\d{0,3}_", "", s)
    s = re.sub(r"\.x?html?$", "", s)
    s = s.replace("_", " ").replace("-", " ").strip()
    return s if len(s) > 2 else None


def _strip_running_lines(chapters: list[Chapter]) -> None:
    """Drop short paragraphs repeated verbatim across many chapters (running headers/footers)."""
    counter: Counter[str] = Counter()
    for ch in chapters:
        seen = set()
        for b in ch.blocks:
            if b.kind == "para" and len(b.text) <= 80:
                key = b.text.lower()
                if key not in seen:
                    seen.add(key)
                    counter[key] += 1
    n_ch = sum(1 for c in chapters if c.words > 0)
    repeated = {k for k, c in counter.items() if c >= 3 and c >= 0.25 * max(n_ch, 4)}
    if not repeated:
        return
    for ch in chapters:
        ch.blocks = [b for b in ch.blocks if not (b.kind == "para" and b.text.lower() in repeated)]


def detect_language_from_text(chapters: list[Chapter]) -> Optional[str]:
    """Fallback language detection via langdetect on a sample of body text -> ISO code."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
    except Exception:
        return None
    sample = []
    total = 0
    for ch in chapters:
        if ch.kind in BODY_KINDS:
            for b in ch.blocks:
                if b.kind == "para":
                    sample.append(b.text)
                    total += len(b.text)
                    if total > 6000:
                        break
        if total > 6000:
            break
    text = " ".join(sample)
    if len(text) < 200:
        return None
    try:
        return detect(text)
    except Exception:
        return None
