"""Command-line interface (handy for batch jobs and for testing without the GUI)."""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from .converter import CancelledError, ConvertOptions, Converter
from .epub_parser import detect_language_from_text, load_book
from .voices import LANGUAGES, VOICES, default_voice, iso_to_kokoro


def parse_ranges(spec: str, n: int) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a else 1
            b = int(b) if b else n
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 1 <= i <= n)


def main(argv=None):
    p = argparse.ArgumentParser(prog="epub2audiobook", description="Convert an EPUB into a chaptered M4B audiobook with Kokoro TTS.")
    p.add_argument("epub", help="input .epub file")
    p.add_argument("-o", "--output", help="output .m4b (default: next to the epub)")
    p.add_argument("-l", "--list", action="store_true", help="list detected sections and exit")
    p.add_argument("--dump-text", metavar="N", type=int, help="print the extracted narration text of section N and exit")
    p.add_argument("-c", "--chapters", help="sections to narrate, e.g. '5-25,27' (default: smart selection)")
    p.add_argument("--all", action="store_true", help="narrate every section")
    p.add_argument("--lang", choices=sorted(LANGUAGES), help="Kokoro language code (default: from EPUB metadata)")
    p.add_argument("-v", "--voice", help="Kokoro voice id (default: best voice for the language)")
    p.add_argument("-s", "--speed", type=float, default=1.0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--bitrate", default="64k")
    p.add_argument("--no-titles", action="store_true", help="do not announce chapter titles")
    p.add_argument("--no-credits", action="store_true", help="skip the opening credits (title, subtitle, author)")
    p.add_argument("--keep", action="store_true", help="keep per-chapter WAV work files")
    p.add_argument("--work-dir")
    p.add_argument("--voices", action="store_true", help="list available voices and exit")
    args = p.parse_args(argv)

    if args.voices:
        for code, name in LANGUAGES.items():
            print(f"{code}  {name}")
            for vid, label in VOICES[code]:
                print(f"     {vid:14s} {label}")
        return 0

    book = load_book(args.epub)
    lang = args.lang or book.kokoro_lang
    if not lang:
        guess = detect_language_from_text(book.chapters)
        lang = iso_to_kokoro(guess)
        if lang:
            print(f"Language detected from text: {guess} -> {LANGUAGES[lang]}")
    if not lang:
        print(f"Language {book.language_iso!r} is not supported by Kokoro; pass --lang.", file=sys.stderr)
        return 2

    print(f"{book.title} — {book.author}  [{book.language_iso} -> {LANGUAGES[lang]}]")
    if args.list or args.dump_text:
        if args.dump_text:
            ch = book.chapters[args.dump_text - 1]
            for b in ch.blocks:
                print("* * *" if b.kind == "break" else (f"## {b.text}" if b.kind == "heading" else b.text))
                print()
            return 0
        print(f"{'#':>3}  {'sel':3}  {'type':14} {'words':>7}  title")
        for ch in book.chapters:
            print(f"{ch.index:>3}  {'[x]' if ch.include_default else '[ ]'}  {ch.kind:14} {ch.words:>7}  {ch.title}")
        return 0

    if args.all:
        chapters = list(book.chapters)
    elif args.chapters:
        idx = set(parse_ranges(args.chapters, len(book.chapters)))
        chapters = [c for c in book.chapters if c.index in idx]
    else:
        chapters = [c for c in book.chapters if c.include_default]
    if not chapters:
        print("No chapters selected.", file=sys.stderr)
        return 2

    voice = args.voice or default_voice(lang)
    out = args.output or str(Path(args.epub).with_suffix(".m4b"))
    opts = ConvertOptions(output_path=out, voice=voice, lang=lang, speed=args.speed, device=args.device,
                          bitrate=args.bitrate, read_titles=not args.no_titles, read_credits=not args.no_credits,
                          keep_work_files=args.keep,
                          work_dir=args.work_dir)

    def on_progress(frac, msg):
        print(f"\r{frac * 100:5.1f}%  {msg:<70}", end="", flush=True)

    def on_log(msg):
        print("\r" + " " * 80 + "\r" + msg, flush=True)

    cancel = threading.Event()
    try:
        Converter(book, chapters, opts, on_progress, on_log, cancel).run()
    except KeyboardInterrupt:
        cancel.set()
        print("\nInterrupted.")
        return 130
    except CancelledError:
        return 130
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
