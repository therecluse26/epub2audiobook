"""Command-line interface (handy for batch jobs and for testing without the GUI)."""
from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path

from .converter import CancelledError, ConvertOptions, Converter
from .epub_parser import detect_language_from_text, load_book
from .keys import load_keys
from .tts import API_FORMATS, SERVER_PRESETS, EngineConfig, EngineError
from .voices import DEFAULT_MODEL, LANGUAGES, MODELS, default_voice, iso_to_kokoro, voices_for


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


def with_saved_key(cfg: EngineConfig) -> EngineConfig:
    """Command line: the key environment variable wins; the key saved by the GUI is the fallback."""
    if cfg.is_server and not cfg.api_key and not os.environ.get(cfg.api_key_env):
        return replace(cfg, api_key=load_keys().get(cfg.provider, ""))
    return cfg


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
    p.add_argument("--bitrate", default="64k")
    p.add_argument("--no-titles", action="store_true", help="do not announce chapter titles")
    p.add_argument("--no-credits", action="store_true", help="skip the opening credits (title, subtitle, author)")
    p.add_argument("--keep", action="store_true", help="keep per-chapter WAV work files")
    p.add_argument("--work-dir")
    p.add_argument("--voices", action="store_true", help="list the voices of --model and exit")
    e = p.add_argument_group("speech engine")
    e.add_argument("--provider", choices=["local", *SERVER_PRESETS], default="local",
                   help="local: Kokoro on this computer (default); openai: any OpenAI-compatible "
                        "/v1/audio/speech server, e.g. Kokoro-FastAPI or a hosted TTS API; openrouter: "
                        "OpenRouter's speech models (Kokoro by default), key in $OPENROUTER_API_KEY")
    e.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL, help="local: Kokoro weights to use")
    e.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="local: where to run the model")
    e.add_argument("--threads", type=int, default=0, help="local: PyTorch CPU threads (default: one per physical core)")
    e.add_argument("--max-vram", type=float, default=0.0, metavar="GB", help="local, CUDA: cap PyTorch's GPU memory")
    e.add_argument("--base-url", help="server URL, e.g. http://localhost:8880/v1 "
                                      "(default: openai: $OPENAI_BASE_URL; openrouter: https://openrouter.ai/api/v1)")
    e.add_argument("--api-model", help="server: model name the server expects "
                                       "(default: openai: kokoro; openrouter: hexgrad/kokoro-82m)")
    e.add_argument("--api-key-env", metavar="VAR", help="server: environment variable that holds the API key "
                                                        "(default: OPENAI_API_KEY or OPENROUTER_API_KEY); if it "
                                                        "is not set, the key saved in the GUI is used")
    e.add_argument("--api-format", choices=API_FORMATS, help="server: audio format to ask for "
                                                            "(default: openai: wav; openrouter: mp3)")
    e.add_argument("--concurrency", type=int, default=4, help="server: parallel requests (default: 4)")
    args = p.parse_args(argv)

    if args.voices:
        for code, voices in voices_for(args.model).items():
            print(f"{code}  {LANGUAGES[code]}")
            for vid, label in voices:
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

    local = args.provider == "local"
    if local and lang not in voices_for(args.model):
        print(f"{MODELS[args.model][1]} cannot narrate {LANGUAGES[lang]}; use another --model.", file=sys.stderr)
        return 2
    voice = args.voice or default_voice(lang, args.model if local else DEFAULT_MODEL)
    out = args.output or str(Path(args.epub).with_suffix(".m4b"))
    base_url = args.base_url
    if base_url is None and args.provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", "")
    engine = EngineConfig.for_provider(args.provider, model=args.model, device=args.device, threads=args.threads,
                                       max_vram_gb=args.max_vram, base_url=base_url, api_model=args.api_model,
                                       api_key_env=args.api_key_env, api_format=args.api_format,
                                       concurrency=args.concurrency)
    engine = with_saved_key(engine)
    opts = ConvertOptions(output_path=out, voice=voice, lang=lang, speed=args.speed, engine=engine,
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
    except EngineError as e:
        print(f"\nSpeech engine error: {e}", file=sys.stderr)
        return 1
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
