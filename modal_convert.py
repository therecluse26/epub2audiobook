"""Convert an EPUB to M4B on a cloud NVIDIA GPU with Modal (https://modal.com), billed per second.

One-time setup:  uv tool install modal  &&  modal setup
Usage:           modal run modal_convert.py --epub book.epub [--out book.m4b] [--args "-v af_bella -s 1.05"] [--gpu L4]

Runs the normal command line (python -m epub2audiobook) in a container that stops when the book is done.
`--args` takes the usual CLI options (see README, "Command line"), except -o (use --out).
The Kokoro model is cached in the Modal Volume "epub2audiobook-cache", so only the first run downloads it.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import modal

GPU = "T4"  # cheapest Modal GPU; Kokoro-82M needs < 2 GB of VRAM
TIMEOUT_H = 3  # hard stop: a stuck run can never bill more than this many GPU-hours

app = modal.App("epub2audiobook")
cache = modal.Volume.from_name("epub2audiobook-cache", create_if_missing=True)


def _requirements() -> list[str]:
    """requirements.txt without torch (CUDA build installed separately) and wxPython (GUI only)."""
    if not modal.is_local():  # the container imports this file too, but has no requirements.txt
        return []
    lines = (Path(__file__).parent / "requirements.txt").read_text().splitlines()
    return [l.strip() for l in lines
            if l.strip() and not l.startswith("#") and not l.lower().startswith(("torch", "wxpython"))]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "build-essential", "cmake")  # compilers: pyopenjtalk (misaki[ja]) has no wheel
    .uv_pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install(*_requirements())
    .run_commands("python -m spacy download en_core_web_sm")  # misaki would download it on every run
    .env({"HF_HOME": "/cache/hf"})  # Kokoro model + voices persist in the volume
    .add_local_python_source("epub2audiobook")
)


@app.function(image=image, gpu=GPU, cpu=2.0, memory=4096, volumes={"/cache": cache}, timeout=TIMEOUT_H * 3600)
def convert(epub: bytes, name: str, cli_args: list[str]) -> str:
    """Narrate one EPUB on the GPU; returns the M4B's path inside the cache volume."""
    from epub2audiobook.app import main as run_app

    src = Path("/tmp") / name
    src.write_bytes(epub)
    out = Path("/cache/out") / (src.stem + ".m4b")
    out.parent.mkdir(parents=True, exist_ok=True)
    rc = run_app([str(src), "-o", str(out), "--work-dir", "/tmp/work", "--device", "cuda", *cli_args])
    if rc:
        raise RuntimeError(f"epub2audiobook exited with code {rc}")
    cache.commit()
    return str(out.relative_to("/cache"))


@app.local_entrypoint()
def main(epub: str, out: str = "", args: str = "", gpu: str = GPU):
    src = Path(epub).expanduser()
    dst = Path(out).expanduser() if out else src.with_suffix(".m4b")
    fn = convert if gpu == GPU else convert.with_options(gpu=gpu)
    rel = fn.remote(src.read_bytes(), src.name, shlex.split(args))
    with dst.open("wb") as f:
        for chunk in cache.read_file(rel):
            f.write(chunk)
    cache.remove_file(rel)
    print(f"saved {dst}")
