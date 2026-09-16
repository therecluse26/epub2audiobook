"""Audio post-processing and M4B assembly with ffmpeg."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


# --------------------------------------------------------------------------------------
# numpy helpers
# --------------------------------------------------------------------------------------

def silence(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sr))), dtype=np.float32)


def trim_silence(a: np.ndarray, sr: int = SAMPLE_RATE, thresh_db: float = -48.0, pad_ms: float = 60.0) -> np.ndarray:
    """Trim leading/trailing near-silence, keeping a short pad, so pauses are controlled by us."""
    if a.size == 0:
        return a
    thresh = 10 ** (thresh_db / 20.0)
    win = max(1, int(sr * 0.01))
    n = a.size // win
    if n == 0:
        return a
    frames = np.abs(a[: n * win]).reshape(n, win).max(axis=1)
    loud = np.where(frames > thresh)[0]
    if loud.size == 0:
        return a[: int(sr * 0.05)]
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, loud[0] * win - pad)
    end = min(a.size, (loud[-1] + 1) * win + pad)
    return a[start:end]


def fade_edges(a: np.ndarray, sr: int = SAMPLE_RATE, ms: float = 8.0) -> np.ndarray:
    n = min(int(sr * ms / 1000.0), a.size // 2)
    if n <= 0:
        return a
    a = a.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a[:n] *= ramp
    a[-n:] *= ramp[::-1]
    return a


def normalize_level(a: np.ndarray, target_rms_db: float = -20.0, peak_db: float = -1.0) -> np.ndarray:
    """Gain to a target RMS, limited so peaks never exceed peak_db."""
    if a.size == 0:
        return a
    rms = float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))
    if rms < 1e-6:
        return a
    gain = (10 ** (target_rms_db / 20.0)) / rms
    peak = float(np.max(np.abs(a)))
    max_gain = (10 ** (peak_db / 20.0)) / max(peak, 1e-6)
    gain = min(gain, max_gain)
    return np.clip(a * gain, -1.0, 1.0).astype(np.float32)


def write_wav(path: str | Path, a: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf.write(str(path), a, sr, subtype="PCM_16")


def wav_duration(path: str | Path) -> float:
    return float(sf.info(str(path)).duration)


# --------------------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------------------

def find_ffmpeg() -> Optional[str]:
    for name in ("ffmpeg", "ffmpeg.exe"):
        p = shutil.which(name)
        if p:
            return p
    here = Path(__file__).resolve().parent.parent
    for cand in (here / "bin" / "ffmpeg.exe", here / "ffmpeg.exe", here / "bin" / "ffmpeg"):
        if cand.exists():
            return str(cand)
    return None


def _ffmeta_escape(s: str) -> str:
    return re.sub(r"([=;#\\\n])", lambda m: "\\" + m.group(1), s)


def _run(cmd: list[str], on_line: Optional[Callable[[str], None]] = None) -> None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", creationflags=_CREATE_NO_WINDOW)
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        lines.append(line)
        if on_line:
            on_line(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + "\n".join(lines[-25:]))


def prepare_cover(cover_bytes: bytes, ext: str, out_dir: Path, ffmpeg: str) -> Optional[str]:
    """Write the cover as a baseline JPEG (Apple Books is picky about cover formats)."""
    if not cover_bytes:
        return None
    src = out_dir / f"cover_src.{ext or 'img'}"
    src.write_bytes(cover_bytes)
    dst = out_dir / "cover.jpg"
    try:
        _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
              "-vf", "scale='min(1400,iw)':-2,format=yuvj420p", "-q:v", "3", "-frames:v", "1", str(dst)])
        return str(dst) if dst.exists() and dst.stat().st_size > 0 else None
    except Exception:
        return None


def fmt_hms(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _require_ffmpeg() -> str:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install it (e.g. `winget install Gyan.FFmpeg`) and make sure it is on PATH.")
    return ffmpeg


def encode_chapter_aac(wav_path: str | Path, out_path: str | Path, bitrate: str = "64k") -> str:
    """Encode one chapter WAV to an AAC .m4a (so the final M4B assembly is a fast stream copy)."""
    ffmpeg = _require_ffmpeg()
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.stem + ".part.m4a")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-i", str(wav_path),
           "-c:a", "aac", "-b:a", bitrate, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "ipod", str(tmp)]
    _run(cmd)
    tmp.replace(out_path)
    return str(out_path)


def build_m4b(chapters: list[tuple[str, str, float]], out_path: str | Path, *, title: str, author: str = "",
              year: Optional[str] = None, cover_path: Optional[str] = None, bitrate: str = "64k",
              work_dir: Optional[Path] = None, on_progress: Optional[Callable[[float, str], None]] = None,
              on_log: Optional[Callable[[str], None]] = None) -> str:
    """Assemble chapter audio files into a chaptered M4B with metadata and cover art.

    chapters: list of (audio_path, chapter_title, duration_seconds) in playback order. If the audio
    files are already AAC (.m4a) they are stream-copied (seconds); WAVs are encoded here (slow).
    """
    ffmpeg = _require_ffmpeg()
    out_path = Path(out_path)
    work_dir = Path(work_dir) if work_dir else out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    all_aac = all(Path(p).suffix.lower() in {".m4a", ".aac", ".mp4"} for p, _, _ in chapters)

    list_file = work_dir / "concat.txt"
    meta_file = work_dir / "ffmeta.txt"
    total_ms = 0
    with list_file.open("w", encoding="utf-8") as lf, meta_file.open("w", encoding="utf-8") as mf:
        mf.write(";FFMETADATA1\n")
        mf.write(f"title={_ffmeta_escape(title)}\n")
        mf.write(f"album={_ffmeta_escape(title)}\n")
        if author:
            mf.write(f"artist={_ffmeta_escape(author)}\n")
            mf.write(f"album_artist={_ffmeta_escape(author)}\n")
            mf.write(f"composer={_ffmeta_escape(author)}\n")
        if year:
            mf.write(f"date={year}\n")
        mf.write("genre=Audiobook\n")
        mf.write("media_type=2\n")
        mf.write("comment=Narrated by Kokoro TTS\n")
        for path, ctitle, dur in chapters:
            p_abs = str(Path(path).resolve()).replace("\\", "/")
            lf.write("file '" + p_abs.replace("'", "'\\''") + "'\n")
            dur_ms = int(round((dur if dur else wav_duration(path)) * 1000))
            mf.write("\n[CHAPTER]\nTIMEBASE=1/1000\n")
            mf.write(f"START={total_ms}\nEND={total_ms + max(dur_ms, 1)}\n")
            mf.write(f"title={_ffmeta_escape(ctitle)}\n")
            total_ms += dur_ms

    have_cover = bool(cover_path and Path(cover_path).exists())
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
           "-f", "concat", "-safe", "0", "-i", str(list_file), "-i", str(meta_file)]
    if have_cover:
        cmd += ["-i", str(cover_path)]
    cmd += ["-map_metadata", "1", "-map", "0:a"]
    if have_cover:
        cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
    if all_aac:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", bitrate, "-ar", str(SAMPLE_RATE), "-ac", "1"]
    cmd += ["-movflags", "+faststart", "-f", "ipod", str(out_path)]

    phase = "Muxing" if all_aac else "Encoding"
    if on_log:
        on_log(f"ffmpeg: {phase.lower()} {len(chapters)} chapters ({fmt_hms(total_ms / 1000)}) into {out_path.name}")
    t0 = time.time()
    last_log = [0.0]

    def on_line(line: str):
        if line.startswith("out_time_us=") and total_ms > 0:
            try:
                secs = int(line.split("=", 1)[1]) / 1e6
            except ValueError:
                return
            frac = min(1.0, secs * 1000.0 / total_ms)
            elapsed = time.time() - t0
            speed = secs / elapsed if elapsed > 0.2 else 0.0
            eta = (total_ms / 1000.0 - secs) / speed if speed > 0 else 0.0
            msg = f"{phase}: {fmt_hms(secs)} of {fmt_hms(total_ms / 1000)}  ({speed:.0f}x, ETA {fmt_hms(eta)})"
            if on_progress:
                on_progress(frac, msg)
            if on_log and elapsed - last_log[0] >= 15:
                last_log[0] = elapsed
                on_log("ffmpeg: " + msg)
        elif on_log and line and not line.startswith(("out_time", "bitrate", "total_size", "frame", "fps", "stream",
                                                        "speed", "progress", "drop", "dup")):
            on_log("ffmpeg: " + line)

    _run(cmd, on_line)
    if on_log:
        on_log(f"ffmpeg: {phase.lower()} finished in {time.time() - t0:.1f}s")
    try:
        tag_as_audiobook(out_path, title=title, author=author, year=year)
        if on_log:
            on_log("tagged as Audiobook for Apple Books (stik=2, brand M4B)")
    except Exception as e:  # tagging is best-effort; the file is still playable
        if on_log:
            on_log(f"warning: could not write iTunes audiobook atoms ({e})")
    if on_progress:
        on_progress(1.0, f"{phase} finished")
    return str(out_path)


def tag_as_audiobook(path: str | Path, *, title: str = "", author: str = "", year: Optional[str] = None,
                     narrator: str = "Kokoro TTS") -> None:
    """Set the iTunes atoms Apple Books relies on to recognise an audiobook.

    ffmpeg writes `media_type` only as a free-text tag; Books needs the binary `stik` atom (2 = Audiobook)
    and prefers the 'M4B ' brand. Also sets 'pgap' (gapless) and the narrator/©nrt fields.
    """
    from mutagen.mp4 import MP4, MP4FreeForm

    path = Path(path)
    mp4 = MP4(str(path))
    tags = mp4.tags if mp4.tags is not None else mp4.add_tags() or mp4.tags
    tags["stik"] = [2]            # media type: Audiobook
    tags["pgap"] = True           # gapless playback between chapters
    tags.setdefault("trkn", [(1, 1)])   # track 1 of 1 (single-file book)
    tags.setdefault("disk", [(1, 1)])
    if title:
        tags["\xa9nam"] = [title]
        tags["\xa9alb"] = [title]
    if author:
        tags["\xa9ART"] = [author]
        tags["aART"] = [author]
        tags["\xa9wrt"] = [author]
    if year:
        tags["\xa9day"] = [str(year)]
    tags["\xa9gen"] = ["Audiobook"]
    if narrator:
        tags["----:com.apple.iTunes:NARRATOR"] = [MP4FreeForm(narrator.encode("utf-8"))]
    mp4.save()
    # brand 'M4A ' -> 'M4B ' in the ftyp box (first 12 bytes: size, 'ftyp', major brand)
    with path.open("r+b") as f:
        head = f.read(12)
        if head[4:8] == b"ftyp" and head[8:12] == b"M4A ":
            f.seek(8)
            f.write(b"M4B ")


def play_wav_blocking_cmd(path: str) -> list[str]:  # helper for CLI preview (uses ffplay if present)
    ffplay = shutil.which("ffplay")
    return [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", path] if ffplay else []


def open_in_explorer(path: str | Path) -> None:
    p = str(path)
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", p])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", p])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(p)])
    except Exception:
        pass
