"""Built-in M4B test player: verify chapters, cover, metadata and audio before copying to the phone.

Decoding is done by ffmpeg (AAC -> float PCM over a pipe) and playback by sounddevice/PortAudio, so it
does not depend on the OS media framework recognising the .m4b extension.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import wx

from .audio import find_ffmpeg, fmt_hms

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
PLAY_SR = 48000
PLAY_CH = 2


# --------------------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------------------

@dataclass
class ChapterInfo:
    index: int
    start: float
    end: float
    title: str

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class M4BInfo:
    path: str
    title: str = ""
    artist: str = ""
    year: str = ""
    genre: str = ""
    duration: float = 0.0
    bit_rate: int = 0
    size: int = 0
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    has_cover: bool = False
    is_audiobook_flagged: bool = False
    cover_bytes: Optional[bytes] = None
    chapters: list[ChapterInfo] = field(default_factory=list)


def _ffprobe_path() -> str:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH.")
    p = Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)
    return str(p) if p.exists() else "ffprobe"


def probe_m4b(path: str) -> M4BInfo:
    out = subprocess.run([_ffprobe_path(), "-v", "error", "-print_format", "json", "-show_format", "-show_chapters",
                          "-show_streams", str(path)], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", creationflags=_CREATE_NO_WINDOW)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "ffprobe failed")
    data = json.loads(out.stdout or "{}")
    fmt = data.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    info = M4BInfo(path=str(path), title=tags.get("title", Path(path).stem), artist=tags.get("artist", ""),
                   year=tags.get("date", ""), genre=tags.get("genre", ""),
                   duration=float(fmt.get("duration", 0) or 0), bit_rate=int(fmt.get("bit_rate", 0) or 0),
                   size=int(fmt.get("size", 0) or 0))
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio" and not info.codec:
            info.codec = s.get("codec_name", "")
            info.sample_rate = int(s.get("sample_rate", 0) or 0)
            info.channels = int(s.get("channels", 0) or 0)
        if s.get("codec_type") == "video" and (s.get("disposition", {}).get("attached_pic") or s.get("codec_name") in ("mjpeg", "png")):
            info.has_cover = True
    for i, c in enumerate(data.get("chapters", []), 1):
        ctags = {k.lower(): v for k, v in (c.get("tags") or {}).items()}
        info.chapters.append(ChapterInfo(i, float(c.get("start_time", 0)), float(c.get("end_time", 0)),
                                         ctags.get("title", f"Chapter {i}")))
    try:  # Apple Books needs the binary 'stik' atom = 2; ffmpeg's text tag is not enough
        from mutagen.mp4 import MP4
        stik = (MP4(str(path)).tags or {}).get("stik")
        info.is_audiobook_flagged = bool(stik and int(stik[0]) == 2)
    except Exception:
        info.is_audiobook_flagged = False
    if info.has_cover:
        try:
            img = subprocess.run([find_ffmpeg() or "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                                  "-c:v", "mjpeg", "-f", "image2pipe", "pipe:1"], capture_output=True,
                                 creationflags=_CREATE_NO_WINDOW, timeout=30)
            if img.returncode == 0 and len(img.stdout) > 500:
                info.cover_bytes = img.stdout
        except Exception:
            pass
    return info


# --------------------------------------------------------------------------------------
# Playback engine
# --------------------------------------------------------------------------------------

class Player:
    """Plays a file from a given position: ffmpeg decodes, sounddevice outputs."""

    def __init__(self, path: str):
        self.path = path
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._start_pos = 0.0
        self._frames = 0
        self._paused_pos = 0.0
        self.playing = False
        self.ended = False
        self.error: Optional[str] = None

    @property
    def position(self) -> float:
        if self.playing:
            return self._start_pos + self._frames / PLAY_SR
        return self._paused_pos

    def play(self, pos: Optional[float] = None) -> None:
        self.stop()
        if pos is None:
            pos = self._paused_pos
        self._start_pos = max(0.0, pos)
        self._frames = 0
        self._stop.clear()
        self.ended = False
        self.error = None
        self.playing = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        if self.playing:
            self._paused_pos = self.position
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        t = self._thread
        if t is not None and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout=2.0)
        if self.playing:
            self._paused_pos = self.position
        self.playing = False

    def seek(self, pos: float) -> None:
        was_playing = self.playing
        self.stop()
        self._paused_pos = max(0.0, pos)
        if was_playing:
            self.play(self._paused_pos)

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except Exception as e:  # pragma: no cover
            self.error = f"sounddevice not available: {e}"
            self.playing = False
            return
        ffmpeg = find_ffmpeg() or "ffmpeg"
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{self._start_pos:.3f}", "-i", self.path, "-vn",
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(PLAY_CH), "-ar", str(PLAY_SR), "pipe:1"]
        block = 2048
        nbytes = block * PLAY_CH * 4
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          creationflags=_CREATE_NO_WINDOW, bufsize=nbytes * 8)
            assert self._proc.stdout is not None
            with sd.OutputStream(samplerate=PLAY_SR, channels=PLAY_CH, dtype="float32", blocksize=block) as stream:
                while not self._stop.is_set():
                    data = self._proc.stdout.read(nbytes)
                    if not data:
                        break
                    arr = np.frombuffer(data, dtype=np.float32)
                    if arr.size % PLAY_CH:
                        arr = arr[: arr.size - arr.size % PLAY_CH]
                    frames = arr.reshape(-1, PLAY_CH)
                    stream.write(frames)
                    self._frames += frames.shape[0]
            if not self._stop.is_set():
                self.ended = True
        except Exception as e:
            if not self._stop.is_set():
                self.error = str(e)
        finally:
            proc = self._proc
            self._proc = None
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
            if not self._stop.is_set():
                self._paused_pos = self._start_pos + self._frames / PLAY_SR
            self.playing = False


# --------------------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------------------

SHORT_CHAPTER_S = 3.0   # chapters shorter than this are flagged (likely an extraction problem)


class PlayerFrame(wx.Frame):
    def __init__(self, parent: Optional[wx.Window], path: Optional[str] = None):
        super().__init__(parent, title="M4B test player", size=wx.Size(980, 700))
        self.SetMinSize(self.FromDIP(wx.Size(760, 520)))
        self.info: Optional[M4BInfo] = None
        self.player: Optional[Player] = None
        self._dragging = False
        self._build()
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._tick, self.timer)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.CreateStatusBar()
        self.Centre()
        if path:
            wx.CallAfter(self.load, path)

    # ---------------------------------------------------------------- UI
    def _build(self):
        panel = wx.Panel(self)
        root = wx.BoxSizer(wx.VERTICAL)

        top = wx.BoxSizer(wx.HORIZONTAL)
        self.cover = wx.StaticBitmap(panel, size=self.FromDIP(wx.Size(140, 200)))
        top.Add(self.cover, 0, wx.ALL, self.FromDIP(8))
        meta = wx.BoxSizer(wx.VERTICAL)
        self.lbl_title = wx.StaticText(panel, label="No file loaded")
        self.lbl_title.SetFont(self.lbl_title.GetFont().Bold().Scaled(1.2))
        self.lbl_artist = wx.StaticText(panel, label="")
        self.lbl_tech = wx.StaticText(panel, label="")
        self.lbl_check = wx.StaticText(panel, label="")
        for l in (self.lbl_title, self.lbl_artist, self.lbl_tech, self.lbl_check):
            meta.Add(l, 0, wx.BOTTOM, self.FromDIP(4))
        meta.AddStretchSpacer()
        self.btn_open = wx.Button(panel, label="Open M4B…")
        meta.Add(self.btn_open, 0, wx.TOP, self.FromDIP(4))
        top.Add(meta, 1, wx.EXPAND | wx.ALL, self.FromDIP(8))
        root.Add(top, 0, wx.EXPAND)

        self.list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for i, (name, w, fmt) in enumerate([("#", 44, wx.LIST_FORMAT_RIGHT), ("Chapter", 520, wx.LIST_FORMAT_LEFT),
                                             ("Start", 90, wx.LIST_FORMAT_RIGHT), ("Length", 90, wx.LIST_FORMAT_RIGHT)]):
            self.list.InsertColumn(i, name, width=self.FromDIP(w), format=fmt)
        root.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, self.FromDIP(8))

        self.slider = wx.Slider(panel, value=0, minValue=0, maxValue=1000)
        root.Add(self.slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, self.FromDIP(8))
        ctl = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_prev = wx.Button(panel, label="|◀ Chapter")
        self.btn_back = wx.Button(panel, label="−15 s")
        self.btn_play = wx.Button(panel, label="▶ Play")
        self.btn_play.SetMinSize(self.FromDIP(wx.Size(110, -1)))
        self.btn_fwd = wx.Button(panel, label="+15 s")
        self.btn_next = wx.Button(panel, label="Chapter ▶|")
        for b in (self.btn_prev, self.btn_back, self.btn_play, self.btn_fwd, self.btn_next):
            ctl.Add(b, 0, wx.RIGHT, self.FromDIP(4))
        ctl.AddStretchSpacer()
        self.lbl_time = wx.StaticText(panel, label="0:00 / 0:00")
        self.lbl_time.SetFont(self.lbl_time.GetFont().Bold())
        ctl.Add(self.lbl_time, 0, wx.ALIGN_CENTER_VERTICAL)
        root.Add(ctl, 0, wx.EXPAND | wx.ALL, self.FromDIP(8))
        self.lbl_chapter = wx.StaticText(panel, label="")
        root.Add(self.lbl_chapter, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, self.FromDIP(8))
        panel.SetSizer(root)

        self.btn_open.Bind(wx.EVT_BUTTON, self._on_open)
        self.btn_play.Bind(wx.EVT_BUTTON, self._on_play)
        self.btn_back.Bind(wx.EVT_BUTTON, lambda e: self._seek_rel(-15))
        self.btn_fwd.Bind(wx.EVT_BUTTON, lambda e: self._seek_rel(15))
        self.btn_prev.Bind(wx.EVT_BUTTON, lambda e: self._chapter_step(-1))
        self.btn_next.Bind(wx.EVT_BUTTON, lambda e: self._chapter_step(1))
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_chapter_activated)
        self.slider.Bind(wx.EVT_SCROLL_THUMBTRACK, lambda e: setattr(self, "_dragging", True))
        self.slider.Bind(wx.EVT_SCROLL_THUMBRELEASE, self._on_slider_release)
        self.slider.Bind(wx.EVT_SCROLL_CHANGED, self._on_slider_release)
        self._enable(False)

    def _enable(self, on: bool):
        for w in (self.btn_play, self.btn_back, self.btn_fwd, self.btn_prev, self.btn_next, self.slider):
            w.Enable(on)

    # ---------------------------------------------------------------- load
    def _on_open(self, _):
        with wx.FileDialog(self, "Open audiobook", wildcard="Audiobooks (*.m4b;*.m4a;*.mp3)|*.m4b;*.m4a;*.mp3|All files|*.*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.load(dlg.GetPath())

    def load(self, path: str):
        if self.player:
            self.player.stop()
        try:
            info = probe_m4b(path)
        except Exception as e:
            wx.MessageBox(f"Could not read this file:\n{e}", "Player", wx.ICON_ERROR)
            return
        self.info = info
        self.player = Player(path)
        self.SetTitle(f"M4B test player — {Path(path).name}")
        self.lbl_title.SetLabel(info.title or Path(path).name)
        self.lbl_artist.SetLabel(info.artist + (f"  ({info.year})" if info.year else ""))
        mb = info.size / 1e6
        self.lbl_tech.SetLabel(f"{fmt_hms(info.duration)}  •  {info.codec.upper()} {info.bit_rate // 1000} kb/s, "
                               f"{info.sample_rate} Hz, {'mono' if info.channels == 1 else str(info.channels) + ' ch'}  •  "
                               f"{mb:.0f} MB  •  {len(info.chapters)} chapters  •  cover: {'yes' if info.has_cover else 'NO'}")
        short = [c for c in info.chapters if c.length < SHORT_CHAPTER_S]
        problems = []
        if not info.chapters:
            problems.append("no chapter markers")
        if short:
            problems.append(f"{len(short)} very short chapter(s)")
        if not info.has_cover:
            problems.append("no cover art")
        if info.genre.lower() != "audiobook":
            problems.append("genre is not 'Audiobook'")
        if not info.is_audiobook_flagged:
            problems.append("missing Apple 'audiobook' media-type flag (stik) — Books won't list it")
        self.lbl_check.SetLabel(("Checks: " + "; ".join(problems)) if problems else "Checks: chapters, cover and tags look good ✓")
        self.lbl_check.SetForegroundColour(wx.Colour(180, 60, 60) if problems else wx.Colour(40, 140, 60))

        bmp = wx.Bitmap(self.FromDIP(wx.Size(140, 200)))
        if info.cover_bytes:
            try:
                img = wx.Image(io.BytesIO(info.cover_bytes))
                if img.IsOk():
                    tw, th = self.FromDIP(wx.Size(140, 200)).Get()
                    sc = min(tw / img.GetWidth(), th / img.GetHeight())
                    img = img.Scale(max(1, int(img.GetWidth() * sc)), max(1, int(img.GetHeight() * sc)), wx.IMAGE_QUALITY_HIGH)
                    bmp = wx.Bitmap(img)
            except Exception:
                pass
        self.cover.SetBitmap(bmp)

        self.list.DeleteAllItems()
        for c in info.chapters:
            i = self.list.InsertItem(self.list.GetItemCount(), str(c.index))
            self.list.SetItem(i, 1, c.title)
            self.list.SetItem(i, 2, fmt_hms(c.start))
            self.list.SetItem(i, 3, fmt_hms(c.length))
            if c.length < SHORT_CHAPTER_S:
                self.list.SetItemTextColour(i, wx.Colour(190, 50, 50))
        self.slider.SetRange(0, max(1, int(info.duration)))
        self.slider.SetValue(0)
        self._enable(True)
        self.timer.Start(250)
        self.Layout()
        self._tick(None)

    # ---------------------------------------------------------------- transport
    def _on_play(self, _):
        if not self.player:
            return
        if self.player.playing:
            self.player.pause()
        else:
            if self.player.ended or (self.info and self.player.position >= self.info.duration - 0.5):
                self.player.play(0.0)
            else:
                self.player.play()
        self._tick(None)

    def _seek_rel(self, delta: float):
        if self.player and self.info:
            self.player.seek(min(max(0.0, self.player.position + delta), max(0.0, self.info.duration - 1)))
            self._tick(None)

    def _current_chapter_idx(self) -> int:
        if not self.info or not self.player:
            return -1
        pos = self.player.position
        for i, c in enumerate(self.info.chapters):
            if c.start <= pos < c.end:
                return i
        return len(self.info.chapters) - 1 if self.info.chapters else -1

    def _chapter_step(self, step: int):
        if not self.info or not self.player or not self.info.chapters:
            return
        i = self._current_chapter_idx()
        if step < 0 and self.player.position - self.info.chapters[i].start > 3:
            target = i
        else:
            target = min(max(0, i + step), len(self.info.chapters) - 1)
        self._go_chapter(target)

    def _go_chapter(self, i: int):
        if self.info and self.player and 0 <= i < len(self.info.chapters):
            self.player.seek(self.info.chapters[i].start)
            if not self.player.playing:
                self.player.play()
            self._tick(None)

    def _on_chapter_activated(self, evt):
        self._go_chapter(evt.GetIndex())

    def _on_slider_release(self, _):
        self._dragging = False
        if self.player:
            self.player.seek(float(self.slider.GetValue()))
            self._tick(None)

    def _tick(self, _):
        if not self.player or not self.info:
            return
        pos = self.player.position
        if not self._dragging:
            self.slider.SetValue(int(min(pos, self.info.duration)))
        self.lbl_time.SetLabel(f"{fmt_hms(pos)} / {fmt_hms(self.info.duration)}")
        self.btn_play.SetLabel("❚❚ Pause" if self.player.playing else "▶ Play")
        i = self._current_chapter_idx()
        if i >= 0:
            c = self.info.chapters[i]
            self.lbl_chapter.SetLabel(f"Chapter {c.index}/{len(self.info.chapters)}: {c.title}   "
                                      f"({fmt_hms(pos - c.start)} / {fmt_hms(c.length)})")
            if self.list.GetFirstSelected() != i and not self.list.HasFocus():
                self.list.Select(i)
                self.list.EnsureVisible(i)
        if self.player.error:
            self.SetStatusText("Playback error: " + self.player.error)
            self.player.error = None
        elif self.player.ended:
            self.SetStatusText("End of book")
        else:
            self.SetStatusText("Playing" if self.player.playing else "Paused")

    def _on_close(self, evt):
        self.timer.Stop()
        if self.player:
            self.player.stop()
        self.Destroy()


def main(path: Optional[str] = None):
    app = wx.App(False)
    PlayerFrame(None, path).Show()
    app.MainLoop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
