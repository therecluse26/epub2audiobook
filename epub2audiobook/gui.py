"""wxPython GUI for the EPUB -> M4B converter."""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Optional

import wx
import wx.adv

from . import __version__
from .audio import find_ffmpeg, open_in_explorer, write_wav
from .converter import CancelledError, ConvertOptions, Converter, preview_audio
from .epub_parser import Book, Chapter, detect_language_from_text, load_book
from .player import PlayerFrame
from .tts import KokoroEngine, device_description
from .voices import LANGUAGES, VOICES, default_voice, iso_to_kokoro

SETTINGS_FILE = Path.home() / ".epub2audiobook.json"
WORDS_PER_MIN = 155  # Kokoro at speed 1.0, roughly
LEFT_COL_DIP = 300   # width of the left (book + narration) column in DIP

STATUS_TEXT = {"synth": "synthesizing…", "encode": "encoding…", "done": "✓ done", "cached": "✓ cached"}


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text("utf-8"))
    except Exception:
        return {}


def save_settings(d: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(d, indent=2), "utf-8")
    except Exception:
        pass


def _wrap_text(win: wx.Window, text: str, width: int) -> str:
    """Word-wrap `text` to `width` pixels using the window's font (never wider than the longest word)."""
    dc = wx.ClientDC(win)
    dc.SetFont(win.GetFont())
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            cand = (cur + " " + word) if cur else word
            if dc.GetTextExtent(cand).width <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return "\n".join(lines)


def fmt_minutes(m: float) -> str:
    if m < 1:
        return f"{int(m * 60)}s"
    h, mm = divmod(int(round(m)), 60)
    return f"{h}h {mm:02d}m" if h else f"{mm}m"


class TextPreviewDialog(wx.Dialog):
    def __init__(self, parent, title: str, text: str):
        super().__init__(parent, title=f"Extracted text — {title}", size=parent.FromDIP(wx.Size(760, 560)),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        s = wx.BoxSizer(wx.VERTICAL)
        tc = wx.TextCtrl(self, value=text, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        tc.SetFont(wx.Font(wx.FontInfo(11)))
        s.Add(tc, 1, wx.EXPAND | wx.ALL, self.FromDIP(8))
        s.Add(self.CreateButtonSizer(wx.OK), 0, wx.ALIGN_RIGHT | wx.ALL, self.FromDIP(8))
        self.SetSizer(s)


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=f"EPUB → Audiobook (Kokoro TTS) v{__version__}")
        self.SetSize(self.FromDIP(wx.Size(1240, 860)))
        self.SetMinSize(self.FromDIP(wx.Size(1000, 700)))
        self.book: Optional[Book] = None
        self.engine: Optional[KokoroEngine] = None
        self.worker: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.settings = load_settings()
        self._preview_sound: Optional[wx.adv.Sound] = None
        self._player: Optional[PlayerFrame] = None
        self._last_output: Optional[str] = None
        self._build_ui()
        self.CreateStatusBar()
        self.SetStatusText(f"Compute: {device_description()}   •   ffmpeg: {find_ffmpeg() or 'NOT FOUND'}")
        self.Centre()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        if not find_ffmpeg():
            wx.CallAfter(wx.MessageBox, "ffmpeg was not found on PATH.\nInstall it (e.g. `winget install Gyan.FFmpeg`) "
                         "and restart, otherwise the M4B cannot be produced.", "ffmpeg missing", wx.ICON_WARNING)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        d = self.FromDIP
        panel = wx.Panel(self)
        root = wx.BoxSizer(wx.VERTICAL)

        # --- file rows
        grid = wx.FlexGridSizer(2, 2, d(6), d(8))
        grid.AddGrowableCol(1, 1)
        grid.Add(wx.StaticText(panel, label="EPUB file:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.epub_picker = wx.FilePickerCtrl(panel, message="Choose an EPUB", wildcard="EPUB files (*.epub)|*.epub|All files|*.*",
                                             style=wx.FLP_OPEN | wx.FLP_FILE_MUST_EXIST | wx.FLP_USE_TEXTCTRL)
        self.epub_picker.SetInitialDirectory(self.settings.get("last_epub_dir", str(Path.home())))
        grid.Add(self.epub_picker, 1, wx.EXPAND)
        grid.Add(wx.StaticText(panel, label="Output M4B:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.out_picker = wx.FilePickerCtrl(panel, message="Save audiobook as", wildcard="M4B audiobook (*.m4b)|*.m4b",
                                            style=wx.FLP_SAVE | wx.FLP_OVERWRITE_PROMPT | wx.FLP_USE_TEXTCTRL)
        grid.Add(self.out_picker, 1, wx.EXPAND)
        root.Add(grid, 0, wx.EXPAND | wx.ALL, d(10))
        self.epub_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_epub_chosen)

        # --- middle: left (scrollable) info/settings, right chapters
        mid = wx.BoxSizer(wx.HORIZONTAL)
        self.left = wx.ScrolledWindow(panel, style=wx.VSCROLL)
        self.left.SetScrollRate(0, d(10))
        self.left.SetMinSize(wx.Size(d(LEFT_COL_DIP), -1))
        left = wx.BoxSizer(wx.VERTICAL)

        info_sz = wx.StaticBoxSizer(wx.VERTICAL, self.left, "Book")
        ib = info_sz.GetStaticBox()
        self.cover = wx.StaticBitmap(ib, size=d(wx.Size(120, 176)))
        self.cover.SetBitmap(wx.Bitmap(d(wx.Size(120, 176))))
        info_sz.Add(self.cover, 0, wx.ALIGN_CENTER | wx.ALL, d(4))
        self.lbl_title = wx.StaticText(ib, label="No book loaded")
        self.lbl_title.SetFont(self.lbl_title.GetFont().Bold())
        self.lbl_author = wx.StaticText(ib, label="")
        self.lbl_lang = wx.StaticText(ib, label="")
        self._wrap_labels = [self.lbl_title, self.lbl_author, self.lbl_lang]
        for l in self._wrap_labels:
            info_sz.Add(l, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, d(6))
        left.Add(info_sz, 0, wx.EXPAND | wx.BOTTOM, d(8))

        set_sz = wx.StaticBoxSizer(wx.VERTICAL, self.left, "Narration")
        sb = set_sz.GetStaticBox()
        g = wx.FlexGridSizer(0, 2, d(6), d(6))
        g.AddGrowableCol(1, 1)
        g.Add(wx.StaticText(sb, label="Language:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.lang_choice = wx.Choice(sb, choices=list(LANGUAGES.values()))
        self._lang_codes = list(LANGUAGES.keys())
        g.Add(self.lang_choice, 1, wx.EXPAND)
        g.Add(wx.StaticText(sb, label="Voice:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.voice_choice = wx.Choice(sb, choices=[])
        g.Add(self.voice_choice, 1, wx.EXPAND)
        g.Add(wx.StaticText(sb, label="Speed:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.speed = wx.SpinCtrlDouble(sb, min=0.6, max=1.5, inc=0.05, initial=float(self.settings.get("speed", 1.0)))
        self.speed.SetDigits(2)
        g.Add(self.speed, 0)
        g.Add(wx.StaticText(sb, label="Device:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.device_choice = wx.Choice(sb, choices=["Auto", "CUDA (GPU)", "CPU"])
        self.device_choice.SetSelection({"auto": 0, "cuda": 1, "cpu": 2}.get(self.settings.get("device", "auto"), 0))
        g.Add(self.device_choice, 1, wx.EXPAND)
        g.Add(wx.StaticText(sb, label="AAC bitrate:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.bitrate_choice = wx.Choice(sb, choices=["48k", "64k", "96k", "128k"])
        self.bitrate_choice.SetStringSelection(self.settings.get("bitrate", "64k"))
        g.Add(self.bitrate_choice, 1, wx.EXPAND)
        for c in (self.lang_choice, self.voice_choice, self.device_choice, self.bitrate_choice):
            c.SetMinSize(d(wx.Size(120, -1)))   # sizer uses this instead of the (wide) best size; EXPAND fills the column
        set_sz.Add(g, 0, wx.EXPAND | wx.ALL, d(4))
        self.chk_credits = wx.CheckBox(sb, label="Read opening credits (title, subtitle, author)")
        self.chk_credits.SetValue(self.settings.get("read_credits", True))
        self.chk_titles = wx.CheckBox(sb, label="Announce chapter titles")
        self.chk_titles.SetValue(self.settings.get("read_titles", True))
        self.chk_keep = wx.CheckBox(sb, label="Keep per-chapter work files (resume)")
        self.chk_keep.SetValue(self.settings.get("keep_work_files", False))
        for c in (self.chk_credits, self.chk_titles, self.chk_keep):
            set_sz.Add(c, 0, wx.LEFT | wx.RIGHT | wx.TOP, d(4))
        prow = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_preview = wx.Button(sb, label="▶ Preview voice")
        self.btn_stop_preview = wx.Button(sb, label="■", size=d(wx.Size(36, -1)))
        prow.Add(self.btn_preview, 1, wx.RIGHT, d(4))
        prow.Add(self.btn_stop_preview, 0)
        set_sz.Add(prow, 0, wx.EXPAND | wx.ALL, d(4))
        left.Add(set_sz, 0, wx.EXPAND | wx.BOTTOM, d(8))

        self.btn_player = wx.Button(self.left, label="Play / test an M4B…")
        left.Add(self.btn_player, 0, wx.EXPAND)
        self.left.SetSizer(left)
        mid.Add(self.left, 0, wx.EXPAND | wx.RIGHT, d(10))

        ch_sz = wx.StaticBoxSizer(wx.VERTICAL, panel, "Chapters (tick the ones to narrate; double-click to see the text)")
        cb = ch_sz.GetStaticBox()
        self.list = wx.ListCtrl(cb, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list.EnableCheckBoxes(True)
        cols = [("#", 46, wx.LIST_FORMAT_RIGHT), ("Title", 400, wx.LIST_FORMAT_LEFT), ("Type", 110, wx.LIST_FORMAT_LEFT),
                ("Words", 70, wx.LIST_FORMAT_RIGHT), ("Est.", 64, wx.LIST_FORMAT_RIGHT), ("Status", 110, wx.LIST_FORMAT_LEFT)]
        for i, (name, w, fmt) in enumerate(cols):
            self.list.InsertColumn(i, name, width=d(w), format=fmt)
        ch_sz.Add(self.list, 1, wx.EXPAND | wx.ALL, d(4))
        brow = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_all = wx.Button(cb, label="Select all")
        self.btn_none = wx.Button(cb, label="Select none")
        self.btn_smart = wx.Button(cb, label="Smart selection")
        self.btn_show = wx.Button(cb, label="Show extracted text…")
        for b in (self.btn_all, self.btn_none, self.btn_smart):
            brow.Add(b, 0, wx.RIGHT, d(4))
        brow.AddStretchSpacer()
        brow.Add(self.btn_show, 0)
        ch_sz.Add(brow, 0, wx.EXPAND | wx.ALL, d(4))
        self.lbl_selection = wx.StaticText(cb, label="")
        ch_sz.Add(self.lbl_selection, 0, wx.LEFT | wx.BOTTOM, d(6))
        mid.Add(ch_sz, 1, wx.EXPAND)
        root.Add(mid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, d(10))

        # --- bottom: progress + log
        prow2 = wx.BoxSizer(wx.HORIZONTAL)
        self.gauge = wx.Gauge(panel, range=1000)
        prow2.Add(self.gauge, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, d(8))
        self.btn_convert = wx.Button(panel, label="Convert to M4B")
        self.btn_convert.SetFont(self.btn_convert.GetFont().Bold())
        self.btn_cancel = wx.Button(panel, label="Cancel")
        self.btn_cancel.Disable()
        prow2.Add(self.btn_convert, 0, wx.RIGHT, d(4))
        prow2.Add(self.btn_cancel, 0)
        root.Add(prow2, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, d(10))
        self.lbl_status = wx.StaticText(panel, label="Load an EPUB to begin.")
        root.Add(self.lbl_status, 0, wx.LEFT | wx.RIGHT | wx.TOP, d(10))
        self.log = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL, size=wx.Size(-1, d(120)))
        self.log.SetFont(wx.Font(wx.FontInfo(9).Family(wx.FONTFAMILY_TELETYPE)))
        root.Add(self.log, 0, wx.EXPAND | wx.ALL, d(10))
        panel.SetSizer(root)

        # events
        self.lang_choice.Bind(wx.EVT_CHOICE, self.on_lang_changed)
        self.btn_all.Bind(wx.EVT_BUTTON, lambda e: self._set_all(True))
        self.btn_none.Bind(wx.EVT_BUTTON, lambda e: self._set_all(False))
        self.btn_smart.Bind(wx.EVT_BUTTON, lambda e: self._smart_select())
        self.btn_show.Bind(wx.EVT_BUTTON, self.on_show_text)
        self.list.Bind(wx.EVT_LIST_ITEM_CHECKED, lambda e: self._update_selection_label())
        self.list.Bind(wx.EVT_LIST_ITEM_UNCHECKED, lambda e: self._update_selection_label())
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.on_show_text)
        self.btn_preview.Bind(wx.EVT_BUTTON, self.on_preview)
        self.btn_stop_preview.Bind(wx.EVT_BUTTON, lambda e: self._stop_preview())
        self.btn_player.Bind(wx.EVT_BUTTON, lambda e: self.open_player(self._last_output))
        self.btn_convert.Bind(wx.EVT_BUTTON, self.on_convert)
        self.btn_cancel.Bind(wx.EVT_BUTTON, lambda e: self.cancel_event.set())
        self.speed.Bind(wx.EVT_SPINCTRLDOUBLE, lambda e: self._update_selection_label())
        self.left.Bind(wx.EVT_SIZE, self._on_left_size)

        self._set_lang(self.settings.get("lang", "a"), voice=self.settings.get("voice"))

    # ------------------------------------------------------------------ helpers
    def _log(self, msg: str):
        self.log.AppendText(msg + "\n")

    def _on_left_size(self, evt):
        evt.Skip()
        wx.CallAfter(self._rewrap)

    def _rewrap(self):
        if not self:
            return
        width = self.left.GetClientSize().width - self.FromDIP(28)
        if width < 60:
            return
        for l in self._wrap_labels:
            text = getattr(l, "_full_text", l.GetLabel())
            l._full_text = text
            l.SetLabel(_wrap_text(l, text, width))
            l.SetMinSize(wx.Size(width, -1))
        self.left.Layout()
        self.left.FitInside()

    def _set_wrapped(self, label: wx.StaticText, text: str):
        label._full_text = text
        label.SetLabel(text)
        self._rewrap()

    def _set_lang(self, code: str, voice: Optional[str] = None):
        if code not in self._lang_codes:
            code = "a"
        self.lang_choice.SetSelection(self._lang_codes.index(code))
        self._fill_voices(code, voice)

    def _fill_voices(self, code: str, voice: Optional[str] = None):
        self.voice_choice.Clear()
        self._voice_ids = [v for v, _ in VOICES[code]]
        for v, label in VOICES[code]:
            self.voice_choice.Append(f"{label}  [{v}]")
        sel = self._voice_ids.index(voice) if voice in self._voice_ids else 0
        self.voice_choice.SetSelection(sel)

    def current_lang(self) -> str:
        return self._lang_codes[max(0, self.lang_choice.GetSelection())]

    def current_voice(self) -> str:
        i = self.voice_choice.GetSelection()
        return self._voice_ids[i] if 0 <= i < len(self._voice_ids) else default_voice(self.current_lang())

    def current_device(self) -> str:
        return ["auto", "cuda", "cpu"][max(0, self.device_choice.GetSelection())]

    def on_lang_changed(self, _):
        self._fill_voices(self.current_lang(), self.settings.get(f"voice_{self.current_lang()}"))

    def _get_engine(self) -> KokoroEngine:
        dev = self.current_device()
        if self.engine is None or (dev != "auto" and self.engine.device != dev):
            if self.engine is not None:
                self.engine.release()
            self.engine = KokoroEngine(dev)
        return self.engine

    def _selected_chapters(self) -> list[Chapter]:
        if not self.book:
            return []
        # CheckItem fires EVT_LIST_ITEM_CHECKED while _book_loaded is still adding rows; the generic
        # list control (GTK/macOS) crashes on IsItemChecked for a row that does not exist yet.
        rows = self.list.GetItemCount()
        return [ch for i, ch in enumerate(self.book.chapters[:rows]) if self.list.IsItemChecked(i)]

    def _set_all(self, value: bool):
        for i in range(self.list.GetItemCount()):
            self.list.CheckItem(i, value)
        self._update_selection_label()

    def _smart_select(self):
        if not self.book:
            return
        for i, ch in enumerate(self.book.chapters):
            self.list.CheckItem(i, ch.include_default)
        self._update_selection_label()

    def _update_selection_label(self):
        chs = self._selected_chapters()
        words = sum(c.words for c in chs)
        est = words / (WORDS_PER_MIN * float(self.speed.GetValue() or 1.0))
        self.lbl_selection.SetLabel(f"{len(chs)} chapters selected — {words:,} words — about {fmt_minutes(est)} of audio")

    def _set_status_col(self, chapter_index: int, state: str):
        """chapter_index is Chapter.index (1-based); 0 = opening credits (not in the list)."""
        if not self.book or chapter_index <= 0:
            return
        row = chapter_index - 1
        if 0 <= row < self.list.GetItemCount():
            self.list.SetItem(row, 5, STATUS_TEXT.get(state, state))
            if state in ("synth", "encode"):
                self.list.EnsureVisible(row)

    def _busy(self, busy: bool):
        for w in (self.epub_picker, self.out_picker, self.btn_convert, self.btn_preview, self.btn_all, self.btn_none,
                  self.btn_smart, self.lang_choice, self.voice_choice, self.speed, self.device_choice, self.bitrate_choice,
                  self.chk_titles, self.chk_credits, self.chk_keep):
            w.Enable(not busy)
        self.btn_cancel.Enable(busy)

    # ------------------------------------------------------------------ load
    def on_epub_chosen(self, _):
        path = self.epub_picker.GetPath()
        if not path or not os.path.exists(path):
            return
        self.settings["last_epub_dir"] = str(Path(path).parent)
        self.lbl_status.SetLabel("Parsing EPUB…")
        self._busy(True)
        self.btn_cancel.Disable()

        def work():
            try:
                book = load_book(path)
                iso_guess = detect_language_from_text(book.chapters) if not book.kokoro_lang else None
                wx.CallAfter(self._book_loaded, book, iso_guess)
            except Exception as e:
                wx.CallAfter(self._load_failed, e)

        threading.Thread(target=work, daemon=True).start()

    def _load_failed(self, e: Exception):
        self._busy(False)
        self.lbl_status.SetLabel("Failed to load EPUB.")
        self._log("ERROR: " + "".join(traceback.format_exception_only(type(e), e)).strip())
        wx.MessageBox(f"Could not read this EPUB:\n{e}", "Error", wx.ICON_ERROR)

    def _book_loaded(self, book: Book, iso_guess: Optional[str]):
        self.book = book
        self._busy(False)
        self._set_wrapped(self.lbl_title, book.title + (f": {book.subtitle}" if book.subtitle else ""))
        self._set_wrapped(self.lbl_author, book.author)
        lang = book.kokoro_lang
        note = f"Language: {book.language_iso or '?'}"
        if not lang and iso_guess:
            lang = iso_to_kokoro(iso_guess)
            note += f" (detected: {iso_guess})"
        if lang:
            self._set_lang(lang, voice=self.settings.get(f"voice_{lang}"))
            note += f" → {LANGUAGES[lang]}"
        else:
            note += " — not supported by Kokoro, pick a language manually"
        self._set_wrapped(self.lbl_lang, note)
        # cover
        tw, th = self.FromDIP(wx.Size(120, 176)).Get()
        bmp = wx.Bitmap(tw, th)
        if book.cover_bytes:
            try:
                img = wx.Image(io.BytesIO(book.cover_bytes))
                if img.IsOk():
                    sc = min(tw / img.GetWidth(), th / img.GetHeight())
                    img = img.Scale(max(1, int(img.GetWidth() * sc)), max(1, int(img.GetHeight() * sc)), wx.IMAGE_QUALITY_HIGH)
                    bmp = wx.Bitmap(img)
            except Exception:
                pass
        self.cover.SetBitmap(bmp)
        # chapters
        self.list.DeleteAllItems()
        for i, ch in enumerate(book.chapters):
            idx = self.list.InsertItem(i, str(ch.index))
            self.list.SetItem(idx, 1, ch.title)
            self.list.SetItem(idx, 2, ch.kind)
            self.list.SetItem(idx, 3, f"{ch.words:,}")
            self.list.SetItem(idx, 4, fmt_minutes(ch.words / WORDS_PER_MIN))
            self.list.CheckItem(idx, ch.include_default)
            if not ch.include_default:
                self.list.SetItemTextColour(idx, wx.Colour(130, 130, 130))
        self._update_selection_label()
        safe = "".join(c for c in book.title if c not in '<>:"/\\|?*').strip() or "audiobook"
        out_dir = self.settings.get("last_out_dir") or str(Path(book.path).parent)
        self.out_picker.SetPath(str(Path(out_dir) / f"{safe}.m4b"))
        skipped = [c.title for c in book.chapters if not c.include_default]
        self._log(f"Loaded '{book.title}': {len(book.chapters)} sections, {sum(c.words for c in book.chapters):,} words. "
                  f"Skipping by default: {len(skipped)} ({', '.join(s[:25] for s in skipped[:8])}{'…' if len(skipped) > 8 else ''})")
        self.lbl_status.SetLabel("Review the chapter selection, then Convert.")
        self.Layout()
        self._rewrap()

    def on_show_text(self, _):
        i = self.list.GetFirstSelected()
        if not self.book or i < 0:
            wx.MessageBox("Select a chapter in the list first.", "Show text", wx.ICON_INFORMATION)
            return
        ch = self.book.chapters[i]
        lines = []
        for b in ch.blocks:
            if b.kind == "break":
                lines.append("\n* * *\n")
            elif b.kind == "heading":
                lines.append(f"\n## {b.text}\n")
            else:
                lines.append(b.text + "\n")
        TextPreviewDialog(self, ch.title, "\n".join(lines) or "(no narratable text)").ShowModal()

    # ------------------------------------------------------------------ preview
    def on_preview(self, _):
        if not self.book:
            wx.MessageBox("Load an EPUB first.", "Preview", wx.ICON_INFORMATION)
            return
        i = self.list.GetFirstSelected()
        chs = self._selected_chapters()
        ch = self.book.chapters[i] if i >= 0 else (chs[0] if chs else None)
        text = " ".join(b.text for b in ch.blocks if b.kind == "para")[:600] if ch else ""
        if not text:
            text = "This is a preview of the selected voice, reading a short sample so you can judge its tone and pacing."
        lang, voice, speed = self.current_lang(), self.current_voice(), float(self.speed.GetValue())
        self._stop_preview()
        self._busy(True)
        self.btn_cancel.Disable()
        self.lbl_status.SetLabel(f"Synthesizing preview with {voice}… (first run downloads the model)")

        def work():
            try:
                audio = preview_audio(lang, voice, text, speed, self._get_engine())
                tmp = Path(tempfile.gettempdir()) / "epub2audiobook_preview.wav"
                write_wav(tmp, audio)
                wx.CallAfter(self._play_preview, tmp)
            except Exception as e:
                wx.CallAfter(self._preview_failed, e)

        threading.Thread(target=work, daemon=True).start()

    def _play_preview(self, path: Path):
        self._busy(False)
        self.lbl_status.SetLabel("Playing preview…")
        self._preview_sound = wx.adv.Sound(str(path))
        if self._preview_sound.IsOk():
            self._preview_sound.Play(wx.adv.SOUND_ASYNC)
        else:
            self.lbl_status.SetLabel("Could not play preview (wx.adv.Sound).")

    def _stop_preview(self):
        try:
            wx.adv.Sound.Stop()
        except Exception:
            pass

    def _preview_failed(self, e: Exception):
        self._busy(False)
        self.lbl_status.SetLabel("Preview failed.")
        self._log("ERROR: " + "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip())

    # ------------------------------------------------------------------ player
    def open_player(self, path: Optional[str] = None):
        if self._player and self._player:
            try:
                self._player.Raise()
                if path:
                    self._player.load(path)
                return
            except RuntimeError:
                self._player = None
        self._player = PlayerFrame(self, path if path and os.path.exists(path) else None)
        self._player.Bind(wx.EVT_WINDOW_DESTROY, self._player_closed)
        self._player.Show()
        if not path:
            wx.CallAfter(self._player._on_open, None)

    def _player_closed(self, evt):
        if evt.GetEventObject() is self._player:
            self._player = None
        evt.Skip()

    # ------------------------------------------------------------------ convert
    def on_convert(self, _):
        if not self.book:
            wx.MessageBox("Load an EPUB first.", "Convert", wx.ICON_INFORMATION)
            return
        chapters = self._selected_chapters()
        if not chapters:
            wx.MessageBox("No chapters are selected.", "Convert", wx.ICON_INFORMATION)
            return
        out = self.out_picker.GetPath()
        if not out:
            wx.MessageBox("Choose an output file.", "Convert", wx.ICON_INFORMATION)
            return
        if not out.lower().endswith(".m4b"):
            out += ".m4b"
        lang = self.current_lang()
        opts = ConvertOptions(output_path=out, voice=self.current_voice(), lang=lang, speed=float(self.speed.GetValue()),
                              device=self.current_device(), bitrate=self.bitrate_choice.GetStringSelection() or "64k",
                              read_titles=self.chk_titles.GetValue(), read_credits=self.chk_credits.GetValue(),
                              keep_work_files=self.chk_keep.GetValue())
        self.settings.update({"lang": lang, "voice": opts.voice, f"voice_{lang}": opts.voice, "speed": opts.speed,
                              "device": opts.device, "bitrate": opts.bitrate, "read_titles": opts.read_titles,
                              "read_credits": opts.read_credits, "keep_work_files": opts.keep_work_files,
                              "last_out_dir": str(Path(out).parent)})
        save_settings(self.settings)
        self._stop_preview()
        self.cancel_event.clear()
        self._busy(True)
        self.gauge.SetValue(0)
        for row in range(self.list.GetItemCount()):
            self.list.SetItem(row, 5, "")
        self._log("=" * 60)

        def on_progress(frac: float, msg: str):
            wx.CallAfter(self._on_progress, frac, msg)

        def on_log(msg: str):
            wx.CallAfter(self._log, msg)

        def on_chapter(idx: int, state: str):
            wx.CallAfter(self._set_status_col, idx, state)

        def work():
            try:
                conv = Converter(self.book, chapters, opts, on_progress, on_log, self.cancel_event, self._get_engine(),
                                 on_chapter=on_chapter)
                result = conv.run()
                wx.CallAfter(self._convert_done, result)
            except CancelledError:
                wx.CallAfter(self._convert_cancelled)
            except Exception as e:
                wx.CallAfter(self._convert_failed, e)

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _on_progress(self, frac: float, msg: str):
        self.gauge.SetValue(int(max(0.0, min(1.0, frac)) * 1000))
        self.lbl_status.SetLabel(msg)

    def _convert_done(self, result: str):
        self._busy(False)
        self.gauge.SetValue(1000)
        self._last_output = result
        self.lbl_status.SetLabel(f"Done: {result}")
        dlg = wx.MessageDialog(self, f"Audiobook created:\n{result}", "Finished",
                               wx.YES_NO | wx.CANCEL | wx.ICON_INFORMATION)
        dlg.SetYesNoCancelLabels("▶ Test in player", "Open folder", "Close")
        r = dlg.ShowModal()
        dlg.Destroy()
        if r == wx.ID_YES:
            self.open_player(result)
        elif r == wx.ID_NO:
            open_in_explorer(result)

    def _convert_cancelled(self):
        self._busy(False)
        self.lbl_status.SetLabel("Cancelled. Chapter files already produced are kept in the work folder for resuming.")
        self._log("Cancelled by user.")

    def _convert_failed(self, e: Exception):
        self._busy(False)
        self.lbl_status.SetLabel("Conversion failed.")
        self._log("ERROR: " + "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip())
        wx.MessageBox(f"Conversion failed:\n{e}", "Error", wx.ICON_ERROR)

    def on_close(self, evt):
        if self.worker and self.worker.is_alive():
            if wx.MessageBox("A conversion is running. Cancel it and quit?", "Quit", wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
                return
            self.cancel_event.set()
        save_settings(self.settings)
        self._stop_preview()
        if self._player:
            try:
                self._player.Close()
            except Exception:
                pass
        self.Destroy()


def main(epub_path: Optional[str] = None):
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    if epub_path and os.path.exists(epub_path):
        frame.epub_picker.SetPath(os.path.abspath(epub_path))
        wx.CallAfter(frame.on_epub_chosen, None)
    app.MainLoop()


if __name__ == "__main__":
    main()
