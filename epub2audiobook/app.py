"""Entry point dispatcher (used by `python -m epub2audiobook` and the `epub2audiobook` console script).

  epub2audiobook                        GUI
  epub2audiobook book.epub              GUI with that book loaded (drag & drop onto run_gui.bat)
  epub2audiobook --play [file.m4b]      built-in M4B test player
  epub2audiobook --fix-tags file.m4b    add the Apple Books audiobook atoms to an existing M4B
  epub2audiobook book.epub [options]    command-line conversion (see --help)
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--fix-tags" and len(args) > 1:
        from .audio import tag_as_audiobook
        tag_as_audiobook(args[1])
        print("tagged as audiobook:", args[1])
        return 0
    if args and args[0] == "--play":
        from .player import main as player_main
        player_main(args[1] if len(args) > 1 else None)
        return 0
    if len(args) == 1 and args[0].lower().endswith(".epub"):
        from .gui import main as gui_main
        gui_main(args[0])
        return 0
    if args:
        from .cli import main as cli_main
        return int(cli_main(args) or 0)
    from .gui import main as gui_main
    gui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
