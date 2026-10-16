#!/usr/bin/env bash
# NixOS launcher (the Linux twin of run_gui.bat). Arguments go to the app:
#   ./run.sh                        GUI
#   ./run.sh book.epub              GUI with that book loaded
#   ./run.sh book.epub -o out.m4b   command line (see README, "Command line")
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$here/.venv/bin/python" || ! -e "$here/.nix-env" ]]; then
  echo "Environment not found. Run ./setup-nixos.sh first." >&2
  exit 1
fi
# The pip wheels (torch, numpy, spacy, sounddevice) need libstdc++, zlib and PortAudio from Nix.
export LD_LIBRARY_PATH="$here/.nix-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# GTK's GSettings schemas (the file chooser aborts without them). An unset XDG_DATA_DIRS means the spec default.
export XDG_DATA_DIRS="$(<"$here/.nix-env/share/epub2audiobook/xdg-data-dirs"):${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
exec "$here/.venv/bin/python" -m epub2audiobook "$@"
