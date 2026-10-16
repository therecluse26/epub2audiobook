#!/usr/bin/env bash
# NixOS setup (the Linux twin of setup.ps1). Run once; safe to re-run.
#   ./setup-nixos.sh
# Then launch with ./run.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "==> Building the Nix environment (Python 3.12, wxPython, libstdc++, PortAudio)"
echo "    The first build compiles wxPython for Python 3.12 (not in the binary cache): 20-40 minutes."
nix-build nixos.nix -o .nix-env

# The pip wheels (and the pyopenjtalk source build) need libstdc++/zlib at install time too.
export LD_LIBRARY_PATH="$PWD/.nix-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x .venv/bin/python ]]; then
  echo "==> Creating .venv"
  "$(readlink -f .nix-env/bin/python3.12)" -m venv .venv
fi
py=.venv/bin/python

pip_install() {
  if command -v uv >/dev/null; then uv pip install --python "$py" "$@"; else "$py" -m pip install "$@"; fi
}

echo "==> Installing PyTorch (CPU build)"
pip_install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing the other packages (wxPython comes from Nix)"
reqs=$(mktemp)
trap 'rm -f "$reqs"' EXIT
grep -iv '^wxpython' requirements.txt > "$reqs"
pip_install -r "$reqs"

echo "==> Linking the Nix wxPython into .venv"
site=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
echo "$PWD/.nix-env/lib/python3.12/site-packages" > "$site/nix-wxpython.pth"

"$py" -c "import torch, kokoro, wx, ebooklib, bs4, soundfile, sounddevice, mutagen; print('Python OK; wxPython', wx.version())"
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg is not on PATH. Add pkgs.ffmpeg to your system packages."
echo "Done. Launch with ./run.sh"
