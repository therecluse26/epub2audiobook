# NixOS support files for epub2audiobook (Windows/macOS do not need this).
#
# PyPI has no Linux wheels for wxPython, so wxPython comes from nixpkgs.
# Everything else is pip-installed into .venv. Build with:
#   nix-build nixos.nix -o .nix-env
# The .nix-env link is a GC root, so the (slow) wxPython build survives `nix-collect-garbage`.
let
  # Pinned to the nixpkgs of the running system (26.11.20260531.331800d).
  nixpkgs = builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/331800de5053fcebacf6813adb5db9c9dca22a0c.tar.gz";
    sha256 = "sha256-x5UQuRsH3MqI0U9afaXSNqzTPSeZlRLvFAav2Ux1pNw=";
  };
in
{ pkgs ? import nixpkgs { } }:
let
  python = pkgs.python312; # kokoro does not support 3.13 yet
  # GTK reads its GSettings schemas from XDG_DATA_DIRS; without the gtk3 ones the file chooser
  # aborts ("Settings schema 'org.gtk.Settings.FileChooser' is not installed"). run.sh prepends these.
  xdgDataDirs = pkgs.writeTextDir "share/epub2audiobook/xdg-data-dirs"
    (pkgs.lib.concatMapStringsSep ":" pkgs.glib.getSchemaDataDirPath [ pkgs.gtk3 pkgs.gsettings-desktop-schemas ]);
in
pkgs.buildEnv {
  name = "epub2audiobook-nixos-env";
  paths = [
    python
    python.pkgs.wxpython # lib/python3.12/site-packages/wx
    pkgs.stdenv.cc.cc.lib # libstdc++ for the torch/numpy/spacy wheels
    pkgs.zlib
    pkgs.portaudio # sounddevice (test player, voice preview)
    xdgDataDirs
  ];
  pathsToLink = [ "/bin" "/lib" "/share/epub2audiobook" ];
}
