<#
.SYNOPSIS
  One-time setup for epub2audiobook on Windows 10/11.

  Creates a private Python 3.12 virtual environment in .\.venv (via the `uv` tool, which downloads
  Python itself - nothing is installed system-wide), installs PyTorch with CUDA (or CPU-only with
  -Cpu), the Kokoro TTS model package, wxPython and the other dependencies, and installs ffmpeg
  through winget if it is not on PATH.

.EXAMPLE
  .\setup.ps1            # NVIDIA GPU (CUDA 12.8 build of PyTorch)
  .\setup.ps1 -Cpu       # no NVIDIA GPU: CPU-only PyTorch (much smaller download, slower synthesis)
  .\setup.ps1 -Locked    # use the exact package versions in requirements-lock.txt
#>
param(
    [switch]$Cpu,
    [switch]$Locked
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# ---------------------------------------------------------------- uv (Python manager)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step "Installing uv (Python package/version manager) with winget"
    winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv was installed but is not on PATH yet. Close this window, open a new PowerShell and run setup.ps1 again."
    }
}

# ---------------------------------------------------------------- Python 3.12 + venv
Step "Installing Python 3.12 (private copy managed by uv) and creating .venv"
uv python install 3.12
if (Test-Path ".venv") { Write-Host "  .venv already exists - reusing it" }
else { uv venv --python 3.12 .venv }
$py = ".venv\Scripts\python.exe"

# ---------------------------------------------------------------- PyTorch
if ($Cpu) {
    Step "Installing PyTorch (CPU-only)"
    uv pip install --python $py torch torchaudio --index-url https://download.pytorch.org/whl/cpu
} else {
    Step "Installing PyTorch with CUDA 12.8 (about 2.5 GB download)"
    uv pip install --python $py torch torchaudio --index-url https://download.pytorch.org/whl/cu128
}

# ---------------------------------------------------------------- the rest
if ($Locked) {
    Step "Installing pinned dependencies from requirements-lock.txt"
    uv pip install --python $py -r requirements-lock.txt
} else {
    Step "Installing dependencies from requirements.txt"
    uv pip install --python $py -r requirements.txt
}

# ---------------------------------------------------------------- ffmpeg
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Step "ffmpeg not found on PATH - installing Gyan.FFmpeg with winget"
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Write-Host "  ffmpeg installed. If the app cannot find it, open a new terminal so PATH is refreshed." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- check
Step "Verifying the installation"
& $py -c "import torch, kokoro, wx, ebooklib, bs4, soundfile, sounddevice, mutagen; print('Python OK; CUDA available:', torch.cuda.is_available())"
Write-Host "`nSetup complete. Launch the app with run_gui.bat (or drag an .epub onto it)." -ForegroundColor Green
Write-Host "The Kokoro model (~330 MB) is downloaded automatically the first time you preview or convert."
