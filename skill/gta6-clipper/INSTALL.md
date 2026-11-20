## Install (Windows, macOS, Linux; Python 3.9 to 3.12)

Open a terminal in the skill folder, then:

| Step | Windows (PowerShell) | macOS (Terminal) | Linux (Debian/Ubuntu) |
|---|---|---|---|
| ffmpeg + ffprobe | `winget install Gyan.FFmpeg` | `brew install ffmpeg` | `sudo apt install ffmpeg` |
| Python venv | `py -3.12 -m venv .venv` | `python3 -m venv .venv` | `python3 -m venv .venv` (`sudo apt install python3-venv`) |
| Activate | `.venv\Scripts\activate` | `source .venv/bin/activate` | `source .venv/bin/activate` |
| Libraries (includes yt-dlp + deno) | `pip install -r requirements.txt` | same | same |

- Open a NEW terminal after installing ffmpeg. Check: `ffmpeg -version`, `python -m yt_dlp --version`. Every script stops with the install line above if ffmpeg is missing.
- GPU is optional. Windows/Linux with an NVIDIA card: `pip install -r requirements-gpu.txt` (Windows is wired in; on Linux also `export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas.lib,nvidia.cudnn.lib,os;print(os.path.dirname(nvidia.cublas.lib.__file__)+':'+os.path.dirname(nvidia.cudnn.lib.__file__))")`). Mac (Intel or Apple Silicon) and no-GPU PCs: nothing to do, transcription falls back to CPU int8 on its own (model "base").
- Colour emoji in captions need Segoe UI Emoji (Windows), Apple Color Emoji (Mac) or `sudo apt install fonts-noto-color-emoji` (Linux); without it the emoji style is skipped, nothing breaks. Caption fonts ship in `fonts/`.
- Scheduled runs (Autopilot tab, always shown as a dry preview first): Windows = Task Scheduler (`schtasks`), macOS = launchd agent `~/Library/LaunchAgents/io.clipper.autopilot.plist` (`launchctl load -w`), Linux = a `crontab` line tagged `# clipper-autopilot` (a systemd user timer preview is shown too). Turn off = the "off" command in the same preview. The computer must be on and logged in at that time.
- Always run with `python` from the venv (the skill calls `sys.executable`, never a hard-coded path).
