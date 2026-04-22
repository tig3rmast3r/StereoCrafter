# Linux/WSL Install (Fork)

Linux and WSL only for this fork atm. Windows partially works (only gui-nogui versions without sh batches, pipeline_master_gui not working as is tied to sh runners), and is not supported here. I strongly suggest using WSL anyway, it works already and have no downsides<br>
<br>
Preset values in this fork are tuned for this reference machine: Intel 265K, 48 GB RAM, RTX 4090. On different hardware, expect to adjust settings.

```bash
git clone https://github.com/tig3rmast3r/StereoCrafter --recursive
cd StereoCrafter
```

## 1) Make Scripts Executable

Run from repo root:

```bash
find . -maxdepth 2 -type f -name "*.sh" -print0 | xargs -0 chmod +x
```

## 2) Install Dependencies (Linux)

Run:

```bash
./install_linux_sh_readmefirst.sh
```

This script will:
- require `uv` as a prerequisite; if missing, it stops and prints a suggested install command instead of auto-installing it;
- optionally install Linux system packages needed for the standard runtime (`apt`);
- detect `ffmpeg` binaries already found in `PATH`, show path/version/NVENC status, and ask whether to keep the current one, install distro `ffmpeg` anyway, or stop;
- sync project dependencies with `uv` without forcing a torch reinstall;
- check current torch/cuda in the project environment;
- if missing, install the recommended stack automatically;
- if different, ask whether you want to align to the recommended stack;
- warn when older stacks (especially CUDA 11.8) are detected because they are usually slower;
- skip the optional Forward-Warp CUDA build by default (`BUILD_FORWARD_WARP=false`);
- show an informational WSL note without installing Linux CUDA drivers/toolkit in the standard flow.

Recommended target stack for this fork:
- `torch==2.9.1`
- `torchvision==0.24.1`
- `torchaudio==2.9.1`
- `CUDA 12.8` wheels

## 3) Mini Guide for Vast.ai

Tested only inpaint and depthcrafting job with docker vastai/pytorch:2.9.1-cuda-12.8.1-py310-24.04

From repo root:

```bash
git clone https://github.com/tig3rmast3r/StereoCrafter --recursive
cd StereoCrafter
find . -maxdepth 2 -type f -name "*.sh" -print0 | xargs -0 chmod +x
./setup_vast_stage1.sh
./setup_vast_stage2_weights.sh
```

Notes:
- Stage 1 installs system deps, validates preinstalled torch stack, and installs Python deps from `requirements.docker.no_torch.txt`.
- The standard Vast flow now skips the optional Forward-Warp CUDA build (`BUILD_FORWARD_WARP=false`).
- Stage 2 downloads model weights from Hugging Face.

## 4) Launch Pipeline GUI

After the installer script completes:

1. Close/reopen the terminal.
2. Reactivate your preferred Python environment.
3. Launch:

```bash
python pipeline_master_gui.py
```

## 5) WSL GUI note:
- The standard Linux installer now includes `python3-tk`, so Tkinter should be available after setup.
- If you are on WSL and the GUI still does not open even though `tkinter` imports correctly, the issue is usually WSLg/session state rather than a missing repo dependency.
- Quick check:

```bash
python -c "import tkinter; print('tk ok')"
```

- If that check passes but GUI windows still do not appear, restart WSL from Windows

```bash
wsl --shutdown
```

- If you have not insalled GUI for WSL yet:

```bash
sudo apt install x11-apps -y
sudo apt install gedit -y
wsl --shutdown
```
### Extra Note about WSL

PYTORCH_ALLOC_CONF expandable_segments:True doesn't work in WSL, at least on my pytorch/cuda combo, if your inpaint/depthcrafting job doesn't start try disabling it in pipeline_master_gui under "Options and RUN" Depthcrafter and Inpainting Retry Policy. you'll get increased chances of VRAM OOM so expect to adjust chunk sizes accordingly
