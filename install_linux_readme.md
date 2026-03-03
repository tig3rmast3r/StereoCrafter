# Linux Install (Fork)

Linux only for this fork. Windows may still work, but it is not supported here.<br>
<br>
Preset values in this fork are tuned for this reference machine: Intel 265K, 48 GB RAM, RTX 4090. On different hardware, expect to adjust settings.

## 1) Make Scripts Executable

Run from repo root:

```bash
find . -maxdepth 1 -type f -name "*.sh" -print0 | xargs -0 chmod +x
find ./Utilities -type f -name "*.sh" -print0 | xargs -0 chmod +x
```

## 2) Install Dependencies (Linux)

Run:

```bash
./install_linux_sh_readmefirst.sh
```

This script will:
- optionally install Linux system packages needed for runtime and Forward-Warp build (`apt`);
  includes `ffmpeg` and `mkvtoolnix`;
- sync project dependencies with `uv` without forcing a torch reinstall;
- check current torch/cuda in the project environment;
- if missing, install the recommended stack automatically;
- if different, ask whether you want to align to the recommended stack;
- warn when older stacks (especially CUDA 11.8) are detected because they are usually slower;
- export `LD_LIBRARY_PATH` for the current shell using the detected torch lib directory;
- optionally build Forward-Warp CUDA extension and print the detected `.so` path.

Recommended target stack for this fork:
- `torch==2.9.1`
- `torchvision==0.24.1`
- `torchaudio==2.9.1`
- `CUDA 12.8` wheels

## 3) Mini Guide for Vast.ai

From repo root:

```bash
chmod +x setup_vast_stage1.sh setup_vast_stage2_weights.sh
./setup_vast_stage1.sh
./setup_vast_stage2_weights.sh
```

Notes:
- Stage 1 installs system deps, validates preinstalled torch stack, and installs Python deps from `requirements.docker.no_torch.txt`.
- Stage 2 downloads model weights from Hugging Face.

## 4) Launch Pipeline GUI

After the installer script completes:

1. Close/reopen the terminal.
2. Reactivate your preferred Python environment.
3. Launch:

```bash
python pipeline_master_gui.py
```
