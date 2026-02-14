#!/usr/bin/env bash
set -euo pipefail

CMD=(python batch_depthcrafter_runner.py --worker_script "./depthcrafter_nogui_batch.py" --input_dir "./work/seg/" --output_dir "./work/depthmap/" --glob "*.mp4" --window_size 70 --overlap 20 --inference_steps 5 --guidance_scale 1.0 --seed 42 --cpu_offload_mode model --decode_chunk_size 2 --debug_mem True --final_upscale False --restart_every 100)

while true; do
  echo "[RUN] ${CMD[*]}"
  "${CMD[@]}" && break

  echo "[CRASH] runner crashed (segfault?). Cleaning + killing GPU python..."
  pkill -TERM -f "python.*(depthcrafter|StereoCrafter|stable-video-diffusion|diffusers)" || true
  sleep 2
  pkill -KILL -f "python.*(depthcrafter|StereoCrafter|stable-video-diffusion|diffusers)" || true

  #rm -rf "/path/out/.tmp_depthcrafter" || true
  sleep 2
done

