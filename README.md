# Fork Info

- added sh + runners for headless batches (mainly for docker usage and handling errors/skip)
- Depthcrafter script memory usage optimization (will allow larger files before OOM)
- Splatting improvement, blur for left borders + clean mask for merge
- Multithreaded Sharpness Analyzer Script - will analyze splat and predict sharpness inside the masked (inpaint) area (to be used relatively with inpaint steps)
- Inpaint with dynamic step selector based on sharpness.csv + Stream behaviour (will allow much longer files)
- Merging_gui using the new clean mask + single process button
- Multithreaded RealESRGAN upscale script
- Rejoin segments ps1 script (ffmpeg)

## Fork workflow example (1080p)

NOTE: always check files after each step
```bash
find ./work/***folder*** -maxdepth 1 -type f -name '*.mp4' -print0 | xargs -0 -n1 -P"$(nproc)" sh -c 'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames -show_entries format=duration -of default=nw=1:nk=1 "$1" >/dev/null || echo "BROKEN: $1"' _
```
NOTE2: this is a very long task, consider ~1 hour for 1 minute with an RTX 5090

### Step 1- SceneDetect

Pick your source.mkv and divide by scenes using SceneDetect (not included), use same pass to crop bars if needed
```
IN="./work/source.mkv"
OUT_SCENES="./work/seg/"
```
for full 1080p (no black bars)
```
scenedetect -i "$IN" detect-adaptive -t 2.0 split-video -o "$OUT_SCENES" -a "-map 0:v:0 -an -dn -sn -c:v libx264 -crf 0 -preset veryfast -pix_fmt yuv420p"
```
1080p IMAX -> 1024
```
scenedetect -i "$IN" detect-adaptive -t 2.0 split-video -o "$OUT_SCENES" -a "-map 0:v:0 -an -dn -sn -vf crop=iw:1024:0:trunc(((ih-1024)/2)/2)*2 -c:v libx264 -crf 0 -preset veryfast -pix_fmt yuv420p"
```
1080p standard -> 832
```
scenedetect -i "$IN" detect-adaptive -t 2.0 split-video -o "$OUT_SCENES" -a "-map 0:v:0 -an -dn -sn -vf crop=iw:832:0:trunc((ih-832)/4)*2 -c:v libx264 -crf 0 -preset veryfast -pix_fmt yuv420p"
```
### Step 2 - DepthCrafter

just run
```bash
chmod +x run_depthcrafter_runner.sh
./run_depthcrafter_runner.sh
```
If your work folder is ./work you don't need to do anything else
It will output depthmap at exactly half size

### Step 3 - Upscale Depthmaps

i use Real ESRGAN (not included) for upscaling, it will help a lot with precision compared to other upscales
5th arg is tile size, i suggest vertical resolution as tile size (faster)
it will launch 4 batches in parallel (can be changed by args or inside the script)
```
./upscale_esrgan.sh "./work/depthmap" "./work/depthmap/upscaled" 2 realesr-animevideov3-x2 416
```

### Step 4 - Splatting

just run with default values
```bash
chmod +x run_splatting_bm_runner.sh
./run_splatting_bm_runner.sh
```
or play with splatting_bm_gui.py to find your own and then change options inside sh and runner accordingly.
this bm_gui version will have optional blur on left edges and optional extra binary mask to be used with merging_gui and mask analyze, saved in ./work/mask

### Step 5 - Analyze mask sharpness

```
python analyze_inpaint_sharpness_newmask.py "./work/splat/hires" "./work/mask" --out_csv "./work/sharpness.csv"
```
Will predict masked zone sharpness and save to csv

### Step 6 - Inpaint

just run and wait (a lot)
for 24GB VRAM (RTX4090) i suggest tile 2, frame chunk size up to 50 and overlap to 4
for 32GB VRAM (RTX5090) you can push up to 80 chunk size
```bash
chmod +x run_inpainting_runner.sh
./run_inpainting_runner.sh
```

### Step 7 - Merge inpaint

run or change settings using gui preview and apply on the runner script
Runner uses ReplaceMask as default option.

```bash
chmod +x run_merging_runner.sh
./run_merging_runner.sh
```

### Step 8 - Merge Scenes and Compress

I merge in Windows with ffmpeg hevc nvenc

```
Rejoin_HEVC_NVENC.ps1
```

### Step 9 - Remux other Streams (audio,subs,etc..)

use mkvtoolnix to remux (i use gui version in Windows)

#### Dependencies (not included)
-RealESRGAN
-ffmpeg
-Scenedetect
-MKVToolNix

# StereoCrafter GUI + DepthCrafter GUI Seg

You can learn more about DepthCrafter GUI Seg <a href="https://github.com/Billynom8/DepthCrafter_GUI_Seg">here</a>.

## Installation

### Option 1: Installer script (Windows)

#### PREREQUISITES:
   - GIT: Ensure Git is installed and added to your system’s PATH.<br>
     Download here: https://git-scm.com/downloads/win<br>
     You can check the installation by running the command:<br>
       `git --version`<br>
     If it shows a version, Git is installed and on PATH.
   
   - CUDA ToolKit: Ensure CUDA 12.8 is installed and added to your PATH.<br>
     Download here: https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64<br>

   - FFMPEG: Ensure FFMpeg is installed and added to your PATH.<br>
     See [Here](https://techtactician.com/how-to-install-ffmpeg-and-add-it-to-path-on-windows/) for a tutorial on how to install.


#### INSTALL:
   - Run <a href="https://github.com/enoky/StereoCrafter/blob/main/_install/StereoCrafter_1click_Installer.bat">script</a> from folder where you want StereoCrafter installed
   - Download and extract <a href="https://mega.nz/file/Fw1GgJrL#bPplu2Y1PT4G-TM29zcGNENUYVySEk2NENT4krkjEso">model</a> "weights" to StereoCrafter folder (use <a href="https://www.qbittorrent.org">qBittorrent</a> to download)

<hr>

### Option 2: Manual Install

For Manual Install Instructions <a href="https://github.com/enoky/StereoCrafter/blob/main/_install/StereoCrafter_Manual_Install.md">Click Here</a>

<hr>
<div align="center">
<h2>StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos</h2>

Sijie Zhao*&emsp;
Wenbo Hu*&emsp;
Xiaodong Cun*&emsp;
Yong Zhang&dagger;&emsp;
Xiaoyu Li&dagger;&emsp;<br>
Zhe Kong&emsp;
Xiangjun Gao&emsp;
Muyao Niu&emsp;
Ying Shan

&emsp;* equal contribution &emsp; &dagger; corresponding author 

<h3>Tencent AI Lab&emsp;&emsp;ARC Lab, Tencent PCG</h3>

<a href='https://arxiv.org/abs/2409.07447'><img src='https://img.shields.io/badge/arXiv-PDF-a92225'></a> &emsp;
<a href='https://stereocrafter.github.io/'><img src='https://img.shields.io/badge/Project_Page-Page-64fefe' alt='Project Page'></a> &emsp;
<a href='https://huggingface.co/TencentARC/StereoCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow'></a>
</div>

## 💡 Abstract

We propose a novel framework to convert any 2D videos to immersive stereoscopic 3D ones that can be viewed on different display devices, like 3D Glasses, Apple Vision Pro and 3D Display. It can be applied to various video sources, such as movies, vlogs, 3D cartoons, and AIGC videos.

![teaser](assets/teaser.jpg)

## 📣 News
- `2024/12/27` We released our inference code and model weights.
- `2024/09/11` We submitted our technical report on arXiv and released our project page.

## 🎞️ Showcases
Here we show some examples of input videos and their corresponding stereo outputs in Anaglyph 3D format.
<div align="center">
    <img src="assets/demo.gif">
</div>


## 🛠️ Installation (for the original repository)

#### 1. Set up the environment
We run our code on Python 3.8 and Cuda 11.8.
You can use Anaconda or Docker to build this basic environment.

#### 2. Clone the repo
```bash
# use --recursive to clone the dependent submodules
git clone --recursive https://github.com/TencentARC/StereoCrafter
cd StereoCrafter
```

#### 3. Install the requirements
```bash
pip install -r requirements.txt
```


#### 4. Install customized 'Forward-Warp' package for forward splatting
```
cd ./dependency/Forward-Warp
chmod a+x install.sh
./install.sh
```


## 📦 Model Weights

#### 1. Download the [SVD img2vid model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for the image encoder and VAE.

```bash
# in StereoCrafter project root directory
mkdir weights
cd ./weights
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
```

#### 2. Download the [DepthCrafter model](https://huggingface.co/tencent/DepthCrafter) for the video depth estimation.
```bash
git clone https://huggingface.co/tencent/DepthCrafter
```

#### 3. Download the [StereoCrafter model](https://huggingface.co/TencentARC/StereoCrafter) for the stereo video generation.
```bash
git clone https://huggingface.co/TencentARC/StereoCrafter
```


## 🔄 Inference

Script:

```bash
# in StereoCrafter project root directory
sh run_inference.sh
```

There are two main steps in this script for generating stereo video.

#### 1. Depth-Based Video Splatting Using the Video Depth from DepthCrafter
Execute the following command:
```bash
python depth_splatting_inference.py --pre_trained_path [PATH] --unet_path [PATH]
                                    --input_video_path [PATH] --output_video_path [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the DepthCrafter model weights (e.g., `./weights/DepthCrafter`).
- `--input_video_path`: Path to the input video (e.g., `./source_video/camel.mp4`).
- `--output_video_path`: Path to the output video (e.g., `./outputs/camel_splatting_results.mp4`).
- `--max_disp`: Parameter controlling the maximum disparity between the generated right video and the input left video. Default value is `20` pixels.

The first step generates a video grid with input video, visualized depth map, occlusion mask, and splatting right video, as shown below:

<img src="assets/camel_splatting_results.jpg" alt="camel_splatting_results" width="800"/> 

#### 2. Stereo Video Inpainting of the Splatting Video
Execute the following command:
```bash
python inpainting_inference.py --pre_trained_path [PATH] --unet_path [PATH]
                               --input_video_path [PATH] --save_dir [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the StereoCrafter model weights (e.g., `./weights/StereoCrafter`).
- `--input_video_path`: Path to the splatting video result generated by the first stage (e.g., `./outputs/camel_splatting_results.mp4`).
- `--save_dir`: Directory for the output stereo video (e.g., `./outputs`).
- `--tile_num`: The number of tiles in width and height dimensions for tiled processing, which allows for handling high resolution input without requiring more GPU memory. The default value is `1` (1 $\times$ 1 tile). For input videos with a resolution of 2K or higher, you could use more tiles to avoid running out of memory.

The stereo video inpainting generates the stereo video result in side-by-side format and anaglyph 3D format, as shown below:

<img src="assets/camel_sbs.jpg" alt="camel_sbs" width="800"/> 

<img src="assets/camel_anaglyph.jpg" alt="camel_anaglyph" width="400"/>

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects:
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models): A latent diffusion model trained to generate video clips from an image or text conditioning.
- [DepthCrafter](https://github.com/Tencent/DepthCrafter): A novel method to generate temporally consistent depth sequences from videos.


## 📚 Citation

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```
