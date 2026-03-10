# Changes in this Fork

This fork is tested only with Linux (ubuntu 22.04 and 24.04 in my case), Windows "should" work but at least you need a Win build for RealESRGAN <br>
Aim to this Fork is to create a full sbs 1080p 3d content as result with a one-click solution but keeping ways to customize it.<br>
Supports only 8 bit format (inpainting is 8 bit anyway so it's actually impossible to get 10 bit hdr end to end) with the following pix formats yuv420p, yuv422p and yuv444p (latest 2 can be achieved downscaling from 4k, no chroma upscaling)
All runner scripts are tuned for intel 265k, 48GB ram and RTX 4090<br>
Most of the job is done by VIBE CODING using VS code + Codex 5.3<br>
All the extra scripts are made multithreading/parallel when possible, helping reducing time (but is still a very very slow process)

## Summarized Changes

### Install and dependencies
- upgrade numpy!! v2.x is much faster and totally compatible with this workflow
- updated pyproject.toml for linux and added sh script for quick install
- Pytorch/cuda will not be overwritten automatically
- changed xformers and Triton-Windows platform dependant (Win)
- added SceneDetect
- added scripts for quick install on docker (tested vast.ai only)
- check your system for use_gpu flag in merging processes, in my case is slower with that

### Depthcrafter
- no_gui headless version with same functionalities from the gui version, segmenting is not supported at this time
- runners for no_gui version, will autostart/retry from errors, skip if done already and supports ctrl-c graceful stop
- small changes to depthcrafter script to reduce Vram spikes during load
- auto routine for depthcrafter running at half size and then upscaling 2x with REALesrgan (build included in this project)

### Splatting
- no_gui wrapper to run gui version headlessly (not a pure headless), it doesn't support sidecars/borders at this time
- runners for no_gui version, with parallel run support, will autostart/retry from errors, skip if done already and supports ctrl-c graceful stop
- New left side smoother to reduce stairs on warped objects
- New binary Replace Mask generation without noise
- New option Single_warped (splatted1), will use Replace Mask instead of the legacy one on the following steps
- New MinBorders auto convergence, will calculate the best convergence value to reduce needed inpaint on borders, will usually get more popout scenes (convergence range 0.2<->0.8)
- optimized all the pipeline and reduced vram usage by up to 90%, allowing parallel processing.

### Inpainting
- no_gui wrapper to run gui version headlessly (not a pure headless), it doesn't support post processing blend and color transfer at this time (a more powerful color transfer has been moved on the merging step)
- runners for no_gui version, will autostart/retry from errors, skip if done already and supports ctrl-c graceful stop
- works perfectly on vast.ai docker
- Implemented full stream end-to-end, will only load required frames for the current chunk, greatly reducing Vram usage on longer files, it basically doesn't matter how long it is.
- Cherry picked chunk overlap from commit https://github.com/Billynom8/StereoCrafter/commit/708bdba3cb86b30ccaa7c6e281e650ec09a7f7ea adapted for my workflow
- Tail-pad will create extra frames for every chunk and at the end of file and then discard them, last generated frame(s) from chunks suffer from terrible color mismatch, with tail pad those frames are discarded and not used for overlap, greatly decreasing color "flashes" on chunk junctions and on the last frame of each scene. Will increase inpaint time a bit but totally worth it
- Option to use Replace Mask as source
- in this Fork version mask is not Pre-Processed and used "as is" for the inference.
- Optional (but recommended) scene analysis, will predict inpaint sharpness to csv and it will be used to regulate inpaint steps automatically, it will basically increase steps only when needed, result are still far from being perfect in some cases, but is a good tradeoff (based on my tests to get near perfect quality you should inpaint at double size, basically MONTHS of encoding with current tech, it's already quite slow as it is).

### Merging
- due to many improvements, and to avoid crashes, and to speed up time, merging is now split in 3 phases, autoct csv (optional), processed mask and merge, you can still use merging_gui as previewer but i strongly discourage processing with it, it will crash.
- Requires Replace Mask for full functionality
- Sliders revamp for mask processing, removed some and improved others, you can now set shadow mask by pixels, binarize mask is not needed (has no effect) for Replace Mask,there is no noise to remove (but i kept it if you want to use the legacy one), also with binary mask you need much less dilate/blur most of the times.
- new option dynamic shadow based on mask thickness, less thick, less shadow, will reduce unneeded inpaint merge zones.
- new option to increase shadow on fast movements, will counter reduced warp precision on fast moving objects automatically
- New mask preprocessing script, will preprocess mask in a separate step, this way you will be able to use more workers and it will take a fraction compared to keeping it into the merging step, also it's stable.
- New GUI Mask For Merge to analyze mask with the new features, with this previewer you change the motion mask behaviour and store settings on the other scripts
- new shadow feature for auto inverting direction, it will invert shadow when mask is touching the right border, we need that when the right border is inpainted
- New Color Transfer panel with 3 working modes and 8 curated presets (ordered by my own tests, the upper ones has more chances to be the good ones)
- First working mode: fixed preset, will use the same preset for all scenes, like it was before, the legacy preset is the one that was present in the older merging_gui, this is the fastest one but be advised that some scenes will work better with other presets.
- Second working mode: AutoCT, it will test and choose the best preset frame by frame on the fly, i've been able to parallelize some stuff but is still quite slow, results are usually very good but sometimes you'll get some flickerings caused by preset changes.
- Third and recommended mode, Auto CSV, you'll need to run a preprocess run that will Auto CT for each frame and save to CSV, having the full CT preset list before the merging process will allow for blending and also it will support oscillating when presets are moving regularly between A B A B pattern (it happens quite often as inpainted scenes usually has this frame color behaviour), this is the slowest one but color fidelity is much higher this way and chances of flickering are almost absent.
- no_gui headless version with same functionalities from the gui version, i've also made a parallel version, but be warned as it requires lot of ram and even with just 2 workers you may go out of ram (with 48GB) if scenes are too long. Best practice is to merge longer scenes (30+ sec) with the single thread script and then use the parallel one for the others. Borders/sidecar are not supported at this time.

### Final rejoin
- included script for rejoin with Nvenc
- other extra script to remux with original file replacing video, with mkvmerge (not included in this project)

### Extra Utilities
- verifyscenes to check for files integrity and length, comparing with reference clips
- mpv player script with ability to click on screen and save annotations to csv (useful for testing/analysis)
- seg mono to sbs, will simply make mono sbs (you may not want to make ending titles in 3d but need that to be sbs before rejoin)

### pipeline master GUI
- made a GUI that will do all the above steps automatically and sequentially, from start to finish with a few clicks, auto presets should be fine in most cases but is quality aimed (slow)
- Main features:
- includes the very first step with Scenedetect (create clips)
- For each step it's possible to switch from "auto" curated preset to manual settings
- tracking of current state + flags (done/verified) and resume
- auto verify for each step and auto delete/retry bad files

### "legacy" fork
- i moved some stuff into legacy folder mainly to cleanup the repo root
- on runners i have changed the "finished" moving thing behaviour in favour of using checks and skips without moving source files
- due to many changes is almost impossible to merge right now but i want to really thank all the guys from TencentARC and the Enoky/Billynom8 fork that made so many improvements, as that fork is still quite active i'll try to cherry pick useful improvements when i see something that may fit this workflow<br>
Long live to 3D!!

# CHANGE LOG

2026-03-06
- First release, now that everything is completed and almost functional i'll track changes from now on. If you tried this fork before the above date, it probably was not working as intended.
