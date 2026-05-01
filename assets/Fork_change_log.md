# Changes in this Fork

This fork is tested only with Linux (ubuntu 22.04 and 24.04 in my case) and WSL 24.04 on Windows, pure Windows "should" work for gui versions but pipeline_master_gui and all auto batches uses bash scripts that needs to be reimplemented using powershell. <br>
All GUI works too in WSL so moving to Windows is quite pointless<br>
Aim to this Fork is to create a full sbs 1080p 3d content as result with a one-click solution but keeping ways to customize it.<br>
Supports only 8 bit format (inpainting is 8 bit anyway so it's actually impossible to get 10 bit hdr end to end) with hardcoded yuv444p during all steps to preserve colors transitions.
All runner scripts are tuned for intel 265k, 48GB ram and RTX 4090 and specifically for 1920*800 content<br>
Most of the job is done by VIBE CODING using VS code + Codex 5.3/5.4<br>
All the extra scripts are made multithreading/parallel when possible, helping reducing time (but is still a very very slow process)

## Summarized Changes

### Install and dependencies
- upgrade numpy!! v2.x is much faster and totally compatible with this workflow
- updated pyproject.toml for linux and added sh script for quick install
- Pytorch/cuda will not be overwritten automatically
- Forward-Warp CUDA build is now treated as optional/experimental and is skipped by default in the standard Linux/Vast install flow
- changed xformers and Triton-Windows platform dependant (Win)
- added SceneDetect
- added scripts for quick install on docker (tested vast.ai only)
- check your system for use_gpu flag in merging processes, in my case is slower with that

### Depthcrafter
- no_gui headless version with same functionalities from the gui version, segmenting is not supported at this time
- runners for no_gui version, will autostart/retry from errors, skip if done already and supports ctrl-c graceful stop
- small changes to depthcrafter script to reduce Vram spikes during load
- alternate script (stream based) will allow depthcrafting from any length source with some compromises

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
- Cherry picked chunk overlap from commit https://github.com/Billynom8/StereoCrafter/commit/708bdba3cb86b30ccaa7c6e281e650ec09a7f7ea adapted for my workflow (now reverted with no crossfade during overlap)
- Tail-pad will create extra frames for every chunk and at the end of file and then discard them, last generated frame(s) from chunks suffer from terrible color mismatch, with tail pad those frames are discarded and not used for overlap, greatly decreasing color "flashes" on chunk junctions and on the last frame of each scene. Will increase inpaint time a bit but totally worth it
- Option to use Replace Mask as source
- in this Fork mask is not Pre-Processed and used "as is" for the inference.
- Optional (but recommended) scene analysis, will predict inpaint sharpness to csv and it will be used to regulate inpaint steps automatically, it will basically increase steps only when needed, result are still far from being perfect in some cases, but is a good tradeoff.
- Dynamic Chunk based on scene sharpness and mask persistence

### Merging
- due to many improvements, and to speed up time, merging is now split in 3 phases, autoct csv (optional), processed mask and merge.
- Requires Replace Mask for full functionality
- Sliders revamp for mask processing, removed some and improved others, you can now set shadow mask by pixels, binarize mask is not needed (has no effect) for Replace Mask,there is no noise to remove (but i kept it if you want to use the legacy one), also with binary mask you need much less dilate/blur most of the times.
- new option dynamic shadow based on mask thickness, less thick, less shadow, will reduce unneeded inpaint merge zones.
- new option to increase shadow on fast movements, will counter reduced warp precision on fast moving objects automatically
- New mask preprocessing script, will preprocess mask in a separate step, this way you will be able to use more workers and it will take a fraction compared to keeping it into the merging step.
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
- requeue annotated scenes will delete/create subset/replace/join using annotations from the csv

### pipeline master GUI
- made a GUI that will do all the above steps automatically and sequentially, from start to finish with a few clicks, auto presets should be fine in most cases but is quality aimed (slow)
- Main features:
- includes the very first step with Scenedetect (create clips)
- For each step it's possible to switch from "auto" curated preset to manual settings
- tracking of current state + flags (done/verified) and resume
- auto verify for each step and auto delete/retry bad files
- Test run support with a few clips (selectable)
- Extra Sharpen step to sharpen inpainted zone selectively

### "legacy" fork
- i moved some stuff into legacy folder mainly to cleanup the repo root
- on runners i have changed the "finished" moving thing behaviour in favour of using checks and skips without moving source files
- due to many changes is almost impossible to merge right now but i want to really thank all the guys from TencentARC and the Enoky/Billynom8 fork that made so many improvements, as that fork is still quite active i'll try to cherry pick useful improvements when i see something that may fit this workflow<br>
Long live to 3D!!

### Time to encode
- in linux with intel 265k 48GB Ram + RTX 4090 total time is 0.016x, that means 1h->60h required time. Half of this time is taken by the inpainting step. A full power RTX 5090 (575+ Watts) with adjusted chunks sizes will gain around 20% on depthmap/inpaint steps

# CHANGE LOG

2026-03-06
- First release, now that everything is completed and almost functional i'll track changes from now on. If you tried this fork before the above date, it probably was not working as intended.

2026-03-12 (Pipeline Master GUI improvements and fixes)
- [new] Scenedetect now detects scenes only, and saves to csv, a new step has been added on pipeline master gui for ffmpeg processing, with multithreading encoding and resume
- [new] New depthcrafter slider allow for larger size instead of a fixed 0.5 for encoding, realesrgan will still double and additionally downscales to fit original size.
- [new] Configurable number of clips for Test run
- [changed] Merging and splatting script will use mono thread script automatically when workers are set to 1, removed checkbox on Merging TAB
- [fix] Test run will keep the jobs if stopped or when finished
- [fix] Minor fixes to popup messages and explicit stop print
- [fix] RealESRGAN error with x264
- [fix] Test run fails on some steps (scripts will now search for simlinks too)
- [fix] Test run will now honor test csv
- [fix] Hardcoded xformers disabled on Depthcrafter no_gui scripts
- [fix] Missing workers option for Mask_for_merge batch
- [fix] Feedback and progress for Splatting, Mask for merge and Merging batches when used in parallel
- [fix] Run should not stop anymore after a script crash/error

2026-03-13 (Pipeline Master GUI improvements and fixes)
- [new] Auto-Retry configurable policies for Depthcrafting and Inpainting steps and skip-to-next on permanent fails
- [changed] tuned some default and auto values (press "Reset Settings" to load new defaults if you have updated the repo from a previous version)
- [fix] Simlink scan for Test Run missing in RealESRGAN scripts
- [fix] Improved error handling and reduced chances of run stop
- [fix] reverted forward-warp to pytorch version for depthcrafter step as i'm experiencing memory leaks with the cuda version, will re-enable once fixed.

2026-03-14 (Pipeline Master GUI fixes)
- [fix] wrong progress feedback for merging script on errors/restarts
- [fix] verifyscenes will also delete apparently good files but with wrong length

2026-03-16 (New Scenes requeue utility, better "Stop" and fixes)
- [new] New Utility that will automatically move/delete selected scenes from the project in order to re-process them again. It accepts annotations csv from Utilities/run_sbs_left_click_logger_player.sh or manual names on the specified TextBox (file name in this case will automatically set the step)
- [new] Start/resume will now become "Stop" when running, in order to stop the run even during verifyscenes
- [fix] VerifyScenes clips deletion was not working during Test Runs
- [fix] Unstable Mask_for_merge workers near run end
- [fix] run suddenly stops after AutoCT step in some cases
- [fix] Verifyscenes on merge tab will exclude files coming from seg-mono during the process
- [fix] latest fixes about Depthcrafter run stability was not implemented on the Inpainting step
- [fix] split scene from csv was using length time instead of length frame

2026-03-20 (Sharpen, PMG changes and fixes)
- [new] Preliminary Sharpen integration, will sharp inpainted mask zones a bit to compensate for blurry inpaint that is more prominent on masked zones, it will sharp only scenes that requires it (based on sharpness.csv)
- [changed] reduced max steps from 11 to 8 for inpainting (sharpen will do a better job than extra inpaint steps)
- [changed] RealESRGAN step is now optional, while sometimes gives good results (eg, with downscaled 4k to yuv444p), some other times it has issues and give results worse than using the original 0.5x depthmap, so for safety i have disabled it as default but can be re-enabled in manual mode
- [changed] Fine tuned several defaults for auto mode according to the above change and to new tests with other sources, defaults are now tuned for a standard 1920*800 window as input
- [fix] depthcrafter automatic download was loading an older model for img2vid

2026-03-21 (Fixes!)
- [fix] updated numpy in vast requirements
- [fix] check files/test run behaviour
- [fix] auto resume if pipeline_master_gui crashes during a test run
- [fix] existing sharpness.csv was not honored on test runs
- [fix] AutoCT csv missing speed optimizations (2.5x faster)
- [fix] Run stops when shapen step throws an error

2026-03-22 (More Fixes)
- [changed] merging step will auto restart workers after each file to prevent RAM buildup
- [fix] Sharpen clips had incorrect fps causing join step to fail
- [fix] seg-mono to flat sbs clips was not identical to merged clips

2026-04-01 (lot of stuff)
- [new] the old crf/qp 1 preset is too lossy if the content is already on the 10mps compress range, so from now on the default preset is lossless for all intermediate steps, with fixed yuv444p, there is now a global preset under "Options and run", the old crf/qp is still available with value 0 and 1, but with hardcoded yuv444p so it will be better than before.
- [new] there is now a global unified script for intermediate steps encoding (dependency/ffmpeg_encoding_profiles.py)
- [new] "Codec Validation" button will test all codec combinations used in the script, run this check to make sure your system ffmpeg will work for every step.
- [new] requeue_annotates_scenes_gui improvements: optional delete and optional create subset with selected clips + final replace into main work folder
- [new] new options for requeue annotaed scenes gui: rejoin in-place, will make the join step preferring the sbs outputs in the subset. Compare join will make a joined sbs with old-new-old-new pattern for a quick compare
- [new] Alternate script (stream based) for Depthcrafter, it will work with files from any size but results will be a bit different, so avoid it until is strictly necessary (it will basically miss the latents_all calculation on the entire clip, results can be more unstable chunk by chunk and have a narrower contrast range), i have included it as last retry (4th attempt).
- [changed] auto preset according to new changes, depthcrafter step will take longer but is crucial to get good results on all the following ones.
- [changed] removed pix_fmt option everywhere (except last join step), it will always use yuv444p to maintain better colors transitions during the steps.
- [changed] Included window,overlap and script selection under the depthcrafter retry policy
- [changed] Removed "inherited" flag for cpu offload on inpaint menus retry and modified it on depthcrafter menu to include all parameters and not just cpu offload
- [changed] vast_stage2 script will now download only required model instead of the full repo, with option for specific steps (inpaint,depthcrafter,all)
- [changed] depthcrafter max resolution slider increased to 1x
- [changed] "join scenes" step will now join an incomplete set with pop-up warning
- [changed] removed VerifyScenes (deep) buttons, the current quick verify is enough, there is still verifyscenes.py script to launch manually.
- [changed] requeue script now support both clicks on the final merged file and clicks on the sbs folder
- [fix] Requeue will delete/move according to the selected step non from the one after it
- [fix] sharpen step was missing in requeue scenes script
- [fix] Scene names are back with 4 digits to avoid incorrect clip sequence during steps
- [fix] Some fields on Merge step was resetting to default on GUI reload
- [fix] Several other minor fixes

2026-04-06 (WSL ready)
- [changed] removed display dependency from splatting step
- [changed] Forward Warp build now optional (default: off)
- [changed] improved install script + WSL friendly + ffmpeg checks
- [changed] removed RealESRGAN step completely
- [changed] repo mini refactor, log files into /logs, configs into /configs, sh scripts and batch runners into /runners
- [changed] pipeline_master_gui now supports multiple configurations, in order to work with more projects simultaneously, to enable custom config (json will be saved and honored into the work folder) launch with arg --work_dir "work path"
- [fix] test run suddenly stops on merge step in some cases

2026-04-16 (Inpainting Changes)
- [new/changed/fix] i want to talk about inpaint and its flaws, there are mainly 2 problems with inpaint atm:<br>
1 - color flashes: at first i tought that color flashes was only on the last frame of each chunk and i fixed it by adding tail_pad so the last frame for every chunk and at the end of each clip is calculated but not sent into the stream. I've discovered later that the same identical color flash is present also on the last overlap frame of the following chunks, so when i've cherry picked the crossfade commit the problem has arised again. for this reason i've removed crossfade when is overlapping and the new chunk starts to be seen after the overlap has done. There are still some traces on the following frames but at least the huge flash is now gone again.<br>
2 - blurry inpaint: when the masked zone is detailed there is a very visible difference between the inpainted zone and the surroudings, the inpainted zone is really blurry compared to the rest, with the older auto setting this issue was more prominent, as the blurry ratio is directly proportional to the chunk size, switching back to tile 1 and shorter chunks helped a lot but there's a downside, it may happen that for static and long scenes the inpainted zone tends to become worse overtime, for this other reason some scenes are better using longer chunks, so in order to let everything going "automatically", the script has to decide if it's better to use shorter or longer chunks depending on 2 factors:<br>
-length of static mask areas<br>
-expected amount of sharpness for the warp areas<br>
when a main mask area is static the max chunk size will be adjusted to avoid the degratation for that area, if the chunk can be used with tile 1 it will be used, (it's faster with tile 1), but if the chunk will require more frames it will fallback to tile 2, you can set the maximum possible frames for your current Vram pool with tile 1 and 2, in my case with 4090 in Linux is 22 and 55 (+1 tail_pad) so this is the default value in pipeline_master_gui<br>
If no static Masks are detected (or are short), chunk size will be inversely proportional to the expected sharpness, paired with number of steps it will give best results.<br>
Still testing, may need some fine tuning<br>
- [fix] Requeue annotated scenes now copy all files from the previous steps and not just the ones required by the selected step
- [fix] stand-alone script settings are now aligned with pipeline_master_gui

2026-04-22 (inpainting changes part 2 + graceful stop fix)
- [new] improved static mask detection, now it takes into account the ROI next to the mask to evaluate if is going to degrade overtime with small chunks
- [new] dynamic chunk settings and static mask divisor are now available as CLI vars and in pipeline_master_gui
- [changed] dynamic chunk preset values
- [changed] added the other auto-convergences in pipeline_master_gui
- [fix] Stop/Graceful Stop should work correctly now on all steps from both scripts and pipeline master gui
- [fix] rare NaN case on AutoCT
- [fix] MinBorder auto convergence clamps was missing

2026-05-01 (headless pipeline + dynamic inpaint resolution + speed optimizations)
- [new] Headless script for Pipeline Master
- [new] Dynamic resolution will now adapt resolution from 50% to 100% dynamically, greatly reducing inference time with abysmal differences in most cases
- [new] resolution drop-down can be used to set the max resolution in dynamic mode and the fixed resolution in static mode
- [new] benchmark mode will test and setup inpaint chunk sizes based on your GPU
- [change] new preset changes, "speed" aimed with minimal quality loss, lowered number of inference steps based on sharpness.csv, lowered depthmap steps from 5 to 4, lowered max inpaint resolution to 90%, depthcrafter retries are now faster if the first run fails, with this new setup the first run will process scenes up to ~10 sec, retry 1 up to ~20 sec at 0.95x speed, retry 2 up to ~30 sec at 0.33x speed, retry 3 any length with 1x speed but with stream script (that misses file level latents, only overlap level) 
- [change] depthcrafter retries are now relative to the chunk size (offsets) so they will adapt automatically based on the chunk size without having to tune them manually
- [fix] Missing requirements for docker setup to enable full headless pipeline