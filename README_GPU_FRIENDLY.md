# GPU-Friendly Orin Nano Run

This branch is intended for NVIDIA Jetson Orin Nano deployment.

What is different from `main`:
- stream-event inference no longer forces CPU for the detector and heavy phases
- size-fill estimation no longer forces CPU
- Linux/Jetson setup and run helpers are included
- event/size artifacts remain disabled by default, so saved photos/videos are not produced unless you explicitly enable them

## Workspace Layout

Use the two repos side by side:

```text
~/truckpipline_with_size/
  repo/         -> truckPipline_April_24
  repo_size/    -> truck_size_2
```

The main repo already auto-discovers size models from:
- local `weights/`
- sibling `../repo_size`

## Jetson Environment

1. Install JetPack and the NVIDIA-provided PyTorch/TorchVision build for your device.
2. Clone this repo into `repo`.
3. Clone `truck_size_2` into `repo_size`.
4. Run:

```bash
cd ~/truckpipline_with_size/repo
bash scripts/setup_orin_nano.sh
```

`requirements.jetson.txt` intentionally does not install:
- `torch`
- `torchvision`
- `opencv-python`

Those should come from the Jetson image / NVIDIA setup you already trust for the target device.

## Run

Recommended edge run:

```bash
cd ~/truckpipline_with_size/repo
bash scripts/run_orin_stream_event.sh /absolute/path/to/video.mp4
```

That script runs:
- `--mode stream_event`
- `--device auto`
- `--show 0`
- `--size-show 0`
- `--summary-only 1`
- `--non-interactive-model-select`

You can append extra flags after the video path if needed.

Example:

```bash
bash scripts/run_orin_stream_event.sh /data/video.mp4 --detect-conf 0.01
```

## Notes

- This branch is for edge deployment, not artifact-heavy debugging.
- Saved event/size photos remain disabled by default unless you pass:
  - `--save-event-artifacts 1`
  - `--save-size-artifacts 1`
- The repo `.gitignore` already excludes output directories, logs, and local environments.
