# Edge Validated Run (Orin Nano 8GB)

This is the validated edge run guide for NVIDIA Jetson Orin Nano 8GB.

It documents the current integrated behavior of this repo:
- repo 1 (`truckPipline_April_24`): compliance/event pipeline (Phase 1..5)
- repo 2 (`truck_size_2`): fill estimation
- merged output: one event line per finalized truck with compliance + fill result

## 1) Target Deployment Layout

Use this side-by-side layout:

```text
~/truckpipline_with_size/
  repo/         -> truckPipline_April_24
  repo_size/    -> truck_size_2
```

Why this layout matters:
- `repo` auto-discovers size models from local `weights/` and sibling `../repo_size`.
- integrated stream-event mode expects the second repo models to be reachable.

## 2) Branches To Use

Use `gpu-friendly` for both repos:
- `truckPipline_April_24`: `gpu-friendly`
- `truck_size_2`: `gpu-friendly`

Clone:

```bash
mkdir -p ~/truckpipline_with_size
cd ~/truckpipline_with_size

git clone --branch gpu-friendly https://github.com/mohammadalkhawaldah/truckPipline_April_24.git repo
git clone --branch gpu-friendly https://github.com/mohammadalkhawaldah/truck_size_2.git repo_size
```

## 3) Jetson Orin Nano 8GB Prerequisites

Before repo setup, the device should already have:
- JetPack installed
- NVIDIA-compatible `torch` / `torchvision`
- working OpenCV in your Jetson environment

Important: `requirements.jetson.txt` in this repo intentionally installs only:
- `ultralytics==8.3.173`
- `numpy==1.26.4`
- `Pillow==10.4.0`
- `pandas==2.2.3`
- `PyYAML==6.0.2`
- `tqdm==4.66.5`

It intentionally does not install/pin `torch`, `torchvision`, or `opencv-python`.

## 4) Setup (Both Repos)

Run the setup helper in each repo:

```bash
cd ~/truckpipline_with_size/repo
bash scripts/setup_orin_nano.sh

cd ~/truckpipline_with_size/repo_size
bash scripts/setup_orin_nano.sh
```

Sanity check CUDA visibility in the main repo venv:

```bash
cd ~/truckpipline_with_size/repo
.venv_orin/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected on Orin: `torch.cuda.is_available()` prints `True`.

## 5) Validated Edge Run Command

Run from the main repo:

```bash
cd ~/truckpipline_with_size/repo
.venv_orin/bin/python main.py \
  --video-path /absolute/path/to/video.mp4 \
  --mode stream_event \
  --every_n 1 \
  --show 0 \
  --size-show 0 \
  --preview-scale 0.25 \
  --size-preview-every 999999 \
  --summary-only 1 \
  --non-interactive-model-select
```

### Effective behavior of this command

- stream-event mode with integrated size tracking is enabled
- runtime device resolves automatically inside pipeline code (`auto -> cuda if available`)
- event and size artifact saving remain disabled by default
- output is optimized for edge storage (summary-first, no heavy preview windows)

## 6) Note About `run_orin_stream_event.sh`

The helper script is still useful, but if your local copy includes `--device auto` in the command line, that can fail with current `main.py` CLI parsing.

Safe options:
1. Use the direct validated command above.
2. Or edit `scripts/run_orin_stream_event.sh` and remove `--device auto`.

## 7) Outputs You Should See

Primary files:
- `outputs/events.jsonl`
- `logs/stream_event.log`

Console behavior with `--summary-only 1`:
- concise event/final stats lines only (not full per-frame debug chatter)

Artifacts (disabled by default):
- event debug images: off unless `--save-event-artifacts 1`
- size candidate/winner images: off unless `--save-size-artifacts 1`

## 8) Quick Validation Checklist

After one test video run:

1. Confirm run starts without model-path errors.
2. Confirm events are emitted in terminal output.
3. Confirm `outputs/events.jsonl` is updated.
4. Confirm no unexpected image dump folders grow (unless explicitly enabled).
5. Confirm no CPU-only fallback warning appears when CUDA is available.

## 9) If Results Are Wrong or Missing

Check these first:
- repo layout is exactly `repo` + `repo_size` side by side
- both repos are on `gpu-friendly`
- `.pt` model files exist under `weights/` and/or `../repo_size`
- Jetson torch build is CUDA-enabled in `.venv_orin`
- your video path is absolute and readable

## 10) Summary

For Orin Nano 8GB edge deployment:
- keep the side-by-side repo layout
- use `gpu-friendly` on both repos
- use Jetson-native torch/torchvision/OpenCV
- run the validated stream-event command from `repo`
- keep artifact-saving disabled unless debugging
