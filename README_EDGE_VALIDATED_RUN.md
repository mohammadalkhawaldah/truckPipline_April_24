# Edge Validated Run

This file is the edge-deployment equivalent of the local validated run guide.

Its purpose is simple:
- pull the `gpu-friendly` branches of both repos
- place them in the expected side-by-side layout
- install only the non-Torch Python dependencies in repo-local environments
- run the pipeline with a storage-light, GPU-usable command on NVIDIA Orin Nano

## Expected Layout

Use this exact folder structure on the edge device:

```text
~/truckpipline_with_size/
  repo/         -> truckPipline_April_24
  repo_size/    -> truck_size_2
```

The `repo` code auto-discovers the size models from:
- local `weights/`
- sibling `../repo_size`

## Required Branches

Pull these branches:

- main repo: `gpu-friendly`
- size repo: `gpu-friendly`

## Clone Commands

```bash
mkdir -p ~/truckpipline_with_size
cd ~/truckpipline_with_size

git clone --branch gpu-friendly https://github.com/mohammadalkhawaldah/truckPipline_April_24.git repo
git clone --branch gpu-friendly https://github.com/mohammadalkhawaldah/truck_size_2.git repo_size
```

## Jetson / Orin Nano Assumption

Before using the repo setup scripts, the device should already have:
- JetPack installed
- NVIDIA-provided `torch` and `torchvision` appropriate for the device
- OpenCV available from the Jetson environment you trust

The repo Jetson requirements intentionally do not pin or install:
- `torch`
- `torchvision`
- `opencv-python`

Those packages are usually device-specific on Jetson and should come from the platform setup.

## Setup Commands

Run both setup scripts:

```bash
cd ~/truckpipline_with_size/repo
bash scripts/setup_orin_nano.sh

cd ~/truckpipline_with_size/repo_size
bash scripts/setup_orin_nano.sh
```

## Edge Run Command

Recommended main run command from the main repo:

```bash
cd ~/truckpipline_with_size/repo
bash scripts/run_orin_stream_event.sh /absolute/path/to/video.mp4
```

That command currently expands to a storage-light stream-event run with:

- `--mode stream_event`
- `--every_n 1`
- `--show 0`
- `--size-show 0`
- `--preview-scale 0.25`
- `--size-preview-every 999999`
- `--summary-only 1`
- `--non-interactive-model-select`
- `--device auto`

## Important Edge Rules

For the edge pull, do not enable extra artifact saving unless you are debugging.

By default this branch keeps edge storage cleaner because:
- event artifacts are disabled by default
- size artifacts are disabled by default
- the size-side helper defaults to `--save-frames 0`

So a normal edge run should not save extra photos or videos.

## Optional Standalone Size Repo Check

If you want to test only the second repo on the edge:

```bash
cd ~/truckpipline_with_size/repo_size
bash scripts/run_auto_select_orin.sh /absolute/path/to/video.mp4 --write-summary-csv
```

## What To Verify

After an edge run:

1. Confirm the process is using the GPU-enabled device path rather than CPU-only forced inference.
2. Confirm terminal output shows emitted event lines.
3. Confirm no unexpected image-dump folders are growing unless you explicitly enabled artifact saving.

## Summary

For the smoothest edge pull:
- use the `gpu-friendly` branch in both repos
- keep the side-by-side `repo` / `repo_size` layout
- use the provided Jetson setup scripts
- use the provided `run_orin_stream_event.sh` command
- do not turn on extra artifact-saving flags unless needed for debugging
