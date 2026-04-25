# Current Validated Run

This file documents the **current validated runtime behavior** of the local integrated pipeline.

The goal is simple:
- run the same code
- with the same runtime parameters
- and get the same behavior that was validated locally

This is the configuration that produced the **good `video_3_min.mp4` run** where the pipeline emitted **20 user-visible truck events**.

## What This README Is For

Use this file when you want another machine to reproduce the same local behavior.

This README is intentionally explicit about:
- the exact command
- the important defaults
- the model files in use
- which parameters should **not** be changed if the goal is identical behavior

## Validated Video

Validated local test video:

```text
C:\Users\moham\Downloads\video_3_min.mp4
```

Expected result for that video:
- **20 emitted truck events**

Important:
- the internal log line `events_finalized=...` is **not** the same as the final emitted event count
- use the actual printed event lines / emitted events as the real result

## Required Working Directory

Run from:

```text
C:\Users\moham\OneDrive\Documents\truckpipline_with_size\repo
```

## Exact Command

Use this exact PowerShell command:

```powershell
Set-Location "C:\Users\moham\OneDrive\Documents\truckpipline_with_size\repo"

python main.py --video-path "C:\Users\moham\Downloads\video_3_min.mp4" --mode stream_event --every_n 1 --show 0 --size-show 1 --preview-scale 0.25 --size-preview-every 999999 --summary-only 1 --non-interactive-model-select
```

## Important Rule

For the validated run, **do not add extra overrides** unless you are intentionally testing changes.

In particular, do **not** add:
- `--detect-conf 0.05`
- `--detect-conf 0.30`
- `--detect-conf 0.35`
- `--every_n 2`
- `--detect-roi-left-ratio ...`
- `--detect-roi-right-ratio ...`

Those were tested separately and changed behavior.

## Effective Runtime Parameters

These are the effective parameters used by the validated run.

Some are passed explicitly by command, and the rest come from current code defaults.

### Explicitly set in the command

- `--mode stream_event`
- `--video-path C:\Users\moham\Downloads\video_3_min.mp4`
- `--every_n 1`
- `--show 0`
- `--size-show 1`
- `--preview-scale 0.25`
- `--size-preview-every 999999`
- `--summary-only 1`
- `--non-interactive-model-select`

### Important defaults that were active

- `detect_conf = 0.01`
- `missed_M = 15`
- `iou_threshold = 0.25`
- `merge_window = 10`
- `merge_iou = 0.20`
- `merge_center_ratio = 0.15`
- `edge_guard = 1`
- `edge_margin = 80`
- `event_infer_mode = finalize`
- `top2 = 0`
- `vote_enable = 0`
- `vote_every = 5`
- `vote_max_samples = 80`
- `track_confirm_hits = 4`
- `track_smooth_alpha = 0.20`
- `track_deadband_px = 4.0`
- `track_max_step_px = 12.0`
- `active_match_center_ratio = 0.080`
- `duplicate_iou_threshold = 0.85`
- `max_detect_fps = 0.00`
- `new_track_ignore_lower_ratio = 0.00`
- `event_dedup_enable = 1`
- `event_dedup_window = 45`
- `event_dedup_iou = 0.55`
- `event_dedup_center_ratio = 0.05`
- `detect_roi_left_ratio = 0.00`
- `detect_roi_right_ratio = 0.00`

### Size-pipeline defaults that were active

- `size_sampling_fps = 5.0`
- `size_precompute_fill = 0`
- `size_precompute_max = 3`
- `size_precompute_gap = 10`
- `size_keep_candidate_frames = 0`
- `size_trigger_fill = 1`
- `size_trigger_bottom_ratio = 0.95`
- `size_trigger_max_candidates = 8`
- `size_trigger_gap = 5`
- `size_trigger_skip_candidates = 4`

### Artifact-saving defaults that were active

- `save_event_artifacts = 0`
- `save_size_artifacts = 0`

So by default this validated run:
- does **not** save event image artifacts
- does **not** save size candidate/winner crop artifacts

It still writes:
- `outputs/events.jsonl`
- `logs/stream_event.log`

## Model Files In Use

These are the currently used default models for the validated run:

- repo 1 truck detector:
  - `weights/weights_March_25/best_Truck_Box_Extraction_March_25.pt`
- Phase 2 coverage classifier:
  - `weights/weights_March_25/best_1st_cls_March_25.pt`
- Phase 3 classifier:
  - `weights/classification#2/best.pt`
- Phase 4 classifier:
  - `weights/classification#3_new1/best_5classes.pt`
- Phase 5 segmentation:
  - `weights/weights_March_25/best_yolo11_seg_march_26v2.pt`
- size-side truck detector:
  - `weights/Yolo-wight/truck.pt`
- size segmentation model:
  - `weights/Yolo-wight/best_size_March_25.pt`

If your machine resolves the size-side models from a sibling `repo_size` folder instead, the filenames must still be:
- `truck.pt`
- `best_size_March_25.pt`

## Environment Notes

The validated runs were performed from the local Python environment already used for this repo.

If another machine pulls the code, the machine must have:
- Python
- the repo dependencies installed
- the default model files present

## How To Verify The Run

After running the command:

1. Watch the printed event lines in the terminal.
2. Do **not** rely only on the internal `events_finalized=...` line.
3. For `video_3_min.mp4`, the validated target is:
   - **20 emitted events**

## Settings That Were Tested And Rejected

These changed behavior away from the validated run:

- increasing repo 1 detector confidence:
  - `--detect-conf 0.05`
  - `--detect-conf 0.30`
  - `--detect-conf 0.35`
- enabling detector ROI cropping
- increasing `--every_n`

So if the goal is to reproduce the validated behavior, do **not** use those.

## Summary

If you want the same current local behavior, use:
- the same code state
- the same models
- the exact command shown above
- no extra runtime overrides

