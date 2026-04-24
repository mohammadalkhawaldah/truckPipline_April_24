from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

from src import config
from src.fill_estimation import ensure_size_model_compat
from src.utils import ensure_dirs


CONF_THRESHOLD = 0.4
SIZE_SEG_CONF_THRESHOLD = 0.05
IOU_MATCH_THRESHOLD = 0.2
MAX_MISSED_SAMPLES = 2
MIN_TRACK_HITS = 2
MIN_RELIABLE_FILL = 5.0
MAX_SELECTION_CANDIDATES = 7


@dataclass
class SizeDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    score: float
    frame_index: int
    timestamp_sec: float
    image_path: Path | None
    fill_percentage: float | None = None
    fill_status: str = "pending"
    raw_output: str = ""


@dataclass
class SizeTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    hits: int
    missed: int
    best_detection: SizeDetection | None
    selected: bool = False
    history: list[SizeDetection] = field(default_factory=list)
    precomputed_fill_count: int = 0
    last_precomputed_fill_frame: int | None = None
    trigger_fill_candidates: list[SizeDetection] = field(default_factory=list)
    trigger_fill_count: int = 0
    trigger_hits_seen: int = 0
    last_trigger_fill_frame: int | None = None
    threshold_reached: bool = False


@dataclass
class SizeEvent:
    track_id: int
    start_frame: int
    end_frame: int
    best_frame: int
    best_timestamp_sec: float
    best_bbox: tuple[int, int, int, int]
    fill_percentage: float | None
    fill_status: str
    best_frame_path: str
    raw_output: str
    history_frames: list[int]
    shortlist_frames: list[int]
    fill_selection_method: str = "fallback_visibility"
    trigger_frames: list[int] = field(default_factory=list)


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_center_distance(box_a, box_b) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def blur_score(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(variance / 400.0, 1.0)


def edge_penalty(bbox, frame_width, frame_height, margin_ratio=0.03):
    x1, y1, x2, y2 = bbox
    x_margin = frame_width * margin_ratio
    y_margin = frame_height * margin_ratio

    touches_edge = (
        x1 <= x_margin
        or y1 <= y_margin
        or x2 >= frame_width - x_margin
        or y2 >= frame_height - y_margin
    )
    return 1.0 if touches_edge else 0.0


def detection_score(frame, bbox, confidence):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = frame_width * frame_height
    area_score = box_area / frame_area if frame_area else 0.0

    box_center_x = (x1 + x2) / 2.0
    box_center_y = (y1 + y2) / 2.0
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0
    distance = ((box_center_x - frame_center_x) ** 2 + (box_center_y - frame_center_y) ** 2) ** 0.5
    max_distance = (frame_center_x ** 2 + frame_center_y ** 2) ** 0.5
    center_score = 1.0 - (distance / max_distance if max_distance else 0.0)

    sharpness_score = blur_score(frame, bbox)
    border_penalty = edge_penalty(bbox, frame_width, frame_height)

    return (
        confidence
        + (2.0 * area_score)
        + center_score
        + sharpness_score
        - (1.5 * border_penalty)
    )


def finalize_track(track, completed_tracks):
    if track.hits >= MIN_TRACK_HITS and track.best_detection is not None:
        track.selected = True
        completed_tracks.append(track)


def resize_mask(mask, target_shape):
    return cv2.resize(
        mask.astype("uint8"),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def calculate_fill_percentage(box_mask, content_mask):
    content_mask = content_mask & box_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    content_mask = cv2.morphologyEx(
        content_mask.astype("uint8"),
        cv2.MORPH_CLOSE,
        kernel,
    ).astype(bool)

    box_rows = box_mask.any(axis=1)
    content_rows = content_mask.any(axis=1)

    if not box_rows.any():
        return 0.0

    box_top = int(box_rows.argmax())
    box_bottom = int(len(box_rows) - 1 - box_rows[::-1].argmax())

    if not content_rows.any():
        return 0.0

    content_top = int(content_rows.argmax())
    content_top = max(content_top, box_top)

    box_height = box_bottom - box_top
    content_height = box_bottom - content_top

    if box_height <= 0:
        return 0.0

    fill = (content_height / box_height) * 100
    return max(0.0, min(fill, 100.0))


def apply_segmentation_overlay(frame, bbox, size_model, size_classes, device):
    x1, y1, x2, y2 = bbox
    truck_crop = frame[y1:y2, x1:x2]
    if truck_crop.size == 0:
        return

    seg_result = size_model(truck_crop, device=device, conf=SIZE_SEG_CONF_THRESHOLD, verbose=False)[0]
    if seg_result.masks is None:
        return

    truck_box_mask = None
    content_mask = None
    masks = seg_result.masks.data.cpu().numpy()
    classes = seg_result.boxes.cls.cpu().numpy()

    for index, cls in enumerate(classes):
        class_name = size_classes[int(cls)]
        if class_name.lower() == "box":
            truck_box_mask = masks[index]
        elif class_name.lower() == "content":
            content_mask = masks[index]

    if truck_box_mask is None:
        return

    overlay = truck_crop.copy()
    box_mask_resized = resize_mask(truck_box_mask, truck_crop.shape)
    overlay[box_mask_resized] = (255, 0, 0)

    if content_mask is not None:
        content_mask_resized = resize_mask(content_mask, truck_crop.shape)
        overlay[content_mask_resized] = (0, 255, 0)

    truck_crop[:] = cv2.addWeighted(overlay, 0.4, truck_crop, 0.6, 0)


def detect_trucks(model, frame, truck_classes, device):
    detections = []
    results = model(frame, device=device, verbose=False)[0]

    for det in results.boxes:
        confidence = float(det.conf[0])
        if confidence < CONF_THRESHOLD:
            continue

        class_id = int(det.cls[0])
        if truck_classes[class_id] != "truck":
            continue

        bbox = tuple(map(int, det.xyxy[0]))
        score = detection_score(frame, bbox, confidence)
        detections.append((bbox, confidence, score))

    detections.sort(key=lambda item: item[2], reverse=True)
    return detections


def save_frame(output_dir, track_id, frame_index, frame, bbox=None):
    image_path = output_dir / f"truck_{track_id:03d}_frame_{frame_index:05d}.jpg"
    to_save = frame
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size != 0:
            to_save = crop
    cv2.imwrite(str(image_path), to_save)
    return image_path


def resize_for_display(image, scale=0.25):
    height, width = image.shape[:2]
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def draw_preview(
    frame,
    active_tracks,
    sampled_frame_index,
    size_model,
    size_classes,
    device,
    preview_scale,
    render_segmentation: bool = True,
):
    preview = frame.copy()
    for track in active_tracks.values():
        x1, y1, x2, y2 = track.bbox
        if render_segmentation and track.threshold_reached:
            apply_segmentation_overlay(preview, track.bbox, size_model, size_classes, device)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if track.best_detection is not None:
            label = f"truck_{track.track_id:03d} best={track.best_detection.score:.2f}"
        else:
            label = f"truck_{track.track_id:03d} waiting_trigger"
        cv2.putText(
            preview,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        preview,
        f"sampled frame: {sampled_frame_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    return resize_for_display(preview, scale=preview_scale)


def match_tracks(active_tracks, detections):
    matches = []
    unmatched_track_ids = set(active_tracks.keys())
    unmatched_detection_ids = set(range(len(detections)))

    scored_pairs = []
    for track_id, track in active_tracks.items():
        for detection_index, (bbox, _, _) in enumerate(detections):
            iou = compute_iou(track.bbox, bbox)
            if iou >= IOU_MATCH_THRESHOLD:
                scored_pairs.append((iou, track_id, detection_index))

    for _, track_id, detection_index in sorted(scored_pairs, reverse=True):
        if track_id not in unmatched_track_ids or detection_index not in unmatched_detection_ids:
            continue
        matches.append((track_id, detection_index))
        unmatched_track_ids.remove(track_id)
        unmatched_detection_ids.remove(detection_index)

    return matches, unmatched_track_ids, unmatched_detection_ids


def estimate_fill_for_crop(truck_crop, size_model, size_classes, device):
    if truck_crop is None or getattr(truck_crop, "size", 0) == 0:
        return None, "Empty truck crop"

    seg_result = size_model(truck_crop, device=device, conf=SIZE_SEG_CONF_THRESHOLD, verbose=False)[0]
    if seg_result.masks is None:
        return None, "No fill detected"

    truck_box_mask = None
    content_mask = None
    masks = seg_result.masks.data.cpu().numpy()
    classes = seg_result.boxes.cls.cpu().numpy()

    for index, cls in enumerate(classes):
        class_name = size_classes[int(cls)]
        if class_name.lower() == "box":
            truck_box_mask = masks[index]
        elif class_name.lower() == "content":
            content_mask = masks[index]

    if truck_box_mask is None:
        return None, "No fill detected"

    box_mask_resized = resize_mask(truck_box_mask, truck_crop.shape)
    if content_mask is not None:
        content_mask_resized = resize_mask(content_mask, truck_crop.shape)
        fill_percentage = calculate_fill_percentage(box_mask_resized, content_mask_resized)
    else:
        fill_percentage = 0.0

    return fill_percentage, f"FINAL FILL: {fill_percentage:.2f}%"


def select_fill_candidates(track):
    ordered_by_score = sorted(track.history, key=lambda detection: detection.score, reverse=True)
    score_candidates = ordered_by_score[:MAX_SELECTION_CANDIDATES]

    midpoint_detection = track.history[len(track.history) // 2]
    midpoint_candidates = sorted(
        track.history,
        key=lambda detection: abs(detection.timestamp_sec - midpoint_detection.timestamp_sec),
    )[:3]

    merged = {}
    for detection in score_candidates + midpoint_candidates:
        merged[detection.image_path] = detection

    return sorted(merged.values(), key=lambda detection: detection.frame_index)


def evaluate_track_detections(detections, truck_model, size_model, truck_classes, size_classes, device):
    for detection in detections:
        if detection.fill_status != "pending":
            continue
        if detection.image_path is None:
            detection.fill_percentage = None
            detection.raw_output = "No saved candidate image"
            detection.fill_status = "failed"
            continue
        frame = cv2.imread(str(detection.image_path))
        if frame is None:
            detection.fill_percentage = None
            detection.raw_output = "Error loading image"
            detection.fill_status = "failed"
            continue

        fill_percentage, raw_output = estimate_fill_for_crop(
            frame,
            size_model,
            size_classes,
            device,
        )
        detection.fill_percentage = fill_percentage
        detection.raw_output = raw_output
        detection.fill_status = "ok" if fill_percentage is not None else "failed"


def select_best_from_detections(detections, history):
    valid_detections = [detection for detection in detections if detection.fill_status == "ok"]
    positive_detections = [
        detection
        for detection in valid_detections
        if detection.fill_percentage is not None and detection.fill_percentage >= MIN_RELIABLE_FILL
    ]

    if len(positive_detections) >= 2:
        positive_values = sorted(
            detection.fill_percentage for detection in positive_detections if detection.fill_percentage is not None
        )
        median_fill = positive_values[len(positive_values) // 2]
        midpoint_time = history[len(history) // 2].timestamp_sec
        return min(
            positive_detections,
            key=lambda detection: (
                abs((detection.fill_percentage or 0.0) - median_fill),
                abs(detection.timestamp_sec - midpoint_time),
                -detection.score,
            ),
        )

    if positive_detections:
        return max(positive_detections, key=lambda detection: detection.score)

    if valid_detections:
        return max(valid_detections, key=lambda detection: detection.score)

    return max(detections or history, key=lambda detection: detection.score)


def select_best_detection(track):
    return select_best_from_detections(track.history, track.history)


def load_models(truck_model_path: Path, size_model_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_size_model_compat()
    truck_model = YOLO(str(truck_model_path))
    size_model = YOLO(str(size_model_path))
    truck_classes = truck_model.names
    size_classes = size_model.names
    return device, truck_model, size_model, truck_classes, size_classes


def analyze_video(video_path, output_dir, sampling_fps, show_preview, preview_scale, device, truck_model, size_model, truck_classes, size_classes):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(round(source_fps / sampling_fps)), 1)

    active_tracks = {}
    completed_tracks = []
    next_track_id = 1
    frame_index = 0
    preview_window_name = "Truck Detection Preview"

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        timestamp_sec = frame_index / source_fps
        detections = detect_trucks(truck_model, frame, truck_classes, device)
        matches, unmatched_track_ids, unmatched_detection_ids = match_tracks(active_tracks, detections)

        for track_id, detection_index in matches:
            bbox, confidence, score = detections[detection_index]
            track = active_tracks[track_id]
            image_path = save_frame(output_dir, track_id, frame_index, frame, bbox)
            detection = SizeDetection(bbox, confidence, score, frame_index, timestamp_sec, image_path)
            track.bbox = bbox
            track.hits += 1
            track.missed = 0
            track.history.append(detection)
            if score > track.best_detection.score:
                track.best_detection = detection

        for track_id in list(unmatched_track_ids):
            track = active_tracks[track_id]
            track.missed += 1
            if track.missed > MAX_MISSED_SAMPLES:
                finalize_track(track, completed_tracks)
                del active_tracks[track_id]

        for detection_index in unmatched_detection_ids:
            bbox, confidence, score = detections[detection_index]
            image_path = save_frame(output_dir, next_track_id, frame_index, frame, bbox)
            detection = SizeDetection(bbox, confidence, score, frame_index, timestamp_sec, image_path)
            active_tracks[next_track_id] = SizeTrack(
                track_id=next_track_id,
                bbox=bbox,
                hits=1,
                missed=0,
                best_detection=detection,
                history=[detection],
            )
            next_track_id += 1

        if show_preview:
            preview = draw_preview(frame, active_tracks, frame_index, size_model, size_classes, device, preview_scale)
            cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_window_name, preview.shape[1], preview.shape[0])
            cv2.imshow(preview_window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_index += 1

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    for track in list(active_tracks.values()):
        finalize_track(track, completed_tracks)

    return completed_tracks


def run_size_event_pipeline(
    video_path: Path | str,
    truck_model_path: Path,
    size_model_path: Path,
    sampling_fps: float,
    output_dir: Path,
    show_preview: bool = False,
    preview_scale: float = 0.25,
    preview_every_samples: int = 1,
) -> list[SizeEvent]:
    ensure_dirs([output_dir])
    device, truck_model, size_model, truck_classes, size_classes = load_models(truck_model_path, size_model_path)
    completed_tracks = analyze_video(
        Path(video_path),
        output_dir,
        sampling_fps,
        show_preview,
        preview_scale,
        device,
        truck_model,
        size_model,
        truck_classes,
        size_classes,
    )
    results: list[SizeEvent] = []
    for track in completed_tracks:
        candidate_detections = select_fill_candidates(track)
        evaluate_track_detections(
            candidate_detections,
            truck_model,
            size_model,
            truck_classes,
            size_classes,
            device,
        )
        track.best_detection = select_best_detection(track)
        best = track.best_detection
        history_frames = [d.frame_index for d in track.history]
        shortlist_frames = [d.frame_index for d in candidate_detections]
        results.append(
            SizeEvent(
                track_id=track.track_id,
                start_frame=min(history_frames) if history_frames else best.frame_index,
                end_frame=max(history_frames) if history_frames else best.frame_index,
                best_frame=best.frame_index,
                best_timestamp_sec=best.timestamp_sec,
                best_bbox=best.bbox,
                fill_percentage=best.fill_percentage,
                fill_status=best.fill_status,
                best_frame_path=str(best.image_path),
                raw_output=best.raw_output,
                history_frames=history_frames,
                shortlist_frames=shortlist_frames,
            )
        )
    return results


class OnlineSizeEventPipeline:
    """Stateful version of the second repo pipeline for per-event merging."""

    def __init__(
        self,
        video_path: Path | str,
        truck_model_path: Path,
        size_model_path: Path,
        sampling_fps: float,
        output_dir: Path,
        source_fps: float,
        show_preview: bool = False,
        preview_scale: float = 0.25,
        preview_every_samples: int = 1,
        precompute_fill: bool = True,
        precompute_max_candidates: int = 3,
        precompute_min_gap_frames: int = 10,
        keep_candidate_frames: bool = False,
        trigger_fill: bool = True,
        trigger_bottom_ratio: float = 0.98,
        trigger_max_candidates: int = 3,
        trigger_gap_frames: int = 5,
        trigger_skip_candidates: int = 0,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = output_dir
        ensure_dirs([self.output_dir])
        self.device, self.truck_model, self.size_model, self.truck_classes, self.size_classes = load_models(
            truck_model_path,
            size_model_path,
        )
        self.source_fps = source_fps or 25.0
        self.frame_interval = max(int(round(self.source_fps / sampling_fps)), 1)
        self.show_preview = show_preview
        self.preview_scale = preview_scale
        self.preview_every_samples = max(1, int(preview_every_samples))
        self.precompute_fill = bool(precompute_fill)
        self.precompute_max_candidates = max(0, int(precompute_max_candidates))
        self.precompute_min_gap_frames = max(0, int(precompute_min_gap_frames))
        self.keep_candidate_frames = bool(keep_candidate_frames)
        self.trigger_fill = bool(trigger_fill)
        self.trigger_bottom_ratio = float(max(0.0, min(1.0, trigger_bottom_ratio)))
        self.trigger_max_candidates = max(0, int(trigger_max_candidates))
        self.trigger_gap_frames = max(0, int(trigger_gap_frames))
        self.trigger_skip_candidates = max(0, int(trigger_skip_candidates))
        self.preview_window_name = "Truck Detection Preview"
        self.active_tracks: dict[int, SizeTrack] = {}
        self.completed_tracks: list[SizeTrack] = []
        self.completed_events: list[SizeEvent] = []
        self.next_track_id = 1
        self.samples_processed = 0
        self.current_frame_height: int = 0

    def process_frame(self, frame, frame_index: int) -> None:
        if frame_index % self.frame_interval != 0:
            return

        self.samples_processed += 1
        self.current_frame_height = int(frame.shape[0]) if frame is not None else self.current_frame_height
        timestamp_sec = frame_index / self.source_fps
        detections = detect_trucks(self.truck_model, frame, self.truck_classes, self.device)
        matches, unmatched_track_ids, unmatched_detection_ids = match_tracks(self.active_tracks, detections)

        for track_id, detection_index in matches:
            bbox, confidence, score = detections[detection_index]
            track = self.active_tracks[track_id]
            track.bbox = bbox
            track.hits += 1
            track.missed = 0
            bottom_ratio = float(bbox[3]) / float(max(1, self.current_frame_height))
            if bottom_ratio >= self.trigger_bottom_ratio:
                track.threshold_reached = True
            if not track.threshold_reached:
                continue
            image_path = save_frame(self.output_dir, track_id, frame_index, frame, bbox)
            detection = SizeDetection(bbox, confidence, score, frame_index, timestamp_sec, image_path)
            track.history.append(detection)
            if track.best_detection is None or score > track.best_detection.score:
                track.best_detection = detection
            self._maybe_collect_trigger_fill(track, detection, frame.shape)
            self._maybe_precompute_fill(track, detection)

        for track_id in list(unmatched_track_ids):
            track = self.active_tracks[track_id]
            track.missed += 1
            if track.missed > MAX_MISSED_SAMPLES:
                self._complete_track(track)
                del self.active_tracks[track_id]

        for detection_index in unmatched_detection_ids:
            bbox, confidence, score = detections[detection_index]
            bottom_ratio = float(bbox[3]) / float(max(1, self.current_frame_height))
            threshold_reached = bottom_ratio >= self.trigger_bottom_ratio
            detection = None
            history: list[SizeDetection] = []
            best_detection: SizeDetection | None = None
            if threshold_reached:
                image_path = save_frame(self.output_dir, self.next_track_id, frame_index, frame, bbox)
                detection = SizeDetection(bbox, confidence, score, frame_index, timestamp_sec, image_path)
                history = [detection]
                best_detection = detection
            self.active_tracks[self.next_track_id] = SizeTrack(
                track_id=self.next_track_id,
                bbox=bbox,
                hits=1,
                missed=0,
                best_detection=best_detection,
                history=history,
                threshold_reached=threshold_reached,
            )
            if detection is not None:
                self._maybe_collect_trigger_fill(self.active_tracks[self.next_track_id], detection, frame.shape)
                self._maybe_precompute_fill(self.active_tracks[self.next_track_id], detection)
            self.next_track_id += 1

        if self.show_preview:
            render_segmentation = (self.samples_processed % self.preview_every_samples) == 0
            preview = draw_preview(
                frame,
                self.active_tracks,
                frame_index,
                self.size_model,
                self.size_classes,
                self.device,
                self.preview_scale,
                render_segmentation=render_segmentation,
            )
            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.preview_window_name, preview.shape[1], preview.shape[0])
            cv2.imshow(self.preview_window_name, preview)
            cv2.waitKey(1)

    def pop_completed_events(self) -> list[SizeEvent]:
        events = self.completed_events
        self.completed_events = []
        return events

    def finalize_all(self) -> list[SizeEvent]:
        for track_id in list(self.active_tracks.keys()):
            self._complete_track(self.active_tracks[track_id])
            del self.active_tracks[track_id]
        return self.pop_completed_events()

    def close(self) -> None:
        if self.show_preview:
            try:
                if cv2.getWindowProperty(self.preview_window_name, cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow(self.preview_window_name)
            except cv2.error:
                pass

    def has_trigger_ready_for_bbox(self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, int] | None = None) -> bool:
        if bbox is None:
            return False
        frame_h = frame_shape[0] if frame_shape is not None and len(frame_shape) >= 1 else self.current_frame_height
        frame_w = frame_shape[1] if frame_shape is not None and len(frame_shape) >= 2 else 0
        diag = float(((max(1, frame_w) ** 2) + (max(1, frame_h) ** 2)) ** 0.5)
        max_center_dist = 0.12 * diag
        for track in self.active_tracks.values():
            if not track.threshold_reached:
                continue
            iou = compute_iou(track.bbox, bbox)
            if iou >= 0.10:
                return True
            center_dist = bbox_center_distance(track.bbox, bbox)
            if center_dist <= max_center_dist:
                return True
        return False

    def force_finalize_best_match(self, candidate_event: dict[str, Any]) -> SizeEvent | None:
        if not self.active_tracks:
            return None
        best_track_id = self._find_best_active_track_id(candidate_event)
        if best_track_id is None:
            return None
        track = self.active_tracks.pop(best_track_id)
        return self._track_to_event(track)

    def force_finalize_ready_trigger_match(self, candidate_event: dict[str, Any]) -> SizeEvent | None:
        best_track_id = self._find_best_active_track_id(candidate_event)
        if best_track_id is None:
            return None
        track = self.active_tracks.get(best_track_id)
        if track is None:
            return None
        if not track.trigger_fill_candidates:
            return None
        if self.trigger_max_candidates > 0 and track.trigger_fill_count < self.trigger_max_candidates:
            return None
        self.active_tracks.pop(best_track_id, None)
        return self._track_to_event(track)

    def _find_best_active_track_id(self, candidate_event: dict[str, Any]) -> int | None:
        start_frame = int(candidate_event.get("start_frame", -1))
        end_frame = int(candidate_event.get("end_frame", -1))
        best_frame = int(candidate_event.get("best_frame", -1))
        best_frame_tolerance = 12
        best_track_id = None
        best_key = None
        for track_id, track in self.active_tracks.items():
            if not track.history or track.best_detection is None:
                continue
            history_frames = [d.frame_index for d in track.history]
            track_start = min(history_frames)
            track_end = max(history_frames)
            if track_end > (end_frame + best_frame_tolerance):
                continue
            if track_start < (start_frame - best_frame_tolerance):
                continue
            overlap = max(0, min(end_frame, track_end) - max(start_frame, track_start) + 1)
            gap = min(abs(best_frame - track.best_detection.frame_index), abs(start_frame - track_start), abs(end_frame - track_end))
            key = (1 if overlap > 0 else 0, overlap, -gap, track.hits)
            if best_key is None or key > best_key:
                best_key = key
                best_track_id = track_id
        return best_track_id

    def _complete_track(self, track: SizeTrack) -> None:
        if track.hits < MIN_TRACK_HITS or track.best_detection is None:
            return
        frame_h = max(1, int(self.current_frame_height))
        if int(track.bbox[3]) < int(frame_h * 0.5):
            return
        self.completed_tracks.append(track)
        self.completed_events.append(self._track_to_event(track))

    def _maybe_collect_trigger_fill(self, track: SizeTrack, detection: SizeDetection, frame_shape) -> None:
        if not self.trigger_fill:
            return
        if self.trigger_max_candidates <= 0:
            return
        if track.trigger_fill_count >= self.trigger_max_candidates:
            return
        frame_h = int(frame_shape[0]) if frame_shape is not None else 0
        if frame_h <= 0:
            return
        bottom_ratio = float(detection.bbox[3]) / float(frame_h)
        if bottom_ratio < self.trigger_bottom_ratio:
            return

        track.trigger_hits_seen += 1
        if track.trigger_hits_seen <= self.trigger_skip_candidates:
            return

        if track.last_trigger_fill_frame is not None:
            if detection.frame_index - track.last_trigger_fill_frame < self.trigger_gap_frames:
                return

        if detection.fill_status == "pending":
            evaluate_track_detections(
                [detection],
                self.truck_model,
                self.size_model,
                self.truck_classes,
                self.size_classes,
                self.device,
            )
        track.trigger_fill_candidates.append(detection)
        track.trigger_fill_count += 1
        track.last_trigger_fill_frame = detection.frame_index

    def _maybe_precompute_fill(self, track: SizeTrack, detection: SizeDetection) -> None:
        if not self.precompute_fill:
            return
        if self.precompute_max_candidates <= 0:
            return
        if track.precomputed_fill_count >= self.precompute_max_candidates:
            return
        if detection.fill_status != "pending":
            return
        if track.last_precomputed_fill_frame is not None:
            if detection.frame_index - track.last_precomputed_fill_frame < self.precompute_min_gap_frames:
                return

        # Prefer candidates that are currently the best visibility sample for the track.
        if track.precomputed_fill_count > 0 and detection is not track.best_detection:
            return

        evaluate_track_detections(
            [detection],
            self.truck_model,
            self.size_model,
            self.truck_classes,
            self.size_classes,
            self.device,
        )
        track.precomputed_fill_count += 1
        track.last_precomputed_fill_frame = detection.frame_index

    def _track_to_event(self, track: SizeTrack) -> SizeEvent:
        trigger_detections = list(track.trigger_fill_candidates)
        if trigger_detections:
            evaluate_track_detections(
                trigger_detections,
                self.truck_model,
                self.size_model,
                self.truck_classes,
                self.size_classes,
                self.device,
            )
        trigger_valid = [d for d in trigger_detections if d.fill_status == "ok"]

        if trigger_detections:
            candidate_detections = trigger_detections
            track.best_detection = select_best_from_detections(trigger_detections, track.history)
            fill_selection_method = "bottom_trigger" if trigger_valid else "bottom_trigger_failed"
        elif self.trigger_fill:
            candidate_detections = [max(track.history, key=lambda detection: detection.frame_index)]
            track.best_detection = candidate_detections[0]
            track.best_detection.fill_percentage = None
            track.best_detection.raw_output = "No bottom-trigger candidate"
            track.best_detection.fill_status = "failed"
            fill_selection_method = "no_bottom_trigger"
        else:
            candidate_detections = select_fill_candidates(track)
            evaluate_track_detections(
                candidate_detections,
                self.truck_model,
                self.size_model,
                self.truck_classes,
                self.size_classes,
                self.device,
            )
            track.best_detection = select_best_from_detections(candidate_detections, track.history)
            fill_selection_method = "fallback_visibility"

        best = track.best_detection
        history_frames = [d.frame_index for d in track.history]
        shortlist_frames = [d.frame_index for d in candidate_detections]
        trigger_frames = [d.frame_index for d in trigger_detections]
        event = SizeEvent(
            track_id=track.track_id,
            start_frame=min(history_frames) if history_frames else best.frame_index,
            end_frame=max(history_frames) if history_frames else best.frame_index,
            best_frame=best.frame_index,
            best_timestamp_sec=best.timestamp_sec,
            best_bbox=best.bbox,
            fill_percentage=best.fill_percentage,
            fill_status=best.fill_status,
            best_frame_path=str(best.image_path),
            raw_output=best.raw_output,
            history_frames=history_frames,
            shortlist_frames=shortlist_frames,
            fill_selection_method=fill_selection_method,
            trigger_frames=trigger_frames,
        )
        if not self.keep_candidate_frames:
            self._cleanup_track_images(track, keep_path=best.image_path)
        return event

    def _cleanup_track_images(self, track: SizeTrack, keep_path: Path) -> None:
        keep_resolved = Path(keep_path).resolve()
        for detection in track.history:
            if detection.image_path is None:
                continue
            path = Path(detection.image_path)
            try:
                if path.resolve() == keep_resolved:
                    continue
                if path.exists():
                    path.unlink()
            except OSError:
                continue
