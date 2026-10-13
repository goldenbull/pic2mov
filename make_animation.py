#!/usr/bin/env python3
"""Align the three target faces and render a cross-fade MP4 animation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from detect_faces import apply_manual_override, load_image, load_manual_config


LABELS = ("person_1", "person_2", "person_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, default=Path("face_review/detections.json"))
    parser.add_argument("--manual-config", type=Path, default=Path("face_review/manual_config.json"))
    parser.add_argument("--output", type=Path, default=Path("family_memory.mp4"))
    parser.add_argument("--reference-image", type=Path, help="Override the reference in detections/manual config")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--transition-seconds", type=float, default=0.5)
    parser.add_argument(
        "--face-size-ratio",
        type=float,
        default=0.18,
        help="Target average face width as a fraction of the shorter output side",
    )
    parser.add_argument("--background", choices=("blur", "black", "white"), default="blur")
    parser.add_argument(
        "--transform",
        choices=("person2", "affine", "similarity"),
        default="person2",
        help="person2 uses only rotation/uniform scale around person_2; other modes use all three faces",
    )
    parser.add_argument("--person2-x", type=float, help="Target person_2 x position in output pixels (default: center)")
    parser.add_argument("--person2-y", type=float, help="Target person_2 y position in output pixels (default: center)")
    parser.add_argument("--max-images", type=int, help="Process only the first N usable images, for testing")
    parser.add_argument("--preview-dir", type=Path, help="Save aligned stills for visual inspection")
    return parser.parse_args()


def read_detections(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Detection file does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid detections JSON: {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise SystemExit("detections.json must contain a records array")
    return data


def face_anchor(face: dict[str, Any]) -> np.ndarray:
    landmarks = face.get("landmarks") or []
    if len(landmarks) >= 2:
        # YuNet's first two landmarks are the left and right eye centers.
        return np.mean(np.asarray(landmarks[:2], dtype=np.float32), axis=0)
    x, y, w, h = face["bbox"]
    return np.asarray([x + w / 2.0, y + h / 2.0], dtype=np.float32)


def selected_faces(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for face in record.get("faces", []):
        label = face.get("identity")
        accepted = face.get("accepted", label is not None)
        if label in LABELS and accepted and label not in selected:
            selected[label] = face
    return selected


def apply_config_to_records(records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    for record in records:
        apply_manual_override(record, config)


def find_record(records: list[dict[str, Any]], image_path: Path) -> dict[str, Any] | None:
    wanted = image_path.resolve()
    for record in records:
        try:
            if Path(record["path"]).resolve() == wanted:
                return record
        except (KeyError, OSError):
            continue
    return None


def canonical_target_points(
    points: np.ndarray,
    reference_faces: dict[str, dict[str, Any]],
    output_size: tuple[int, int],
    face_size_ratio: float,
) -> np.ndarray:
    """Build a stable face layout centered on the output canvas.

    Using the reference face geometry directly as a full-image fit makes a
    portrait reference produce tiny faces in a 16:9 video. Instead, preserve
    the relative triangle of the three face anchors and choose its scale from
    the desired face width.
    """
    out_w, out_h = output_size
    average_face_width = float(
        np.mean([float(reference_faces[label]["bbox"][2]) for label in LABELS])
    )
    desired_face_width = face_size_ratio * min(out_w, out_h)
    scale = desired_face_width / max(average_face_width, 1.0)
    center = np.mean(points, axis=0)
    output_center = np.asarray([out_w / 2.0, out_h / 2.0], dtype=np.float32)
    return (points.astype(np.float32) - center) * scale + output_center


def make_transform(
    source_faces: dict[str, dict[str, Any]],
    target_points: np.ndarray,
    transform_mode: str,
) -> np.ndarray:
    source_points = np.asarray([face_anchor(source_faces[label]) for label in LABELS], dtype=np.float32)
    if transform_mode == "similarity":
        matrix, _ = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError("Could not estimate similarity transform")
        return matrix
    # Three non-collinear face anchors define an exact affine transform. This
    # fixes all three face positions, including when camera distance changes.
    return cv2.getAffineTransform(source_points, target_points.astype(np.float32))


def make_person2_transform(
    source_face: dict[str, Any],
    reference_face: dict[str, Any],
    output_size: tuple[int, int],
    face_size_ratio: float,
    target_xy: tuple[float, float] | None,
) -> np.ndarray:
    """Return a rotation + uniform-scale transform anchored on person_2.

    The translation term only places the transformed person_2 anchor at the
    requested output position. The 2x2 part contains a pure rotation and one
    shared scale, so there is no shear or non-uniform stretching.
    """
    source_anchor = face_anchor(source_face)
    source_width = float(source_face["bbox"][2])
    target_width = face_size_ratio * min(output_size)
    if source_width <= 0 or target_width <= 0:
        raise ValueError("person_2 face width must be positive")

    def eye_angle(face: dict[str, Any]) -> float:
        landmarks = face.get("landmarks") or []
        if len(landmarks) >= 2:
            dx = float(landmarks[1][0] - landmarks[0][0])
            dy = float(landmarks[1][1] - landmarks[0][1])
            return float(np.arctan2(dy, dx))
        return 0.0

    angle = eye_angle(reference_face) - eye_angle(source_face)
    scale = target_width / source_width
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    linear = scale * np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    if target_xy is None:
        target = np.asarray([output_size[0] / 2.0, output_size[1] / 2.0], dtype=np.float32)
    else:
        target = np.asarray(target_xy, dtype=np.float32)
    translation = target - linear @ source_anchor
    return np.column_stack([linear, translation]).astype(np.float32)


def background_canvas(image: np.ndarray, output_size: tuple[int, int], mode: str) -> np.ndarray:
    out_w, out_h = output_size
    if mode == "black":
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if mode == "white":
        return np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = max(out_w / w, out_h / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    y = max(0, (resized.shape[0] - out_h) // 2)
    x = max(0, (resized.shape[1] - out_w) // 2)
    crop = resized[y : y + out_h, x : x + out_w]
    return cv2.GaussianBlur(crop, (0, 0), sigmaX=35, sigmaY=35)


def warp_image(image: np.ndarray, matrix: np.ndarray, output_size: tuple[int, int], background: str) -> np.ndarray:
    out_w, out_h = output_size
    canvas = background_canvas(image, output_size, background)
    warped = cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_LANCZOS4)
    mask = cv2.warpAffine(
        np.full(image.shape[:2], 255, dtype=np.uint8), matrix, (out_w, out_h), flags=cv2.INTER_NEAREST
    )
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.2)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(warped.astype(np.float32) * alpha + canvas.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)


def open_ffmpeg(output: Path, width: int, height: int, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to create MP4 output")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def write_frame(process: subprocess.Popen, frame: np.ndarray) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write(frame.tobytes())
    except BrokenPipeError as exc:
        error = process.stderr.read().decode(errors="replace") if process.stderr else ""
        raise RuntimeError(f"ffmpeg stopped while writing frames:\n{error}") from exc


def write_still(process: subprocess.Popen, frame: np.ndarray, count: int) -> None:
    for _ in range(count):
        write_frame(process, frame)


def write_crossfade(process: subprocess.Popen, first: np.ndarray, second: np.ndarray, count: int) -> None:
    if count <= 0:
        return
    for index in range(1, count + 1):
        alpha = index / count
        blended = cv2.addWeighted(first, 1.0 - alpha, second, alpha, 0.0)
        write_frame(process, blended)


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height, and fps must be positive")
    if args.hold_seconds <= 0 or args.transition_seconds < 0:
        raise SystemExit("hold-seconds must be positive and transition-seconds cannot be negative")
    if args.face_size_ratio <= 0:
        raise SystemExit("face-size-ratio must be positive")
    detections = read_detections(args.detections)
    manual_config = load_manual_config(args.manual_config if args.manual_config.exists() else None)
    records = detections["records"]
    apply_config_to_records(records, manual_config)

    reference_path = args.reference_image or Path(manual_config.get("reference_image", detections.get("reference_image", "")))
    if not reference_path or not reference_path.exists():
        raise SystemExit("Reference image is missing; pass --reference-image")
    reference_record = find_record(records, reference_path)
    if reference_record is None:
        raise SystemExit(f"Reference image is not present in detections.json: {reference_path}")
    reference_faces = selected_faces(reference_record)
    if args.transform == "person2":
        if "person_2" not in reference_faces:
            raise SystemExit(f"Reference image must contain an accepted person_2 face: {reference_path}")
    elif set(reference_faces) != set(LABELS):
        raise SystemExit(f"Reference image must contain all three accepted faces: {reference_path}")
    reference_image = load_image(reference_path, cv2, np, Image, ImageOps)
    output_size = (args.width, args.height)
    if args.transform == "person2":
        target_points = None
    else:
        ref_points = np.asarray([face_anchor(reference_faces[label]) for label in LABELS], dtype=np.float32)
        target_points = canonical_target_points(ref_points, reference_faces, output_size, args.face_size_ratio)
    target_xy = (
        args.person2_x if args.person2_x is not None else output_size[0] / 2.0,
        args.person2_y if args.person2_y is not None else output_size[1] / 2.0,
    )

    usable: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        faces = selected_faces(record)
        required_labels = {"person_2"} if args.transform == "person2" else set(LABELS)
        if not required_labels.issubset(faces):
            skipped.append({"path": record.get("path", ""), "reason": "missing required target face(s)"})
            continue
        usable.append((record, faces))
        if args.max_images and len(usable) >= args.max_images:
            break
    if not usable:
        raise SystemExit("No usable images with all three accepted faces")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = args.preview_dir
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    hold_frames = round(args.hold_seconds * args.fps)
    transition_frames = round(args.transition_seconds * args.fps)
    if hold_frames < 1:
        raise SystemExit("hold-seconds multiplied by fps must produce at least one frame")
    process = open_ffmpeg(args.output, args.width, args.height, args.fps)
    used_paths: list[str] = []
    try:
        previous: np.ndarray | None = None
        for index, (record, faces) in enumerate(usable, 1):
            path = Path(record["path"])
            image = load_image(path, cv2, np, Image, ImageOps)
            if args.transform == "person2":
                matrix = make_person2_transform(
                    faces["person_2"],
                    reference_faces["person_2"],
                    output_size,
                    args.face_size_ratio,
                    target_xy,
                )
            else:
                assert target_points is not None
                matrix = make_transform(faces, target_points, args.transform)
            frame = warp_image(image, matrix, output_size, args.background)
            used_paths.append(str(path))
            if preview_dir:
                cv2.imwrite(str(preview_dir / f"{index:04d}_{path.stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if previous is None:
                write_still(process, frame, hold_frames)
            else:
                write_crossfade(process, previous, frame, transition_frames)
                write_still(process, frame, hold_frames)
            previous = frame
        assert process.stdin is not None
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}:\n{stderr}")
    except Exception:
        if process.poll() is None:
            process.kill()
        raise

    report = {
        "output": str(args.output.resolve()),
        "reference_image": str(reference_path.resolve()),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "hold_seconds": args.hold_seconds,
        "transition_seconds": args.transition_seconds,
        "background": args.background,
        "transform": args.transform,
        "used_images": used_paths,
        "skipped_images": skipped,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote animation: {args.output}")
    print(f"Used {len(used_paths)} images; skipped {len(skipped)} incomplete images")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted")
