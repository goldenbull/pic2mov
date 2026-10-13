#!/usr/bin/env python3
"""Detect and label the three target family members in a photo series.

This is the review stage of the family-memory animation workflow. It writes:

* detections.json: machine-readable detections and identity assignments;
* annotated/: thumbnails with face boxes and labels;
* manual_config.json: a template for correcting ambiguous frames.

The script uses OpenCV Zoo's YuNet detector and SFace recognizer. Model files
are downloaded automatically on first use unless --model-dir is supplied.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
RECOGNIZER_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
DETECTOR_NAME = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_NAME = "face_recognition_sface_2021dec.onnx"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def require_dependencies():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image, ImageOps  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: python3 -m pip install -r requirements.txt"
        ) from exc
    return cv2, np, Image, ImageOps


def download_if_missing(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # YuNet is intentionally compact (about 230 KB); SFace is much larger.
    # A lower bound avoids repeatedly trying to re-download a valid YuNet
    # model in restricted/offline environments.
    if destination.exists() and destination.stat().st_size > 100_000:
        return destination
    print(f"Downloading model: {destination.name}")
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Download it manually into {destination.parent}."
        ) from exc
    return destination


def load_image(path: Path, cv2, np, Image, ImageOps):
    """Read an image and apply its EXIF orientation before converting to BGR."""
    with Image.open(path) as pil_image:
        oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
        rgb = np.asarray(oriented)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_thumbnail(image, destination: Path, cv2, max_width: int = 900) -> None:
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def detect_faces(image, detector, cv2, max_detect_size: int, min_face_size: int):
    """Detect faces on a bounded-size copy and map boxes back to the source."""
    h, w = image.shape[:2]
    scale = min(1.0, max_detect_size / float(max(h, w)))
    if scale < 1.0:
        detect_image = cv2.resize(
            image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        detect_image = image
    dh, dw = detect_image.shape[:2]
    detector.setInputSize((dw, dh))
    _, faces = detector.detect(detect_image)
    if faces is None:
        return []
    # YuNet returns x, y, width, height, then five (x, y) landmarks,
    # followed by score. Keep coordinates in original-image pixels.
    results = []
    for row in faces:
        row = [float(v) for v in row]
        x, y, fw, fh = [value / scale for value in (row[0], row[1], row[2], row[3])]
        score = row[-1]
        landmarks = [[row[i] / scale, row[i + 1] / scale] for i in range(4, 14, 2)]
        if min(fw, fh) < min_face_size:
            continue
        results.append(
            {
                "bbox": [x, y, fw, fh],
                "landmarks": landmarks,
                "score": score,
                "feature": None,
                "identity": None,
                "similarity": None,
            }
        )
    return results


def feature_for_face(image, face, recognizer, np):
    # FaceRecognizerSF expects the complete YuNet row, including landmarks.
    row = np.asarray(face["bbox"] + sum(face["landmarks"], []) + [face["score"]], dtype=np.float32)
    aligned = recognizer.alignCrop(image, row)
    return recognizer.feature(aligned)


def cosine_similarity(a, b, np) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a.ravel(), b.ravel()) / denom) if denom else -1.0


def best_identity_assignment(faces, reference_features, recognizer, image, np, identity_threshold):
    """Assign labels uniquely, allowing fewer or more than three detections."""
    labels = list(reference_features)
    if not faces:
        return
    for face in faces:
        try:
            face["feature"] = feature_for_face(image, face, recognizer, np)
            face["similarities"] = {
                label: cosine_similarity(face["feature"], ref, np)
                for label, ref in reference_features.items()
            }
        except Exception:
            face["similarities"] = {}

    # For 1-3 faces, try every injective assignment. For 4+ faces, choose
    # the three faces and labels that maximize the total similarity.
    best = None
    count = min(len(faces), len(labels))
    for face_indices in itertools.combinations(range(len(faces)), count):
        for chosen_labels in itertools.permutations(labels, count):
            score = sum(
                faces[i].get("similarities", {}).get(label, -1.0)
                for i, label in zip(face_indices, chosen_labels)
            )
            if best is None or score > best[0]:
                best = (score, face_indices, chosen_labels)
    if best is None:
        return
    _, face_indices, chosen_labels = best
    for i, label in zip(face_indices, chosen_labels):
        faces[i]["identity"] = label
        faces[i]["similarity"] = faces[i].get("similarities", {}).get(label)
        faces[i]["accepted"] = faces[i]["similarity"] >= identity_threshold


def draw_annotations(image, faces, cv2, np):
    out = image.copy()
    colors = {
        "person_1": (50, 210, 50),
        "person_2": (40, 170, 240),
        "person_3": (220, 80, 210),
        None: (80, 80, 240),
    }
    for index, face in enumerate(faces, 1):
        # Unknown detections are usually texture/background false positives.
        # Keep low-confidence identity matches visible for manual review, but
        # omit unassigned candidates from the preview to keep it readable.
        if face.get("identity") is None:
            continue
        x, y, w, h = [round(v) for v in face["bbox"]]
        color = colors.get(face.get("identity"), colors[None]) if face.get("accepted") else (130, 130, 130)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(2, round(min(out.shape[:2]) / 500)))
        for px, py in face["landmarks"]:
            cv2.circle(out, (round(px), round(py)), max(2, round(min(out.shape[:2]) / 250)), color, -1)
        label = face.get("identity") or f"unknown_{index}"
        if not face.get("accepted"):
            label = "review:" + label
        similarity = face.get("similarity")
        if similarity is not None:
            label += f" {similarity:.2f}"
        baseline_y = max(25, y - 8)
        cv2.putText(out, label, (x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def serializable_faces(faces) -> list[dict[str, Any]]:
    output = []
    for face in faces:
        item = {key: value for key, value in face.items() if key not in {"feature", "similarities"}}
        output.append(item)
    return output


def load_manual_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Manual config does not exist: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid manual config JSON: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit("Manual config root must be a JSON object")
    return config


def apply_manual_override(record: dict[str, Any], config: dict[str, Any]) -> bool:
    """Apply a per-file face-index/identity override."""
    basename = Path(record["path"]).name
    override = config.get("overrides", {}).get(basename)
    if override is None:
        return False
    if not isinstance(override, dict):
        raise SystemExit(f"Override for {basename} must be an object")
    indices = override.get("use_face_indices", [])
    labels = override.get("labels_in_face_order", [])
    if len(indices) != len(labels):
        raise SystemExit(
            f"Override for {basename} has {len(indices)} indices but {len(labels)} labels"
        )
    faces = record["faces"]
    for face in faces:
        face["identity"] = None
        face["similarity"] = None
        face["accepted"] = False
    for index, label in zip(indices, labels):
        if not isinstance(index, int) or index < 0 or index >= len(faces):
            raise SystemExit(
                f"Override for {basename} refers to face index {index}; "
                f"valid range is 0..{len(faces) - 1}"
            )
        if label not in {"person_1", "person_2", "person_3"}:
            raise SystemExit(f"Override for {basename} has unknown label: {label}")
        faces[index]["identity"] = label
        faces[index]["accepted"] = True
    record["manual_override_applied"] = True
    return True


def iter_images(input_dir: Path) -> Iterable[Path]:
    return sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )


def choose_reference(records: list[dict[str, Any]], requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        if not path.exists():
            raise SystemExit(f"Reference image does not exist: {path}")
        return path
    for record in records:
        if len(record["faces"]) == 3:
            return Path(record["path"])
    raise SystemExit("No image with exactly three detected faces. Use --reference-image explicitly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("face_review"))
    parser.add_argument("--reference-image", type=str, help="Image containing the three target people")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--manual-config",
        type=Path,
        help="JSON file containing per-image face-index and identity overrides",
    )
    parser.add_argument("--score-threshold", type=float, default=0.85)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument(
        "--max-detect-size",
        type=int,
        default=1600,
        help="Resize the longest image side before detection (0 keeps original size)",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=48,
        help="Ignore detections smaller than this many source-image pixels",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.30,
        help="Cosine similarity below this is marked review instead of accepted",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv2, np, Image, ImageOps = require_dependencies()
    manual_config = load_manual_config(args.manual_config)
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    files = list(iter_images(args.input_dir))
    if not files:
        raise SystemExit(f"No JPG/JPEG images found in {args.input_dir}")

    detector_path = download_if_missing(DETECTOR_URL, args.model_dir / DETECTOR_NAME)
    recognizer_path = download_if_missing(RECOGNIZER_URL, args.model_dir / RECOGNIZER_NAME)
    detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (320, 320), args.score_threshold, args.nms_threshold, args.top_k
    )
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    records: list[dict[str, Any]] = []
    print(f"Scanning {len(files)} images...")
    for number, path in enumerate(files, 1):
        try:
            image = load_image(path, cv2, np, Image, ImageOps)
            faces = detect_faces(image, detector, cv2, args.max_detect_size, args.min_face_size)
            records.append({"path": str(path), "faces": faces, "error": None})
            print(f"[{number:03d}/{len(files):03d}] {path.name}: {len(faces)} candidate face(s)")
        except Exception as exc:
            records.append({"path": str(path), "faces": [], "error": repr(exc)})
            print(f"[{number:03d}/{len(files):03d}] {path.name}: ERROR {exc}", file=sys.stderr)

    reference_path = choose_reference(records, args.reference_image)
    ref_record = next((r for r in records if Path(r["path"]).resolve() == reference_path.resolve()), None)
    if ref_record is None:
        image = load_image(reference_path, cv2, np, Image, ImageOps)
        ref_faces = detect_faces(image, detector, cv2, args.max_detect_size, args.min_face_size)
    else:
        ref_faces = ref_record["faces"]
        image = load_image(reference_path, cv2, np, Image, ImageOps)
    if len(ref_faces) != 3:
        raise SystemExit(
            f"Reference image must contain exactly three faces; found {len(ref_faces)}: {reference_path}"
        )
    # Reference labels are left-to-right. The manual config can rename or
    # override these labels after reviewing the annotated reference image.
    ref_faces = sorted(ref_faces, key=lambda f: f["bbox"][0])
    reference_features = {}
    for index, face in enumerate(ref_faces, 1):
        reference_features[f"person_{index}"] = feature_for_face(image, face, recognizer, np)

    output_dir = args.output_dir
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record["error"]:
            continue
        path = Path(record["path"])
        image = load_image(path, cv2, np, Image, ImageOps)
        if not apply_manual_override(record, manual_config):
            best_identity_assignment(
                record["faces"], reference_features, recognizer, image, np, args.identity_threshold
            )
        record["raw_face_count"] = len(record["faces"])
        record["target_face_count"] = sum(1 for face in record["faces"] if face.get("accepted"))
        record["review_face_count"] = sum(
            1 for face in record["faces"] if face.get("identity") and not face.get("accepted")
        )
        print(
            f"[review] {path.name}: target={record['target_face_count']}/3, "
            f"candidates={record['raw_face_count']}, review={record['review_face_count']}"
        )
        annotated = draw_annotations(image, record["faces"], cv2, np)
        save_thumbnail(annotated, annotated_dir / f"{path.stem}.jpg", cv2)
        record["faces"] = serializable_faces(record["faces"])

    for record in records:
        record["path"] = str(Path(record["path"]).resolve())
    json_records = {
        "reference_image": str(reference_path.resolve()),
        "labels": ["person_1", "person_2", "person_3"],
        "records": records,
    }
    (output_dir / "detections.json").write_text(
        json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manual_template = {
        "reference_image": str(reference_path.resolve()),
        "renamed_labels": {
            "person_1": "父亲",
            "person_2": "孩子",
            "person_3": "母亲",
        },
        "overrides": {
            "example.jpg": {
                "use_face_indices": [0, 1, 2],
                "labels_in_face_order": ["person_1", "person_2", "person_3"]
            }
        },
    }
    manual_template_path = output_dir / "manual_config.json"
    if not manual_template_path.exists():
        manual_template_path.write_text(
            json.dumps(manual_template, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("target_face_count", 0)) if not record["error"] else "error"
        counts[key] = counts.get(key, 0) + 1
    print(f"Reference: {reference_path}")
    print(f"Wrote: {output_dir / 'detections.json'}")
    print(f"Wrote annotated previews: {annotated_dir}")
    print("Accepted target-face count summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("Raw candidate counts are retained in detections.json as raw_face_count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
