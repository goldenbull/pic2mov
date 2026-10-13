# Family-memory face alignment

`detect_faces.py` is the review stage for the family photo animation. It
detects faces, assigns the three target people relative to a reference image,
and writes annotated thumbnails for inspection. It does not create the final
video yet.

## Install with uv

```bash
uv sync
```

`uv sync` creates and manages the project environment in `.venv` and installs
the dependencies declared in `pyproject.toml`. Activate it only when you want
an interactive shell:

```bash
source .venv/bin/activate
```

The first run downloads the YuNet and SFace ONNX models into `models/`.

## Run on the supplied directory

```bash
uv run detect_faces.py \
  --input-dir /Users/cjn/Downloads/family_memories \
  --reference-image /Users/cjn/Downloads/family_memories/20180529.jpg \
  --output-dir face_review
```

`requirements.txt` is retained for tools that do not use `uv`; normal project
work should use `uv sync` and `uv run`.

The output contains:

* `detections.json`: all face boxes, landmarks, assigned labels, and scores;
* `annotated/`: review thumbnails with colored labels;
* `manual_config.json`: a template for correcting ambiguous photos.

The reference faces receive `person_1`, `person_2`, and `person_3` from left to
right. After reviewing the thumbnails, edit `manual_config.json` for frames
where a person is missing, a fourth person is present, or the identity matcher
chooses the wrong face. The animation stage will consume this corrected file.

## Recommended review order

1. Open `annotated/` as a contact sheet or inspect the files with low
   similarity scores.
2. Confirm which label is the father, child, and mother in
   `manual_config.json`.
3. Add overrides for early photos that contain grandparents or only the baby.
4. Use the corrected configuration when the alignment/video stage is added.
