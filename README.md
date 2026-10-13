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

The terminal output distinguishes `candidate face(s)` from the final
`target=x/3` count. Candidate detections include extra people and background
false positives. The target count only includes faces matched to the three
reference people; `review=n` marks low-similarity assignments that should be
checked manually. The raw candidate count is retained as `raw_face_count` in
`detections.json`.

Detection tuning options are available when needed:

```bash
uv run detect_faces.py ... \
  --score-threshold 0.85 \
  --max-detect-size 1600 \
  --min-face-size 48 \
  --identity-threshold 0.30
```

The reference faces receive `person_1`, `person_2`, and `person_3` from left to
right. After reviewing the thumbnails, edit `manual_config.json` for frames
where a person is missing, a fourth person is present, or the identity matcher
chooses the wrong face. Then pass it back to the detector with
`--manual-config`; the selected face indices will be marked as accepted and all
other candidates will be excluded from the target selection.

For example, face indices are zero-based and follow the order in the image's
`faces` array in `detections.json`:

```json
{
  "overrides": {
    "20250129.jpg": {
      "use_face_indices": [0, 1, 2],
      "labels_in_face_order": ["person_2", "person_1", "person_3"]
    }
  }
}
```

If the only problem is two `identity: null` candidates, such as the last two
records for `20250129.jpg`, no override is needed: they are already excluded
from `target_face_count`. Deleting them from `detections.json` is unnecessary
and will be lost when detection is rerun.

## Recommended review order

1. Open `annotated/` as a contact sheet or inspect the files with low
   similarity scores.
2. Confirm which label is the father, child, and mother in
   `manual_config.json`.
3. Add overrides for early photos that contain grandparents or only the baby.
4. Run the animation command below with the corrected configuration.

## Create the animation

After reviewing `detections.json` and applying any overrides in
`manual_config.json`, render the MP4 with:

```bash
uv run make_animation.py \
  --detections face_review/detections.json \
  --manual-config face_review/manual_config.json \
  --output family_memory.mp4 \
  --width 1920 \
  --height 1080 \
  --hold-seconds 1.5 \
  --transition-seconds 0.5 \
  --background blur
```

The three face anchors from the reference image are mapped to the same output
positions in every frame. `--transform affine` (the default) fixes all three
anchors exactly; `--transform similarity` reduces background stretching while
allowing small residual face-position error. Portrait images use a blurred
background by default; use `--background black` or `--background white` for a
plain fill. Increase `--face-size-ratio` if the faces should appear larger.

Images without all three accepted target faces are skipped and listed in the
`.report.json` file next to the MP4. The report also records the exact input
order and all animation parameters. Use `--preview-dir aligned_preview` to
save the aligned stills before reviewing the video.
