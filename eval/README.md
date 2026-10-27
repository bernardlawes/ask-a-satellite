# Evaluation set

A hand-labelled test set that every detector (classical, YOLO) is scored against. No numbers go in
the README or a post without coming from `ask-a-satellite evaluate`.

## Make the kit

```bash
uv run ask-a-satellite label-kit --aoi port-everglades --start 2026-04-01 --end 2026-10-27 --max-scenes 5
uv run ask-a-satellite label-kit --aoi port-of-miami  --start 2026-04-01 --end 2026-10-27 --max-scenes 5
```

This writes `images/` (3× upscaled true colour), `labels/` (pre-annotations from the current
classical detector), `meta/` (scene and AOI, so evaluation re-runs on exactly the same pixels), and
`data.yaml`.

## Label in Roboflow

1. Create an Object Detection project with one class, `vessel`.
2. Upload `images/` and `labels/` together (Roboflow pairs them by file name) along with `data.yaml`.
3. Correct every image. **Delete** false boxes (surf, piers, cloud edges), **add** missed ships, and
   tighten boxes to the visible hull.
4. Labelling rules, so the numbers mean something:
   - Label a vessel if you can see a distinct hull of about 50 m or more (5+ px at native resolution,
     15+ px in these 3× images).
   - Label berthed ships too, even though the classical detector cannot find them yet. Recall should
     show that.
   - Don't label wakes. Label the hull at the head of the wake only.
   - If you can't decide, label it and add the tag `uncertain` in Roboflow (it's kept in the export
     notes, and the YOLO txt treats it as a vessel).
5. Generate a version with **no preprocessing and no augmentation** (no resize, no tiling), then export
   it in **YOLO** format.
6. Copy the exported `.txt` label files into `truth/`, keeping the original file stems
   (`port-everglades_20261021.txt` and so on). Roboflow may append `_png.rf.<hash>`, so rename those
   back to the stem.

## Score

```bash
uv run ask-a-satellite evaluate
```

A prediction matches a label if its centroid is inside the label box grown by 2 px (20 m). Matching
is one-to-one. IoU is not used because at 10 m a 1-pixel shift halves it.

`meta/` and `truth/` are committed. `images/` and `labels/` are regenerable and git-ignored.
Images contain modified Copernicus Sentinel-2 data (2026).
