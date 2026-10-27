# Dev log

Running notes on what actually happened. Useful raw material for the LinkedIn posts. Numbers here come
from real runs. Visual judgements are labelled as such until the hand-labelled test set exists (M2).

## 2026-10-27: first real-imagery runs (M1)

Scene: `S2B_17RNJ_20261021_0_L2A` (2026-10-21 16:06 UTC). AOI cloud 3% (Port Everglades), 2% (PortMiami).

1. **Double-applied reflectance offset.** Earth Search items for processing baseline 05.13 carry
   `earthsearch:boa_offset_applied: true` while `raster:bands` still lists `offset: -0.1`. Applying it
   made open-water reflectance negative (NIR ≈ −0.09). NDWI then classed only 4% of the Everglades AOI
   as water, and the detector boxed 18 warehouse roofs. Fixed in `scene_from_item` with a regression test.
2. **White hulls look like water in NDWI.** Bright white ships reflect green ≈ NIR, so NDWI ≈ 0 and
   many hull pixels passed the water test. Requiring candidates to be "not water" discarded the actual
   ships offshore. Candidates are now any NIR-bright pixel inside the sea mask.
3. **Morphology leaks onto land.** Closing (r = 50 m) plus unrestricted hole-filling turned canal-laced
   neighbourhoods into "sea", which produced 120 detections at Everglades and 164 at PortMiami. Fixes:
   keep only water bodies ≥ 0.2 km², close with r = 20 m, fill only holes ≤ 600 px.
4. **Shoreline clutter.** A 30 m coastal buffer cut detections to 15 (Everglades) and 65 (PortMiami).
   On visual inspection, the ships in open water east of Port Everglades are found. The remaining false
   alarms are mostly surf strips along the beach, the inlet jetty, and marinas and seawalls in Biscayne
   Bay. **These counts are not yet vessel counts.** Precision and recall wait for hand labels (M2).

5. **Measured, not guessed.** I computed features for every remaining detection on the same scene and
   compared the ones that look like ships with the obvious false alarms (visual judgement):
   - At Port Everglades, the likely ships sit 24–264 px from land (median), and every false alarm is
     within 5.5 px.
   - Surf strips and the jetty are about 1 px wide with elongation of 12–25. Ships are 2.7–4.3 px wide
     with elongation 2–4.
   - The likely ships have **positive** mean NDWI (0.1–0.4), which confirms finding 2.
   - At PortMiami, many false alarms touch land: roofs, piers, marinas. Two are vegetated islets
     (red/NIR ≈ 0.1), and Miami Beach surf has NDWI 0.5–0.6 with blue/NIR ≈ 3.

   The resulting rules reject a blob if it touches land, has a median distance to land under 60 m, is
   a thin strip, has NDVI over 0.5, or has NDWI over 0.45. The last two thresholds are fitted to one
   scene and are provisional until the labels exist. Result: **6 detections at Port Everglades and 19
   at PortMiami**, with rejection counts recorded in each result's `params`. A known cost: cruise ships
   berthed along Dodge Island's quay are now excluded. At 10 m they can't be separated from the quay by
   this method.
6. **Small cumulus beats the cloud mask.** PortMiami 2026-10-11 passed the AOI cloud check (≤ 10%) but
   produced 56 pre-annotations. Small fair-weather clouds over the bay are only partly flagged by the
   20 m SCL band, so their bright edges look like ships. That's a candidate for the Post 2 failure
   gallery. Possible fixes: buffer the cloud mask, or add a brightness/whiteness test.

## 2026-10-27: evaluation tooling

- Crop cache (`data/cache/*.npz`): reruns take about 5 s instead of re-reading COGs.
- `label-kit` exported 10 scenes (5 per port, April–October 2026) for labelling in Roboflow. See
  `eval/README.md`.
- `evaluate` scores precision, recall, and F1 against `eval/truth/`, using centroid-in-box matching
  with 2 px tolerance.

## 2026-10-27: small clouds (detector v2), M1 closed

The PortMiami 2026-10-11 scene had 56 detections, many on small cumulus clouds.

1. **SCL misses the clouds entirely.** Almost every cloud false alarm sat 10–90 px from the nearest SCL
   cloud pixel, on pixels SCL calls water. Buffering the cloud mask would not have helped.
2. **SCL flags bright ships as cloud.** The two largest ships at Port Everglades on 2026-10-21 contained
   SCL class 8 ("cloud, medium probability") pixels, and the candidate step was cutting holes in them.
   Small SCL cloud specks (< 50 px) are now ignored. Large SCL clouds still mask.
3. **Visible brightness doesn't separate cloud from hull, but shortwave infrared does.** On true blob
   pixels across 5 scenes, cloud fragments measure 0.72–1.17 SWIR/NIR (B11/B08) and hulls mostly
   0.23–0.61. The one exception is a container ship at 0.81, probably from its deck cargo, which is
   elongated (4.6) where cloud fragments reach 3.6 at most. Rule: reject if SWIR/NIR ≥ 0.70 **and**
   elongation < 4.0.
4. **A measurement mistake.** My first feature table measured every above-threshold pixel in each
   bounding box, not each blob's own pixels. That made cloud fragments look rounder than they are
   (#30: 1.5 instead of 2.5), and the first version of the rule missed it. I re-measured on blob pixels
   and refitted. Lesson for Post 2: check what your feature code actually measures.

Result (v1 → v2 pre-annotations): PortMiami 2026-10-11 went 56 → 36, and the cloud puffs are gone. The
other 9 scenes changed by 0 to −4. Port Everglades 2026-10-21 is unchanged at 6. What remains is mostly
4–6 px specks, likely small boats at anchor, which is a size-floor question for M2. Counts are still
**not validated**.

Next (M2): hand-label the 10 scenes, run `evaluate`, and only then tune.
