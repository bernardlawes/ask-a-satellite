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

Next: hand-label 8–12 crops, then tune shape filters (surf strips are long, thin, and shore-parallel)
and the shore buffer against the labels instead of by eye. Consider an AOI-specific "navigable water"
polygon for Biscayne Bay.
