"""Compare vessel counts between two observations of the same AOI."""

from __future__ import annotations

from ..models import CountComparison, DetectionResult


def compare_counts(before: DetectionResult, after: DetectionResult,
                   cloud_gap: float = 0.1, water_gap: float = 0.1) -> CountComparison:
    if before.aoi_id != after.aoi_id:
        raise ValueError(f"cannot compare different AOIs: {before.aoi_id} vs {after.aoi_id}")
    if before.datetime > after.datetime:
        before, after = after, before

    caveats = []
    if before.method != after.method:
        caveats.append(f"Different detection methods ({before.method} vs {after.method}).")
    if abs(before.aoi_cloud_fraction - after.aoi_cloud_fraction) > cloud_gap:
        caveats.append(
            f"Cloud cover differs ({before.aoi_cloud_fraction:.0%} vs {after.aoi_cloud_fraction:.0%}); "
            "the cloudier date may undercount."
        )
    if abs(before.water_fraction - after.water_fraction) > water_gap:
        caveats.append(
            f"Visible water area differs ({before.water_fraction:.0%} vs {after.water_fraction:.0%})."
        )
    caveats.append("Each count is a single snapshot (Sentinel-2 passes at ~10:30 local solar time), "
                   "not an average over the period.")

    diff = after.count - before.count
    pct = round(100 * diff / before.count, 1) if before.count else None
    return CountComparison(
        aoi_id=before.aoi_id,
        before_scene_id=before.scene_id, before_datetime=before.datetime, before_count=before.count,
        after_scene_id=after.scene_id, after_datetime=after.datetime, after_count=after.count,
        difference=diff, percent_change=pct, caveats=caveats,
    )
