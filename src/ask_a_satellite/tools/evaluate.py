"""Score detections against hand labels.

At 10 m a ship is a few pixels, so IoU is unstable (a 1-px shift halves it). A prediction matches a
label when its centroid falls inside the label box grown by ``tol_px``; matching is one-to-one and
greedy by centroid distance.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..models import DetectionResult
from .labels import BoxPx


class EvalScore(BaseModel):
    name: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else None


def match(pred_centroids: list[tuple[float, float]], truth: list[BoxPx], tol_px: float = 2.0,
          ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return (matched (pred, truth) index pairs, unmatched preds, unmatched truths)."""
    candidates = []
    for i, (r, c) in enumerate(pred_centroids):
        for j, (r0, c0, r1, c1) in enumerate(truth):
            if r0 - tol_px <= r <= r1 + tol_px and c0 - tol_px <= c <= c1 + tol_px:
                d = ((r - (r0 + r1) / 2) ** 2 + (c - (c0 + c1) / 2) ** 2) ** 0.5
                candidates.append((d, i, j))
    pairs, used_p, used_t = [], set(), set()
    for _, i, j in sorted(candidates):
        if i not in used_p and j not in used_t:
            pairs.append((i, j))
            used_p.add(i)
            used_t.add(j)
    return (pairs, [i for i in range(len(pred_centroids)) if i not in used_p],
            [j for j in range(len(truth)) if j not in used_t])


def score(result: DetectionResult, truth: list[BoxPx], name: str = "", tol_px: float = 2.0) -> EvalScore:
    # Detection.row/col are pixel indices of the centroid; boxes use exclusive max edges -> +0.5
    cents = [(d.row + 0.5, d.col + 0.5) for d in result.detections]
    pairs, fp, fn = match(cents, truth, tol_px)
    return EvalScore(name=name or result.aoi_id, tp=len(pairs), fp=len(fp), fn=len(fn))


def total(scores: list[EvalScore], name: str = "total") -> EvalScore:
    return EvalScore(name=name, tp=sum(s.tp for s in scores), fp=sum(s.fp for s in scores),
                     fn=sum(s.fn for s in scores))


def markdown_table(scores: list[EvalScore]) -> str:
    def f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    rows = ["| Scene | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    rows += [f"| {s.name} | {s.tp} | {s.fp} | {s.fn} | {f(s.precision)} | {f(s.recall)} | {f(s.f1)} |"
             for s in scores]
    return "\n".join(rows)
