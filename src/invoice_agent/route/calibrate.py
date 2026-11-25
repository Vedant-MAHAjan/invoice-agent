"""Turn signals into P(posting-critical fields are all correct), then pick an auto-post threshold.

Small L2 logistic regression in pure Python (a dozen features, a few hundred docs: no need for
numpy/sklearn). Dev scores are out-of-fold so the threshold isn't picked on scores the model saw.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from invoice_agent.eval.scorer import wilson_interval


@dataclass
class Model:
    features: list[str]
    means: list[float]
    stds: list[float]
    weights: list[float]
    bias: float

    def score(self, x: dict[str, float]) -> float:
        z = self.bias
        for name, m, s, w in zip(self.features, self.means, self.stds, self.weights):
            z += w * (x.get(name, 0.0) - m) / s
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def fit(xs: list[dict[str, float]], ys: list[int], features: list[str], l2: float = 1.0,
        lr: float = 0.1, epochs: int = 2000) -> Model:
    n = len(xs)
    cols = [[x.get(f, 0.0) for x in xs] for f in features]
    means = [sum(c) / n for c in cols]
    stds = [math.sqrt(sum((v - m) ** 2 for v in c) / n) or 1.0 for c, m in zip(cols, means)]
    rows = [[(x.get(f, 0.0) - m) / s for f, m, s in zip(features, means, stds)] for x in xs]
    w = [0.0] * len(features)
    pos = sum(ys) / n
    b = math.log(pos / (1 - pos)) if 0 < pos < 1 else 0.0
    for _ in range(epochs):
        gw = [l2 * wi / n for wi in w]
        gb = 0.0
        for r, y in zip(rows, ys):
            z = b + sum(wi * ri for wi, ri in zip(w, r))
            p = 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
            d = (p - y) / n
            gb += d
            for i, ri in enumerate(r):
                gw[i] += d * ri
        b -= lr * gb
        w = [wi - lr * g for wi, g in zip(w, gw)]
    return Model(features, means, stds, w, b)


def out_of_fold_scores(xs: list[dict[str, float]], ys: list[int], features: list[str], k: int = 5,
                       seed: int = 0) -> list[float]:
    idx = list(range(len(xs)))
    random.Random(seed).shuffle(idx)
    scores = [0.0] * len(xs)
    for fold in range(k):
        test = set(idx[fold::k])
        train = [i for i in idx if i not in test]
        model = fit([xs[i] for i in train], [ys[i] for i in train], features)
        for i in test:
            scores[i] = model.score(xs[i])
    return scores


@dataclass
class OperatingPoint:
    threshold: float
    auto_posted: int
    errors: int
    n: int

    @property
    def coverage(self) -> float:
        return self.auto_posted / self.n if self.n else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.auto_posted if self.auto_posted else 0.0

    def as_dict(self) -> dict:
        lo, hi = wilson_interval(self.errors, self.auto_posted)
        return {
            "threshold": round(self.threshold, 4),
            "auto_post_rate": round(self.coverage, 4),
            "auto_posted": self.auto_posted,
            "review": self.n - self.auto_posted,
            "errors_among_auto_posted": self.errors,
            "error_rate_among_auto_posted": round(self.error_rate, 4),
            "error_rate_ci95": [round(lo, 4), round(hi, 4)],
        }


def risk_coverage(scores: list[float], ok: list[int], eligible: list[bool]) -> list[OperatingPoint]:
    """All thresholds, highest first. Gate-failing docs are never auto-posted."""
    n = len(scores)
    order = sorted((i for i in range(n) if eligible[i]), key=lambda i: -scores[i])
    points, errors = [], 0
    for k, i in enumerate(order, 1):
        errors += 1 - ok[i]
        if k == len(order) or scores[order[k]] < scores[i]:  # only cut between distinct scores
            points.append(OperatingPoint(scores[i], k, errors, n))
    return points


def pick(points: list[OperatingPoint], target_error: float) -> OperatingPoint | None:
    """Largest coverage whose observed error among auto-posted is within the target."""
    ok = [p for p in points if p.error_rate <= target_error]
    return max(ok, key=lambda p: p.auto_posted) if ok else None
