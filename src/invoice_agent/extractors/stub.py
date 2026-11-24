"""Gold-reading stubs for exercising the harness end to end. Never use these for reported numbers."""

from __future__ import annotations

import random
from decimal import Decimal

from invoice_agent.extractors.base import Extraction
from invoice_agent.schema import HEADER_FIELDS, Example, FieldKind, Invoice


class OracleExtractor:
    name = "oracle"

    def extract(self, example: Example) -> Extraction:
        return Extraction(invoice=example.gold)


class EmptyExtractor:
    name = "empty"

    def extract(self, example: Example) -> Extraction:
        return Extraction(invoice=Invoice())


class NoisyOracleExtractor:
    """Corrupts each non-empty header field with probability `p`, deterministically per doc."""

    name = "noisy-oracle"

    def __init__(self, p: float = 0.1, seed: int = 0):
        self.p = p
        self.seed = seed

    def extract(self, example: Example) -> Extraction:
        rng = random.Random(f"{self.seed}:{example.doc_id}")
        updates = {}
        for name, kind in HEADER_FIELDS.items():
            value = getattr(example.gold, name)
            if value is None or rng.random() >= self.p:
                continue
            if rng.random() < 0.5:
                updates[name] = None
            elif kind is FieldKind.MONEY:
                updates[name] = value + Decimal("1.00")
            elif kind is FieldKind.DATE:
                updates[name] = value.replace(day=1 if value.day != 1 else 2)
            else:
                updates[name] = f"{value}X"
        return Extraction(invoice=example.gold.model_copy(update=updates))


STUBS = {cls.name: cls for cls in (OracleExtractor, EmptyExtractor, NoisyOracleExtractor)}
