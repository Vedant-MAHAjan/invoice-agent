"""Wraps any extractor with deterministic repairs, so the report can show raw-LLM vs LLM+code."""

from __future__ import annotations

from invoice_agent.extractors.base import Extraction, Extractor
from invoice_agent.schema import Example
from invoice_agent.validate.gstin import repair_party_tax_ids


class RepairedExtractor:
    def __init__(self, inner: Extractor):
        self.inner = inner
        self.name = f"{inner.name}+repair"

    def extract(self, example: Example) -> Extraction:
        out = self.inner.extract(example)
        invoice, notes = repair_party_tax_ids(out.invoice)
        return Extraction(
            invoice=invoice,
            cost_usd=out.cost_usd,
            latency_s=out.latency_s,
            meta={**out.meta, "repairs": notes},
        )
