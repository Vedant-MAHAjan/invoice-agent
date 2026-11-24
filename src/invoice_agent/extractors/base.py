from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from invoice_agent.schema import Example, Invoice


@dataclass
class Extraction:
    invoice: Invoice
    cost_usd: float = 0.0
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)


class Extractor(Protocol):
    name: str

    def extract(self, example: Example) -> Extraction:
        """Extract from example.document_path / example.ocr_text. Must never read example.gold."""
        ...
