"""Supplier master and resolution, the way accounts payable identifies who sent an invoice.

AP doesn't post the raw supplier string; it matches the invoice to an onboarded supplier record,
by tax ID when there's a valid one, otherwise by name. A supplier that matches nothing is a new
vendor and goes to review (onboarding), never auto-posted.

The master is built from labeled documents as a stand-in for the ERP's curated supplier list.
Gold names carry OCR variants ("SHAL MARKETING" for "SHAH MARKETING"), so entries are merged by
valid GSTIN and the most common name becomes the canonical one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz

from invoice_agent.eval.normalize import normalize
from invoice_agent.schema import Example, FieldKind
from invoice_agent.validate.gstin import is_valid_gstin, normalize_gstin

NAME_THRESHOLD = 90.0
NAME_MARGIN = 5.0  # best match must beat the runner-up by this much


@dataclass(frozen=True)
class Supplier:
    supplier_id: str
    name: str
    gstins: frozenset[str]
    aliases: frozenset[str]  # normalized name variants seen for this supplier


@dataclass(frozen=True)
class Resolution:
    supplier_id: str | None
    method: str  # "gstin" | "name" | "unknown"
    score: float = 0.0
    gstin_name_conflict: bool = False


def _name(value: str | None) -> str | None:
    return normalize(FieldKind.TEXT, value)


class SupplierMaster:
    def __init__(self, suppliers: Iterable[Supplier]):
        self.suppliers = {s.supplier_id: s for s in suppliers}
        self.by_gstin = {g: s.supplier_id for s in self.suppliers.values() for g in s.gstins}

    def __len__(self) -> int:
        return len(self.suppliers)

    @classmethod
    def from_examples(cls, examples: Iterable[Example]) -> SupplierMaster:
        # Union-find over names and valid GSTINs: anything sharing a GSTIN is one supplier.
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            parent[find(a)] = find(b)

        name_counts: Counter = Counter()
        for ex in examples:
            name = _name(ex.gold.vendor_name)
            gstin = normalize_gstin(ex.gold.vendor_tax_id)
            gstin = gstin if is_valid_gstin(gstin) else None
            if not name and not gstin:
                continue
            if name:
                name_counts[name] += 1
                find(f"n:{name}")
            if gstin:
                find(f"g:{gstin}")
            if name and gstin:
                union(f"n:{name}", f"g:{gstin}")

        groups: dict[str, list[str]] = defaultdict(list)
        for key in list(parent):
            groups[find(key)].append(key)
        suppliers = []
        for i, keys in enumerate(sorted(groups.values(), key=lambda ks: sorted(ks))):
            names = [k[2:] for k in keys if k.startswith("n:")]
            gstins = frozenset(k[2:] for k in keys if k.startswith("g:"))
            canonical = max(names, key=lambda n: (name_counts[n], n)) if names else ""
            suppliers.append(Supplier(f"S{i:04d}", canonical, gstins, frozenset(names)))
        return cls(suppliers)

    def _by_name(self, name: str | None) -> tuple[str | None, float]:
        n = _name(name)
        if not n:
            return None, 0.0
        scored = []
        for s in self.suppliers.values():
            # Subset matching ("shah marketing mangalwar peth" contains "shah marketing") only against
            # the canonical name; OCR-variant aliases like "marketing" must match nearly exactly.
            best = max(
                fuzz.token_set_ratio(n, s.name) if s.name else 0.0,
                max((fuzz.ratio(n, a) for a in s.aliases), default=0.0),
            )
            scored.append((best, s.supplier_id))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < NAME_THRESHOLD:
            return None, scored[0][0] if scored else 0.0
        if len(scored) > 1 and scored[0][0] - scored[1][0] < NAME_MARGIN:
            return None, scored[0][0]
        return scored[0][1], scored[0][0]

    def resolve(self, vendor_name: str | None, vendor_tax_id: str | None) -> Resolution:
        by_name, score = self._by_name(vendor_name)
        gstin = normalize_gstin(vendor_tax_id)
        if gstin and is_valid_gstin(gstin) and gstin in self.by_gstin:
            sid = self.by_gstin[gstin]
            return Resolution(sid, "gstin", 100.0, gstin_name_conflict=by_name is not None and by_name != sid)
        if by_name:
            return Resolution(by_name, "name", score)
        return Resolution(None, "unknown", score)

    def resolve_gold(self, ex: Example) -> str | None:
        return self.resolve(ex.gold.vendor_name, ex.gold.vendor_tax_id).supplier_id
