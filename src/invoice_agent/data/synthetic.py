"""Deterministic, arithmetically consistent synthetic invoices for harness and validator tests.

    python -m invoice_agent.data.synthetic fixtures/synthetic.jsonl --n 40
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
from decimal import ROUND_HALF_UP, Decimal

from invoice_agent.data.loader import write_examples
from invoice_agent.schema import HEADER_FIELDS, LINE_ITEMS, Example, Invoice, LineItem, Split, TaxLine

CENT = Decimal("0.01")
VENDORS = [
    ("Northwind Traders Ltd", "GB123456789", "GB29NWBK60161331926819"),
    ("Acme Industrial Supply", "US-87-1234567", None),
    ("Contoso Office GmbH", "DE811907980", "DE89370400440532013000"),
    ("Fabrikam Logistics", "FR40303265045", "FR1420041010050500013M02606"),
]
PRODUCTS = ["Copy paper A4", "Toner cartridge", "Freight charge", "Consulting hours", "Pallet rental"]


def make_invoice(rng: random.Random, i: int) -> Invoice:
    vendor, tax_id, iban = rng.choice(VENDORS)
    rate = rng.choice([Decimal(0), Decimal(5), Decimal(19), Decimal(20)])
    items = []
    for _ in range(rng.randint(1, 5)):
        qty = Decimal(rng.randint(1, 20))
        price = Decimal(rng.randint(100, 50000)) / 100
        items.append(
            LineItem(
                description=rng.choice(PRODUCTS),
                quantity=qty,
                unit_price=price,
                amount=(qty * price).quantize(CENT, ROUND_HALF_UP),
                tax_rate=rate,
            )
        )
    subtotal = sum((li.amount for li in items), Decimal(0))
    tax = (subtotal * rate / 100).quantize(CENT, ROUND_HALF_UP)
    issued = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randint(0, 300))
    return Invoice(
        vendor_name=vendor,
        vendor_tax_id=tax_id,
        buyer_name="Example Buyer Inc",
        invoice_number=f"INV-{2025}-{i:05d}",
        po_number=f"PO{rng.randint(10000, 99999)}" if rng.random() < 0.6 else None,
        invoice_date=issued,
        due_date=issued + dt.timedelta(days=30),
        payment_terms="Net 30",
        currency=rng.choice(["USD", "EUR", "GBP"]),
        subtotal=subtotal,
        tax_total=tax,
        total=subtotal + tax,
        amount_due=subtotal + tax,
        bank_iban=iban,
        line_items=tuple(items),
        tax_lines=(TaxLine(rate=rate, base=subtotal, amount=tax),),
    )


def make_examples(n: int, seed: int = 7) -> list[Example]:
    rng = random.Random(seed)
    labeled = frozenset(HEADER_FIELDS) | {LINE_ITEMS}
    return [
        Example(
            doc_id=f"synthetic-{i:05d}",
            source="synthetic",
            split=Split.DEV if i % 2 == 0 else Split.TEST,
            labeled_fields=labeled,
            gold=make_invoice(rng, i),
        )
        for i in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    print(f"wrote {write_examples(args.out, make_examples(args.n, args.seed))} examples to {args.out}")


if __name__ == "__main__":
    main()
