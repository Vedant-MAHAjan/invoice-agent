"""Canonical invoice schema shared by datasets, extractors, validators and the ERP poster."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Money = Annotated[Decimal, Field(max_digits=18, decimal_places=4)]


class FieldKind(str, Enum):
    TEXT = "text"  # fuzzy-ish: case/whitespace/punctuation-insensitive
    ID = "id"  # alphanumeric only, case-insensitive
    DATE = "date"
    MONEY = "money"
    CURRENCY = "currency"
    NUMBER = "number"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LineItem(Strict):
    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Money | None = None
    amount: Money | None = None
    tax_rate: Decimal | None = Field(default=None, description="Percent, e.g. 18 for 18%")


class TaxLine(Strict):
    rate: Decimal | None = Field(default=None, description="Percent")
    base: Money | None = None
    amount: Money | None = None


class Invoice(Strict):
    vendor_name: str | None = None
    vendor_tax_id: str | None = None
    vendor_address: str | None = None
    buyer_name: str | None = None
    buyer_tax_id: str | None = None

    invoice_number: str | None = None
    po_number: str | None = None
    invoice_date: dt.date | None = None
    due_date: dt.date | None = None
    payment_terms: str | None = None

    currency: str | None = Field(default=None, description="ISO 4217 code")
    subtotal: Money | None = None
    tax_total: Money | None = None
    total: Money | None = None
    amount_due: Money | None = None

    bank_iban: str | None = None
    bank_account_number: str | None = None

    line_items: tuple[LineItem, ...] = ()
    tax_lines: tuple[TaxLine, ...] = ()


HEADER_FIELDS: dict[str, FieldKind] = {
    "vendor_name": FieldKind.TEXT,
    "vendor_tax_id": FieldKind.ID,
    "vendor_address": FieldKind.TEXT,
    "buyer_name": FieldKind.TEXT,
    "buyer_tax_id": FieldKind.ID,
    "invoice_number": FieldKind.ID,
    "po_number": FieldKind.ID,
    "invoice_date": FieldKind.DATE,
    "due_date": FieldKind.DATE,
    "payment_terms": FieldKind.TEXT,
    "currency": FieldKind.CURRENCY,
    "subtotal": FieldKind.MONEY,
    "tax_total": FieldKind.MONEY,
    "total": FieldKind.MONEY,
    "amount_due": FieldKind.MONEY,
    "bank_iban": FieldKind.ID,
    "bank_account_number": FieldKind.ID,
}

LINE_ITEM_FIELDS: dict[str, FieldKind] = {
    "description": FieldKind.TEXT,
    "quantity": FieldKind.NUMBER,
    "unit_price": FieldKind.MONEY,
    "amount": FieldKind.MONEY,
    "tax_rate": FieldKind.NUMBER,
}

LINE_ITEMS = "line_items"


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    OOD = "ood"


class Example(Strict):
    """One labeled document. `labeled_fields` says which fields the source dataset annotates,
    so a dataset that never labels e.g. due_date isn't penalised for it."""

    doc_id: str
    source: str
    split: Split
    document_path: str | None = None
    ocr_text: str | None = None
    labeled_fields: frozenset[str]
    gold: Invoice
