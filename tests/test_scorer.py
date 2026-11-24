import datetime as dt
from decimal import Decimal

from invoice_agent.eval.scorer import aggregate, compare_value, score_document, wilson_interval
from invoice_agent.schema import Example, FieldKind, Invoice, LineItem, Split

GOLD = Invoice(
    vendor_name="Acme Industrial Supply",
    invoice_number="INV-001",
    invoice_date=dt.date(2025, 2, 1),
    total=Decimal("110.00"),
    line_items=(
        LineItem(description="Toner cartridge", quantity=Decimal(2), amount=Decimal("60.00")),
        LineItem(description="Copy paper A4", quantity=Decimal(1), amount=Decimal("40.00")),
    ),
)


def ex(labeled=("vendor_name", "invoice_number", "invoice_date", "total", "line_items")):
    return Example(doc_id="d1", source="t", split=Split.TEST, labeled_fields=frozenset(labeled), gold=GOLD)


def test_compare_value_outcomes():
    assert compare_value(FieldKind.ID, None, None).__dict__ == {"tp": 0, "fp": 0, "fn": 0}
    assert compare_value(FieldKind.ID, None, "X").fp == 1
    assert compare_value(FieldKind.ID, "X", None).fn == 1
    wrong = compare_value(FieldKind.ID, "X", "Y")
    assert (wrong.fp, wrong.fn) == (1, 1)


def test_perfect_prediction():
    r = score_document(ex(), GOLD)
    assert r.all_correct and r.errors == []


def test_line_items_order_independent():
    pred = GOLD.model_copy(update={"line_items": tuple(reversed(GOLD.line_items))})
    assert score_document(ex(), pred).all_correct


def test_missing_line_item_counts_fn():
    pred = GOLD.model_copy(update={"line_items": GOLD.line_items[:1]})
    r = score_document(ex(), pred)
    assert r.fields["line_items.amount"].fn == 1
    assert not r.all_correct


def test_unlabeled_fields_not_scored():
    pred = GOLD.model_copy(update={"total": Decimal("1.00"), "due_date": dt.date(2025, 3, 1)})
    r = score_document(ex(labeled=("vendor_name", "invoice_number")), pred)
    assert r.all_correct


def test_wrong_total_breaks_doc():
    pred = GOLD.model_copy(update={"total": Decimal("111.00")})
    r = score_document(ex(), pred)
    assert r.errors == ["total"]


def test_aggregate_and_wilson():
    good = score_document(ex(), GOLD)
    bad = score_document(ex(), Invoice())
    agg = aggregate([good, bad])
    assert agg["doc_all_fields_correct"]["correct"] == 1
    assert agg["micro_header"]["recall"] == 0.5
    header_only = aggregate([score_document(ex(labeled=("total",)), GOLD)])
    assert header_only["micro_line_items"] is None
    lo, hi = wilson_interval(0, 30)
    assert lo == 0.0 and 0.10 < hi < 0.12
