import datetime as dt
from decimal import Decimal

from invoice_agent.data.midd import Stats, convert, spans
from invoice_agent.schema import Split

HEADER = "Text,Tag\n"
DOC = HEADER + "\n".join(
    [
        "Acme,B-SUPP_N",
        "Traders,I-SUPP_N",
        "GSTIN,B-GSTL",
        "27AARFA8194J1ZP,B-SUPP_G",
        "Invoice,B-INV_L",
        "No,I-INV_L",
        "TI-113/19,B-INV_NO",
        "-,I-INV_NO",
        "20,I-INV_NO",
        "05/04/2019,B-INV_DT",
        "Buyer,O",
        "Total,B-GT_AMTL",
        '"1,23,450.00",B-GT_AMT',
    ]
)


def write(root, layout, name, text):
    d = root / layout
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")


def test_spans_handles_stray_inside_tag():
    toks = [("a", "I-X"), ("b", "I-X"), ("c", "O"), ("d", "B-X")]
    assert spans(toks) == {"X": ["a b", "d"]}


def test_convert(tmp_path):
    write(tmp_path, "Layout1", "1131GVDTD.csv", DOC)
    write(tmp_path, "Layout1", "113 (1)1GVDTD.csv", DOC)  # exact duplicate
    write(tmp_path, "Layout2", "9GVDTD.csv", "text,tag\nFoo,B-SUPP_N\n")
    stats = Stats()
    examples = convert(tmp_path, stats)
    assert (stats.files, stats.duplicates, stats.written) == (3, 1, 2)

    e = next(x for x in examples if x.source == "midd-layout1")
    g = e.gold
    assert g.vendor_name == "Acme Traders"
    assert g.vendor_tax_id == "27AARFA8194J1ZP"
    assert g.invoice_number == "TI-113/19 - 20"
    assert g.invoice_date == dt.date(2019, 4, 5)  # day-first
    assert g.total == Decimal("123450.00")
    assert g.buyer_name is None and "buyer_name" not in e.labeled_fields
    assert e.split in (Split.DEV, Split.TEST)
    assert "Invoice No" in e.ocr_text
