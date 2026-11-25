from invoice_agent.data.ocr import Box, boxes_to_text


def box(text, x, y, h=0.02):
    return Box(text=text, conf=1.0, x=x, y=y, w=0.1, h=h)


def test_rows_and_columns_reconstructed():
    boxes = [
        box("Total", 0.1, 0.802),
        box("INVOICE", 0.4, 0.05),
        box("$1,200.00", 0.7, 0.800),
        box("Bill To:", 0.1, 0.20),
        box("Acme Ltd", 0.1, 0.23),
    ]
    assert boxes_to_text(boxes) == "INVOICE\nBill To:\nAcme Ltd\nTotal   $1,200.00"


def test_empty():
    assert boxes_to_text([]) == ""


def wbox(text, x, y, w=0.2, h=0.02):
    return Box(text=text, conf=1.0, x=x, y=y, w=w, h=h)


TWO_COL = [
    wbox("TAX INVOICE", 0.02, 0.01, w=0.3),
    wbox("Acme Supplies", 0.05, 0.10),
    wbox("Bill to: Globex", 0.55, 0.10),
    wbox("1 Main St", 0.05, 0.13),
    wbox("9 Side Rd", 0.55, 0.13),
    wbox("GSTIN 27AAA", 0.05, 0.16),
    wbox("Tel: 555", 0.55, 0.19),
    wbox("Qty", 0.05, 0.30, w=0.05), wbox("Item", 0.30, 0.30, w=0.05), wbox("Amount", 0.80, 0.30, w=0.1),
    wbox("2", 0.05, 0.33, w=0.05), wbox("Bolts", 0.30, 0.33, w=0.05), wbox("10.00", 0.80, 0.33, w=0.1),
]


def test_two_column_header_kept_as_blocks():
    assert boxes_to_text(TWO_COL, columns=True).splitlines() == [
        "TAX INVOICE",
        "Acme Supplies", "1 Main St", "GSTIN 27AAA",
        "",
        "Bill to: Globex", "9 Side Rd", "Tel: 555",
        "Qty   Item   Amount",
        "2   Bolts   10.00",
    ]


def test_row_mode_unchanged_and_interleaves():
    assert boxes_to_text(TWO_COL).splitlines()[1] == "Acme Supplies   Bill to: Globex"


def test_single_two_segment_row_is_not_a_region():
    boxes = [wbox("Total", 0.05, 0.5), wbox("100.00", 0.7, 0.5), wbox("Thanks", 0.05, 0.6, w=0.9)]
    assert boxes_to_text(boxes, columns=True) == "Total   100.00\nThanks"
