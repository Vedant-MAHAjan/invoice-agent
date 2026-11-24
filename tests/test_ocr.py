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
