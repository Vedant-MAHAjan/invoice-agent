import datetime as dt
from collections import Counter
from decimal import Decimal

from invoice_agent.data.fatura import parse_annotation, parse_buyer, parse_gstin, parse_number, sample
from invoice_agent.schema import Split

ANN = {
    "NUMBER": {"text": "INVOICE ID INV/11-05/138"},
    "DATE": {"text": "Invoice Date: 17-Nov-2002"},
    "DUE_DATE": {"text": "Due Date : 14-Jan-2019"},
    "SELLER_NAME": {"text": "Phillips, Thomas and Malone"},
    "SELLER_ADDRESS": {"text": "Address:744 Hernandez Hollow\nRossville, UT 53295 US"},
    "BUYER": {"text": "Bill to:Beth Brennan\n416 Grimes Mall\nTel:+(648)618-8332"},
    "SUB_TOTAL": {"text": "SUB_TOTAL : 1,082.94  EUR"},
    "TAX": {"text": "TAX:VAT (5.75%):  28.96 EUR"},
    "TOTAL": {"text": "TOTAL : 1113.61 EUR"},
    "PO_NUMBER": {"text": "PO Number :39"},
    "PAYMENT_DETAILS": {"text": "Bank Name   X\nBank Account Number  18160778\nBank Swift Code  SBIN"},
    "TABLE": [[{"bbox": [[0, 0], [1, 1]]}]],
    "INVOICE_INFO": [],
}


def test_parse_annotation():
    v = parse_annotation(ANN, Counter())
    assert v["invoice_number"] == "INV/11-05/138"
    assert v["invoice_date"] == dt.date(2002, 11, 17)
    assert v["due_date"] == dt.date(2019, 1, 14)
    assert v["vendor_name"] == "Phillips, Thomas and Malone"
    assert v["vendor_address"] == "744 Hernandez Hollow Rossville, UT 53295 US"
    assert v["buyer_name"] == "Beth Brennan"
    assert v["subtotal"] == Decimal("1082.94")
    assert v["tax_total"] == Decimal("28.96")  # last number, not the 5.75% rate
    assert v["total"] == Decimal("1113.61")
    assert v["currency"] == "EUR"
    assert v["po_number"] == "39"
    assert v["bank_account_number"] == "18160778"


def test_caption_variants():
    assert parse_number("Invoice number 6Y9M4d-779") == "6Y9M4d-779"
    assert parse_number("INVOICE ID 2076-675") == "2076-675"
    assert parse_buyer("BILL_TO:\nRobert English\n983 Micheal Extensions") == "Robert English"
    assert parse_gstin("(GSTIN: 12345670 00070007") == "12345670 00070007"
    assert parse_gstin("GSTIN/UIN: 09AABCS142961ZS") == "09AABCS142961ZS"


def test_sample_is_per_template_and_split():
    stems = [f"Template{t}_Instance{i}" for t in (1, 2) for i in range(10)]
    splits = {s: (Split.DEV if int(s.rsplit("Instance", 1)[1]) < 5 else Split.TEST) for s in stems}
    picked = sample(stems, splits, per_template=2, seed=0)
    assert len(picked) == 8
    assert picked == sample(stems, splits, per_template=2, seed=0)
