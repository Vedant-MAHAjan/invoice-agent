from invoice_agent.schema import Invoice
from invoice_agent.validate.gstin import (
    check_digit,
    is_valid_gstin,
    owner_matches,
    repair_party_tax_ids,
)

AVIVA = "27AARFA8194J1ZP"  # PAN initial A
SIDDHARTH = "27AAZFS3607C1ZY"  # PAN initial S


def test_check_digit_and_validity():
    assert check_digit(AVIVA[:14]) == AVIVA[14]
    assert is_valid_gstin(AVIVA) and is_valid_gstin("27 AARFA8194J1ZP")
    assert not is_valid_gstin("27AAZFS3607CIZY")  # OCR I-for-1
    assert not is_valid_gstin("27AARFA8194J1ZQ")
    assert not is_valid_gstin(None)


def test_owner_matches():
    assert owner_matches(AVIVA, "AVIVA DOORS") is True
    assert owner_matches(AVIVA, "M/s. SIDDHARTH PROPERTIES") is False
    assert owner_matches(AVIVA, None) is None


def test_swap_when_both_reversed():
    inv = Invoice(vendor_name="AVIVA DOORS", buyer_name="SIDDHARTH PROPERTIES", vendor_tax_id=SIDDHARTH, buyer_tax_id=AVIVA)
    fixed, notes = repair_party_tax_ids(inv)
    assert (fixed.vendor_tax_id, fixed.buyer_tax_id, notes) == (AVIVA, SIDDHARTH, ["gstin_swapped"])


def test_move_single_id_to_right_party():
    inv = Invoice(vendor_name="AVIVA DOORS", buyer_name="SIDDHARTH PROPERTIES", vendor_tax_id=SIDDHARTH)
    fixed, notes = repair_party_tax_ids(inv)
    assert fixed.vendor_tax_id is None and fixed.buyer_tax_id == SIDDHARTH and notes == ["gstin_moved_to_buyer"]


def test_no_change_when_correct_or_ambiguous():
    ok = Invoice(vendor_name="AVIVA DOORS", buyer_name="SIDDHARTH PROPERTIES", vendor_tax_id=AVIVA, buyer_tax_id=SIDDHARTH)
    assert repair_party_tax_ids(ok) == (ok, [])
    same_initial = Invoice(vendor_name="SHAH MARKETING", buyer_name="SIDDHARTH PROPERTIES", vendor_tax_id=SIDDHARTH, buyer_tax_id="27ADXPS3921J1ZT")
    assert repair_party_tax_ids(same_initial) == (same_initial, [])
