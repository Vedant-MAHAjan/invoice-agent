from invoice_agent.erp.vendor_master import SupplierMaster
from invoice_agent.schema import Example, Invoice, Split

SHAH = "27ADXPS3921J1ZT"
AVIVA = "27AARFA8194J1ZP"


def ex(i, name, gstin=None):
    return Example(doc_id=f"d{i}", source="t", split=Split.DEV, labeled_fields=frozenset({"vendor_name"}),
                   gold=Invoice(vendor_name=name, vendor_tax_id=gstin))


MASTER = SupplierMaster.from_examples([
    ex(1, "SHAH MARKETING", SHAH), ex(2, "SHAH MARKETING", SHAH), ex(3, "SHAL MARKETING", SHAH),
    ex(4, "MARKETING", SHAH), ex(5, "AVIVA DOORS", AVIVA), ex(6, "INDIA CABLES"),
])


def test_ocr_variants_merge_by_gstin():
    assert len(MASTER) == 3
    shah = MASTER.suppliers[MASTER.by_gstin[SHAH]]
    assert shah.name == "shah marketing" and "shal marketing" in shah.aliases


def test_resolution_paths():
    assert MASTER.resolve("anything", SHAH).method == "gstin"
    by_name = MASTER.resolve("SHAH MARKETING (MANGALWAR PETH)", None)  # superset of canonical name
    assert by_name.method == "name" and by_name.supplier_id == MASTER.by_gstin[SHAH]
    assert MASTER.resolve("Shal Marketing", None).supplier_id == MASTER.by_gstin[SHAH]


def test_unknowns_are_not_forced():
    # A different marketing firm must not match the bare "marketing" OCR alias.
    assert MASTER.resolve("XYZ MARKETING", None).supplier_id is None
    assert MASTER.resolve(None, None).method == "unknown"
    assert MASTER.resolve("Brand New Supplier Pvt Ltd", "27AAAAA0000A1Z5").supplier_id is None


def test_gstin_name_conflict_flagged():
    r = MASTER.resolve("AVIVA DOORS", SHAH)
    assert r.supplier_id == MASTER.by_gstin[SHAH] and r.gstin_name_conflict
