import datetime as dt
import json
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from invoice_agent.data.loader import write_examples
from invoice_agent.extractors.ollama import ExtractionError, OllamaExtractor, inline_refs
from invoice_agent.extractors.wire import wire_json_schema
from invoice_agent.harness import run
from invoice_agent.schema import Example, Invoice, Split

WIRE = {
    "vendor_name": "AVIVA DOORS",
    "vendor_tax_id": "27AARFA8194J1ZP",
    "vendor_address": None,
    "buyer_name": "SIDDHARTH PROPERTIES",
    "buyer_tax_id": None,
    "invoice_number": "TI-113/19-20",
    "po_number": None,
    "invoice_date": "2019-04-05",
    "due_date": None,
    "payment_terms": None,
    "currency": "inr",
    "subtotal": None,
    "tax_total": None,
    "total": "33800.00",
    "amount_due": None,
    "bank_iban": None,
    "bank_account_number": None,
    "line_items": [],
    "tax_lines": [],
    "suspicious_instructions": False,
}


class FakeOllama:
    def __init__(self, content=None, done_reason="stop"):
        self.requests = []
        content = json.dumps(WIRE) if content is None else content
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps(
                    {"message": {"content": content}, "done_reason": done_reason, "prompt_eval_count": 900, "eval_count": 300}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def ollama():
    servers = []

    def make(**kw):
        s = FakeOllama(**kw)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def example(doc_id="d1"):
    return Example(
        doc_id=doc_id,
        source="midd-layout1",
        split=Split.DEV,
        ocr_text="AVIVA DOORS GSTIN 27AARFA8194J1ZP Invoice No TI-113/19-20 Total 33,800.00",
        labeled_fields=frozenset({"vendor_name", "invoice_number", "invoice_date", "total"}),
        gold=Invoice(
            vendor_name="AVIVA DOORS",
            invoice_number="TI-113/19 - 20",
            invoice_date=dt.date(2019, 4, 5),
            total=Decimal("33800.00"),
        ),
    )


def test_schema_strict_and_inlined():
    schema = wire_json_schema()
    for obj in [schema, *schema["$defs"].values()]:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    flat = json.dumps(inline_refs(schema))
    assert "$ref" not in flat and "$defs" not in flat
    assert "unit_price" in flat


def test_extract_parses_and_sends_constrained_request(ollama, tmp_path):
    srv = ollama()
    out = OllamaExtractor(base_url=srv.url, cache_dir=tmp_path).extract(example())
    assert out.invoice.total == Decimal("33800.00")
    assert out.invoice.invoice_date == dt.date(2019, 4, 5)
    assert out.invoice.currency == "INR"
    assert out.cost_usd == 0.0
    req = srv.requests[0]
    assert req["format"]["type"] == "object" and "tools" not in req
    assert req["options"]["temperature"] == 0
    assert "<document>" in req["messages"][1]["content"]


def test_cache_prevents_second_call(ollama, tmp_path):
    srv = ollama()
    ext = OllamaExtractor(base_url=srv.url, cache_dir=tmp_path)
    ext.extract(example())
    second = ext.extract(example())
    assert len(srv.requests) == 1 and second.meta["cached"]


@pytest.mark.parametrize("kw", [{"done_reason": "length"}, {"content": "not json"}])
def test_bad_outputs_raise(ollama, tmp_path, kw):
    srv = ollama(**kw)
    with pytest.raises(ExtractionError):
        OllamaExtractor(base_url=srv.url, cache_dir=tmp_path).extract(example())


def test_harness_scores_and_records_failures(ollama, tmp_path):
    data = tmp_path / "d.jsonl"
    write_examples(data, [example("a"), example("b")])
    srv = ollama()
    ok = run(OllamaExtractor(base_url=srv.url, cache_dir=tmp_path / "c1"), data, None, tmp_path / "ok", workers=2)
    assert ok["overall"]["doc_all_fields_correct"]["correct"] == 2
    down = run(OllamaExtractor(base_url="http://127.0.0.1:1", cache_dir=None, timeout_s=2), data, None, tmp_path / "down")
    assert down["extraction_failures"]["count"] == 2


def test_garbage_numbers_dropped_not_fatal():
    from invoice_agent.extractors.wire import WireInvoice, to_invoice

    bad = {**WIRE, "total": "3261002349234930798.0023492349", "line_items": [
        {"description": "x", "quantity": "1", "unit_price": None, "amount": "9" * 25, "tax_rate_percent": None}
    ]}
    inv = to_invoice(WireInvoice.model_validate(bad))
    assert inv.total is None and inv.line_items[0].amount is None
