import datetime as dt
import json
from decimal import Decimal

from invoice_agent.route.calibrate import fit, out_of_fold_scores, pick, risk_coverage
from invoice_agent.route.cli import main as route_main
from invoice_agent.route.signals import agreement, compute_signals, grounded
from invoice_agent.schema import Example, Invoice, Split

TODAY = dt.date(2026, 11, 25)
TEXT = "AVIVA DOORS TAX INVOICE Invoice No . : TI-122/19 - 20 Date : 12-04-2019 Grand Total 2,80,250.00"
GOOD = Invoice(
    vendor_name="AVIVA DOORS", invoice_number="TI-122/19 - 20",
    invoice_date=dt.date(2019, 4, 12), total=Decimal("280250.00"),
)


def test_grounded():
    assert grounded("TI-122/19-20", TEXT) is True  # spacing/punctuation ignored
    assert grounded("INV-999", TEXT) is False
    assert grounded(None, TEXT) is None


def test_clean_doc_passes_gates():
    s = compute_signals(GOOD, TEXT, [], False, TODAY)
    assert s.gates == [] and s.features["total_not_printed"] == 0


def test_gates_fire():
    hallucinated = GOOD.model_copy(update={"invoice_number": "INV-999", "total": None})
    s = compute_signals(hallucinated, TEXT, [], False, TODAY)
    assert any(g.startswith("missing:") for g in s.gates)
    assert any(g.startswith("ungrounded:") for g in s.gates)
    future = GOOD.model_copy(update={"invoice_date": dt.date(2031, 1, 1)})
    assert "invoice_date_out_of_range" in compute_signals(future, TEXT, [], False, TODAY).gates


def test_suspicious_flag_is_feature_not_gate():
    s = compute_signals(GOOD, TEXT, [], True, TODAY)
    assert s.gates == [] and s.features["suspicious_instructions"] == 1


def test_agreement():
    other = GOOD.model_copy(update={"total": Decimal("1.00")})
    a = agreement(GOOD, other)
    assert a["agree_total"] == 0 and a["agree_vendor_name"] == 1 and a["critical_disagreements"] == 1


def test_fit_learns_a_separating_feature():
    xs = [{"bad": float(i % 2), "noise": float(i % 3)} for i in range(200)]
    ys = [1 - int(x["bad"]) for x in xs]
    model = fit(xs, ys, ["bad", "noise"])
    assert model.score({"bad": 0}) > 0.9 > 0.1 > model.score({"bad": 1})
    oof = out_of_fold_scores(xs, ys, ["bad", "noise"])
    assert all((s > 0.5) == bool(y) for s, y in zip(oof, ys))


def test_risk_coverage_and_pick():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    ok = [1, 1, 0, 1, 0]
    eligible = [True, True, True, True, False]  # last one gated: never auto-posted
    points = risk_coverage(scores, ok, eligible)
    assert [(p.auto_posted, p.errors) for p in points] == [(1, 0), (2, 0), (3, 1), (4, 1)]
    assert pick(points, 0.0).auto_posted == 2
    assert pick(points, 0.25).auto_posted == 4
    assert pick([], 0.05) is None


def test_cli_fit_then_apply(tmp_path):
    data, preds = tmp_path / "d.jsonl", tmp_path / "p.jsonl"
    with open(data, "w") as fd, open(preds, "w") as fp:
        for i in range(40):
            good = i % 4 != 0
            inv = GOOD if good else GOOD.model_copy(update={"invoice_number": f"INVOICE ID {i}"})
            ex = Example(doc_id=f"d{i}", source="t", split=Split.DEV, ocr_text=TEXT,
                         labeled_fields=frozenset({"invoice_number"}), gold=GOOD)
            fd.write(ex.model_dump_json() + "\n")
            fp.write(json.dumps({
                "doc_id": f"d{i}", "prediction": inv.model_dump(mode="json"),
                "errors": [] if good else ["invoice_number"], "meta": {"repairs": []},
            }) + "\n")
    route_main(["fit", "--predictions", str(preds), "--data", str(data), "--today", "2026-11-25", "--out", str(tmp_path / "fit")])
    report = json.loads((tmp_path / "fit" / "routing.json").read_text())
    assert report["auto_post_everything"]["error_rate_among_auto_posted"] == 0.25
    assert report["target_1%"]["error_rate_among_auto_posted"] == 0.0
    assert report["target_1%"]["auto_posted"] > 0
    route_main(["apply", "--model", str(tmp_path / "fit" / "model.json"), "--predictions", str(preds),
                "--data", str(data), "--today", "2026-11-25", "--out", str(tmp_path / "apply")])
    applied = json.loads((tmp_path / "apply" / "routing.json").read_text())
    assert applied["target_1%"]["errors_among_auto_posted"] == 0
