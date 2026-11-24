import json

from invoice_agent.data.loader import read_examples, write_examples
from invoice_agent.data.synthetic import make_examples
from invoice_agent.extractors.stub import EmptyExtractor, NoisyOracleExtractor, OracleExtractor
from invoice_agent.harness import run
from invoice_agent.schema import Split


def test_synthetic_is_arithmetically_consistent():
    for e in make_examples(30):
        g = e.gold
        assert sum(li.amount for li in g.line_items) == g.subtotal
        assert g.subtotal + g.tax_total == g.total


def test_roundtrip(tmp_path):
    path = tmp_path / "d.jsonl"
    examples = make_examples(6)
    write_examples(path, examples)
    assert list(read_examples(path)) == examples
    assert len(list(read_examples(path, Split.TEST))) == 3


def test_harness_ordering(tmp_path):
    data = tmp_path / "d.jsonl"
    write_examples(data, make_examples(20))
    f1 = {}
    for extractor in (OracleExtractor(), NoisyOracleExtractor(p=0.2), EmptyExtractor()):
        report = run(extractor, data, None, tmp_path / extractor.name)
        f1[extractor.name] = report["overall"]["micro_header"]["f1"]
        lines = (tmp_path / extractor.name / "predictions.jsonl").read_text().splitlines()
        assert len(lines) == 20 and "prediction" in json.loads(lines[0])
    assert f1["oracle"] == 1.0 > f1["noisy-oracle"] > f1["empty"] == 0.0
