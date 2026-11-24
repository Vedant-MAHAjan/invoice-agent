"""Run an extractor over a canonical dataset and write predictions + a metrics report.

    invoice-eval --data fixtures/synthetic.jsonl --extractor noisy-oracle --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from invoice_agent.data.loader import read_examples
from invoice_agent.eval.scorer import aggregate, score_document
from invoice_agent.extractors.base import Extractor
from invoice_agent.extractors.stub import STUBS
from invoice_agent.schema import Split


def run(extractor: Extractor, data: Path, split: Split | None, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results, cost, latency = [], 0.0, 0.0
    by_source: dict[str, list] = {}
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as preds:
        for ex in read_examples(data, split):
            extraction = extractor.extract(ex)
            doc = score_document(ex, extraction.invoice)
            results.append(doc)
            by_source.setdefault(ex.source, []).append(doc)
            cost += extraction.cost_usd
            latency += extraction.latency_s
            preds.write(
                json.dumps(
                    {
                        "doc_id": ex.doc_id,
                        "source": ex.source,
                        "prediction": extraction.invoice.model_dump(mode="json"),
                        "all_correct": doc.all_correct,
                        "errors": doc.errors,
                        "cost_usd": extraction.cost_usd,
                        "latency_s": extraction.latency_s,
                        "meta": extraction.meta,
                    }
                )
                + "\n"
            )

    n = len(results)
    report = {
        "extractor": extractor.name,
        "data": str(data),
        "split": split.value if split else "all",
        "overall": aggregate(results),
        "by_source": {s: aggregate(r) for s, r in sorted(by_source.items())},
        "cost_usd_total": round(cost, 6),
        "cost_usd_per_doc": round(cost / n, 6) if n else None,
        "latency_s_per_doc": round(latency / n, 3) if n else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def build_extractor(name: str) -> Extractor:
    if name in STUBS:
        return STUBS[name]()
    raise SystemExit(f"unknown extractor {name!r}; available: {', '.join(STUBS)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--extractor", default="oracle")
    ap.add_argument("--split", type=Split, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    extractor = build_extractor(args.extractor)
    out = args.out or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{extractor.name}"
    report = run(extractor, args.data, args.split, out)

    o = report["overall"]
    d = o["doc_all_fields_correct"]
    print(f"extractor={report['extractor']} docs={o['n_docs']} -> {out}")
    for label, key in (("header micro F1", "micro_header"), ("line-item micro F1", "micro_line_items")):
        print(f"  {label:<20} {f'{o[key]['f1']:.3f}' if o[key] else 'n/a (not labeled)'}")
    print(f"  all-fields-correct   {d['rate']:.3f}  (95% CI {d['ci95'][0]:.3f}-{d['ci95'][1]:.3f})")


if __name__ == "__main__":
    main()
