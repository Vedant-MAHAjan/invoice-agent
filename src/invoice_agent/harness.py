"""Run an extractor over a canonical dataset and write predictions + a metrics report.

    invoice-eval --data fixtures/synthetic.jsonl --extractor noisy-oracle --split test
    invoice-eval --data data/canonical/midd.jsonl --extractor ollama --model qwen2.5-coder:7b \
        --split dev --limit 20 --workers 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from invoice_agent.data.loader import read_examples
from invoice_agent.eval.scorer import aggregate, score_document
from invoice_agent.extractors.base import Extraction, Extractor
from invoice_agent.extractors.stub import STUBS
from invoice_agent.schema import Example, Invoice, Split


def _safe_extract(extractor: Extractor, ex: Example) -> tuple[Extraction, str | None]:
    try:
        return extractor.extract(ex), None
    except Exception as e:  # noqa: BLE001 - one bad doc must not kill a paid run
        return Extraction(invoice=Invoice()), f"{type(e).__name__}: {e}"


def run(
    extractor: Extractor,
    data: Path,
    split: Split | None,
    out_dir: Path,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = list(itertools.islice(read_examples(data, split), limit))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = list(pool.map(lambda ex: _safe_extract(extractor, ex), examples))

    results, by_source, failures = [], {}, []
    cost = latency = 0.0
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as preds:
        for ex, (extraction, error) in zip(examples, outcomes):
            doc = score_document(ex, extraction.invoice)
            results.append(doc)
            by_source.setdefault(ex.source, []).append(doc)
            cost += extraction.cost_usd
            latency += extraction.latency_s
            if error:
                failures.append({"doc_id": ex.doc_id, "error": error})
            preds.write(
                json.dumps(
                    {
                        "doc_id": ex.doc_id,
                        "source": ex.source,
                        "prediction": extraction.invoice.model_dump(mode="json"),
                        "gold": ex.gold.model_dump(mode="json"),
                        "all_correct": doc.all_correct,
                        "errors": doc.errors,
                        "extraction_error": error,
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
        "limit": limit,
        "overall": aggregate(results),
        "by_source": {s: aggregate(r) for s, r in sorted(by_source.items())},
        "extraction_failures": {"count": len(failures), "examples": failures[:20]},
        "cost_usd_total": round(cost, 6),
        "cost_usd_per_doc": round(cost / n, 6) if n else None,
        "latency_s_per_doc": round(latency / n, 3) if n else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def build_extractor(args: argparse.Namespace) -> Extractor:
    if args.extractor in STUBS:
        return STUBS[args.extractor]()
    if args.extractor == "ollama":
        from invoice_agent.extractors.ollama import OllamaExtractor

        extractor = OllamaExtractor(model=args.model, base_url=args.ollama_url)
        if args.repair:
            from invoice_agent.extractors.repaired import RepairedExtractor

            return RepairedExtractor(extractor)
        return extractor
    raise SystemExit(f"unknown extractor {args.extractor!r}; available: ollama, {', '.join(STUBS)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--extractor", default="oracle")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--repair", action="store_true", help="apply deterministic post-extraction repairs")
    ap.add_argument("--split", type=Split, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    extractor = build_extractor(args)
    safe_name = extractor.name.replace(":", "_")
    out = args.out or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe_name}"
    report = run(extractor, args.data, args.split, out, args.limit, args.workers)

    o = report["overall"]
    d = o["doc_all_fields_correct"]
    print(f"extractor={report['extractor']} docs={o['n_docs']} -> {out}")
    for label, key in (("header micro F1", "micro_header"), ("line-item micro F1", "micro_line_items")):
        print(f"  {label:<20} {f'{o[key]['f1']:.3f}' if o[key] else 'n/a (not labeled)'}")
    print(f"  all-fields-correct   {d['rate']:.3f}  (95% CI {d['ci95'][0]:.3f}-{d['ci95'][1]:.3f})")
    print(f"  failures             {report['extraction_failures']['count']}")
    print(f"  latency              {report['latency_s_per_doc'] or 0:.2f}s/doc")


if __name__ == "__main__":
    main()
