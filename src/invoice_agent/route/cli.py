"""Confidence scoring and review routing over an extraction run.

    invoice-route fit   --predictions runs/X/predictions.jsonl --data data/canonical/midd.jsonl \
        [--second runs/Y/predictions.jsonl] --out runs/route-midd-dev
    invoice-route apply --model runs/route-midd-dev/model.json --predictions runs/Z/predictions.jsonl \
        --data data/canonical/midd.jsonl [--second ...] --out runs/route-midd-test

`fit` trains on dev with out-of-fold scores and picks thresholds per target error rate;
`apply` scores a held-out run with that frozen model and those frozen thresholds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any

from invoice_agent.erp.vendor_master import SupplierMaster
from invoice_agent.route.calibrate import Model, OperatingPoint, fit, out_of_fold_scores, pick, risk_coverage
from invoice_agent.route.signals import CRITICAL_FIELDS, agreement, compute_signals
from invoice_agent.schema import Example, Invoice, Split

TARGETS = (0.01, 0.02, 0.05)


MASTER_FOLDS = 5


def master_for(examples: dict[str, Example], doc_ids: list[str], master_split: Split | None,
               cross_fit: bool) -> dict[str, SupplierMaster]:
    """Which supplier master each document is resolved against. When the documents being scored
    are themselves the master's source (fit on dev), each fold gets a master built from the other
    folds, so no document is matched against its own label."""
    if master_split is None:
        return {}
    source = [e for e in examples.values() if e.split == master_split]
    if not cross_fit:
        m = SupplierMaster.from_examples(source)
        return {d: m for d in doc_ids}
    ids = sorted(doc_ids)
    random.Random(0).shuffle(ids)
    fold = {d: i % MASTER_FOLDS for i, d in enumerate(ids)}
    masters = {
        k: SupplierMaster.from_examples(e for e in source if fold.get(e.doc_id, -1) != k)
        for k in range(MASTER_FOLDS)
    }
    return {d: masters[fold[d]] for d in doc_ids}


def load(predictions: Path, data: Path, second: Path | None, today: dt.date,
         master_split: Split | None = None, cross_fit: bool = False) -> list[dict[str, Any]]:
    examples = {}
    with open(data) as f:
        for line in f:
            e = Example.model_validate_json(line)
            examples[e.doc_id] = e
    texts = {d: e.ocr_text for d, e in examples.items()}
    with open(predictions) as f:
        pred_rows = [json.loads(line) for line in f]
    masters = master_for(examples, [r["doc_id"] for r in pred_rows], master_split, cross_fit)
    others = {}
    if second:
        with open(second) as f:
            for line in f:
                r = json.loads(line)
                others[r["doc_id"]] = Invoice.model_validate(r["prediction"])
    docs = []
    for r in pred_rows:
        inv = Invoice.model_validate(r["prediction"])
        sig = compute_signals(
            inv, texts.get(r["doc_id"]), r["meta"].get("repairs", []),
            r["meta"].get("suspicious_instructions", False), today,
        )
        feats = dict(sig.features)
        if second:
            feats.update(agreement(inv, others[r["doc_id"]]) if r["doc_id"] in others else {"critical_disagreements": 5.0})
        if r.get("extraction_error"):
            sig.gates.append("extraction_failed")

        critical_errors = set(r["errors"]) & set(CRITICAL_FIELDS)
        strict_ok = not critical_errors
        ok = strict_ok
        if r["doc_id"] in masters:
            master, ex = masters[r["doc_id"]], examples[r["doc_id"]]
            res = master.resolve(inv.vendor_name, inv.vendor_tax_id)
            feats["supplier_by_gstin"] = float(res.method == "gstin")
            feats["supplier_by_name"] = float(res.method == "name")
            feats["supplier_gstin_name_conflict"] = float(res.gstin_name_conflict)
            if res.supplier_id is None:
                sig.gates.append("supplier_not_in_master")
            # Posting uses the resolved supplier record, so that's what must be right.
            if "vendor_name" in ex.labeled_fields:
                critical_errors.discard("vendor_name")
                if res.supplier_id != master.resolve_gold(ex):
                    critical_errors.add("supplier")
            ok = not critical_errors
        docs.append({
            "doc_id": r["doc_id"],
            "features": feats,
            "gates": sig.gates,
            "ok": int(ok),
            "strict_ok": int(strict_ok),
            "critical_errors": sorted(critical_errors),
        })
    return docs


def baselines(docs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(docs)
    all_err = sum(1 - d["ok"] for d in docs)
    passed = [d for d in docs if not d["gates"]]
    return {
        "auto_post_everything": OperatingPoint(0.0, n, all_err, n).as_dict(),
        "hard_gates_only": OperatingPoint(0.0, len(passed), sum(1 - d["ok"] for d in passed), n).as_dict(),
    }


def summarize(docs, scores, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    ok = [d["ok"] for d in docs]
    eligible = [not d["gates"] for d in docs]
    points = risk_coverage(scores, ok, eligible)
    out: dict[str, Any] = {
        "n_docs": len(docs),
        "critical_correct_rate": round(sum(ok) / len(ok), 4),
        "critical_correct_rate_strict_strings": round(sum(d["strict_ok"] for d in docs) / len(docs), 4),
        **baselines(docs),
    }
    err_counts: dict[str, int] = {}
    for d in docs:
        for e in d["critical_errors"]:
            err_counts[e] = err_counts.get(e, 0) + 1
    out["critical_error_counts"] = err_counts
    chosen = {}
    for t in TARGETS:
        key = f"target_{t:.0%}"
        if thresholds is None:
            p = pick(points, t)
            chosen[key] = p.threshold if p else None
        else:
            thr = thresholds.get(key)
            sel = [i for i in range(len(docs)) if eligible[i] and thr is not None and scores[i] >= thr]
            p = OperatingPoint(thr or 1.0, len(sel), sum(1 - ok[i] for i in sel), len(docs))
        out[key] = p.as_dict() if p else None
    out["risk_coverage"] = [p.as_dict() for p in points]
    gate_counts: dict[str, int] = {}
    for d in docs:
        for g in d["gates"]:
            k = g.split(":")[0]
            gate_counts[k] = gate_counts.get(k, 0) + 1
    out["gate_counts"] = gate_counts
    return out, chosen


def print_summary(title: str, s: dict[str, Any]) -> None:
    print(f"{title}: {s['n_docs']} docs, posting-critical fields all correct on {s['critical_correct_rate']:.1%}"
          f" (strict strings: {s['critical_correct_rate_strict_strings']:.1%}); errors {s['critical_error_counts']}")
    rows = [("auto-post everything", s["auto_post_everything"]), ("hard gates only", s["hard_gates_only"])]
    rows += [(f"score, target error {k[7:]}", s[k]) for k in s if k.startswith("target_")]
    for label, p in rows:
        if p is None:
            print(f"  {label:28} no threshold meets the target")
            continue
        lo, hi = p["error_rate_ci95"]
        print(f"  {label:28} auto-post {p['auto_post_rate']:6.1%}  error among auto-posted "
              f"{p['error_rate_among_auto_posted']:6.1%} (CI {lo:.1%}-{hi:.1%})  [{p['errors_among_auto_posted']}/{p['auto_posted']}]")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fit", "apply"])
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--second", type=Path, default=None, help="independent second extraction, for agreement")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--master-split", type=Split, default=None,
                    help="build the supplier master from this split's labels in --data (e.g. dev)")
    ap.add_argument("--today", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    preds_split = {json.loads(line)["doc_id"] for line in open(args.predictions)}
    cross_fit = args.master_split is not None and any(
        Example.model_validate_json(line).split == args.master_split
        for line in open(args.data) if json.loads(line)["doc_id"] in preds_split
    )
    docs = load(args.predictions, args.data, args.second, args.today, args.master_split, cross_fit)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "fit":
        features = sorted({k for d in docs for k in d["features"]})
        xs, ys = [d["features"] for d in docs], [d["ok"] for d in docs]
        scores = out_of_fold_scores(xs, ys, features)
        summary, chosen = summarize(docs, scores)
        model = fit(xs, ys, features)
        (args.out / "model.json").write_text(json.dumps({**model.to_dict(), "thresholds": chosen}, indent=2))
        summary["weights"] = dict(sorted(zip(model.features, (round(w, 3) for w in model.weights)), key=lambda kv: kv[1]))
        title = "dev (out-of-fold)"
    else:
        saved = json.loads(args.model.read_text())
        thresholds = saved.pop("thresholds")
        model = Model(**saved)
        scores = [model.score(d["features"]) for d in docs]
        summary, _ = summarize(docs, scores, thresholds)
        title = "held-out (frozen model and thresholds)"
    (args.out / "routing.json").write_text(json.dumps(summary, indent=2))
    with open(args.out / "decisions.jsonl", "w") as f:
        for d, s in zip(docs, scores):
            f.write(json.dumps({"doc_id": d["doc_id"], "score": round(s, 4), "gates": d["gates"], "ok": d["ok"]}) + "\n")
    print_summary(title, summary)
    if "weights" in summary:
        print("  weights:", summary["weights"])


if __name__ == "__main__":
    main()
