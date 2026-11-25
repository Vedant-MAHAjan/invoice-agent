# Invoice agent

Extract invoices to a strict schema, validate deterministically, auto-post only what clears a
calibrated error budget, and post idempotently to ERPNext. Everything else goes to a review queue.

## Setup (macOS, local)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Eval harness

Datasets are converted to one canonical JSONL format (`invoice_agent.schema.Example`).
`labeled_fields` records which fields each source annotates, so a dataset is only scored on what it labels.

```bash
.venv/bin/python -m invoice_agent.data.synthetic fixtures/synthetic.jsonl --n 40
.venv/bin/invoice-eval --data fixtures/synthetic.jsonl --extractor noisy-oracle --split test
```

Each run writes `runs/<ts>-<extractor>/predictions.jsonl` and `report.json`. The report includes
per-field P/R/F1, header and line-item micro F1, and the all-fields-correct rate with a 95% Wilson CI,
broken down by source.

## Local LLM (zero cost)

Extraction runs entirely on the Mac through [Ollama](https://ollama.com): no API keys, no spend.
The model gets the document as untrusted data, no tools, and must return JSON matching the wire
schema (`format` = JSON schema, temperature 0).

```bash
ollama pull qwen2.5-coder:7b
.venv/bin/invoice-eval --data data/canonical/midd.jsonl --extractor ollama --model qwen2.5-coder:7b \
    --split dev --workers 2            # add --repair for deterministic post-extraction fixes
```

Responses are cached in `runs/cache/`, so rescoring or adding `--repair` makes no new model calls.

`--repair` applies code, not prompts:
- **GSTIN ownership.** An Indian GSTIN embeds the holder's PAN, whose 5th character is the first
  letter of the holder's name, so tax IDs the model put on the wrong party are moved back.
- **Amount in words.** Many invoices print the total in words ("Indian Rupees Seventy Seven Only").
  It's parsed (lakh/crore/thousand, paise), tax-labelled phrases are skipped, and the total is
  replaced only when the words amount is *also* printed as a number, so two renderings agree.

## Results so far (MIDD dev, 323 docs, qwen2.5-coder:14b, local)

| | Header micro F1 | Total F1 | All fields correct (95% CI) |
|---|---|---|---|
| LLM only | 0.771 | 0.433 | 12.1% (9.0–16.1) |
| + GSTIN ownership repair | 0.859 | 0.433 | 39.6% (34.4–45.1) |
| + amount-in-words repair | **0.926** | **0.949** | **62.2% (56.8–67.3)** |

The prompt was told which party is which and not to compute totals; it still swapped tax IDs and
picked the wrong number on the dominant layout. A few lines of deterministic code fixed both.
The amount-in-words check agrees with the gold total on 409 of 423 MIDD docs; several
disagreements are OCR errors in the gold labels themselves.

## Results so far (FATURA dev, 200 docs from 50 templates, qwen2.5-coder:14b, Apple Vision OCR)

| | Header micro F1 | All fields correct (95% CI) |
|---|---|---|
| LLM only, row-by-row OCR text | 0.908 | 46.0% (39.2–52.9) |
| + repairs | 0.908 | 46.0% (no-op: no amount in words, placeholder tax IDs) |
| LLM only, column-aware OCR text (`--columns`) | 0.912 | 48.0% (41.2–54.9) |

Column-aware text keeps side-by-side supplier/buyer blocks together instead of interleaving them.
It helped the fields it targets (supplier name 0.88 → 0.91, invoice number 0.92 → 0.95) and hurt
no money field, but at the document level it fixed 12 and broke 8, which is **not a significant
difference** on 200 docs. Most remaining supplier-address errors aren't layout: 19 of 73 are
addresses the OCR never read, 22 are partial spans, 15 are the wrong text block picked.

Money fields are near perfect (subtotal, total, amount due 1.00; tax 0.97). The weak fields are:
- **Supplier address (0.57) and name (0.88):** two-column headers. Rebuilding OCR lines row by row
  interleaves the supplier and buyer blocks, and the model often returns null for the supplier
  (see the column-aware row above).
- **Tax IDs (0.10–0.56):** FATURA's GSTINs are template placeholders (`O4AROPO98743125`), and
  OCR reads letter O as zero; there's no real check digit to recover them.

## Confidence scoring and review routing (dev, out-of-fold)

An invoice is safe to post only if its **posting-critical fields** (supplier name, invoice number,
invoice date, total, currency) are all correct. Signals are gold-free, so the same code runs in production:

- **Hard gates** (always review): missing supplier/number/date/total, a critical value that doesn't
  appear in the document (hallucination), total disagreeing with the amount in words, impossible date.
- **Soft signals:** captions leaking into values, total not printed, repaired fields, GSTIN check
  digit/ownership, subtotal + tax ≠ total, and field agreement with an independent second
  extraction (qwen2.5-coder:7b).
- A small logistic regression (pure Python) scores P(all critical fields correct); scores are
  5-fold out-of-fold, and the threshold is the largest auto-post share whose error stays within target.

```bash
.venv/bin/invoice-route fit --predictions runs/<14b>/predictions.jsonl --second runs/<7b>/predictions.jsonl \
    --data data/canonical/midd.jsonl --out runs/route-midd-dev
.venv/bin/invoice-route apply --model runs/route-midd-dev/model.json --predictions runs/<test>/predictions.jsonl ...
```

Posting raw supplier strings (strict):

| Dev set | Auto-post everything | Hard gates only | Target ≤5% error | Target ≤1% error |
|---|---|---|---|---|
| MIDD (323) | 100%, 27.9% wrong | 87.0%, 26.3% wrong | 31.6% auto-posted, 4.9% wrong | 3.4%, 0 wrong |
| FATURA (200) | 100%, 16.0% wrong | 49.5%, 13.1% wrong | 13.5%, 3.7% wrong | 8.5%, 0 wrong |

Resolving the supplier against a **supplier master** first (`--master-split dev`), the way AP
actually posts:

| Dev set | Auto-post everything | Hard gates only | Target ≤2% error | Target ≤5% error |
|---|---|---|---|---|
| MIDD (323) | 100%, 20.4% wrong | 84.2%, 15.4% wrong | **40.9%** auto-posted, 1.5% wrong | **62.8%**, 4.9% wrong |
| FATURA (200) | 100%, 15.0% wrong | 43.5%, 9.2% wrong | 7.0%, 0 wrong | **41.5%**, 4.8% wrong |

### Supplier master (`erp/vendor_master.py`)

AP doesn't post the raw supplier string; it matches the invoice to an onboarded supplier, by
valid GSTIN first and then by name, and sends anything unmatched to review as a possible new vendor.
The master is built from labeled documents as a stand-in for the ERP's supplier list; OCR variants
in the labels (`SHAL MARKETING`, `AH MARKETING`) are merged by shared GSTIN, so MIDD collapses to
5 suppliers. To keep it leak-free, the test run uses a master built from dev labels only, and on
dev each of 5 folds uses a master built from the other folds.

On dev, resolution was right for 311/323 MIDD and 117/136 FATURA invoices, and **never picked the
wrong supplier**: every miss was "unknown → review". A bare OCR alias like `marketing` can only match
nearly exactly, so a different "XYZ MARKETING" isn't absorbed into SHAH MARKETING.

What we learned:
- **The model's self-reported "suspicious instructions" flag is useless as a gate:** it fires on
  about half of clean invoices (payment-terms boilerplate). It's kept as a weak feature only.
- **Second-model agreement is strong per field, not overall.** When 14B and 7B disagree on the
  MIDD invoice date, 14B is wrong 37% of the time vs 0.7% when they agree (total: 31% vs 2.4%).
  But on supplier name the relation inverts, because both models make the same "mistake".
- **Most remaining MIDD "errors" are span conventions, not wrong values:** 61 of 105 are a supplier
  name with a branch suffix (`SHAH MARKETING (MANGALWAR PETH)` vs `SHAH MARKETING`) or an invoice
  number with/without a `/ GST` suffix. Ignoring those, posting-critical correctness is 88.5%, not
  72.1%. Resolving the supplier against a master (below) handles the supplier half of that; the
  invoice-number half (`/ GST` suffixes) remains and is now the largest MIDD error class.

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| MIDD | [Zenodo 5113009](https://zenodo.org/record/5113009), CC-BY-4.0 | Token text only, 7 header fields, no line items. 708 files → 684 after exact-duplicate removal. Fields with no annotation in a document aren't scored for it (e.g. total is unlabeled on 98 docs). |
| FATURA | [Zenodo 10371464](https://zenodo.org/records/10371464), CC-BY-4.0 | 10,000 synthetic images from 50 templates. Values are parsed out of caption-bearing annotation text; no line-item labels. We sample 4 docs per template per split (400) and OCR them locally with Apple Vision. **Totals don't reconcile** (subtotal − discount + tax = total holds on 11 of 5,597), so FATURA is used for extraction, not for arithmetic validation. |

```bash
mkdir -p data/raw/midd/archive data/raw/midd/extracted
curl -L -o data/raw/midd/archive/IOB.rar https://zenodo.org/api/records/5113009/files/IOB.rar/content
tar -xf data/raw/midd/archive/IOB.rar -C data/raw/midd/extracted
.venv/bin/python -m invoice_agent.data.midd data/raw/midd/extracted/IOB data/canonical/midd.jsonl

# FATURA (691 MB download; needs the OCR tool: swiftc -O tools/ocr/vision_ocr.swift -o tools/ocr/vision_ocr)
mkdir -p data/raw/fatura/archive data/raw/fatura/extracted
curl -L -o data/raw/fatura/archive/FATURA2.zip https://zenodo.org/api/records/10371464/files/FATURA2.zip/content
unzip -q data/raw/fatura/archive/FATURA2.zip 'invoices_dataset_final/images/*' \
    'invoices_dataset_final/Annotations/Original_Format/*' 'invoices_dataset_final/strat*' -d data/raw/fatura/extracted
.venv/bin/python -m invoice_agent.data.fatura data/raw/fatura/extracted/invoices_dataset_final data/canonical/fatura.jsonl
# column-aware text variant
.venv/bin/python -m invoice_agent.data.fatura data/raw/fatura/extracted/invoices_dataset_final data/canonical/fatura-columns.jsonl --columns
```

## Data policy

- `data/raw/`: downloaded public datasets. Gitignored.
- `data/private/`: personal bills (OOD set). Gitignored, never published.
