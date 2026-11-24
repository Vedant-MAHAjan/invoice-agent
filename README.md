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
```

## Data policy

- `data/raw/`: downloaded public datasets. Gitignored.
- `data/private/`: personal bills (OOD set). Gitignored, never published.
