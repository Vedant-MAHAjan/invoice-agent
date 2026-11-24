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

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| MIDD | [Zenodo 5113009](https://zenodo.org/record/5113009), CC-BY-4.0 | Token text only, 7 header fields, no line items. 708 files → 684 after exact-duplicate removal. Fields with no annotation in a document aren't scored for it (e.g. total is unlabeled on 98 docs). |

```bash
mkdir -p data/raw/midd/archive data/raw/midd/extracted
curl -L -o data/raw/midd/archive/IOB.rar https://zenodo.org/api/records/5113009/files/IOB.rar/content
tar -xf data/raw/midd/archive/IOB.rar -C data/raw/midd/extracted
.venv/bin/python -m invoice_agent.data.midd data/raw/midd/extracted/IOB data/canonical/midd.jsonl
```

## Data policy

- `data/raw/`: downloaded public datasets. Gitignored.
- `data/private/`: personal bills (OOD set). Gitignored, never published.
