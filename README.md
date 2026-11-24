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

## Data policy

- `data/raw/`: downloaded public datasets. Gitignored.
- `data/private/`: personal bills (OOD set). Gitignored, never published.
