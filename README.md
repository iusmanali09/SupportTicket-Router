# SupportTicket Router

**Owner: Usman**

SupportTicket Router is a small, local NLP baseline that classifies a support ticket from its subject and description, then suggests a destination team. It uses TF-IDF word features and logistic regression. Predictions are recommendations for human review, not automated assignment decisions.

## Problem and taxonomy

Support queues often receive tickets that need to be triaged to a suitable team. The baseline maps each ticket to one category and one suggested destination:

| Category | Suggested team | Scope |
| --- | --- | --- |
| `account_access` | Identity & Access | Sign-in, password reset, MFA, account lockout |
| `billing` | Billing Operations | Invoices, charges, refunds, payment methods |
| `bug_report` | Product Support Engineering | Unexpected product behavior or reproducible defects |
| `feature_request` | Product Management | Requests for new capabilities or product changes |
| `technical_support` | Technical Support | Setup, configuration, integrations, and usage help |

These labels are an initial example taxonomy. Adapt them and the team mapping to the actual organization before routing live tickets.

## Dataset

No dataset is included, and no model performance is claimed. Prepare a CSV locally as described in [DATASET.md](DATASET.md). It must have `subject`, `description`, and `category` columns. Raw and processed data, and model files, are excluded by `.gitignore` by default.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Train and evaluate

After preparing `data/raw/tickets.csv`:

```bash
support-ticket-router train
```

This creates deterministic stratified train, validation, and test splits (70/15/15, seed 42 by default). The classifier is evaluated on validation data, then fitted on train plus validation and evaluated once on the held-out test split. Output includes routing accuracy, per-category precision, recall and F1, and a confusion matrix. `macro avg` and `weighted avg` are included in sklearn's per-category report. Specify another seed or paths as needed:

```bash
support-ticket-router train --data data/raw/tickets.csv --model models/router.joblib --seed 17
```

Do not tune repeatedly against the test metrics; use validation data for model selection. Evaluation estimates depend on data quality, label consistency, representativeness, and dataset size.

## Predict

Pass both fields to get the predicted category, suggested team, predicted-class confidence, and scores for all categories:

```bash
support-ticket-router predict \
  --subject "Unable to sign in" \
  --description "My account is locked after the password reset flow."
```

For an interactive prompt:

```bash
support-ticket-router demo
```

The model artifact must first be trained and available at `models/router.joblib` or supplied via `--model`. Inputs must be non-empty strings no longer than 20,000 characters each. Confidence is a classifier probability, not a guarantee of correctness or a calibrated risk score.

## Configuration and limitations

- `--seed` controls split and model randomness; default is `42`.
- `--data` selects the CSV; `--model` selects the serialized model path.
- Categories and destination teams are defined in `src/support_ticket_router/taxonomy.py`.
- Text is whitespace-normalized and vectorized with word unigrams and bigrams. There is no language detection, entity redaction, multilabel prediction, class-specific thresholding, or confidence calibration.
- Results may reflect imbalanced or mislabeled data, language and terminology drift, and category overlap. Rare or unseen cases need human review.
- Joblib artifacts should only be loaded from trusted sources; loading a malicious pickle-based file can execute code.
- This is a local command-line baseline; it does not provide authentication, an HTTP API, audit logging, or production ticket-system integration.

## Repository layout

```text
src/support_ticket_router/  data, model, taxonomy, and CLI
data/raw/                   local input CSV (ignored by Git)
data/processed/             reserved for local prepared datasets (ignored)
models/                     local model artifacts (ignored)
```

## License

No license has been selected. Add an appropriate license before distributing or reusing the project.
