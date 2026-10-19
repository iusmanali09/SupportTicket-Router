# Dataset

This repository does not include customer support messages or a bundled training dataset. Ticket text may contain personal or confidential information, so obtain authorization before using real data and keep raw files local. The `data/raw/` directory is ignored by Git.

## Prepare a CSV

Create `data/raw/tickets.csv` in UTF-8 with a header and these columns:

| Column | Meaning |
| --- | --- |
| `subject` | Short ticket title; required |
| `description` | Ticket body; required |
| `category` | One label from the taxonomy below; required |

Rows with empty text, unknown labels, or invalid CSV fields are rejected. Remove personal data that is not needed, de-duplicate repeated tickets, and make labels consistent before training. This project does not download an external dataset automatically: choose a dataset whose license and intended use permit your use, then map its labels to this taxonomy and preserve its attribution and license terms.

The stratified split requires at least three examples in each of two or more categories. For useful evaluation, prepare substantially more varied and representative examples per category; tiny datasets yield unstable estimates. Review split sizes and class balance for your own dataset.

## Privacy

Training and CLI inference run locally. Ticket text is not printed or logged by the application. Model artifacts contain vocabulary and learned classifier weights, not training rows by design; nevertheless, treat trained artifacts as internal data and review your organization's privacy policy before sharing them. Do not commit raw tickets or model artifacts unless approved.
