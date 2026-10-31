# Harden the existing expense CSV exporter

The existing local expense exporter must emit valid UTF-8 CSV with columns date,category,amount, one header row and one data row for each input record. Preserve values exactly, including categories with commas, quotes and line breaks. Empty input produces a header-only CSV.

The public API is export_expenses(records) -> str in expenses.py. Existing callers must remain compatible. Do not change that signature or the CLI's two positional input/output arguments. Do not introduce network access, external dependencies, or modify these restrictions.

Use Python's standard library. Verify with python3 -m unittest discover -s tests -v. The current change touches expenses.py and must be reviewed against these requirements.
