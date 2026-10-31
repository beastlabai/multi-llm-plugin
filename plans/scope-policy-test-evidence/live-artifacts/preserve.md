# Local CSV export

## Goal
Implement a local command that converts a UTF-8 JSON array of expense records to CSV.
The command takes input and output file paths. Each record has date, category and amount.
CSV columns are date,category,amount in that order, with one header row.
Keep the current CLI arguments. Existing callers must keep working.

## Implementation
Read the JSON, iterate records, and build each row from its three fields using a CSV writer that follows RFC 4180 (e.g. Python's csv module). Fields containing commas, double quotes or line breaks must be quoted, with embedded double quotes escaped by doubling them. Do not join fields with raw commas.
Open the destination for writing and write the header row followed by the data rows.

## Checks
Use a happy-path record and confirm the file has a header and one data row.
