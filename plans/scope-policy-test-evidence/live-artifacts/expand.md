# Local CSV export

## Goal
Implement a local command that converts a UTF-8 JSON array of expense records to CSV.
The command takes input and output file paths. Each record has date, category and amount.
CSV columns are date,category,amount in that order, with one header row.
Keep the current CLI arguments. Existing callers must keep working.

## Implementation
Read the JSON, iterate records, and join each row's three fields with commas.
Open the destination for writing and write the resulting lines.

## Checks
Use a happy-path record and confirm the file has a header and one data row.
