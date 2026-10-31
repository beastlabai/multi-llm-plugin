import csv
import io


def export_expenses(records):
    """Return a CSV document containing the supplied expense records."""
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["date", "category", "amount"])
    for record in records:
        writer.writerow([record["date"], record["category"], record["amount"]])
    return out.getvalue()
