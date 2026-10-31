def export_expenses(records):
    """Return a CSV document containing the supplied expense records."""
    lines = ["date,category,amount"]
    for record in records:
        lines.append(",".join(str(record[key]) for key in ("date", "category", "amount")))
    return "\n".join(lines) + "\n"
