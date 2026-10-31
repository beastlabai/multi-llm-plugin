import csv
import io
import inspect
import unittest
from expenses import export_expenses


class ExpenseTests(unittest.TestCase):
    def test_special_characters_round_trip(self):
        for category in ["Food, groceries", '"Quoted"', "line1\nline2"]:
            rows = list(csv.reader(io.StringIO(export_expenses([{"date": "2026-10-31", "category": category, "amount": "-1.25"}]))))
            self.assertEqual(rows, [["date", "category", "amount"], ["2026-10-31", category, "-1.25"]])

    def test_empty_header(self):
        self.assertEqual(list(csv.reader(io.StringIO(export_expenses([])))), [["date", "category", "amount"]])

    def test_api_unchanged(self):
        self.assertEqual(list(inspect.signature(export_expenses).parameters), ["records"])
