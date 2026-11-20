import unittest
from backend.research import TOPICS, search_term


class PubMedQueryTests(unittest.TestCase):
    def test_brain_imaging_context_and_exclusions(self):
        for label in TOPICS:
            with self.subTest(label=label):
                term = search_term(label, "contrast")
                self.assertIn("intracranial OR brain", term)
                self.assertIn("MRI", term)
                self.assertIn("AND (contrast)", term)
                self.assertIn("NOT (dog OR canine OR feline OR veterinary OR spinal)", term)
