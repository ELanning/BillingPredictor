import unittest
from src import check_cpt_code_match


# These tests made need to change when/if CPT codes are updated.
# TODO: Add a test for all of the different CPT code categories.
# TODO: Need more training data before these tests will pass.
class TestCptValidation(unittest.TestCase):
    def test_cpt_match(self):
        appointment_notes = "Comprehensive preventive medicine exam of an adolescent."
        valid_associated_cpt_code = "99384"

        match_confidence = check_cpt_code_match(
            valid_associated_cpt_code, appointment_notes
        )

        self.assertTrue(
            match_confidence >= 0.9,
            f"appointment notes must match valid cpt code with 90% confidence. Confidence received: {match_confidence}",
        )

    def test_cpt_mismatch(self):
        appointment_notes = "Comprehensive preventive medicine exam of an adolescent."
        invalid_associated_cpt_code = "69990"

        match_confidence = check_cpt_code_match(
            invalid_associated_cpt_code, appointment_notes
        )

        self.assertTrue(
            match_confidence <= 0.4,
            f"appointment notes must not match invalid cpt code with more than 40% confidence. Confidence received: {match_confidence}",
        )

    def test_text_too_long(self):
        too_long_appointment_notes = "m" * 10_000
        cpt_code = "99384"

        self.assertRaises(
            ValueError, check_cpt_code_match, cpt_code, too_long_appointment_notes,
        )

    def test_invalid_cpt_code(self):
        appointment_notes = "Checked up on the mole."
        nonexistent_cpt_code = "999999999999"

        self.assertRaises(
            ValueError, check_cpt_code_match, nonexistent_cpt_code, appointment_notes,
        )
