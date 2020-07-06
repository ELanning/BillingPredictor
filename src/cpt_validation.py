r"""
Module for CPT code validation tasks.
"""
from torch import softmax, squeeze
from .nlp_models.cpt_code_labels import Labels
from .nlp_models.cpt_sequence_classification import (
    cpt_sequence_classification,
    tokenizer,
    max_character_length,
)


# Based on the CPT 2020. New editions are released each October.
# TODO: Setup alert job to periodically remind us to update codes, or let a 3rd party library handle it.
def get_class_index(cpt_code: int) -> int:
    if 99201 <= cpt_code <= 99499:
        return Labels.evaluationAndManagement.value
    elif (100 <= cpt_code <= 1999) or (99100 <= cpt_code <= 99140):
        return Labels.anesthesia.value
    elif 10021 <= cpt_code <= 69990:
        return Labels.surgery.value
    elif 70010 <= cpt_code <= 79999:
        return Labels.radiology.value
    elif 80047 <= cpt_code <= 89398:
        return Labels.pathologyAndLaboratory.value
    elif (90281 <= cpt_code <= 99607) or (99500 <= cpt_code <= 99607):
        return Labels.medicine.value
    raise ValueError(f"invalid cpt_code: {cpt_code}")


# TODO: Support CPT modifiers and category II CPT codes.
def check_cpt_code_match(cpt_code: str, appointment_notes: str) -> float:
    """Calculates the confidence that the CPT code matches the appointment text.
    Can be used to flag potential CPT code mismatches.

    :param cpt_code: the CPT code to check against.
        Eg "98960", "98962", "00100" etc
    :param appointment_notes: a summary of what occurred at the appointment.
        Eg "Discussed use of ACE inhibitors for renal protection in diabetic patients"
        Max string length given in cpt_sequence_classification module.
    :returns: a confidence interval in the format of 0.0 to 1.0
    :raises:
        ValueError: if cpt_code is invalid.
    """
    # Fail-fast instead of silently truncating, which can hide bugs.
    # Ref: https://wiki.c2.com/?FailFast
    if len(appointment_notes) > max_character_length:
        raise ValueError(
            f"appointment_notes must not exceed {max_character_length}."
            f"cpt_code: {cpt_code}"
            f"appoint_notes: {appointment_notes}"
        )

    cpt_code_index = get_class_index(int(cpt_code))

    # Encode the text and extract the probability that the CPT code matches.
    encoded_notes = tokenizer.encode(appointment_notes, return_tensors="pt", padding=True, truncation=True)
    cpt_classification_logits = cpt_sequence_classification(encoded_notes)[0]
    results = softmax(cpt_classification_logits, dim=1).tolist()[0]

    return results[cpt_code_index]
