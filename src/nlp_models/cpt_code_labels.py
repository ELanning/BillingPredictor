from enum import Enum

cpt_code_labels = (
    "evaluation and management",
    "anesthesia",
    "surgery",
    "radiology",
    "pathology and laboratory",
    "medicine",
)


class Labels(Enum):
    evaluationAndManagement = 0
    anesthesia = 1
    surgery = 2
    radiology = 3
    pathologyAndLaboratory = 4
    medicine = 5
