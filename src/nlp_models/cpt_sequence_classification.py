from pathlib import Path
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer,
)
from src.nlp_models.cpt_code_labels import cpt_code_labels

cache_dir = (str(Path(__file__).parent) + '/cache').replace("\\", "/")
saved_model_dir = (str(Path(__file__).parent) + "/results").replace("\\", "/")

max_character_length = 512

# DistilBert was chosen based on its speed and lightweight footprint.
# Future work could explore different models.
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased",
    cache_dir=cache_dir,
    max_character_length=max_character_length,
)

# Change "distilbert-base-uncased" to save_model_dir after running the training script once.
# Unfortunately cannot upload the pre-trained model to git due to the file size.
# Will find a better workaround later.
cpt_sequence_classification = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(cpt_code_labels), cache_dir=cache_dir
)
