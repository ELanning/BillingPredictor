"""Small python app for training the CptSequenceClassification model.
Should be used as a standalone app, and not as part of the library.
TODO: Move it to a "tools" folder outside of the project.
"""
from transformers import (
    Trainer,
    TrainingArguments,
)
from src.nlp_models.cpt_sequence_dataset import CptSequenceDataset
from src.nlp_models.cpt_code_labels import Labels
from src.nlp_models.cpt_sequence_classification import (
    cpt_sequence_classification,
    tokenizer,
)


def noop():
    return


cpt_sequence_classification.train()  # Doesn't do actual training. Sets training mode to on.

# Only train the top layer. This is for speed reasons.
# Future work could explore entire model training, as well as unsupervised pre-training on medical text.
for param in cpt_sequence_classification.base_model.parameters():
    param.requires_grad = False

# HACK: Prevent trainer from changing train state. This is because we only want to train the top layer.
cpt_sequence_classification.train = noop

train_dataset = CptSequenceDataset(
    tokenizer,
    (Labels.evaluationAndManagement, "Checked for cancer"),
    (Labels.evaluationAndManagement, "Ran biopsy"),
)

eval_dataset = CptSequenceDataset(
    tokenizer,
    (Labels.evaluationAndManagement, "Diabetes medications and side effects reviewed",),
    (
        Labels.evaluationAndManagement,
        "Discussed use of ACE inhibitors for renal protection in diabetic patients",
    ),
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # TODO: Change to realistic size, eg 32-64, once we have more data.
    per_device_eval_batch_size=1,  # TODO: Change to realistic size, eg 32-64, once we have more data.
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=cpt_sequence_classification,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
trainer.save_model()
