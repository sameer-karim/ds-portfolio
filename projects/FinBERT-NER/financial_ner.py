import json
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

# 1. Load and Split the Dataset

orig_dataset = load_dataset("Josephgflowers/Financial-NER-NLP")
# Create an 80/20 train-test split (for evaluation on unseen data)
dataset = orig_dataset["train"].train_test_split(test_size=0.2, seed=42)

# 2. Convert Each Example into Token-Level Format

def convert_to_token_level(example):
    text = example["user"]
    tokens = text.split()  # Simple whitespace splitting
    ner_tags = ["O"] * len(tokens)

    assistant_str = example["assistant"]
    try:
        gold_data = json.loads(assistant_str)
    except json.JSONDecodeError:
        try:
            gold_data = json.loads(assistant_str.replace("'", "\""))
        except Exception:
            gold_data = {}

    if not gold_data or (isinstance(gold_data, str) and "No XBRL" in gold_data):
        example["tokens"] = tokens
        example["ner_tags"] = ner_tags
        return example

    for entity_type, values in gold_data.items():
        for val in values:
            pattern = r'\b' + re.escape(val) + r'\b'
            for i, token in enumerate(tokens):
                cleaned_token = re.sub(r'[,\.:;]', '', token)
                if re.fullmatch(pattern, cleaned_token):
                    ner_tags[i] = "B-" + entity_type
    example["tokens"] = tokens
    example["ner_tags"] = ner_tags
    return example

dataset = dataset.map(convert_to_token_level)

# 3. Build a Label Mapping from the Training Split

unique_labels = set()
for ex in dataset["train"]:
    unique_labels.update(ex["ner_tags"])
unique_labels = sorted(list(unique_labels), key=lambda x: (x != "O", x))
label_list = unique_labels
label_to_id = {label: idx for idx, label in enumerate(label_list)}
num_labels = len(label_list)
print("Label list:", label_list)

# 4. Convert String Labels to Integer IDs
def convert_labels_to_ids(example):
    example["ner_tags"] = [label_to_id[tag] for tag in example["ner_tags"]]
    return example

dataset = dataset.map(convert_labels_to_ids)

# 5. Tokenize and Align Labels Using FinBERT's Tokenizer

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(labels[word_idx])
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# 6. Downsample the Dataset for Faster Training

small_train_dataset = tokenized_dataset["train"].select(range(50000))
small_test_dataset = tokenized_dataset["test"].select(range(10000))

# 7. Load the Model with the Updated Number of Labels

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# 8. Define Evaluation Metrics Using seqeval

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[pred] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lab] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# 9. Set Up Training Arguments and the Trainer

training_args = TrainingArguments(
    output_dir="./finbert_ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # increased batch size if possible
    per_device_eval_batch_size=32,
    num_train_epochs=5,              # fewer epochs for a quick experiment
    weight_decay=0.01,
    fp16=True,                       # use mixed precision training
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# 10. Train the Model

trainer.train()

eval_metrics = trainer.evaluate()
print(eval_metrics)

######################################################

import unicodedata
import nltk
from nltk.tokenize import word_tokenize

from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

nltk.download('punkt_tab')


# Preprocess the Text Data
def preprocess_text(text):
    # Normalize Unicode to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Replace multiple whitespace with a single space and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    # Use NLTK's word_tokenize for a more robust tokenization
    return word_tokenize(text)

def preprocess_example(example):
    # Preprocess the "user" field text and tokenize it
    preprocessed_text = preprocess_text(example["user"])
    tokens = tokenize_text(preprocessed_text)
    # Update the example with the preprocessed text and tokens
    example["user"] = preprocessed_text
    example["tokens"] = tokens
    return example

# Apply preprocessing to both train and test splits
dataset = dataset.map(preprocess_example)

# Convert Each Example to Token-Level Format for NER
def convert_to_token_level(example):
    tokens = example["tokens"]
    # Initialize all tokens with the "O" label (no entity)
    ner_tags = ["O"] * len(tokens)

    assistant_str = example["assistant"]
    try:
        gold_data = json.loads(assistant_str)
    except json.JSONDecodeError:
        try:
            gold_data = json.loads(assistant_str.replace("'", "\""))
        except Exception:
            gold_data = {}

    # If there's no valid annotation, keep all labels as "O"
    if not gold_data or (isinstance(gold_data, str) and "No XBRL" in gold_data):
        example["ner_tags"] = ner_tags
        return example

    # Mark tokens for each extracted entity
    for entity_type, values in gold_data.items():
        for val in values:
            # Build a regex pattern to match the whole value
            pattern = r'\b' + re.escape(val) + r'\b'
            for i, token in enumerate(tokens):
                # Clean token of punctuation for matching
                cleaned_token = re.sub(r'[,\.:;]', '', token)
                if re.fullmatch(pattern, cleaned_token):
                    ner_tags[i] = "B-" + entity_type
    example["ner_tags"] = ner_tags
    return example

dataset = dataset.map(convert_to_token_level)

# Build a Label Mapping from the Training Data
unique_labels = set()
for ex in dataset["train"]:
    unique_labels.update(ex["ner_tags"])
# Ensure "O" is first, then sort remaining labels
unique_labels = sorted(list(unique_labels), key=lambda x: (x != "O", x))
label_list = unique_labels
label_to_id = {label: idx for idx, label in enumerate(label_list)}
num_labels = len(label_list)
print("Label list:", label_list)

# Convert String Labels to Integer IDs
def convert_labels_to_ids(example):
    example["ner_tags"] = [label_to_id[tag] for tag in example["ner_tags"]]
    return example

dataset = dataset.map(convert_labels_to_ids)

# Tokenize and Align Labels Using FinBERT's Tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    # Tokenize pre-split tokens; padding and truncation enabled.
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens in loss
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(labels[word_idx])
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Adjust the number of examples to reduce training time
small_train_dataset = tokenized_dataset["train"].select(range(50000))
small_test_dataset = tokenized_dataset["test"].select(range(10000))

# Load the Model (with a Reinitialized Classification Head)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# Define Evaluation Metrics Using seqeval
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[pred] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lab] for pred, lab in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# Set Up Training Arguments and the Trainer
training_args = TrainingArguments(
    output_dir="./finbert_ner",  # local output; we'll later copy to Drive
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,  # For quick experimentation; adjust as needed
    weight_decay=0.01,
    fp16=True,         # Enable mixed precision for faster training
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the Model
trainer.train()

######################################################

improved_training_args = TrainingArguments(
    output_dir="./finbert_ner_improved",
    evaluation_strategy="steps",  # Evaluate every fixed number of steps
    eval_steps=500,               # Evaluate every 500 steps
    logging_steps=100,            # Log every 100 steps
    learning_rate=1e-5,           # Lower learning rate for finer updates
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,          # Increase the number of epochs for more training
    weight_decay=0.01,
    fp16=True,                    # Use mixed precision for faster training
    warmup_steps=500,             # Warmup steps to stabilize training early on
)

trainer = Trainer(
    model=model,
    args=improved_training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


# ===================================
# Finer hyperparameter tuning
# ===================================


improved_training_args = TrainingArguments(
    output_dir="./finbert_ner_improved",
    evaluation_strategy="steps",  # Evaluate every fixed number of steps
    eval_steps=500,               # Evaluate every 500 steps
    logging_steps=100,            # Log every 100 steps
    learning_rate=1e-5,           # Lower learning rate for finer updates
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,          # Increase the number of epochs for more training
    weight_decay=0.01,
    fp16=True,                    # Use mixed precision for faster training
    warmup_steps=500,             # Warmup steps to stabilize training early on
)

trainer = Trainer(
    model=model,
    args=improved_training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

