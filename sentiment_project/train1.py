import pandas as pd
import numpy as np
import torch
import emoji

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv(
    "data.csv"
)

# Rename columns
df = df.rename(columns={
    "comment": "text",
    "sentiment_class": "label"
})

# Keep required columns only
df = df[["text", "label"]].dropna()

print("✅ Original Dataset Shape:", df.shape)

# =========================================================
# 2. DOMAIN CONTEXT
# =========================================================
COURSE_CONTEXT = (
    "This comment is about an online government learning "
    "course from the iGOT Karmayogi platform. Comment: "
)

df["text"] = COURSE_CONTEXT + df["text"].astype(str)

# =========================================================
# 3. EMOJI → TEXT CONVERSION
# =========================================================
df["text"] = df["text"].apply(
    lambda x: emoji.demojize(str(x))
)

# =========================================================
# 4. BASIC CLEANING
# =========================================================
df["text"] = df["text"].str.strip()

# Remove empty rows
df = df[df["text"] != ""]

# =========================================================
# 5. SMART DATA AUGMENTATION
# =========================================================
def augment_text(text):
    augmented = []

    replacements = {
        "good": "very good",
        "bad": "very bad",
        "nice": "really nice",
        "excellent": "really excellent",
        "acha": "bahut acha",
        "helpful": "very helpful",
        "useful": "extremely useful"
    }

    for old, new in replacements.items():
        if old in text.lower():
            augmented.append(text.replace(old, new))

    return augmented

augmented_rows = []

for _, row in df.iterrows():

    aug_texts = augment_text(row["text"])

    for aug in aug_texts:
        augmented_rows.append({
            "text": aug,
            "label": row["label"]
        })

df_aug = pd.DataFrame(augmented_rows)

# Combine original + augmented
df = pd.concat([df, df_aug], ignore_index=True)

print("✅ After Augmentation:", df.shape)

# =========================================================
# 6. HANDLE CLASS IMBALANCE (OVERSAMPLING)
# =========================================================
df_pos = df[df.label == 0]
df_neu = df[df.label == 1]
df_neg = df[df.label == 2]

max_size = max(len(df_pos), len(df_neu), len(df_neg))

df_pos = resample(
    df_pos,
    replace=True,
    n_samples=max_size,
    random_state=42
)

df_neu = resample(
    df_neu,
    replace=True,
    n_samples=max_size,
    random_state=42
)

df_neg = resample(
    df_neg,
    replace=True,
    n_samples=max_size,
    random_state=42
)

df = pd.concat([df_pos, df_neu, df_neg])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Balanced Dataset Shape:", df.shape)

print("\n✅ Label Distribution:")
print(df["label"].value_counts())

# =========================================================
# 7. TRAIN / VALIDATION SPLIT
# =========================================================
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# =========================================================
# 8. TOKENIZER
# =========================================================
model_name = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

val_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

# =========================================================
# 9. CLASS WEIGHTS
# =========================================================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label"]),
    y=df["label"]
)

class_weights = torch.tensor(weights, dtype=torch.float)

print("\n✅ Class Weights:", class_weights)

# =========================================================
# 10. MODEL
# =========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# =========================================================
# 11. CUSTOM TRAINER
# =========================================================
class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        labels = inputs.get("labels")

        outputs = model(**inputs)

        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(model.device)
        )

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# =========================================================
# 12. TRAINING SETTINGS
# =========================================================
training_args = TrainingArguments(
    output_dir="model",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,

    save_total_limit=2,

    fp16=torch.cuda.is_available()
)

# =========================================================
# 13. TRAIN
# =========================================================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("\n🚀 Starting Training...\n")

trainer.train()

# =========================================================
# 14. SAVE MODEL
# =========================================================
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("\n✅ Training Completed Successfully!")
print("✅ Model saved in: model/")
