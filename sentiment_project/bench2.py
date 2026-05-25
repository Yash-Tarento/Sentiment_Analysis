import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import emoji
import time

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm

# =========================================================
# 1. LOAD DATASET
# =========================================================
print("🚀 Loading evaluation dataset...")

df = pd.read_csv(
    "data.csv"
)

df = df.rename(columns={
    "comment": "text",
    "sentiment_class": "label"
})

df = df[["text", "label"]].dropna()

print("✅ Dataset loaded:", df.shape)

# =========================================================
# 2. PREPROCESSING
# =========================================================
COURSE_CONTEXT = (
    "This comment is about an online government learning "
    "course from the iGOT Karmayogi platform. Comment: "
)

df["text"] = df["text"].astype(str)

# Emoji conversion
df["text"] = df["text"].apply(
    lambda x: emoji.demojize(x)
)

# Add domain context
df["text"] = COURSE_CONTEXT + df["text"]

texts = df["text"].tolist()
labels = df["label"].tolist()

# =========================================================
# 3. MODELS TO BENCHMARK
# =========================================================
models = {

    # Your trained model
    "Custom_XLMR":
    "model",

    # Twitter multilingual sentiment model
    "Twitter_XLMR":
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",

    # Multilingual BERT
    "mBERT":
    "bert-base-multilingual-cased",

    # IndicBERT
    "IndicBERT":
    "ai4bharat/indic-bert",

    # Multilingual MiniLM
    "MiniLM_Multilingual":
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

# =========================================================
# 4. LABEL MAPPING
# =========================================================
# Your labels:
# 0 = positive
# 1 = neutral
# 2 = negative

twitter_label_map = {
    0: 2,   # negative
    1: 1,   # neutral
    2: 0    # positive
}

# =========================================================
# 5. BENCHMARK LOOP
# =========================================================
benchmark_results = []

for model_name, model_path in models.items():

    print("\n================================================")
    print(f"🚀 Benchmarking: {model_name}")
    print("================================================")

    try:

        start_time = time.time()

        # -------------------------------------------------
        # LOAD TOKENIZER + MODEL
        # -------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3,
            ignore_mismatched_sizes=True
        )

        model.eval()

        predictions = []
        confidence_scores = []

        batch_size = 32

        # -------------------------------------------------
        # INFERENCE LOOP
        # -------------------------------------------------
        for i in tqdm(
            range(0, len(texts), batch_size)
        ):

            batch = texts[i:i+batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            with torch.no_grad():

                outputs = model(**inputs)

                probs = F.softmax(
                    outputs.logits,
                    dim=1
                )

                preds = torch.argmax(
                    probs,
                    dim=1
                )

            preds = preds.numpy()

            # ---------------------------------------------
            # Twitter model label correction
            # ---------------------------------------------
            if model_name == "Twitter_XLMR":

                preds = [
                    twitter_label_map[p]
                    for p in preds
                ]

            predictions.extend(preds)

            confidence_scores.extend(
                torch.max(
                    probs,
                    dim=1
                ).values.numpy()
            )

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------
        accuracy = accuracy_score(
            labels,
            predictions
        )

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro"
        )

        weighted_f1 = f1_score(
            labels,
            predictions,
            average="weighted"
        )

        end_time = time.time()

        inference_time = round(
            end_time - start_time,
            2
        )

        # -------------------------------------------------
        # PRINT RESULTS
        # -------------------------------------------------
        print("\n✅ RESULTS")

        print(f"Accuracy      : {accuracy:.4f}")
        print(f"Macro F1      : {macro_f1:.4f}")
        print(f"Weighted F1   : {weighted_f1:.4f}")
        print(f"Inference Time: {inference_time}s")

        print("\n✅ Classification Report:\n")

        print(classification_report(
            labels,
            predictions,
            target_names=[
                "positive",
                "neutral",
                "negative"
            ]
        ))

        # -------------------------------------------------
        # SAVE MODEL RESULTS
        # -------------------------------------------------
        benchmark_results.append({

            "Model": model_name,

            "Accuracy":
            round(accuracy, 4),

            "Macro_F1":
            round(macro_f1, 4),

            "Weighted_F1":
            round(weighted_f1, 4),

            "Inference_Time_Seconds":
            inference_time
        })

        # -------------------------------------------------
        # SAVE WRONG PREDICTIONS
        # -------------------------------------------------
        temp_df = df.copy()

        temp_df["predicted"] = predictions

        temp_df["confidence"] = confidence_scores

        temp_df["correct"] = (
            temp_df["label"] ==
            temp_df["predicted"]
        )

        wrong_df = temp_df[
            temp_df["correct"] == False
        ]

        wrong_df.to_csv(
            f"wrong_predictions_{model_name}.csv",
            index=False
        )

        print(
            f"\n✅ Wrong predictions saved for {model_name}"
        )

    except Exception as e:

        print(
            f"\n❌ Error benchmarking {model_name}"
        )

        print(e)

# =========================================================
# 6. FINAL BENCHMARK TABLE
# =========================================================
results_df = pd.DataFrame(
    benchmark_results
)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)

# Save benchmark results
results_df.to_csv(
    "final_benchmark_results.csv",
    index=False
)

print("\n================================================")
print("🏆 FINAL BENCHMARK RESULTS")
print("================================================\n")

print(results_df)

print(
    "\n✅ Benchmark results saved to:"
)

print("final_benchmark_results.csv")

print(
    "\n✅ Benchmarking completed successfully!"
)
