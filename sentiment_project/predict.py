import pandas as pd
import torch
import torch.nn.functional as F
import emoji

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from tqdm import tqdm

print("🚀 Starting prediction pipeline...")

# =========================================================
# 1. LOAD MODEL
# =========================================================
model_path = "model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

model.eval()

print("✅ Model loaded successfully!")

# =========================================================
# 2. LABEL MAPPING
# =========================================================
label_map = {
    0: "positive",
    1: "neutral",
    2: "negative"
}

# =========================================================
# 3. LOAD TEST DATA
# =========================================================
df = pd.read_csv(
    "test.csv"
)

print("✅ Test data loaded:", df.shape)

# =========================================================
# 4. DOMAIN CONTEXT
# =========================================================
COURSE_CONTEXT = (
    "This comment is about an online government learning "
    "course from the iGOT Karmayogi platform. Comment: "
)

# =========================================================
# 5. PREPROCESS COMMENTS
# =========================================================
comments = df["comment"].fillna("").astype(str)

# Emoji → text
comments = comments.apply(
    lambda x: emoji.demojize(x)
)

# Add domain context
comments = comments.apply(
    lambda x: COURSE_CONTEXT + x
)

comments = comments.tolist()

# =========================================================
# 6. PREDICTION SETTINGS
# =========================================================
batch_size = 64

predictions = []
prediction_labels = []
confidence_scores = []
review_status = []

manual_review_rows = []

# Confidence thresholds
THRESHOLD = 0.60
REVIEW_MARGIN = 0.75

# =========================================================
# 7. PREDICTION LOOP
# =========================================================
print("\n🚀 Running predictions...\n")

for i in tqdm(range(0, len(comments), batch_size)):

    batch = comments[i:i + batch_size]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        preds = torch.argmax(probs, dim=1)

    for j in range(len(batch)):

        pred_class = preds[j].item()

        confidence = probs[j][pred_class].item()

        label = label_map[pred_class]

        # =================================================
        # CONFIDENCE / REVIEW LOGIC
        # =================================================
        if confidence < THRESHOLD:

            status = "manual_review"

            manual_review_rows.append({
                "comment": batch[j],
                "predicted_label": label,
                "confidence": round(confidence, 4)
            })

        elif confidence < REVIEW_MARGIN:

            status = "review_recommended"

            manual_review_rows.append({
                "comment": batch[j],
                "predicted_label": label,
                "confidence": round(confidence, 4)
            })

        else:

            status = "auto_approved"

        predictions.append(pred_class)
        prediction_labels.append(label)
        confidence_scores.append(round(confidence, 4))
        review_status.append(status)

# =========================================================
# 8. ADD RESULTS TO DATAFRAME
# =========================================================
df["sentiment_class"] = predictions

df["sentiment_label"] = prediction_labels

df["confidence_score"] = confidence_scores

df["review_status"] = review_status

# =========================================================
# 9. SAVE MAIN OUTPUT
# =========================================================
output_path = "data/output.csv"

df.to_csv(output_path, index=False)

print(f"\n✅ Predictions saved to: {output_path}")

# =========================================================
# 10. SAVE MANUAL REVIEW FILE
# =========================================================
review_df = pd.DataFrame(manual_review_rows)

review_output_path = "data/manual_review.csv"

review_df.to_csv(review_output_path, index=False)

print(f"✅ Manual review file saved to: {review_output_path}")

# =========================================================
# 11. PRINT SUMMARY
# =========================================================
print("\n📊 Prediction Summary\n")

print(df["sentiment_label"].value_counts())

print("\n📌 Review Status Summary\n")

print(df["review_status"].value_counts())

print("\n✅ Prediction pipeline completed successfully!")
