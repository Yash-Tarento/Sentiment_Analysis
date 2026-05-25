# Sentiment Intelligence System

# Overview

✅ Training pipeline 
✅ Benchmarking framework 
✅ Optimized inference pipeline (Kaggle) 
✅ Incremental caching system 
✅ Recommendation engine

***

# Project Journey

## Phase 1 — Initial Attempts

The first version of the code was a simple prediction script:

```text
Load data → Tokenize → Predict → Save results
```

### ✅ Features in Early Version

* Batch predictions
* Confidence scoring
* Manual review system

#### Confidence Logic

| Confidence | Action             |
| ---------- | ------------------ |
| < 0.60     | Manual review      |
| 0.60–0.75  | Review recommended |
| > 0.75     | Auto-approved      |

But this version had limitations:

* Recomputed data every time ❌
* No scalability ❌
* No structured insights ❌

***

# 📊 Phase 2 — Benchmarking Models

Before committing to a model, multiple architectures were tested.

## 🧪 Models Compared

* ✅ Custom XLM-R (final choice)
* Twitter XLM-R
* Multilingual BERT (mBERT)
* IndicBERT
* Multilingual MiniLM

***

## 📊 Benchmark Results

| Model               | Accuracy | Macro F1 | Weighted F1 | Inference Time |
| ------------------- | -------- | -------- | ----------- | -------------- |
| 🏆 Custom XLM-R     | 98.63%   | 96.83%   | 98.67%      | 131.69 sec     |
| Twitter XLM-R       | 46.90%   | 41.32%   | 56.88%      | 77.77 sec      |
| MiniLM Multilingual | 80.29%   | 33.42%   | 74.32%      | 50.77 sec      |
| mBERT               | 45.90%   | 27.78%   | 52.88%      | 109.54 sec     |

***

## ✅ Key Takeaway

The fine-tuned **Custom XLM-R** model achieved the best overall performance across all evaluation metrics, especially in:

* Sentiment classification accuracy
* Multilingual understanding
* Handling noisy course feedback
* Balanced class prediction performance

While some baseline models had faster inference times, they struggled significantly with real-world domain-specific comments.

***

## 📈 Evaluation Metrics

* Accuracy
* Macro F1 (important for imbalance)
* Weighted F1
* Inference time
* Classification reports

***

## 🏆 Outcome

The **fine-tuned XLM-R model** was selected because:

✅ Best Macro F1  
✅ Stable on noisy data  
✅ Strong multilingual performance

***

# 🏋️ Training Pipeline

## 🔄 Flow

```text
Load Data
↓
Add Domain Context
↓
Emoji Conversion
↓
Text Cleaning
↓
Data Augmentation
↓
Class Balancing
↓
Train / Validation Split
↓
Tokenization
↓
Weighted Training
↓
Save Model
```

***

## 🧠 Key Techniques

### ✅ Domain Context Injection

```text
"This comment is about an online government learning course..."
```

👉 Gives the model context about what the text means.

***

### ✅ Emoji Normalization

```text
👍 → :thumbs_up:
```

👉 Converts emojis into meaningful tokens.

***

### ✅ Data Augmentation

```text
good → very good
bad → very bad
```

👉 Improves robustness.

***

### ✅ Weighted Loss

```python
CrossEntropyLoss(weight=class_weights)
```

👉 Fixes class imbalance.

***

# ⚙️ Final Inference Pipeline (Kaggle Implementation)

This is the main production pipeline.

***

## 🔄 Full Flow

```text
Load Dataset
↓
Normalize columns (comment → text)
↓
Clean text + emoji conversion
↓
Add timestamp (if missing)
↓
Generate hash (row_id)
↓
Load cache
↓
Split data:
    old → reuse
    new → infer
↓
Run model (only new rows)
↓
Update cache
↓
Merge results
↓
Compute metrics
↓
Generate recommendation
↓
Save output CSV
```

***

## ⚡ Key Features in the Final Code

### ✅ 1. Multi-Format Input

```python
def load_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
```

👉 Flexible ingestion pipeline.

***

### ✅ 2. Emoji-Aware Processing

```python
df["text"] = df["text"].apply(emoji.demojize)
```

👉 Improves sentiment detection.

***

### ✅ 3. Automatic Timestamp Generation

```python
df["timestamp"] = pd.date_range(...)
```

👉 Enables trend analysis.

***

### ✅ 4. Incremental Caching

```python
df["row_id"] = df["text"].apply(hash)
```

👉 Prevents recomputation.

***

## Cache Logic

```text
If row exists → reuse result ✅
If new → run model ✅
```

***

## ✅ Result

| Scenario  | Behavior          |
| --------- | ----------------- |
| First run | Full inference    |
| Same file | Instant ⚡         |
| New data  | Partial inference |

***

### ✅ 5. Batch + GPU Inference

```python
batch_size = 256 (GPU)
batch_size = 128 (CPU)
```

👉 Efficient large-scale processing.

***

### ✅ 6. Confidence Scoring

```text
confidence = top1_prob - top2_prob
```

👉 Measures prediction certainty.

***

# 🎯 Recommendation Engine

## Formula

```text
Recommendation Score =
(Positive% × 1) + (100 - Negative%) × 0.5
-----------------------------------------
               1.5
```

***

## Output

| Score | Recommendation       |
| ----- | -------------------- |
| > 70  | ✅ Highly Recommended |
| 40–70 | ✅ Recommended        |
| 10–40 | ⚠️ Average           |
| < 10  | ❌ Needs Improvement  |

***

# 📊 Sample Output

```text
Total comments: 100000
Positive: 72.4%
Neutral: 12.1%
Negative: 15.5%
Average Confidence: 0.81

Recommendation Score: 74.22
✅ Highly Recommended
```

***

# 📂 Input Format

Supports:

```text
CSV | JSON | Parquet
```

***

# 📤 Output

Saved to:

```text
/kaggle/working/final_output.csv
```

***

# ⚡ Performance

✅ GPU accelerated  
✅ Handles 100k+ rows  
✅ Incremental processing  
✅ Fast repeated runs

