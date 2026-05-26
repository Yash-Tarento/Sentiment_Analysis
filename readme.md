
````markdown
# 🚀 Sentiment Intelligence System

> Transforming raw course feedback into actionable insights using NLP, deep learning, and scalable engineering.

---

## ✨ Overview

- 🧠 Training pipeline  
- 📊 Benchmarking framework  
- ⚡ Optimized inference pipeline (Kaggle)  
- 🔁 Incremental caching system  
- 🎯 Recommendation engine  

---

## 🧭 Project Journey

### 🪜 Phase 1 — Initial Attempts

The early system was simple:

```text
Load data → Tokenize → Predict → Save results
````

#### ✅ Features

* Batch predictions
* Confidence scoring
* Manual review system

#### 🎯 Confidence Logic

| Confidence  | Action                |
| ----------- | --------------------- |
| < 0.60      | 🛑 Manual review      |
| 0.60 – 0.75 | ⚠️ Review recommended |
| > 0.75      | ✅ Auto-approved       |

#### ❌ Limitations

* Recomputed data every time
* No scalability
* No structured insights

***

### 📊 Phase 2 — Benchmarking Models

We evaluated multiple multilingual transformer models.

#### 🧪 Models Compared

* 🏆 Custom XLM-R (Final Model)
* Twitter XLM-R
* mBERT
* IndicBERT
* MiniLM Multilingual

***

### 📈 Benchmark Results

| Model               |  Accuracy  |  Macro F1  | Weighted F1 | Inference Time |
| :------------------ | :--------: | :--------: | :---------: | :------------: |
| 🏆 **Custom XLM-R** | **98.63%** | **96.83%** |  **98.67%** |   131.69 sec   |
| Twitter XLM-R       |   46.90%   |   41.32%   |    56.88%   |    77.77 sec   |
| MiniLM Multilingual |   80.29%   |   33.42%   |    74.32%   |    50.77 sec   |
| mBERT               |   45.90%   |   27.78%   |    52.88%   |   109.54 sec   |

***

### ✅ Key Takeaway

The **fine-tuned XLM-R model** stood out due to:

* ✅ High accuracy and Macro F1
* ✅ Strong multilingual understanding
* ✅ Robust performance on noisy real-world data
* ✅ Balanced class predictions

***

## 🏋️ Training Pipeline

### 🔄 Flow

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

### 🧠 Key Techniques

#### 🏷️ Domain Context Injection

```
"This comment is about an online government learning course..."
```

👉 Helps the model understand context better

***

#### 😊 Emoji Normalization

```
👍 → :thumbs_up:
```

👉 Retains sentiment signals

***

#### 🔁 Data Augmentation

```
good → very good
bad → very bad
```

👉 Improves generalization

***

#### ⚖️ Weighted Loss

```python
CrossEntropyLoss(weight=class_weights)
```

👉 Handles class imbalance

***

## ⚙️ Final Inference Pipeline

### 🔄 System Flow

```text
Load Dataset
   ↓
Normalize Columns (comment → text)
   ↓
Clean Text + Emoji Conversion
   ↓
Add Timestamp (if missing)
   ↓
Generate Hash (row_id)
   ↓
Load Cache
   ↓
Split:
   Old → reuse
   New → infer
   ↓
Run Model (only new rows)
   ↓
Update Cache
   ↓
Merge Results
   ↓
Compute Metrics
   ↓
Generate Recommendation
   ↓
Save Output
```

***

## ⚡ Key Features

### 📦 Multi-Format Input

```python
def load_data(path):
```

✅ Supports CSV / JSON / Parquet

***

### 😊 Emoji-Aware Processing

```python
emoji.demojize()
```

***

### ⏱️ Timestamp Generation

```python
pd.date_range(...)
```

***

### 🔁 Incremental Caching

```python
df["row_id"] = df["text"].apply(hash)
```

#### Logic

```text
Old rows → reuse ✅
New rows → compute ✅
```

```


