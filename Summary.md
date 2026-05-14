# AML Detection Pipeline — Summary

## 🔴 Phase 1 — Understanding the Problem

### Fraud Detection

- Fraud = unauthorized financial activity such as stolen cards or hacked accounts.
    
- Fraud Detection = identifying suspicious financial transactions using machine learning models.
    

### Money Laundering

Money laundering is the process of converting illegal money into seemingly legal money.

### Stages of Money Laundering

1. **Placement** → inserting illegal money into the financial system
    
2. **Layering** → moving money across many accounts to hide its source
    
3. **Integration** → reintroducing the money as “clean” money into the economy
    

### Why AML Systems Are Important

- Banks are legally required to monitor suspicious transactions.
    
- Failure to detect laundering can lead to huge fines and legal penalties.
    
- Suspicious activities are reported using SARs (Suspicious Activity Reports).
    

### Fraud Detection vs AML

|Fraud Detection|AML|
|---|---|
|Detects individual unauthorized transactions|Detects long-term laundering behavior|
|Usually real-time|May require days or weeks|
|Focuses on theft/fraud|Focuses on criminal financial networks|

---

# 🟠 Phase 2 — Dataset & Features

## Dataset: PaySim

- Synthetic mobile-money transaction dataset
    
- ~6.3 million transactions
    
- Target column:
    

```text
isFraud
```

## Important Features

- transaction type
    
- transaction amount
    
- sender balance
    
- receiver balance
    

## Class Imbalance Problem

- Legitimate transactions ≈ 98.7%
    
- Fraudulent transactions ≈ 1.3%
    

Because of this imbalance, **Accuracy is misleading**.

## Feature Engineering

We created additional features:

- `balance_change_orig`
    
- `balance_change_dest`
    
- `amount_balance_error`
    
- `dest_was_empty`
    

These features help detect suspicious financial inconsistencies.

---

# 🟡 Phase 3 — Preprocessing

## Steps

1. Handle missing values
    
2. Encode categorical values using LabelEncoder
    
3. Apply StandardScaler
    

## Why Scaling Is Important

Scaling is necessary for distance-based algorithms such as PCA and K-Means because large-value features can dominate distance calculations.

---

# 🟢 Phase 4 — PCA (Principal Component Analysis)

## Purpose of PCA

Reduce dimensionality while preserving the most important information.

## Benefits

- Removes correlated features
    
- Reduces noise
    
- Improves clustering quality
    

## Component Selection

We selected the number of principal components that preserve at least:

[  
85%  
]

of the total explained variance.

---

# 🔵 Phase 5 — K-Means Clustering

## What Is K-Means?

An unsupervised learning algorithm that groups similar transactions into behavioral clusters.

## Workflow

1. Initialize K centroids
    
2. Assign each point to the nearest centroid
    
3. Update centroids
    
4. Repeat until convergence
    

## Choosing K

We used the Elbow Method and selected:

[  
K = 4  
]

## Output Feature

K-Means generated a new feature:

```text
behavioral_segment
```

This represents the behavioral profile of each transaction.

---

# 🟣 Phase 6 — XGBoost Classifier

## What Is XGBoost?

An ensemble boosting algorithm built from many decision trees.

## Why XGBoost?

- Handles class imbalance efficiently
    
- High performance and speed
    
- Prevents overfitting
    
- Provides feature importance scores
    

## Important Parameters

- `max_depth`
    
- `learning_rate`
    
- `subsample`
    
- `scale_pos_weight`
    

## Evaluation Metrics

Instead of Accuracy, we used:

- Precision
    
- Recall
    
- F1-Score
    
- Confusion Matrix
    

## Most Important Metric

In AML systems:

Recall = \frac{TP}{TP + FN}

Recall is critical because missing a true laundering case is more dangerous than generating a false alert.

---

# 🔴 Phase 7 — Continuous Learning

## Problem: Concept Drift

Criminals continuously change laundering strategies over time.

## Solution: Warm-Start Incremental Learning

```python
updated_model = xgb.train(
    params,
    d_new,
    num_boost_round=20,
    xgb_model=model
)
```

## Benefits

- Updates the model without retraining from scratch
    
- Faster and cheaper
    
- Compatible with GDPR data-retention rules
    

---

# ⚫ Phase 8 — Production AML System

## Real-Time Pipeline

```text
Transaction
    ↓
Preprocessing
    ↓
PCA
    ↓
K-Means
    ↓
XGBoost Scoring
    ↓
Alert Generation
    ↓
Human Analyst Review
    ↓
SAR Filing
```

---

# ✅ Final Pipeline Overview

```text
Raw Data (PaySim)
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Scaling
        ↓
PCA
        ↓
K-Means Clustering
        ↓
behavioral_segment Feature
        ↓
XGBoost Classification
        ↓
Evaluation (Recall, F1-Score)
        ↓
Continuous Learning
```

---

# 🎓 Key Concepts

- AML focuses on behavioral patterns, not only individual transactions.
    
- PCA reduces dimensionality and improves clustering.
    
- K-Means provides behavioral profiling.
    
- XGBoost performs the final fraud classification.
    
- Recall is the most important evaluation metric.
    
- Continuous learning helps the model adapt to evolving laundering techniques.
    

---

# 🏆 Two-Line Project Description

“We built an end-to-end AML detection pipeline using PCA and K-Means for behavioral profiling, followed by XGBoost for fraud classification. The system supports continuous learning through warm-start incremental updates to adapt to evolving money laundering behaviors.”