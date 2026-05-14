# 🏦 Anti-Money Laundering (AML) Detection Pipeline

<div dir="rtl">

## 🇪🇬 نظام كشف غسيل الأموال باستخدام التعلم الآلي

مشروع متكامل لكشف عمليات غسيل الأموال باستخدام خوارزميات متقدمة في التعلم الآلي

</div>

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents / المحتويات

- [Overview](#overview--نظرة-عامة)
- [Features](#features--المميزات)
- [Architecture](#architecture--البنية-المعمارية)
- [Installation](#installation--التثبيت)
- [Usage](#usage--الاستخدام)
- [Dataset](#dataset--البيانات)
- [Model Performance](#model-performance--أداء-النموذج)
- [Project Structure](#project-structure--هيكل-المشروع)
- [Algorithms Explained](#algorithms-explained--شرح-الخوارزميات)
- [Continuous Learning](#continuous-learning--التعلم-المستمر)
- [Contributing](#contributing--المساهمة)

---

## 🎯 Overview / نظرة عامة

<div dir="rtl">

### بالعربي

هذا المشروع يقدم نظام متكامل للكشف عن عمليات غسيل الأموال في المعاملات المالية باستخدام:
- **PCA** لتقليل الأبعاد
- **K-Means** للتجميع السلوكي
- **XGBoost** للتصنيف النهائي
- **Continuous Learning** للتكيف مع الأنماط الجديدة

### In English

An end-to-end Anti-Money Laundering detection system using state-of-the-art machine learning techniques:
- **PCA** for dimensionality reduction
- **K-Means** for behavioral clustering
- **XGBoost** for fraud classification
- **Continuous Learning** for model adaptation

</div>

---

## ✨ Features / المميزات

### 🔍 Core Features

- ✅ **Class Imbalance Handling**: Addresses the 98.7% / 1.3% imbalance using `scale_pos_weight`
- ✅ **Feature Engineering**: Creates 4 new financial signal features
- ✅ **Dimensionality Reduction**: PCA preserving 85%+ variance
- ✅ **Behavioral Profiling**: K-Means clustering for contextual features
- ✅ **Explainable AI**: Built-in feature importance for regulatory compliance
- ✅ **Continuous Learning**: Warm-start incremental training on new data batches

<div dir="rtl">

### المميزات الرئيسية

- ✅ **معالجة اختلال البيانات**: يعالج النسبة 98.7% / 1.3% باستخدام scale_pos_weight
- ✅ **هندسة الخصائص**: ينشئ 4 خصائص مالية جديدة
- ✅ **تقليل الأبعاد**: PCA مع الحفاظ على 85%+ من التباين
- ✅ **التنميط السلوكي**: K-Means للتجميع السياقي
- ✅ **قابلية التفسير**: أهمية الميزات للامتثال التنظيمي
- ✅ **التعلم المستمر**: تدريب تدريجي على دفعات البيانات الجديدة

</div>

---

## 🏗️ Architecture / البنية المعمارية

```
┌─────────────────────────────────────────────────────┐
│                 RAW TRANSACTION DATA                │
│              (PaySim Dataset - 6.3M rows)           │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              PREPROCESSING PIPELINE                 │
│  • Handle Missing Values                            │
│  • Feature Engineering (+4 features)                │
│  • Label Encoding (text → numbers)                  │
│  • Standard Scaling (mean=0, std=1)                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│          PCA - DIMENSIONALITY REDUCTION             │
│  • Reduce from 11 → 4-6 dimensions                  │
│  • Preserve 85%+ variance                           │
│  • Remove correlated features                       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│         K-MEANS CLUSTERING (K=4)                    │
│  • Behavioral profiling                             │
│  • Unsupervised learning                            │
│  • Output: behavioral_segment feature               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│      XGBOOST CLASSIFIER (Supervised)                │
│  • Input: All features + behavioral_segment         │
│  • scale_pos_weight for imbalance                   │
│  • 100 boosting rounds                              │
│  • Output: Fraud probability [0-1]                  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              EVALUATION METRICS                     │
│  • Precision, Recall, F1-Score                      │
│  • Confusion Matrix                                 │
│  • Feature Importance                               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│         CONTINUOUS LEARNING MODULE                  │
│  • Monthly batch updates                            │
│  • Warm-start training (xgb_model=)                 │
│  • No retraining from scratch                       │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Installation / التثبيت

### Prerequisites / المتطلبات

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- 2GB free disk space

### Quick Start / البدء السريع

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/aml-detection-pipeline.git
cd aml-detection-pipeline

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Option A: Download PaySim from Kaggle
# https://www.kaggle.com/datasets/ealaxi/paysim1
# Place paysim.csv in the project root

# Option B: The notebook will auto-generate synthetic data if file not found

# 5. Run the Jupyter Notebook
jupyter notebook AML_Detection_Pipeline.ipynb
```

---

## 📊 Dataset / البيانات

### PaySim Dataset

<div dir="rtl">

**البيانات المستخدمة:**
- **المصدر:** PaySim - محاكاة معاملات الأموال عبر الهاتف المحمول
- **الحجم:** 6.3 مليون صف (نستخدم 200,000 في المشروع لسرعة التدريب)
- **نسبة الاحتيال:** ~1.3% (imbalanced dataset)
- **الأعمدة:** 11 عمود بعد Feature Engineering

</div>

### Features Description

| Column | Type | Description (EN) | الوصف (AR) |
|--------|------|-----------------|------------|
| `step` | int | Hour of simulation (1-744) | رقم الساعة في المحاكاة |
| `type` | str | Transaction type (PAYMENT, TRANSFER, etc.) | نوع العملية |
| `amount` | float | Transaction amount | مبلغ العملية |
| `oldbalanceOrg` | float | Origin balance before | رصيد المرسل قبل |
| `newbalanceOrig` | float | Origin balance after | رصيد المرسل بعد |
| `oldbalanceDest` | float | Destination balance before | رصيد المستقبل قبل |
| `newbalanceDest` | float | Destination balance after | رصيد المستقبل بعد |
| `isFraud` | int | **TARGET**: 1=fraud, 0=legitimate | **الهدف**: 1=احتيال، 0=سليم |

### Engineered Features

```python
# 4 new features created:
balance_change_orig  = oldbalanceOrg - newbalanceOrig
balance_change_dest  = newbalanceDest - oldbalanceDest
amount_balance_error = abs(amount - balance_change_orig)
dest_was_empty       = 1 if oldbalanceDest == 0 else 0
```

---

## 📈 Model Performance / أداء النموذج

### Evaluation Metrics

<div dir="rtl">

#### النتائج النموذجية (على بيانات الاختبار)

</div>

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Precision** | ~0.85-0.92 | 85-92% of flagged transactions are actual fraud |
| **Recall** | ~0.75-0.85 | Catches 75-85% of all fraud cases |
| **F1-Score** | ~0.80-0.88 | Balanced performance |
| **Accuracy** | ~98.5% | ⚠️ Misleading due to class imbalance |

### Why We Don't Use Accuracy

<div dir="rtl">

**لماذا لا نستخدم Accuracy؟**

نموذج يتنبأ بأن كل معاملة "سليمة" سيحصل على 98.7% accuracy ولكنه لن يكتشف أي جريمة!

لذلك نستخدم **Recall** (نسبة الجرائم المكتشفة) و**F1-Score** (التوازن).

</div>

A model predicting "legitimate" for everything achieves 98.7% accuracy while catching **zero fraud**. That's why we focus on **Recall** and **F1-Score**.

---

## 📁 Project Structure / هيكل المشروع

```
aml-detection-pipeline/
│
├── 📓 AML_Detection_Pipeline.ipynb    # Main Jupyter Notebook
├── 📄 README.md                        # This file
├── 📄 requirements.txt                 # Python dependencies
├── 📄 LICENSE                          # MIT License
├── 📄 .gitignore                       # Git ignore rules
│
├── 📂 docs/                            # Documentation
│   ├── ARCHITECTURE.md                 # System architecture details
│   ├── ALGORITHMS_EXPLAINED.md         # Deep dive into algorithms
│   └── FAQ_AR.md                       # FAQ in Arabic
│
├── 📂 data/                            # Data directory
│   ├── .gitkeep                        # Keep folder in git
│   └── README.md                       # Data instructions
│
├── 📂 models/                          # Saved models
│   ├── .gitkeep
│   └── README.md                       # Model versioning info
│
├── 📂 src/                             # Source code (if modularized)
│   ├── __init__.py
│   ├── preprocessing.py                # Preprocessing functions
│   ├── feature_engineering.py          # Feature creation
│   ├── clustering.py                   # K-Means module
│   └── continuous_learning.py          # Incremental training
│
├── 📂 notebooks/                       # Additional notebooks
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_Feature_Analysis.ipynb       # Feature importance study
│   └── 03_Model_Comparison.ipynb       # Compare different models
│
└── 📂 presentation/                    # Presentation materials
    ├── AML_Project_Slides.pdf          # Project presentation
    └── Demo_Video.mp4                  # Demo video link
```

---

## 🧠 Algorithms Explained / شرح الخوارزميات

<div dir="rtl">

### 1️⃣ PCA (Principal Component Analysis) - تحليل المكونات الرئيسية

**ماذا يفعل؟**
يحول 11 عمود متشابك ومترابط إلى 4-6 أعمدة مستقلة، مع الاحتفاظ بـ 85%+ من المعلومات.

**لماذا نحتاجه؟**
- يحل مشكلة **Curse of Dimensionality**
- يزيل **الارتباط** بين الأعمدة
- يحضّر البيانات لـ K-Means

**كيف يعمل؟**
يبحث عن الاتجاهات ذات أعلى تباين (variance) في البيانات ويُسقط عليها.

---

### 2️⃣ K-Means Clustering - التجميع

**ماذا يفعل؟**
يجمّع المعاملات المتشابهة في 4 مجموعات سلوكية.

**لماذا نحتاجه؟**
يوفر **السياق السلوكي** - يخبر XGBoost: "هذه المعاملة من نوع سلوكي عالي الخطورة".

**كيف يعمل؟**
- يختار K=4 مراكز عشوائية
- يُعيّن كل نقطة لأقرب مركز
- يحسب متوسط كل مجموعة = مركز جديد
- يكرر حتى الاستقرار

**الإخراج:**
عمود جديد `behavioral_segment` بقيم (0, 1, 2, 3)

---

### 3️⃣ XGBoost - التصنيف النهائي

**ماذا يفعل؟**
يتوقع: هل المعاملة احتيال أم لا؟

**لماذا نحتاجه؟**
- يتعامل مع **Class Imbalance** عبر `scale_pos_weight`
- سريع في التدريب والتوقع
- قابل للتفسير (Feature Importance)
- يدعم **Continuous Learning**

**كيف يعمل؟**
يبني 100 شجرة قرار صغيرة، كل شجرة تتعلم من أخطاء السابقة.

</div>

### Algorithm Comparison

| Feature | PCA | K-Means | XGBoost |
|---------|-----|---------|---------|
| **Type** | Dimensionality Reduction | Unsupervised Clustering | Supervised Classification |
| **Input** | 11 scaled features | PCA components | All features + cluster |
| **Output** | 4-6 components | Cluster labels (0-3) | Fraud probability |
| **Learning** | N/A (transformation) | Unsupervised | Supervised |
| **Purpose** | Reduce noise & correlation | Add behavioral context | Final fraud detection |

---

## 🔄 Continuous Learning / التعلم المستمر

<div dir="rtl">

### المشكلة

المجرمون يغيرون أساليبهم باستمرار (**Concept Drift**). نموذج ثابت سيضعف مع الوقت.

### الحل

كل شهر نستقبل دفعة جديدة من البيانات ونحدّث النموذج:

</div>

```python
# ❌ WRONG: Retrain from scratch
model = xgb.train(params, old_data + new_data)  # Slow + requires storing old data

# ✅ CORRECT: Warm-start incremental training
updated_model = xgb.train(
    params,
    new_data_only,           # Only new batch
    num_boost_round=20,      # Add 20 new trees
    xgb_model=existing_model # Start from existing model
)
```

### Benefits / المميزات

- ⚡ **Fast**: Only trains on new data
- 💾 **GDPR Compliant**: No need to retain old data
- 🎯 **Adaptive**: Learns new fraud patterns
- 🔒 **Stable**: Doesn't forget old patterns

---

## 🤝 Contributing / المساهمة

<div dir="rtl">

### نرحب بمساهماتكم!

1. Fork المشروع
2. أنشئ فرع جديد (`git checkout -b feature/AmazingFeature`)
3. Commit التغييرات (`git commit -m 'Add some AmazingFeature'`)
4. Push للفرع (`git push origin feature/AmazingFeature`)
5. افتح Pull Request

</div>

---

## 📚 References / المراجع

### Papers & Research

1. **PaySim Dataset**: E. A. Lopez-Rojas et al., "PaySim: A financial mobile money simulator for fraud detection" (2016)
2. **XGBoost**: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016)
3. **K-Means**: MacQueen, "Some Methods for Classification and Analysis of Multivariate Observations" (1967)
4. **PCA**: Pearson, "On Lines and Planes of Closest Fit to Systems of Points in Space" (1901)

### Useful Links

- [PaySim Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [FATF AML Guidelines](https://www.fatf-gafi.org/)

---

## 📞 Contact / التواصل

<div dir="rtl">

**المطور:** [اسمك هنا]

**البريد الإلكتروني:** your.email@example.com

**LinkedIn:** [linkedin.com/in/yourprofile](https://linkedin.com)

**رابط المشروع:** [https://github.com/YOUR_USERNAME/aml-detection-pipeline](https://github.com/YOUR_USERNAME/aml-detection-pipeline)

</div>

---

## 📄 License / الترخيص

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div dir="rtl">

هذا المشروع مرخص بموجب رخصة MIT - انظر ملف [LICENSE](LICENSE) للحصول على التفاصيل.

</div>

---

## 🙏 Acknowledgments / شكر وتقدير

<div dir="rtl">

- جامعة [اسم جامعتك] - قسم علوم الحاسب
- الدكتور [اسم دكتورك] - المشرف على المشروع
- مجموعة PaySim - لتوفير البيانات المحاكاة
- مجتمع علوم البيانات المفتوح المصدر

</div>

---

<div align="center">

**⭐ إذا أعجبك المشروع، لا تنسَ إعطائه نجمة! ⭐**

**Made with ❤️ by [Your Name]**

</div>
