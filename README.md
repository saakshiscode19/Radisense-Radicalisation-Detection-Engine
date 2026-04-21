<div align="center">

<img src="https://img.shields.io/badge/RadiSense-Radicalisation%20Detection-e74c3c?style=for-the-badge&logo=shield&logoColor=white" alt="RadiSense"/>

# RadiSense
### *Leveraging RoBERTa-Based Transfer Learning for Explainable Tri-Class Radicalisation Detection in Social Media*

> **An intelligent, explainable NLP system that automatically classifies social media content into**  
> **Neutral · Non-Radical · Radical** — powered by fine-tuned RoBERTa and LIME explainability.

<br/>

| 🎯 Accuracy | 📊 Macro F1 | 🔵 ROC-AUC | 📦 Dataset | Presecision|
|:-----------:|:-----------:|:----------:|:----------:|:------------:|
| **95.78%** | **95.62%** | **99.46%** | **13,567 samples** | **95.79%** |

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training Pipeline](#-training-pipeline)
- [Explainability — LIME](#-explainability--lime)
- [Streamlit Deployment](#-streamlit-deployment)
- [Tech Stack](#-tech-stack)


---

## 🔍 Overview

The proliferation of extremist content on social media platforms poses a severe and escalating threat to global public safety. Traditional content moderation systems relying on keyword filters or classical machine learning pipelines **fail to capture the contextually nuanced and linguistically evolving nature of online radicalisation**.

**RadiSense** addresses this gap by fine-tuning [RoBERTa-base](https://huggingface.co/roberta-base) — a 125M parameter pre-trained transformer — on a curated multi-source dataset of labelled social media content. The system classifies any input text into one of three categories:

| Label | Class | Description |
|:-----:|:-----:|:-----------|
| 🟢 `0` | **Neutral** | Factual, objective content — news headlines, informational posts |
| 🟡 `1` | **Non-Radical** | Ordinary social interactions — opinions, casual conversations, personal posts |
| 🔴 `2` | **Radical** | Extremist propaganda, calls to violence, ideological hate speech |

RadiSense goes beyond classification — it integrates **LIME (Local Interpretable Model-agnostic Explanations)** to reveal *which exact words* drove each prediction, enabling transparent, auditable, and responsible AI deployment.

---

## 🚀 Live Demo

> **Try it right now — no installation required:**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://radisense-radicalisation-detection-engine-qnfr4yjtjs7p4mv6va3a.streamlit.app/)

**🔗 https://radisense-radicalisation-detection-engine-qnfr4yjtjs7p4mv6va3a.streamlit.app/**

Paste any social media post, news headline, or text snippet and get:
- Predicted class with confidence score
- Full probability distribution across all 3 classes
- LIME word-importance explanation

---

## ✨ Features

- 🤖 **Transfer Learning** — Fine-tuned RoBERTa-base with 125M pre-trained parameters
- 🧠 **Bidirectional Context** — Full sentence understanding via self-attention (not just keywords)
- 🔍 **LIME Explainability** — Word-level feature importance for every prediction
- ⚖️ **Class-Weighted Training** — Handles class imbalance without SMOTE
- 📊 **Comprehensive EDA** — Word clouds, KDE plots, frequency charts, class distribution
- 🆚 **Baseline Comparisons** — Evaluated against TF-IDF + Logistic Regression, Naive Bayes, Linear SVM
- 🌐 **Streamlit Web App** — One-click deployment, zero-friction inference
- 💾 **Exportable Model** — Save/load ready for production deployment

---

## 🏗️ Model Architecture

<img width="703" height="569" alt="image" src="https://github.com/user-attachments/assets/a65cab01-11aa-4508-b769-de9dc8b2289a" />


### Key Architectural Choices

| Component | Choice | Why |
|-----------|--------|-----|
| **Base Model** | RoBERTa-base | Removes NSP task, dynamic masking, 160GB training data — consistently outperforms BERT-base on classification |
| **Tokeniser** | Byte-Pair Encoding | Handles OOV words via subword decomposition; codes extremist slang at subword level |
| **Sequence Rep.** | [CLS] token | Aggregates full bidirectional context across all 12 attention layers |
| **Loss Function** | Weighted Cross-Entropy | `compute_class_weight('balanced')` compensates for mild class imbalance |
| **Optimiser** | AdamW | Weight decay decoupled from adaptive updates; standard for transformer fine-tuning |
| **Mixed Precision** | FP16 (when CUDA) | 2-3× speedup; halves GPU memory requirement |

---

## 📦 Dataset

RadiSense is trained on a curated **multi-source, multi-register** corpus:

| Class | Source | Platform | Language | Samples |
|:-----:|--------|----------|----------|:-------:|
| 🟢 **Neutral** | ABC News Headlines CSV | News Portal | English | 5,000 |
| 🟡 **Non-Radical** | Twitter Training CSV | Twitter | English | 5,000 |
| 🔴 **Radical** | ISIS Tweets + Religious Texts v1 | Twitter / Dark Web | English / Arabic | 4,000 |
| | **Total (after cleaning)** | | | **13,567** |

### Class Distribution
```
Neutral      ████████████████████████  36.8%  (4,998)
Non-Radical  ███████████████████████   35.6%  (4,828)
Radical (C.) ██████████████████        27.6%  (3,741)
```

### Dataset Splitting (Stratified)
```
┌─────────────────────────────────────────────────────────┐
│  TRAIN  70%  (9,497)  │  VAL  15%  (2,035)  │  TEST 15% │
└─────────────────────────────────────────────────────────┘
```

### Preprocessing Pipeline
```
Raw Text
  │
  ├── 1. Lowercase            → "ISIS" → "isis"
  ├── 2. URL Removal          → strip http/https/www links
  ├── 3. Mention Strip        → remove @handles, strip # symbol
  ├── 4. Char Filter          → keep only [a-z] and whitespace
  ├── 5. Whitespace Collapse  → "  " → " "
  └── 6. Length Filter        → drop if len < 5 chars
              │
              └── RoBERTa BPE Tokenisation (max_length=128)
```

> **Note:** Stemming, lemmatisation, and stopword removal are intentionally **not applied** to transformer inputs — RoBERTa handles morphological variation via BPE and learns token importance through attention weights during fine-tuning.

---

## 📈 Results

<img width="530" height="273" alt="image" src="https://github.com/user-attachments/assets/8ff6e443-e45e-4224-a71c-81c1bacccbe3" />
<img width="1612" height="594" alt="image" src="https://github.com/user-attachments/assets/a473ffb8-d8cd-4ad0-a565-339f38ee0658" />
<img width="1067" height="587" alt="image" src="https://github.com/user-attachments/assets/81491c0f-80d5-4a4e-8559-6c7c983c6a03" />


### Comparison with Baselines
<img width="879" height="118" alt="image" src="https://github.com/user-attachments/assets/7afdd53d-15de-488d-9d06-289facb262b2" />
<img width="1908" height="711" alt="image" src="https://github.com/user-attachments/assets/8a6c9f07-d997-4444-83dc-4a99060d8aa9" />





> RadiSense outperforms the best classical baseline (Linear SVM) by **+3.08% accuracy** and **+3.05% F1-Macro**.

### Confusion Matrix (Raw Counts)
```
                 Predicted
              Neutral  Non-Rad  Radical
Actual Neutral   734      11       5
   Non-Radical    14     695      16
      Radical     13      29     519

<img width="2115" height="594" alt="image" src="https://github.com/user-attachments/assets/aaf53a35-617b-47ed-abed-6c0b30cf8498" />
```
*Diagonal values represent correct predictions. The highest off-diagonal error (29) is Radical→Non-Radical, reflecting the challenge of distinguishing aggressive social speech from genuine extremist ideology.*

---

## 📁 Project Structure

```
RadiSense/
│
├── 📓 RadiSense_NLP.ipynb          # Main Colab notebook (full pipeline)
│
├── 🌐 streamlit_app.py             # Streamlit web application
│
├── 📦 radical_model/               # Saved model weights (after training)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
│
├── 📊 plots/                       # All generated EDA and evaluation plots
│   ├── eda_overview.png
│   ├── wordclouds.png
│   ├── top_words_per_class.png
│   ├── kde_distributions.png
│   ├── dataset_split.png
│   ├── hyperparameter_table.png
│   ├── training_curves_roberta.png
│   ├── confusion_matrix_roberta.png
│   ├── roc_auc_roberta.png
│   ├── pr_curve_roberta.png
│   ├── per_class_metrics_roberta.png
│   ├── confusion_matrix_baselines.png
│   ├── model_comparison.png
│   ├── radar_comparison.png
│   ├── lime_explanation.png
│   └── inference_probabilities.png
│
├── 📄 requirements.txt             # All Python dependencies
├── 📝 README.md                    # This file
└── 📜 LICENSE                      # MIT License
```

---

## ⚡ Quick Start

### Option 1 — Use the Live Demo (Recommended)
Just go to: **https://radisense-radicalisation-detection-engine-qnfr4yjtjs7p4mv6va3a.streamlit.app/**

### Option 2 — Run Locally

**Prerequisites:** Python 3.10+, pip, (optional but recommended) CUDA GPU

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/radisense.git
cd radisense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run streamlit_app.py
```

## 🔬 Training Pipeline

The full training pipeline runs end-to-end in the Colab notebook. Here is a summary:

```
1. DATA COLLECTION
   └── Load ABCNews (Neutral) + Twitter (Non-Radical) + ISIS corpus (Radical)
   
2. PREPROCESSING
   └── clean_text() → lowercase, strip URLs/mentions/special chars, length filter
   
3. EDA
   └── Class distribution · Word clouds · KDE plots · Frequency charts
   
4. SPLITTING  (stratified 70 / 15 / 15)
   └── train_test_split(stratify=y) → 9,497 train | 2,035 val | 2,035 test
   
5. TOKENISATION
   └── RobertaTokenizer · BPE · max_length=128 · padding · truncation
   
6. FINE-TUNING (Google Colab T4 GPU — ~40 min)
   ├── Model   : roberta-base + 3-class head
   ├── Epochs  : 3  (best checkpoint saved)
   ├── LR      : 2e-5 (AdamW + linear warmup 500 steps)
   ├── Batch   : 16 (train) / 32 (eval)
   ├── Loss    : Weighted Cross-Entropy (balanced class weights)
   └── FP16    : Enabled
   
7. EVALUATION
   └── Accuracy · Macro F1 · ROC-AUC · Confusion Matrix · PR Curves
   
8. BASELINES
   └── TF-IDF (10K features, bigrams) + LR · NB · SVM
   
9. EXPLAINABILITY
   └── LIME → word-level importance for each prediction
   
10. DEPLOYMENT
    └── Streamlit Cloud → live web application
```

### Hyperparameter Configuration
<img width="1667" height="827" alt="image" src="https://github.com/user-attachments/assets/c5530731-e2e5-4579-bc5a-d36088b9f976" />

---

## 🔎 Explainability — LIME

RadiSense integrates LIME to provide **word-level transparency** for every prediction.

**How it works:**
```
Input Text
    │
    ├── Generate 300 perturbed variants (random word masking)
    │
    ├── Feed all 300 through RoBERTa → get probability vectors
    │
    ├── Fit local linear model weighted by proximity to original
    │
    └── Coefficients = word importance scores
              ├── Positive → pushes toward Radical class
              └── Negative → pushes away from Radical class

```
<img width="756" height="570" alt="image" src="https://github.com/user-attachments/assets/22497ed0-ba7d-4a55-92f5-4f0ee9dd88d7" />

```

*The model attends to combinations of violent agency words + reclamation narrative + ideological structure — not isolated religious vocabulary.*

> **Why LIME and not attention weights?**  
> Attention scores show which tokens are *attended to*, not which ones *caused* the prediction. LIME measures actual causal influence by observing how predictions change when words are removed — a more faithful indicator of decision-driving features.

---

## 🌐 Streamlit Deployment

The app was trained in **Google Colab** (T4 GPU) and deployed to **Streamlit Cloud**.

**Deployment steps:**
```bash
# 1. After training in Colab, download radical_model.zip
# 2. Create a GitHub repository and push:
git init
git add .
git commit -m "Initial RadiSense deployment"
git remote add origin https://github.com/YOUR_USERNAME/radisense.git
git push -u origin main

# 3. Go to share.streamlit.io
#    → Connect your GitHub repo
#    → Set main file: streamlit_app.py
#    → Deploy
```

**requirements.txt for deployment:**
```
transformers==4.40.0
torch==2.2.0
streamlit==1.33.0
lime==0.2.0.1
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.4
seaborn==0.13.2
matplotlib==3.8.0
wordcloud==1.9.3
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
|----------|-----------|
| **Model** | RoBERTa-base (HuggingFace Transformers) |
| **Deep Learning** | PyTorch 2.x |
| **Classical ML** | scikit-learn (TF-IDF, LR, NB, SVM) |
| **Explainability** | LIME (lime-text) |
| **Data Processing** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, WordCloud |
| **Web App** | Streamlit |
| **Training Environment** | Google Colab (NVIDIA T4 GPU) |
| **Deployment** | Streamlit Cloud |
| **Language** | Python 3.10+ |

</div>

## ⚠️ Limitations & Ethics

- **Monolingual:** English-dominant; Arabic script is removed during preprocessing
- **ISIS-centric Radical class:** Model has learned ISIS-specific patterns; may underperform on far-right, eco-terrorism, or other ideological extremism
- **Not a moderation tool in isolation:** Designed as a triage aid for human moderators — not an autonomous removal system
- **Potential religious bias:** Certain Islamic terminology may be over-represented in Radical predictions due to training data composition; this is a known limitation requiring broader dataset diversity
- **Local LIME only:** LIME provides local explanations for individual predictions — not global model interpretability

---

## 🔭 Future Work

- [ ] **Multilingual support** — XLM-RoBERTa for Arabic, Urdu, French extremist content
- [ ] **Expanded radical taxonomy** — far-right, incel, eco-terrorism, and other ideological classes
- [ ] **Real-time streaming** — Apache Kafka pipeline for live social media feed processing
- [ ] **Adversarial robustness** — training against evasion via misspellings and code-switching
- [ ] **Federated learning** — privacy-preserving training across distributed organisations
- [ ] **ONNX export** — reduce inference latency from ~50ms to <10ms for production scale

---



---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---



