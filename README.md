# 🇧🇷🇵🇹 Portuguese Variant Classification with LSTM & CNN

This project was developed as part of a Machine Learning course challenge and submitted to the Kaggle competition:

🔗 [Portuguese Variant Identification – Kaggle](https://www.kaggle.com/competitions/portuguese-variant-identification)

The goal was to **build a classifier that distinguishes between Brazilian Portuguese (pt-BR) and European Portuguese (pt-PT)** using natural language processing techniques. The best model achieved a **precision score of 0.93009**.

---

## 🎯 Objective

Train deep learning models to classify short text inputs as either pt-BR or pt-PT, using a combined dataset from Hugging Face and DSL-TL, and apply tuning strategies to improve classification performance in an unbalanced dataset.

---

## 🧪 Datasets

- **Hugging Face Dataset** (from the competition)
- **DSL-TL Dataset**: [GitHub](https://github.com/LanguageTechnologyLab/DSL-TL)

Both datasets were merged to increase the linguistic coverage and training corpus.

---

## 📚 Pipeline

### 1. Dataset Preparation
- Loaded datasets from Hugging Face and DSL-TL.
- Assigned class labels:
  - `0` → Portuguese (pt-PT)
  - `1` → Brazilian Portuguese (pt-BR)
- Cleaned and converted data to usable format.
- Merged the datasets into a single corpus.

### 2. Tokenization
- Applied tokenizer (BERT-based or custom, depending on cell).
- Resulted in ~110 tokens per input (avg).

### 3. Modeling Strategy

#### 🔹 LSTM Model
- Used for initial training and evaluation.
- Defined `F1-score` as main metric due to class imbalance.
- Applied hyperparameter tuning.
- Saved best model for prediction and comparison.

#### 🔹 CNN Model
- Trained a 1D CNN as an alternative approach.
- Tuned model and compared performance with LSTM.
- Final predictions made using best CNN model.

---

## 📊 Results

- Best model: CNN with tuning  
- **F1 Score (best model):** 0.93879  
- Evaluation focused on generalization and class imbalance

---

## 🔧 Technologies Used

- Python, Pandas, NumPy
- PyTorch / Keras
- Scikit-learn
- Hugging Face Transformers
- Matplotlib / Seaborn

---


