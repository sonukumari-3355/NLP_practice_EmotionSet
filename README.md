# Emotion Detection from Text Using NLP and Machine Learning
A Natural Language Processing (NLP) text classification project that predicts human emotions (such as sadness, anger, and love) from raw text data using classical machine learning algorithms.
---
## 📌 Project Overview
The objective of this project is to build an end-to-end NLP pipeline that cleans unstructured text, extracts features using Bag-of-Words and TF-IDF, and evaluates multiple classification models to determine the most accurate approach for multi-class emotion classification.
---
## ⚙️ Workflow & Preprocessing Pipeline
1. **Data Ingestion & Inspection:**
   - Loaded semicolon-separated dataset (`train.txt`) into a Pandas DataFrame.
   - Verified that no missing or null values exist in the dataset.
   - Mapped unique string labels to numerical representations.
2. **Text Cleaning & Normalization:**
   - **Lowercasing:** Standardized all text to lower case.
   - **Punctuation Removal:** Stripped punctuation symbols using Python's `string.punctuation`.
   - **Digit Removal:** Filtered out numerical characters.
   - **Emoji & Non-ASCII Cleaning:** Removed non-ASCII characters and emojis.
   - **URL Removal:** Identified and removed web links (`http`/`https`).
   - **Stopwords Removal:** Filtered standard English stopwords using NLTK (`stopwords.words('english')`).
3. **Feature Extraction:**
   - **Bag of Words (`CountVectorizer`):** Encoded raw term frequencies.
   - **TF-IDF (`TfidfVectorizer`):** Weighted words by Term Frequency-Inverse Document Frequency.
4. **Model Training & Evaluation:**
   - Dataset split: **80% Training**, **20% Testing** (`test_size=0.20`).
   - Evaluated models using accuracy score on unseen test data.
---
## 📊 Model Performance Comparison

| Model | Feature Extraction | Accuracy |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | CountVectorizer (BoW) | **76.88%** (`0.76875`) |
| **Multinomial Naive Bayes** | TfidfVectorizer | **66.13%** (`0.66125`) |
| **Logistic Regression** (`max_iter=1000`) | TfidfVectorizer | **86.22%** (`0.8621875`) |

> **Best Model:** **Logistic Regression** trained on TF-IDF features yielded the highest accuracy of **~86.22%**.
---
## 🛠️ Tech Stack & Dependencies
- **Python 3.x**
- **Libraries:**
  - `pandas` & `numpy` (Data manipulation)
  - `matplotlib` & `seaborn` (Visualization)
  - `nltk` (Tokenization and Stopwords)
  - `scikit-learn` (Vectorization, Model Training, Metrics)
---
## 🚀 How to Run
1. **Clone the repository:**
2. **Install requires packages: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn**
3. **Download NLTK data**
4. **Run the Notebook**
   
   ```bash
   git clone [https://github.com/yourusername/emotion-text-classification.git](https://github.com/yourusername/emotion-text-classification.git)
   cd emotion-text-classification
