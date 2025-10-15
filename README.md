# Twitter Sentiment Analysis with Machine Learning and BERT

This project performs sentiment analysis on the Sentiment140 dataset, which contains 1.6 million tweets. It explores various techniques ranging from traditional rule-based models to classic machine learning classifiers and a fine-tuned DistilBERT model for sequence classification. The entire workflow is documented in the `Sentiment_BERT.ipynb` Jupyter Notebook.


## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Trained Model Access](#trained-model-access)

---

## 📝 Project Overview

The goal of this project is to classify tweets as either **positive** or **negative**. The notebook covers a complete data science workflow:
1.  **Data Ingestion:** Downloading the dataset directly from Kaggle.
2.  **Data Cleaning:** Removing duplicates, non-ASCII characters, and irrelevant entries.
3.  **Text Preprocessing:** Cleaning tweet text by removing URLs, mentions, hashtags, and performing lemmatization.
4.  **Exploratory Data Analysis (EDA):** Visualizing the data distribution, word clouds, and n-grams.
5.  **Modeling:**
    -   Baseline models (TextBlob, VADER).
    -   Classic Machine Learning models (Logistic Regression, Linear SVM).
    -   A fine-tuned **DistilBERT** model using the Hugging Face `transformers` library.
6.  **Evaluation:** Comparing all models based on accuracy, precision, recall, and F1-score.

---

## ✨ Features
- **Data Cleaning & Preprocessing Pipeline:** Robust functions to handle noisy social media text.
- **In-depth EDA:** Rich visualizations including word clouds, frequency distributions, and n-grams.
- **Multi-Model Comparison:** Benchmarks rule-based, traditional ML, and deep learning models.
- **BERT Fine-Tuning:** Leverages the power of transformers for high-accuracy sentiment classification.
- **Ready-to-Use Inference:** Includes a function to predict sentiment on new text using the trained model.

---

## 💾 Dataset

The project uses the **Sentiment140 dataset**.
-   **Source:** [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
-   **Content:** 1,600,000 tweets with sentiment labels.
-   **Labels:** `0` = Negative, `1` = Positive.

The notebook downloads this dataset automatically using the `kagglehub` library.

---

## 🛠️ Technologies Used
- **Python 3.x**
- **Core Libraries:** Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn
- **NLP & Deep Learning:** Hugging Face `transformers`, `datasets`, PyTorch
- **Environment:** Jupyter Notebook / Google Colab

---

## ⚙️ Setup and Installation

To run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/sentiment-analysis-bert.git](https://github.com/your-username/sentiment-analysis-bert.git)
    cd sentiment-analysis-bert
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:** Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    ```
---

## 🚀 How to Run

The entire project is contained within the `Sentiment_BERT.ipynb` Jupyter Notebook.
1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open the `Sentiment_BERT.ipynb` file and run the cells sequentially to reproduce the analysis and model training.

---

## 📊 Model Performance

| Model                       | Accuracy   |
| --------------------------- | ---------- |
| TextBlob (Rule-based)       | ~62.3%     |
| VADER (Rule-based)          | ~68.9%     |
| Logistic Regression (TF-IDF)| ~80.2%     |
| Linear SVM (TF-IDF)         | ~80.1%     |
| **DistilBERT (Fine-tuned)** | **~85.5%** |

The fine-tuned DistilBERT model significantly outperforms all other methods.

---

## 🧠 Trained Model Access

The final trained DistilBERT model is too large for GitHub.

**Option 1: Download the Pre-trained Model**

You can download the model files from this link:

[**➡️ Download Fine-Tuned BERT Model**](https://www.dropbox.com/scl/fi/sb5pjgpmvcgstb870nspa/bert.zip?rlkey=95fwfll66t407qjzuz2r2vwj8&st=nshccini&dl=1)  *(This is the link from your code. You can host it on Google Drive, Dropbox, etc.)*

After downloading, unzip it and place the `bert_sentiment_model` folder in your project directory.

**Option 2: Train the Model Yourself**
Run the training cells in the Jupyter notebook. This will generate the model files for you.