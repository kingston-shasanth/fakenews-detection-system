"""
=============================================================================
FAKE JOB DETECTION - MODEL TRAINING SCRIPT
=============================================================================
This script trains a machine learning model to detect fake job postings.
It uses TF-IDF vectorization and Logistic Regression for classification.

Dataset: Real-world fake job postings dataset with text features
Model: Logistic Regression (explainable, good for text classification)

How to use:
    1. Place your dataset (fake_jobs.csv) in the data/ folder
    2. Run: python train_model.py
    3. The model will be saved to model/fake_job_model.pkl and model/vectorizer.pkl

Author: Student Project
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import string
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "data/fake_jobs.csv"
MODEL_PATH = "model/fake_job_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

# =============================================================================
# TEXT PREPROCESSING FUNCTIONS
# =============================================================================


def clean_text(text):
    """
    Clean and preprocess text data for the ML model.

    Steps:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Remove extra whitespace
    4. Remove common words (stopwords) - simple version

    Args:
        text (str): Raw job description text

    Returns:
        str: Cleaned text
    """
    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def combine_text_features(row):
    """
    Combine multiple text columns into one for better prediction.

    We combine: title, company_profile, description, requirements, benefits
    This gives the model more context about the job posting.

    Args:
        row: DataFrame row with text columns

    Returns:
        str: Combined text
    """
    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
    ]
    combined = ""

    for col in text_columns:
        if col in row and pd.notna(row[col]):
            combined += str(row[col]) + " "

    return combined.strip()


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def train_model():
    """
    Main function to train the fake job detection model.

    Pipeline:
    1. Load dataset
    2. Preprocess text data
    3. Split into train/test (80/20)
    4. Vectorize using TF-IDF
    5. Train Logistic Regression
    6. Evaluate model performance
    7. Save model and vectorizer

    Returns:
        dict: Model performance metrics
    """
    print("=" * 60)
    print("FAKE JOB DETECTION - MODEL TRAINING")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load Dataset
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading dataset...")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("\nPlease download the dataset from:")
        print(
            "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-dataset"
        )
        print("\nOr create a fake_jobs.csv with these columns:")
        print("  - title, company_profile, description, requirements, benefits")
        print("  - fraudulent (target: 1=fake, 0=real)")
        return None

    # Try different encodings for the CSV
    try:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")
    except:
        df = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip")

    print(f"   Dataset loaded: {len(df)} job postings")
    print(f"   Columns: {list(df.columns)}")

    # -------------------------------------------------------------------------
    # Step 2: Preprocess Data
    # -------------------------------------------------------------------------
    print("\n[2/7] Preprocessing text data...")

    # Check if 'fraudulent' column exists (target variable)
    if "fraudulent" not in df.columns:
        print("ERROR: 'fraudulent' column not found in dataset!")
        print("Available columns:", list(df.columns))
        return None

    # Check class distribution
    fraud_count = df["fraudulent"].sum()
    real_count = len(df) - fraud_count
    print(f"   Real jobs: {real_count} ({real_count / len(df) * 100:.1f}%)")
    print(f"   Fake jobs: {fraud_count} ({fraud_count / len(df) * 100:.1f}%)")

    # Combine text features
    print("   Combining text features...")
    df["combined_text"] = df.apply(combine_text_features, axis=1)

    # Clean the combined text
    print("   Cleaning text...")
    df["cleaned_text"] = df["combined_text"].apply(clean_text)

    # Remove empty text rows
    df = df[df["cleaned_text"].str.len() > 10].reset_index(drop=True)
    print(f"   After cleaning: {len(df)} job postings")

    # -------------------------------------------------------------------------
    # Step 3: Split Data
    # -------------------------------------------------------------------------
    print("\n[3/7] Splitting data into train/test sets...")

    X = df["cleaned_text"]  # Features (text)
    y = df["fraudulent"]  # Target (0=real, 1=fake)

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # Maintain class distribution
    )

    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # -------------------------------------------------------------------------
    # Step 4: Vectorize Text using TF-IDF
    # -------------------------------------------------------------------------
    print("\n[4/7] Vectorizing text using TF-IDF...")

    # TF-IDF: Converts text to numerical features
    # - Term Frequency (TF): How often a word appears
    # - Inverse Document Frequency (IDF): How unique the word is
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit to top 5000 features
        min_df=2,  # Ignore very rare words
        max_df=0.95,  # Ignore very common words
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words="english",  # Remove common English words
    )

    # Fit on training data, transform both train and test
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Feature matrix shape: {X_train_tfidf.shape}")

    # -------------------------------------------------------------------------
    # Step 5: Train Logistic Regression
    # -------------------------------------------------------------------------
    print("\n[5/7] Training Logistic Regression model...")

    # Logistic Regression:
    # - Good for binary classification
    # - Provides probability scores
    # - Easy to explain (weights can be interpreted)
    model = LogisticRegression(
        max_iter=1000,  # Maximum iterations for convergence
        class_weight="balanced",  # Handle class imbalance
        random_state=42,  # For reproducibility
        C=1.0,  # Regularization strength
        solver="lbfgs",  # Optimization algorithm
    )

    model.fit(X_train_tfidf, y_train)
    print("   Model trained successfully!")

    # -------------------------------------------------------------------------
    # Step 6: Evaluate Model
    # -------------------------------------------------------------------------
    print("\n[6/7] Evaluating model performance...")

    # Predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n   MODEL PERFORMANCE METRICS:")
    print("   " + "-" * 40)
    print(f"   Accuracy:  {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}%")
    print(f"   F1-Score:  {f1 * 100:.2f}%")

    print("\n   CONFUSION MATRIX:")
    print("   " + "-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(f"   True Negatives (Real correctly identified):  {cm[0][0]}")
    print(f"   False Positives (Real predicted as Fake):     {cm[0][1]}")
    print(f"   False Negatives (Fake predicted as Real):     {cm[1][0]}")
    print(f"   True Positives (Fake correctly identified):  {cm[1][1]}")

    print("\n   CLASSIFICATION REPORT:")
    print("   " + "-" * 40)
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # -------------------------------------------------------------------------
    # Step 7: Save Model and Vectorizer
    # -------------------------------------------------------------------------
    print("\n[7/7] Saving model and vectorizer...")

    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    # Save using joblib (better for large numpy arrays)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Vectorizer saved to: {VECTORIZER_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTo run the web application:")
    print("   python app.py")
    print("\nThen open your browser at: http://localhost:5000")

    return metrics


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    metrics = train_model()
