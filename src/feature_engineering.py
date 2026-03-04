# =============================================================================
# src/feature_engineering.py
# PURPOSE: Convert cleaned text into numerical features that ML models understand
# This is called "Feature Engineering" — one of the most important ML skills
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib
import os


def build_tfidf_vectorizer(max_features: int = 5000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Create and configure a TF-IDF Vectorizer.

    TF-IDF Explained Simply:
    - TF (Term Frequency)  = How often a word appears in ONE email
    - IDF (Inverse Document Frequency) = How rare the word is across ALL emails
    - TF-IDF score = TF × IDF
    - HIGH score = word is important in THIS email but rare in other emails
    - LOW score  = word appears everywhere (like "the", "is") — not useful

    Args:
        max_features (int): Maximum number of words to keep in vocabulary
                           5000 means keep the top 5000 most useful words
        ngram_range (tuple): (1,2) means use single words AND word pairs
                            e.g., "free money" as one feature

    Returns:
        TfidfVectorizer: Configured but NOT yet fitted vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,        # (1,2) = unigrams + bigrams
        min_df=2,                        # Ignore words appearing in < 2 documents
                                         # This removes extremely rare/typo words
        max_df=0.95,                     # Ignore words in > 95% of documents
                                         # These are too common to be useful
        sublinear_tf=True,               # Apply log(TF) instead of raw TF
                                         # Prevents very common words dominating
        strip_accents='unicode',         # Handle accented characters é → e
        analyzer='word',                 # Analyze at word level (not character level)
    )
    return vectorizer


def fit_vectorizer(vectorizer: TfidfVectorizer, train_texts: pd.Series) -> TfidfVectorizer:
    """
    Fit the vectorizer on TRAINING data only.

    CRITICAL RULE: NEVER fit on test data — this causes "data leakage"
    Data leakage = model accidentally learns from test data → fake high accuracy

    Args:
        vectorizer: Unfitted TfidfVectorizer
        train_texts: Series of cleaned training messages

    Returns:
        TfidfVectorizer: Fitted vectorizer (knows the vocabulary now)
    """
    print(f"[FEATURES] Fitting TF-IDF vectorizer on {len(train_texts)} training samples...")
    vectorizer.fit(train_texts)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[FEATURES] Vocabulary size: {vocab_size} words")
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: pd.Series):
    """
    Transform text into TF-IDF numerical matrix.
    Uses the vocabulary learned during fit() — does NOT relearn.

    Args:
        vectorizer: Already fitted TfidfVectorizer
        texts: Series of cleaned messages to transform

    Returns:
        scipy sparse matrix: Shape (n_samples, n_features)
    """
    return vectorizer.transform(texts)


def add_handcrafted_features(df: pd.DataFrame) -> np.ndarray:
    """
    Add extra manually-crafted features alongside TF-IDF.
    These capture things TF-IDF alone might miss.

    Features added:
    - message_length   : Long spam often has more text
    - word_count       : Word count
    - has_number       : "Win $500" — spam uses numbers a lot
    - has_currency     : '$', '£', '€' symbols are big spam indicators
    - uppercase_ratio  : "CLICK NOW WIN FREE" — spam uses CAPS a lot

    Args:
        df: DataFrame containing the engineered feature columns

    Returns:
        np.ndarray: Array of shape (n_samples, 5)
    """
    features = df[['message_length', 'word_count', 'has_number',
                   'has_currency', 'uppercase_ratio']].values
    return features


def combine_features(tfidf_matrix, handcrafted_array: np.ndarray):
    """
    Combine TF-IDF features + handcrafted features into one matrix.
    This usually improves accuracy slightly.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix
        handcrafted_array: Dense numpy array of extra features

    Returns:
        Combined sparse matrix
    """
    # Convert handcrafted features to sparse format for efficient combination
    handcrafted_sparse = csr_matrix(handcrafted_array)
    # hstack = horizontal stack (add columns side by side)
    combined = hstack([tfidf_matrix, handcrafted_sparse])
    return combined


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str):
    """
    Save the fitted vectorizer to disk using joblib.

    WHY SAVE IT?
    The vectorizer learned the vocabulary from training data.
    When a NEW email comes in at prediction time, we must transform it
    using the SAME vocabulary — so we save and reload it.

    Args:
        vectorizer: Fitted TfidfVectorizer
        filepath: Where to save (e.g., 'models/tfidf_vectorizer.pkl')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(vectorizer, filepath)
    print(f"[SAVE] Vectorizer saved to: {filepath}")


def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """
    Load a previously saved vectorizer from disk.

    Args:
        filepath: Path to the saved .pkl file

    Returns:
        TfidfVectorizer: The loaded fitted vectorizer
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Vectorizer not found at: {filepath}\n"
            "Run 'python run.py' first to train and save the model!"
        )
    vectorizer = joblib.load(filepath)
    print(f"[LOAD] Vectorizer loaded from: {filepath}")
    return vectorizer
