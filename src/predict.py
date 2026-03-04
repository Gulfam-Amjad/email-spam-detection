# =============================================================================
# src/predict.py
# PURPOSE: Single place for all prediction logic
# The Streamlit app imports from here — clean separation of concerns
# =============================================================================

import os
import re
from src.feature_engineering import load_vectorizer
from src.model_training import load_model
from src.data_preprocessing import clean_text, STOPWORDS


# Paths to saved model files
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
MODEL_PATH      = os.path.join("models", "spam_classifier.pkl")

# Cache loaded models in memory (so we don't reload from disk on every prediction)
_vectorizer = None
_model      = None


def load_pipeline():
    """
    Load both the vectorizer and model into memory.
    Uses global caching so files are loaded only ONCE per session.

    This is called automatically by predict_email() — you don't need to call it manually.
    """
    global _vectorizer, _model

    if _vectorizer is None:
        _vectorizer = load_vectorizer(VECTORIZER_PATH)

    if _model is None:
        _model = load_model(MODEL_PATH)


def predict_email(email_text: str) -> dict:
    """
    Predict whether a single email is spam or ham.
    This is the MAIN function called by the Streamlit app.

    Pipeline:
    1. Load model and vectorizer (if not already loaded)
    2. Clean the input text (same steps as training)
    3. Convert to TF-IDF numbers
    4. Run model prediction
    5. Return result with confidence score

    Args:
        email_text (str): Raw email message from user input

    Returns:
        dict with keys:
            - 'label'      : 'spam' or 'ham'
            - 'confidence' : probability % (0-100)
            - 'spam_prob'  : raw spam probability (0.0 - 1.0)
            - 'ham_prob'   : raw ham probability (0.0 - 1.0)
            - 'cleaned'    : what the cleaned text looked like
            - 'word_count' : number of words in original message
            - 'char_count' : number of characters in original message
    """
    # Step 1: Load pipeline (cached after first call)
    load_pipeline()

    # Step 2: Clean the text — MUST use SAME cleaning as training
    cleaned = clean_text(email_text)

    # Step 3: Vectorize — MUST use SAME fitted vectorizer as training
    X = _vectorizer.transform([cleaned])

    # Step 4: Predict
    prediction = _model.predict(X)[0]       # 'spam' or 'ham'
    probabilities = _model.predict_proba(X)[0]   # [ham_prob, spam_prob]

    # Get class order from model (could be ['ham','spam'] or ['spam','ham'])
    classes   = list(_model.classes_)
    spam_idx  = classes.index('spam')
    ham_idx   = classes.index('ham')

    spam_prob = probabilities[spam_idx]
    ham_prob  = probabilities[ham_idx]

    # Step 5: Return structured result
    return {
        'label'     : prediction,                        # 'spam' or 'ham'
        'confidence': round(max(spam_prob, ham_prob) * 100, 2),  # % confidence
        'spam_prob' : round(spam_prob * 100, 2),         # spam %
        'ham_prob'  : round(ham_prob * 100, 2),          # ham %
        'cleaned'   : cleaned,                            # cleaned version
        'word_count': len(email_text.split()),            # original word count
        'char_count': len(email_text),                    # original char count
        'has_currency': bool(re.search(r'[$£€]', email_text)),
        'uppercase_ratio': round(
            sum(1 for c in email_text if c.isupper()) / max(len(email_text), 1), 3
        )
    }


def batch_predict(emails: list) -> list:
    """
    Predict multiple emails at once (more efficient than one-by-one).
    Useful for testing or bulk processing.

    Args:
        emails: List of email strings

    Returns:
        List of prediction dicts
    """
    return [predict_email(email) for email in emails]


def get_spam_keywords(top_n: int = 15) -> list:
    """
    Return the top words most associated with spam (for display in the app).
    Uses the model's learned weights to rank words.

    Works with Logistic Regression (uses coefficients).
    Falls back to empty list if model doesn't support it.

    Args:
        top_n: How many top words to return

    Returns:
        List of (word, score) tuples sorted by spam association
    """
    load_pipeline()

    try:
        feature_names = _vectorizer.get_feature_names_out()

        # Logistic Regression has .coef_ — positive = spam, negative = ham
        if hasattr(_model, 'coef_'):
            coefs = _model.coef_[0]
            top_indices = coefs.argsort()[-top_n:][::-1]
            return [(feature_names[i], round(coefs[i], 3)) for i in top_indices]

        # Naive Bayes — use log probability difference
        elif hasattr(_model, 'feature_log_prob_'):
            classes   = list(_model.classes_)
            spam_idx  = classes.index('spam')
            ham_idx   = classes.index('ham')
            diff      = _model.feature_log_prob_[spam_idx] - _model.feature_log_prob_[ham_idx]
            top_indices = diff.argsort()[-top_n:][::-1]
            return [(feature_names[i], round(diff[i], 3)) for i in top_indices]

    except Exception as e:
        print(f"[WARN] Could not extract keywords: {e}")

    return []
