# =============================================================================
# src/model_training.py
# PURPOSE: Train multiple models, compare them, save the best one
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (required for servers)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import MaxAbsScaler

import joblib
import os


# =============================================================================
# MODEL DEFINITIONS
# Why these 2 models?
#   Naive Bayes        — fastest, best for short texts, great baseline
#   Logistic Regression— most reliable, interpretable, production-grade
# =============================================================================
MODELS = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    # alpha=0.1 (Laplace smoothing) — prevents zero-probability for unseen words
    # Lower alpha = more sensitive to training data patterns

    "Logistic Regression": LogisticRegression(
        C=1.0,              # Regularization strength — prevents overfitting
                            # C=1.0 is a balanced default
        max_iter=1000,      # Max optimization iterations
        solver='lbfgs',     # Optimization algorithm — good for small-medium data
        random_state=42     # Reproducibility — same results every run
    ),
}


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and labels into train and test sets.

    Rule of thumb:
    - 80% training (model learns from this)
    - 20% testing  (we evaluate on this — model has never seen it)

    stratify=y ensures EQUAL RATIO of spam/ham in both train and test sets.
    Without stratify, you might get 100% ham in test set by chance.

    Args:
        X: Feature matrix (TF-IDF output)
        y: Labels array (spam/ham or 0/1)
        test_size: Fraction for test set (0.2 = 20%)
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y              # Maintain class ratio in both splits
    )
    print(f"[SPLIT] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"[SPLIT] Train spam ratio: {y_train.mean():.1%} | Test spam ratio: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features so all values are in the same range [0, 1].
    Required for Logistic Regression to work properly.
    NOT needed for Naive Bayes (it uses probabilities internally).

    MaxAbsScaler is used because:
    - It preserves sparsity of TF-IDF matrices (important for memory)
    - It scales each feature by its max absolute value
    - It doesn't shift/center data (no negative values from TF-IDF)

    Args:
        X_train, X_test: Feature matrices

    Returns:
        X_train_scaled, X_test_scaled, fitted scaler
    """
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit ONLY on train
    X_test_scaled  = scaler.transform(X_test)        # Transform test with train's scale
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model and return all performance metrics.

    Metrics explained:
    - Accuracy   : % of total correct predictions (can be misleading if data is imbalanced)
    - Precision  : Of emails predicted SPAM, how many actually were? (avoid false alarms)
    - Recall     : Of actual SPAM emails, how many did we catch? (don't miss spam)
    - F1 Score   : Harmonic mean of precision and recall (best overall metric)
    - ROC-AUC    : Area under ROC curve — 1.0 is perfect, 0.5 is random

    For spam detection: HIGH RECALL is most important
    (better to flag a real email as spam than let spam through)

    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: True test labels
        model_name: Name for display

    Returns:
        dict: All computed metrics
    """
    y_pred = model.predict(X_test)

    # Get probability scores for ROC-AUC
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    metrics = {
        'model'    : model_name,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall'   : recall_score(y_test, y_pred, zero_division=0),
        'f1'       : f1_score(y_test, y_pred, zero_division=0),
        'auc'      : auc,
        'y_pred'   : y_pred,
    }

    print(f"\n{'='*55}")
    print(f"  {model_name} Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    if auc:
        print(f"  ROC-AUC   : {metrics['auc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Ham','Spam'])}")

    return metrics


def cross_validate_model(model, X_train, y_train, model_name: str, cv: int = 5):
    """
    Cross-validation — a more reliable evaluation method than a single train/test split.

    HOW IT WORKS:
    - Splits training data into 5 equal parts (folds)
    - Trains on 4 parts, tests on 1 part — repeated 5 times
    - Each fold gets to be the "test set" exactly once
    - Returns average score across all 5 folds

    WHY USE IT?
    - Single train/test split can be lucky or unlucky
    - Cross-validation gives a more stable, reliable estimate of real performance

    Args:
        model: sklearn model to cross-validate
        X_train, y_train: Training data
        model_name: Name for display
        cv: Number of folds (5 is standard)

    Returns:
        dict: Mean and std of CV scores
    """
    print(f"[CV] Running {cv}-fold cross-validation for {model_name}...")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    # n_jobs=-1 uses all available CPU cores for faster computation
    print(f"[CV] {model_name} F1 scores: {scores.round(3)}")
    print(f"[CV] Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")
    return {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}


def train_all_models(X_train, y_train) -> dict:
    """
    Train all models defined in MODELS dictionary.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        dict: {model_name: trained_model}
    """
    trained = {}
    for name, model in MODELS.items():
        print(f"\n[TRAIN] Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"[TRAIN] {name} ✅")
    return trained


def select_best_model(results: list, metric: str = 'f1') -> dict:
    """
    Compare all model results and select the best one.

    Args:
        results: List of metric dicts from evaluate_model()
        metric: Which metric to use for comparison (default: f1)

    Returns:
        dict: The metrics dict of the best model
    """
    best = max(results, key=lambda x: x[metric])
    print(f"\n🏆 BEST MODEL: {best['model']} with {metric.upper()}={best[metric]:.4f}")
    return best


def save_model(model, filepath: str):
    """
    Save a trained model to disk using joblib.
    joblib is better than pickle for large numpy arrays (like sklearn models).

    Args:
        model: Trained sklearn model
        filepath: Where to save (e.g., 'models/spam_classifier.pkl')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[SAVE] Model saved to: {filepath}")


def load_model(filepath: str):
    """
    Load a previously saved model from disk.

    Args:
        filepath: Path to the saved .pkl file

    Returns:
        Loaded sklearn model ready for predictions
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model not found at: {filepath}\n"
            "Run 'python run.py' first to train and save the model!"
        )
    model = joblib.load(filepath)
    print(f"[LOAD] Model loaded from: {filepath}")
    return model


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str = None):
    """
    Plot and optionally save the confusion matrix as an image.

    Confusion Matrix layout:
                  Predicted Ham    Predicted Spam
    Actual Ham  [ True Negative    False Positive ]
    Actual Spam [ False Negative   True Positive  ]

    True Negative  = Ham correctly identified ✅
    True Positive  = Spam correctly caught ✅
    False Positive = Ham wrongly flagged as spam ❌ (false alarm)
    False Negative = Spam missed, went to inbox ❌ (dangerous)
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Ham (Not Spam)', 'Spam'],
        yticklabels=['Ham (Not Spam)', 'Spam'],
        linewidths=1, annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results: list, save_path: str = None):
    """
    Bar chart comparing all models on multiple metrics side by side.
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    model_names     = [r['model'] for r in results]
    n_models        = len(model_names)
    n_metrics       = len(metrics_to_plot)
    x               = np.arange(n_metrics)
    width           = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    colors  = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, result in enumerate(results):
        values = [result[m] for m in metrics_to_plot]
        bars   = ax.bar(x + i * width, values, width, label=result['model'],
                        color=colors[i], edgecolor='black', alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Model comparison chart saved to: {save_path}")
