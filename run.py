# =============================================================================
# run.py
# PURPOSE: ONE command to rule them all
# Run this once to train everything and save the model files
#
# USAGE:
#   python run.py                     ← uses built-in sample data
#   python run.py --data data/raw/spam.csv  ← uses your Kaggle dataset
# =============================================================================

import os
import sys
import argparse
import time

# Add project root to Python path so 'src' imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import (
    load_raw_data, create_sample_data,
    preprocess_dataframe, save_processed_data
)
from src.feature_engineering import (
    build_tfidf_vectorizer, fit_vectorizer,
    transform_texts, save_vectorizer
)
from src.model_training import (
    split_data, scale_features, train_all_models,
    evaluate_model, select_best_model, save_model,
    plot_confusion_matrix, plot_model_comparison, cross_validate_model
)


def main(data_path: str = None):
    start_time = time.time()

    print("\n" + "=" * 60)
    print("   SPAM EMAIL CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1: LOAD DATA
    # ----------------------------------------------------------------
    print("\n[STEP 1/6] Loading Data...")

    if data_path and os.path.exists(data_path):
        df = load_raw_data(data_path)
        print(f"✅ Loaded real dataset: {data_path}")
    else:
        print("⚠️  No data path provided. Using built-in sample data.")
        print("   To use real data: python run.py --data data/raw/spam.csv")
        df = create_sample_data()

    # ----------------------------------------------------------------
    # STEP 2: PREPROCESS
    # ----------------------------------------------------------------
    print("\n[STEP 2/6] Preprocessing Text...")
    df = preprocess_dataframe(df)

    # Save cleaned data
    processed_path = os.path.join("data", "processed", "cleaned_data.csv")
    save_processed_data(df, processed_path)
    print(f"✅ Preprocessed {len(df)} messages")

    # ----------------------------------------------------------------
    # STEP 3: FEATURE ENGINEERING (TF-IDF)
    # ----------------------------------------------------------------
    print("\n[STEP 3/6] Building TF-IDF Features...")

    # Labels — use encoded (0/1) for sklearn
    y = df['label_encoded']

    # Split BEFORE vectorizing — prevents data leakage
    # We split on indices, then fit TF-IDF on training text only
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )

    train_df = df.loc[train_idx]
    test_df  = df.loc[test_idx]
    y_train  = y.loc[train_idx]
    y_test   = y.loc[test_idx]

    # Build and fit TF-IDF ONLY on training data
    vectorizer = build_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2))
    vectorizer = fit_vectorizer(vectorizer, train_df['cleaned_message'])

    # Transform both sets using the fitted vectorizer
    X_train = transform_texts(vectorizer, train_df['cleaned_message'])
    X_test  = transform_texts(vectorizer, test_df['cleaned_message'])

    print(f"✅ TF-IDF matrix: Train {X_train.shape} | Test {X_test.shape}")

    # ----------------------------------------------------------------
    # STEP 4: TRAIN MODELS
    # ----------------------------------------------------------------
    print("\n[STEP 4/6] Training Models...")
    trained_models = train_all_models(X_train, y_train)
    print(f"✅ Trained {len(trained_models)} models")

    # ----------------------------------------------------------------
    # STEP 5: EVALUATE & COMPARE MODELS
    # ----------------------------------------------------------------
    print("\n[STEP 5/6] Evaluating Models...")

    all_results = []
    for name, model in trained_models.items():
        result = evaluate_model(model, X_test, y_test, name)
        result['model_obj'] = model
        all_results.append(result)

    # Cross-validation for reliable estimate
    print("\n[CV] Running Cross-Validation...")
    for name, model in trained_models.items():
        cross_validate_model(model, X_train, y_train, name, cv=5)

    # Pick the best model
    best_result = select_best_model(all_results, metric='f1')
    best_model  = best_result['model_obj']

    # Save charts
    os.makedirs("models", exist_ok=True)

    # Confusion matrix for best model
    plot_confusion_matrix(
        y_test, best_result['y_pred'],
        best_result['model'],
        save_path=os.path.join("models", "confusion_matrix.png")
    )

    # Model comparison chart
    plot_model_comparison(
        all_results,
        save_path=os.path.join("models", "model_comparison.png")
    )

    # ----------------------------------------------------------------
    # STEP 6: SAVE BEST MODEL + VECTORIZER
    # ----------------------------------------------------------------
    print("\n[STEP 6/6] Saving Model Files...")

    # Save vectorizer (the fitted vocabulary)
    save_vectorizer(vectorizer, os.path.join("models", "tfidf_vectorizer.pkl"))

    # Save the best model
    # Convert best_model to use string labels for the app
    # We retrain on full data with string labels for cleaner app predictions
    final_model = trained_models[best_result['model']]

    # Refit on string labels so .predict() returns 'spam'/'ham' not 0/1
    full_texts     = df['cleaned_message']
    full_labels    = df['label']             # 'spam' / 'ham' strings
    X_full         = transform_texts(vectorizer, full_texts)
    final_model.fit(X_full, full_labels)     # Retrain on ALL data with string labels

    save_model(final_model, os.path.join("models", "spam_classifier.pkl"))

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("   ✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"""
  Dataset Size : {len(df)} messages
  Train Size   : {len(train_idx)} samples
  Test Size    : {len(test_idx)} samples
  Best Model   : {best_result['model']}
  Accuracy     : {best_result['accuracy'] * 100:.2f}%
  F1 Score     : {best_result['f1']:.4f}
  Time Taken   : {elapsed:.1f} seconds

  Files Saved:
    📄 data/processed/cleaned_data.csv
    🔧 models/tfidf_vectorizer.pkl
    🤖 models/spam_classifier.pkl
    📊 models/confusion_matrix.png
    📊 models/model_comparison.png

  Next Step:
    👉 streamlit run app/streamlit_app.py
""")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the Spam Email Classifier")
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help="Path to dataset CSV file (e.g., data/raw/spam.csv)"
    )
    args = parser.parse_args()
    main(data_path=args.data)
