# =============================================================================
# src/data_preprocessing.py
# PURPOSE: Load raw data and clean all text messages
# This file is responsible for ONLY one thing: cleaning text
# In real projects, each file has ONE responsibility (called Single Responsibility Principle)
# =============================================================================

import re
import string
import pandas as pd
import os

# -------------------------------------------------------
# STOPWORDS — Common words that carry no useful meaning
# In real projects with NLTK available, use:
#   from nltk.corpus import stopwords
#   STOPWORDS = set(stopwords.words('english'))
# We define them manually here for zero-dependency running
# -------------------------------------------------------
STOPWORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','is','it',
    'its','that','this','was','are','be','been','have','has','had','do','does',
    'did','will','would','could','should','may','might','shall','can','need',
    'you','your','we','our','they','their','i','my','me','him','his','her',
    'he','she','us','them','what','which','who','how','when','where','why',
    'not','no','so','if','as','with','by','from','up','out','about','into',
    'then','than','just','also','am','were','any','all','more','some','such',
    'only','own','same','too','very','here','there','now','get','go','got',
    'let','see','say','said','know','think','come','want','use','make','take'
])


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset into a pandas DataFrame.

    SUPPORTS TWO FORMATS:
    1. Kaggle SMS Spam Collection (spam.csv)
       Columns: v1 (label), v2 (message) + extra unnamed columns
    2. Simple format with columns already named 'label' and 'message'

    Args:
        filepath (str): Full path to the CSV file

    Returns:
        pd.DataFrame: DataFrame with exactly two columns: 'label' and 'message'
    """
    print(f"[DATA] Loading data from: {filepath}")

    # Read CSV with latin-1 encoding (Kaggle spam.csv needs this)
    # latin-1 handles special characters that utf-8 might crash on
    df = pd.read_csv(filepath, encoding='latin-1')

    # --- Handle Kaggle format (columns named v1, v2) ---
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]                    # Keep only label and message columns
        df.columns = ['label', 'message']         # Rename to standard names

    # --- Handle simple format (already has label, message) ---
    elif 'label' in df.columns and 'message' in df.columns:
        df = df[['label', 'message']]

    else:
        raise ValueError(
            f"Unexpected columns: {df.columns.tolist()}. "
            "Expected 'v1'/'v2' (Kaggle format) or 'label'/'message' columns."
        )

    # Drop any rows where label or message is missing (NaN)
    df.dropna(subset=['label', 'message'], inplace=True)

    # Remove duplicate messages (same email appearing multiple times)
    df.drop_duplicates(subset=['message'], inplace=True)

    # Reset index after dropping rows (so index goes 0, 1, 2, ... again)
    df.reset_index(drop=True, inplace=True)

    print(f"[DATA] Loaded {len(df)} messages ({df['label'].value_counts().to_dict()})")
    return df


def create_sample_data() -> pd.DataFrame:
    """
    Create a built-in sample dataset for testing/demo purposes.
    Use this when you don't have the Kaggle dataset yet.
    Replace with load_raw_data() once you have spam.csv

    Returns:
        pd.DataFrame: Sample DataFrame with 'label' and 'message' columns
    """
    data = {
        'label': ['spam'] * 20 + ['ham'] * 20,
        'message': [
            # --- SPAM emails ---
            "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
            "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
            "You have been selected for a cash prize of $500. Call 08701417779 to claim!",
            "URGENT: Your account has been compromised. Verify immediately at this link.",
            "Buy cheap medicines online! No prescription needed. 90% off today only!",
            "You are a lucky winner! Claim your free iPhone 15 now. Offer expires tonight.",
            "Hot singles in your area are waiting! Click here to meet them now.",
            "Get rich quick! Earn $5000/week from home. No experience needed. Apply now!",
            "Your loan has been APPROVED. $10,000 deposited today. Call 0800-LOAN-NOW.",
            "Win big at our online casino! 100% bonus on your first deposit today.",
            "Earn money from home! Join our network now and make thousands monthly.",
            "Your package delivery failed. Pay $2.99 to reschedule. Click link now.",
            "You won the national lottery! Send bank details to claim your prize immediately.",
            "SPECIAL DEAL: Buy 2 get 10 FREE. Order in next 10 minutes. Call 0800-DEAL.",
            "Your credit card has been suspended. Call urgently to reactivate your account.",
            "Claim your FREE holiday voucher! Text YES to 88600. T&C apply. £3/msg.",
            "SIX chances to win CASH! From £100 to £20,000. Text WIN to 89010 now.",
            "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & FREE.",
            "Get paid £350 for every survey completed. Register at surveys4cash.com today!",
            "PRIVATE! Your 2003 Account Statement shows 800 un-redeemed S&H greenpoints.",
            # --- HAM emails ---
            "Hey, are you coming to the office tomorrow? Team meeting at 10am.",
            "Can you send me the project report by end of day? Thanks so much.",
            "Mom wants to know if you're joining us for dinner this Sunday evening.",
            "The flight is delayed by 2 hours. We land at 8pm instead of 6pm.",
            "Happy birthday! Hope you have a wonderful day with your family.",
            "I reviewed your resume. Let's schedule an interview for next week.",
            "The meeting is rescheduled to Thursday 3pm. Please update your calendar.",
            "Can you recommend a good book to read this weekend? Looking for suggestions.",
            "Your order has shipped. Expected delivery in 3-5 business days.",
            "Please find the attached invoice for services rendered in January.",
            "The project deadline has been extended to next Friday. Great news!",
            "Thanks for coming to the event yesterday. It was great meeting you.",
            "Just checking in to see how you are doing. It's been a while!",
            "I will be 15 minutes late for the meeting. Please start without me.",
            "New policy documents uploaded to the shared drive. Please review.",
            "The code review comments have been addressed. Ready for another look?",
            "Lunch at 1pm? The new Italian place near the office looks great.",
            "Your bank statement for November is ready to view in your online account.",
            "Don't forget to submit your timesheet by Friday 5pm. HR reminder.",
            "The conference call has been moved to 2pm due to a scheduling conflict.",
        ]
    }
    df = pd.DataFrame(data)
    print(f"[DATA] Using built-in sample data: {len(df)} messages")
    return df


def clean_text(text: str) -> str:
    """
    Clean a single email message through 5 steps.

    Step 1: Lowercase everything         "Hello WORLD" → "hello world"
    Step 2: Remove URLs                  "visit http://spam.com" → "visit"
    Step 3: Remove special chars/digits  "Win $100!" → "Win "
    Step 4: Tokenize (split to words)    "win now" → ["win", "now"]
    Step 5: Remove stopwords + rejoin    ["win", "the", "now"] → "win now"

    Args:
        text (str): Raw email message string

    Returns:
        str: Cleaned, processed text string
    """
    # Guard against non-string input (NaN, numbers, etc.)
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs (http://... or www....)
    # re.sub replaces pattern matches with empty string
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Remove everything that's NOT a letter or space
    # [^a-z\s] = "not (a-z or whitespace)"
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 4: Split into words
    words = text.split()

    # Step 5: Filter out stopwords and very short words (< 3 chars)
    cleaned = [w for w in words if w not in STOPWORDS and len(w) > 2]

    return ' '.join(cleaned)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the full DataFrame.

    Adds new columns:
    - 'cleaned_message': cleaned version of original message
    - 'message_length': character count of original message
    - 'word_count': word count of original message
    - 'label_encoded': 0 for ham, 1 for spam (for ML models)

    Args:
        df (pd.DataFrame): Raw DataFrame with 'label' and 'message' columns

    Returns:
        pd.DataFrame: Preprocessed DataFrame with extra feature columns
    """
    print("[PREPROCESS] Cleaning text messages...")

    # Apply clean_text to every row in 'message' column
    df['cleaned_message'] = df['message'].apply(clean_text)

    # Feature engineering — extra numerical features
    df['message_length'] = df['message'].apply(len)          # char count
    df['word_count']     = df['message'].apply(lambda x: len(x.split()))  # word count
    df['has_number']     = df['message'].apply(lambda x: int(bool(re.search(r'\d', x))))
    df['has_currency']   = df['message'].apply(lambda x: int(bool(re.search(r'[$£€]', x))))
    df['uppercase_ratio']= df['message'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )   # Ratio of uppercase letters — spam uses CAPS a lot

    # Encode labels: ham=0, spam=1
    df['label_encoded'] = (df['label'] == 'spam').astype(int)

    # Remove rows where cleaned_message is empty after cleaning
    df = df[df['cleaned_message'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    print(f"[PREPROCESS] Done. Shape: {df.shape}")
    return df


def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Save the cleaned DataFrame to CSV for later reuse.
    This avoids re-cleaning data every time you run the project.

    Args:
        df (pd.DataFrame): Processed DataFrame
        output_path (str): Where to save the CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Processed data saved to: {output_path}")
