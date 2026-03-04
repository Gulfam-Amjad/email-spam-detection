# 📧 Spam Email Classifier
### Production-Level NLP Project | Python + Scikit-learn + Streamlit

**💻 Live Demo:** [Click here to try the app online](https://email-spam-detection-6buvueq6rcqp2xjtp7o2uo.streamlit.app)


# 📧 Spam Email Classifier
### Production-Level NLP Project | Python + Scikit-learn + Streamlit

---

## 🗂️ Project Structure
```
spam-email-classifier/
│
├── data/
│   ├── raw/                        ← Put spam.csv here (from Kaggle)
│   │   └── spam.csv
│   └── processed/                  ← Auto-generated after running run.py
│       └── cleaned_data.csv
│
├── notebooks/                      ← For experiments (optional)
│   └── 01_EDA_and_Experiments.ipynb
│
├── src/                            ← Core ML engine (Python modules)
│   ├── __init__.py
│   ├── data_preprocessing.py       ← Load + clean text data
│   ├── feature_engineering.py      ← TF-IDF vectorization
│   ├── model_training.py           ← Train, evaluate, save models
│   └── predict.py                  ← Prediction functions for the app
│
├── models/                         ← Auto-generated saved model files
│   ├── tfidf_vectorizer.pkl        ← Saved TF-IDF vocabulary
│   ├── spam_classifier.pkl         ← Saved trained model
│   ├── confusion_matrix.png        ← Model evaluation chart
│   └── model_comparison.png        ← Model comparison chart
│
├── app/
│   └── streamlit_app.py            ← Web application (run this!)
│
├── tests/
│   └── test_predict.py             ← Unit tests
│
├── run.py                          ← Master training script
├── requirements.txt                ← All Python dependencies
└── README.md                       ← This file
```

---

## ⚡ QUICK START — Run the Project in 4 Steps

### Step 1 — Get the Dataset (2 options)

**Option A: Download manually (Beginner — Recommended)**
1. Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Create a free Kaggle account if you don't have one
3. Click the blue **Download** button
4. Unzip the file
5. Copy `spam.csv` into `data/raw/spam.csv`

**Option B: Download via terminal (Professional)**
```bash
pip install kaggle
# Put your kaggle.json API key in ~/.kaggle/kaggle.json
kaggle datasets download -d uciml/sms-spam-collection-dataset
unzip sms-spam-collection-dataset.zip -d data/raw/
```

---

### Step 2 — Install Dependencies
```bash
# Navigate to the project folder
cd spam-email-classifier

# Install all required libraries
pip install -r requirements.txt
```

---

### Step 3 — Train the Model
```bash
# Option A: With real Kaggle dataset (recommended)
python run.py --data data/raw/spam.csv

# Option B: Without dataset (uses built-in sample data for testing)
python run.py
```

**What this does:**
- Cleans and preprocesses all emails
- Trains Naive Bayes and Logistic Regression models
- Evaluates and compares both models
- Saves the best model to `models/spam_classifier.pkl`
- Saves the vectorizer to `models/tfidf_vectorizer.pkl`
- Saves evaluation charts to `models/`

**Expected output:**
```
============================================================
   SPAM EMAIL CLASSIFIER — TRAINING PIPELINE
============================================================
[STEP 1/6] Loading Data...
[DATA] Loaded 5572 messages ({'ham': 4825, 'spam': 747})
[STEP 2/6] Preprocessing Text...
[STEP 3/6] Building TF-IDF Features...
[FEATURES] Vocabulary size: 5000 words
[STEP 4/6] Training Models...
[STEP 5/6] Evaluating Models...
  Naive Bayes        Accuracy: 98.12%
  Logistic Regression Accuracy: 98.74%
🏆 BEST MODEL: Logistic Regression with F1=0.9641
[STEP 6/6] Saving Model Files...
✅ TRAINING COMPLETE! Time: 8.3 seconds
```

---

### Step 4 — Launch the Web App
```bash
streamlit run app/streamlit_app.py
```

Then open your browser and go to:
```
http://localhost:8501
```

**That's it! 🎉 Your app is running.**

---

## 🧪 Run Unit Tests
```bash
# Using pytest (recommended)
pip install pytest
python -m pytest tests/ -v

# Without pytest
python tests/test_predict.py
```

---

## 📁 What Each File Does

| File | Purpose |
|------|---------|
| `src/data_preprocessing.py` | Load CSV, clean text (lowercase, remove stopwords, etc.) |
| `src/feature_engineering.py` | TF-IDF vectorization — converts text to numbers |
| `src/model_training.py` | Train models, evaluate metrics, save/load model files |
| `src/predict.py` | Single function `predict_email()` used by the app |
| `run.py` | Runs the full training pipeline with one command |
| `app/streamlit_app.py` | The web application users interact with |
| `tests/test_predict.py` | Automated tests to verify functions work correctly |
| `requirements.txt` | List of all Python packages needed |

---

## 🔄 How the System Works (Data Flow)

```
1. User downloads spam.csv from Kaggle
         ↓
2. python run.py --data data/raw/spam.csv
         ↓
3. data_preprocessing.py    → cleans text
         ↓
4. feature_engineering.py   → TF-IDF: text → numbers
         ↓
5. model_training.py        → trains + evaluates 2 models
         ↓
6. models/spam_classifier.pkl + tfidf_vectorizer.pkl (saved)
         ↓
7. streamlit run app/streamlit_app.py
         ↓
8. User types email → predict.py loads .pkl files → returns SPAM/HAM
```

---

## 🤖 Models Used & Why

### Naive Bayes (MultinomialNB)
- Best for text classification tasks
- Very fast to train (< 1 second)
- Works well even with small datasets
- Used in real email spam filters historically

### Logistic Regression ← FINAL DEPLOYED MODEL
- More accurate than Naive Bayes on this dataset
- Interpretable — we can see which words cause spam flag
- Industry-standard for binary text classification
- 97-99% accuracy on the Kaggle dataset

---

## 📊 Expected Results (with Kaggle Dataset)

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Naive Bayes | ~97-98% | ~93-95% |
| Logistic Regression | ~98-99% | ~96-97% |

---

## 🔧 Troubleshooting

**Error: "Model not found"**
```bash
# You need to train the model first
python run.py
```

**Error: "Module not found"**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**Error: "Port already in use"**
```bash
# Run on a different port
streamlit run app/streamlit_app.py --server.port 8502
```

**Error reading spam.csv**
```bash
# Make sure the file is in the right place
ls data/raw/        # Should show spam.csv
```

---

## 🚀 Future Improvements (Next Steps)

1. Use BERT transformer model for higher accuracy
2. Add email header analysis (From, Subject features)
3. Deploy to Streamlit Cloud (free hosting) — share with anyone
4. Add a feedback button — let users flag wrong predictions
5. Add support for multiple languages

---

## 📚 Learning Resources

- Scikit-learn docs: https://scikit-learn.org
- Streamlit docs: https://docs.streamlit.io
- Kaggle dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- TF-IDF explanation: https://en.wikipedia.org/wiki/Tf–idf

---

*This project is part of an NLP learning roadmap. Build it, understand it, then improve it.*
