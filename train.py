import pandas as pd
import numpy as np
import os
import joblib
import json
import torch
import warnings
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
from collections import Counter

# --- NEW IMPORTS FOR SMOTE ---
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
except ImportError:
    print("❌ Error: 'imbalanced-learn' is not installed.")
    print("   Please run: pip install imbalanced-learn")
    exit()

warnings.filterwarnings("ignore")

# --- SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# --- LOAD DATA ---
def load_data():
    print("🔄 Loading Dataset...")
    file_path = os.path.join(os.path.dirname(__file__), "amazon_reviews.txt")

    try:
        df = pd.read_csv(file_path,
                         sep='\t',
                         engine='python',
                         on_bad_lines='skip')
    except FileNotFoundError:
        print(f"❌ Error: '{file_path}' not found. Please make sure the file is in the project folder.")
        return None

    print(f"✅ Raw Dataset Loaded: {len(df)} rows (malformed lines skipped)")

    # Map labels: Fake=0, Real=1
    df['target'] = df['LABEL'].map({'__label1__': 0, '__label2__': 1})

    # Feature Engineering
    df['is_verified'] = df['VERIFIED_PURCHASE'].map({'Y': 1, 'N': 0, 'y': 1, 'n': 0}).fillna(0)
    df['review_len'] = df['REVIEW_TEXT'].astype(str).fillna('').apply(lambda x: len(x.split()))
    df['rating'] = pd.to_numeric(df['RATING'], errors='coerce').fillna(3)

    print(f"✅ Full Dataset Processed: {len(df)} rows ready for training")
    return df

# --- SKLEARN MODELS ---
def train_sklearn_models(X_train_df, X_test_df, y_train, y_test):
    print("\n🚀 Training Traditional Models (with Oversampling + Metadata)...")

    # Vectorize Text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_df['REVIEW_TEXT'].astype(str).fillna(''))
    X_test_tfidf = vectorizer.transform(X_test_df['REVIEW_TEXT'].astype(str).fillna(''))

    # Combine Metadata
    meta_cols = ['is_verified', 'rating', 'review_len']
    X_train_meta = X_train_df[meta_cols].values
    X_test_meta = X_test_df[meta_cols].values

    X_train_combined = hstack([X_train_tfidf, X_train_meta])
    X_test_combined = hstack([X_test_tfidf, X_test_meta])

    # Convert to numpy 221
    X_train_array = np.asarray(X_train_combined.todense())
    X_test_array = np.asarray(X_test_combined.todense())

    # Check class distribution
    class_counts = Counter(y_train)
    print("   > Class distribution in y_train:", class_counts)

    # Apply oversampling only if more than 1 class exists
    if len(class_counts) > 1:
        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_resample(X_train_array, y_train)
        print(f"     -> Resampled Size: {X_train_res.shape[0]}")
    else:
        print("⚠️ Only one class detected. Skipping oversampling.")
        X_train_res, y_train_res = X_train_array, y_train

    # Train models
    results = {}
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'Linear SVM': LinearSVC(dual=False, max_iter=3000),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB()
    }

    for name, model in models.items():
        print(f"   > Training {name}...")
        model.fit(X_train_res, y_train_res)
        acc = accuracy_score(y_test, model.predict(X_test_array))
        results[name] = {'model': model, 'accuracy': acc, 'vectorizer': vectorizer, 'type': 'sklearn'}
        print(f"     Accuracy: {acc*100:.2f}%")

    return results

# --- BERT MODEL ---
class AmazonReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def train_bert_model(X_train_text, X_test_text, y_train, y_test):
    print("\n🚀 Training DistilBERT (Deep Learning with Balancing)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   > Using Device: {device}")

    X_train_text = pd.Series(X_train_text).astype(str).fillna('').tolist()
    X_test_text = pd.Series(X_test_text).astype(str).fillna('').tolist()

    class_counts = Counter(y_train)
    if len(class_counts) > 1:
        ros = RandomOverSampler(random_state=42)
        X_train_np = np.array(X_train_text).reshape(-1, 1)
        X_train_res, y_train_res = ros.fit_resample(X_train_np, y_train)
        X_train_res = X_train_res.flatten().tolist()
        print(f"     -> Training Samples after balancing: {len(X_train_res)}")
    else:
        print("⚠️ Only one class detected. Skipping oversampling for BERT.")
        X_train_res, y_train_res = X_train_text, y_train

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_enc = tokenizer(X_train_res, truncation=True, padding=True, max_length=128)
    test_enc = tokenizer(X_test_text, truncation=True, padding=True, max_length=128)

    train_dataset = AmazonReviewDataset(train_enc, y_train_res)
    test_dataset = AmazonReviewDataset(test_enc, y_test)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

    args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=(device == "cuda"),
        logging_steps=50,
        report_to="none"
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()
    metrics = trainer.evaluate()
    acc = metrics['eval_accuracy']
    print(f"     Accuracy: {acc*100:.2f}%")

    return {
        'BERT': {
            'model': model,
            'tokenizer': tokenizer,
            'accuracy': acc,
            'type': 'bert'
        }
    }

# --- MAIN DRIVER ---
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        y = df['target'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)

        sklearn_results = train_sklearn_models(X_train, X_test, y_train, y_test)

        try:
            bert_results = train_bert_model(
                X_train['REVIEW_TEXT'].astype(str).tolist(),
                X_test['REVIEW_TEXT'].astype(str).tolist(),
                y_train, y_test
            )
        except Exception as e:
            print(f"⚠️ BERT Training skipped due to error: {e}")
            bert_results = {}

        # Pick champion
        all_models = {**sklearn_results, **bert_results}
        best_name = max(all_models, key=lambda k: all_models[k]['accuracy'])
        champion = all_models[best_name]

        print(f"\n🏆 AND THE WINNER IS: {best_name} ({champion['accuracy']*100:.2f}%)")

        # Save champion
        config = {'model_type': champion['type'], 'accuracy': champion['accuracy'], 'name': best_name}
        if champion['type'] == 'sklearn':
            joblib.dump(champion['model'], os.path.join(MODEL_DIR, 'champion_model.pkl'))
            joblib.dump(champion['vectorizer'], os.path.join(MODEL_DIR, 'champion_vectorizer.pkl'))
        else:
            bert_dir = os.path.join(MODEL_DIR, 'champion_bert')
            champion['model'].save_pretrained(bert_dir)
            champion['tokenizer'].save_pretrained(bert_dir)

        with open(os.path.join(MODEL_DIR, 'model_config.json'), 'w') as f:
            json.dump(config, f)

        print("💾 Best model saved to 'saved_models/'. Run 'python app.py' to start the app.")