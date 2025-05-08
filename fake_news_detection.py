import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
def load_data(file_path):
    # Example dataset with 'text' and 'label' columns
    data = pd.read_csv(file_path)
    return data['text'], data['label']

# Preprocess text (basic example)
def preprocess_text(texts):
    return texts.str.lower().str.replace(r'\W', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

# Feature extraction
def extract_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

# Train model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Predict and assign scores
def predict_and_score(model, X):
    probabilities = model.predict_proba(X)[:, 1]  # Fake news probability
    return probabilities

# Fake news detection pipeline
def fake_news_pipeline(file_path):
    # Step 1: Load data
    texts, labels = load_data(file_path)

    # Step 2: Preprocess text
    texts = preprocess_text(texts)

    # Step 3: Train/test split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Step 4: Feature extraction
    X_train, X_test, vectorizer = extract_features(X_train_texts, X_test_texts)

    # Step 5: Train model
    model = train_model(X_train, y_train)

    # Step 6: Predict and score
    scores = predict_and_score(model, X_test)

    # Step 7: Evaluate
    predictions = (scores > 0.5).astype(int)
    print("Classification Report:\n", classification_report(y_test, predictions))
    
    return scores

# Run the pipeline
if __name__ == "__main__":
    file_path = "fake_news_dataset.csv"  # Replace with your dataset path
    scores = fake_news_pipeline(file_path)
    print("Fake News Scores:\n", scores)