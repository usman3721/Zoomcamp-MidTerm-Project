import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

# -----------------------------
# Helper functions
# -----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'[\/`!@#$%^&*()_+{}<>,.?/":;0-9]', ' ', text)
    return text.lower()

def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    df["cleaned_data"] = df["Text"].apply(clean_text)
    df = df.drop(columns=["Text"])
    return df

# -----------------------------
# Training
# -----------------------------
def train_model(data_path: str, model_dir: str = "models"):
    df = load_and_prepare_data(data_path)
    
    X = df["cleaned_data"].values
    y = df["language"].values
    
    # 60/20/20 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=29)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=29)
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,3), analyzer='char_wb', min_df=10)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train_encoded)
    
    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, "tfidf.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    print("✅ Training complete. Model, vectorizer, and label encoder saved.")
    return model, vectorizer, label_encoder

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_model(r"\Users\DELL\Desktop\ML ZoomCamp\Mid_Term_Project\Data\dataset.csv")
