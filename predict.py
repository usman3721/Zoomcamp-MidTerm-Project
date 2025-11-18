import pickle
from typing import List

# -----------------------------
# Load model artifacts
# -----------------------------
def load_components(model_dir: str = "models"):
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

# -----------------------------
# Prediction
# -----------------------------
def predict(texts: List[str], model, vectorizer, label_encoder):
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    labels = label_encoder.inverse_transform(preds)
    return labels

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    model, vectorizer, label_encoder = load_components()
    
    texts = ["Hello world", "Bonjour tout le monde"]
    predictions = predict(texts, model, vectorizer, label_encoder)
    for text, pred in zip(texts, predictions):
        print(f"{text} --> {pred}")
