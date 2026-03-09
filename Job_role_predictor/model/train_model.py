import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# -----------------------------
# Step 1: Set paths correctly
# -----------------------------
# Get the current script's folder
current_folder = os.path.dirname(os.path.abspath(__file__))

# Dataset path
dataset_path = os.path.join(current_folder, "../dataset/jobs.csv")

# Model save paths
model_path = os.path.join(current_folder, "job_model.pkl")
vectorizer_path = os.path.join(current_folder, "tfidf_vectorizer.pkl")

# -----------------------------
# Step 2: Load dataset
# -----------------------------
data = pd.read_csv(dataset_path)
print("Dataset loaded:")
print(data.head())

# -----------------------------
# Step 3: Prepare features
# -----------------------------
X = data['skills']
y = data['job_role']

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)
print("TF-IDF shape:", X_vectors.shape)

# -----------------------------
# Step 4: Train Naive Bayes model
# -----------------------------
model = MultinomialNB()
model.fit(X_vectors, y)
print("Model trained successfully!")

# -----------------------------
# Step 5: Save model & vectorizer
# -----------------------------
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
print("Model and vectorizer saved successfully!")