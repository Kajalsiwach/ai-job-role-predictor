import joblib
import os

# -----------------------------
# Step 1: Paths
# -----------------------------
current_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_folder, "../model/job_model.pkl")
vectorizer_path = os.path.join(current_folder, "../model/tfidf_vectorizer.pkl")

# -----------------------------
# Step 2: Load model and vectorizer
# -----------------------------
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# -----------------------------
# Step 3: Get user input
# -----------------------------
skills_input = input("Enter your skills (comma separated): ")

# -----------------------------
# Step 4: Preprocess input
# -----------------------------
# Convert to lowercase and remove extra spaces
skills_input = skills_input.lower().strip()

# -----------------------------
# Step 5: Convert input & predict
# -----------------------------
skills_vector = vectorizer.transform([skills_input])
predicted_role = model.predict(skills_vector)

print("Predicted Job Role:", predicted_role[0])
