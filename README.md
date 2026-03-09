# AI Job Role Predictor

This project predicts suitable job roles based on skills using Machine Learning.

## Features
- Skill-based job prediction
- Machine Learning model using TF-IDF
- Streamlit web interface

## Tech Stack
- Python
- Scikit-learn
- Pandas
- Streamlit

## Project Structure
Job_role_predictor/
 ├── app/predict.py
 ├── dataset/jobs.csv
 ├── model/job_model.pkl
 ├── model/tfidf_vectorizer.pkl
 └── job_role_app.py

## Run the Project
pip install -r requirements.txt
streamlit run job_role_app.py
