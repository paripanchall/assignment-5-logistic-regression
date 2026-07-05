Airbnb Superhost Prediction

Overview
  A machine learning pipeline that trains, tunes, and evaluates a Logistic Regression model to predict whether an Airbnb host is a "super host" (a binary classification problem).

Key Highlights
  Hyperparameter Tuning: Used GridSearchCV to find the optimal regularization parameter (C).
  Evaluation: Analyzed model performance across various thresholds using Confusion Matrices, Precision-Recall curves, and ROC/AUC curves.
  Feature Selection: Extracted the top 5 most impactful features using SelectKBest.
  Deployment: Packaged the best-performing model into a .pkl file for persistent, out-of-the-box predictions.

Tech Stack
  Python, Scikit-learn, Pandas, Matplotlib, Seaborn

How to Use the Saved Model
import pickle

# Load the model
with open('my_best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions on new data
predictions = loaded_model.predict(X_test)
