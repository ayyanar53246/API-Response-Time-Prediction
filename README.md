
# API Response-Time Prediction System

## Overview
An ML-based system that predicts API response time and classifies API performance using Regression and Classification.

## Machine Learning Models
- Random Forest Regressor
- Random Forest Classifier

## Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

## Dataset
Synthetic web server access logs containing 5,000 records.

## Model Evaluation
- Regression MAE: 174.49 ms
- Regression RMSE: 284.77 ms
- Classification Accuracy: 64%

Note: Classification performance varies by category. Further improvements are required for minority classes.

## Features
- API response-time prediction
- Performance classification
- Interactive Streamlit dashboard
- API analytics and visualizations

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Business Use Case
Helps e-commerce and SaaS development teams identify potentially slow API requests and investigate application performance.
