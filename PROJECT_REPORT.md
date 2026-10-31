API RESPONSE-TIME PREDICTION SYSTEM USING MACHINE LEARNING

1. Abstract

The API Response-Time Prediction System is a machine learning-based academic project designed to estimate the response time of API requests and classify API performance. It uses Random Forest Regression to predict response time in milliseconds and Random Forest Classification to categorize performance as Fast, Normal, Slow, or Critical. A Streamlit dashboard allows users to enter request details and view predictions, evaluation metrics, and feature importance. The system is a prototype developed using sample server-log data.

2. Problem Statement

Slow API responses can affect application performance and user experience. This project explores how machine learning can estimate response time and identify potentially slow requests using available server-log features.

3. Objectives

- Predict API response time.
- Classify API performance into four categories.
- Evaluate model performance using suitable metrics.
- Display predictions and analytics through an interactive dashboard.

4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Random Forest Regression
- Random Forest Classification

5. Dataset

The project uses a sample server-log dataset containing approximately 5,000 records and 10 columns. Features used by the models include server ID, HTTP method, API path, and bytes sent. The target for regression is response time in milliseconds. Classification labels are derived from response-time thresholds.

6. System Methodology

1. Load the server-log dataset.
2. Prepare input features and target values.
3. Encode categorical features using One-Hot Encoding.
4. Split the data into training and testing sets.
5. Train Random Forest Regression and Classification models.
6. Evaluate the models.
7. Save trained models and evaluation results.
8. Display predictions and charts in a Streamlit dashboard.

7. Evaluation Results

Record the values from the final run of "train_model.py".

- Regression MAE: 178.03 ms (previous recorded run)
- Regression RMSE: 304.59 ms (previous recorded run)
- Regression R²: -0.445 (previous recorded run)
- Classification accuracy: 52.5% (previous run with the original category thresholds)

These are historical results from different runs and must not be presented as one final evaluation. Replace them with the matching values from the final run before submission. The later experiment using different category thresholds achieved 35.2% classification accuracy.

8. Limitations

- The model's current predictive performance needs improvement.
- Sample server logs may not represent real production traffic.
- Feature importance indicates model reliance, not necessarily causation.
- Predictions are estimates and should not be treated as production service-level guarantees.

9. Future Enhancements

- Train and validate using representative real API monitoring data.
- Improve features and compare multiple machine learning algorithms.
- Tune model parameters and evaluate class imbalance.
- Add real-time API monitoring and alerting.
- Deploy the application in a suitable hosted environment.

10. Conclusion

This project demonstrates an end-to-end machine learning prototype for API response-time estimation and performance classification. It includes data preprocessing, model training, evaluation, saved models, and an interactive dashboard. Further validation and model improvement are necessary before real-world deployment.