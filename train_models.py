
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("access_logs.csv")

print("Dataset loaded:", df.shape)

# 2. Prepare target
df = df.dropna(subset=["response_time_ms"])
df["response_time_ms"] = pd.to_numeric(
    df["response_time_ms"], errors="coerce"
)
df = df.dropna(subset=["response_time_ms"])

# 3. Extract useful time information
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)
df = df.dropna(subset=["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

# 4. Select input features
features = [
    "server_id",
    "method",
    "path",
    "user_agent",
    "referrer",
    "hour",
    "day_of_week"
]

X = df[features].copy()
y = df["response_time_ms"]

# 5. Create classification categories
def classify_response(time):
    if time <= 200:
        return "Fast"
    elif time <= 500:
        return "Normal"
    elif time <= 1000:
        return "Slow"
    else:
        return "Critical"

y_class = y.apply(classify_response)

# 6. Preprocessing
categorical_features = [
    "server_id", "method", "path",
    "user_agent", "referrer"
]

numerical_features = ["hour", "day_of_week"]

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    ("numerical", "passthrough", numerical_features)
])

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

_, _, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# 8. Regression model
regression_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

regression_model.fit(X_train, y_train)

reg_predictions = regression_model.predict(X_test)

mae = mean_absolute_error(y_test, reg_predictions)
rmse = np.sqrt(mean_squared_error(y_test, reg_predictions))

print("\n--- REGRESSION RESULTS ---")
print("MAE:", round(mae, 2), "ms")
print("RMSE:", round(rmse, 2), "ms")

# 9. Classification model
classification_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

classification_model.fit(X_train, yc_train)

class_predictions = classification_model.predict(X_test)

accuracy = accuracy_score(yc_test, class_predictions)

print("\n--- CLASSIFICATION RESULTS ---")
print("Accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(
    yc_test,
    class_predictions,
    zero_division=0
))

# 10. Save both trained models
joblib.dump(regression_model, "regression_model.pkl")
joblib.dump(classification_model, "classification_model.pkl")

print("\nBoth ML models saved successfully!")
print("PROJECT MODEL TRAINING COMPLETED!")
