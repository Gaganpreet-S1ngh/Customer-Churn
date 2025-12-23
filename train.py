import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn


df = pd.read_csv("data/churn.csv")


df.drop("customerID", axis=1, inplace=True)


label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    if col != "Churn": 
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le


X = df.drop("Churn", axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, params, cv=3, n_jobs=-1)


with mlflow.start_run():
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(best_model, artifact_path="model")

    print("Accuracy:", accuracy)

joblib.dump(best_model, "model/churn_model.pkl")

joblib.dump(label_encoders, "model/label_encoders.pkl")


default_values = {}
for col in X.columns:
    if col in label_encoders:
        default_values[col] = df[col].mode()[0]  
    else:
        default_values[col] = df[col].median()   

joblib.dump(default_values, "model/default_values.pkl")
print("Training complete. Model, encoders, and defaults saved.")
