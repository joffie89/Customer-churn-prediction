import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load engineered data
data = pd.read_csv('data/processed/engineered_churn_data.csv')

# Splitting the data into training and test data set
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the models using ML algorithms
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"{model_name} Precision: {precision_score(y_test, y_pred)}")
    print(f"{model_name} Recall: {recall_score(y_test, y_pred)}")
    print(f"{model_name} F1 Score: {f1_score(y_test, y_pred)}")

# Saving the best model
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
import joblib
joblib.dump(best_model, 'models/churn_model.pkl')
