import pandas as pd

# Load preprocessed data
data = pd.read_csv('data/processed/preprocessed_churn_data.csv')

# Example feature engineering
data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']
data.drop(['customerID', 'tenure'], axis=1, inplace=True)

# Save engineered data
data.to_csv('data/processed/engineered_churn_data.csv', index=False)
