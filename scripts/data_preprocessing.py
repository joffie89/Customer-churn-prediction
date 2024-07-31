import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Loading the customer churn dataset from Kaggle
data = pd.read_csv('data/raw/telco_customer_churn.csv')

# Next step is to handle missing values
data.fillna(method='ffill', inplace=True)

# Encoding categorical variables
label_encoder = LabelEncoder()
categorical_columns = data.select_dtypes(include=['object']).columns

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Scaling the numerical features
scaler = StandardScaler()
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Finally saving the preprocessed data
data.to_csv('data/processed/preprocessed_churn_data.csv', index=False)

