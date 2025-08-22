import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data = pd.read_csv('E:/IDS-FL-CSE400/IDS-FL-CSE400/DL/ton_iot_pre-proc.py')

data.drop_duplicates(inplace=True)

null_columns = data.isnull().sum()
print(f"Null values per column:\n{null_columns}")

data.replace(['-', np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

data = data.apply(pd.to_numeric, errors='ignore')

label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = data[column].astype(str)
    data[column] = label_encoder.fit_transform(data[column])

# Exclude the 'type' and 'label' columns from scaling
exclude_columns = ['type', 'label']

scaler = MinMaxScaler()
# Apply scaling only to the numeric columns, excluding 'type' and 'label'
numeric_columns = data.select_dtypes(include=[np.number]).columns.difference(exclude_columns)
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

data.to_csv('E:/IDS-FL-CSE400/IDS-FL-CSE400/DL', index=False)

print("Preprocessing completed and saved to 'processed_train_test_network.csv'")
