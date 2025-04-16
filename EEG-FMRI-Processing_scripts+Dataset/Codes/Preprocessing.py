import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the combined dataset
data = pd.read_csv('/content/sample_data/Testing_raw.csv')  # Replace with your actual file path

# Remove unnecessary columns

# Separate features and target variable (assuming 'Task' is the target)
X = data.drop(columns=['Task'])
y = data['Task']

# Fill missing values in the features (numerical columns) with the median of that column
X = X.apply(lambda col: col.fillna(col.median()) if col.isna().sum() > 0 else col, axis=0)


# Encoding categorical target variable (if applicable)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scaling features (MinMax scaling the data)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Combine the scaled features with the encoded target variable
preprocessed_df = pd.DataFrame(X_scaled, columns=X.columns)
preprocessed_df['Task'] = y_encoded

# Save the preprocessed dataset
preprocessed_df.to_csv('Tes_preprocessed_data.csv', index=False)
print("Preprocessing completed and saved as 'Tes_preprocessed_data.csv'.")
