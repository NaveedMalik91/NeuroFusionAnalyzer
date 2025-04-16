import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load trained model
fusion_model = tf.keras.models.load_model('/content/fusion_model.h5')

# Load processed features
data = pd.read_csv('/content/Tes_preprocessed_data.csv')
X = data.drop(columns=['Task'])
y = data['Task']

# Split into train and test (80-20 split for example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN and RNN
X_train_cnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

X_train_rnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Evaluate the model
loss, accuracy = fusion_model.evaluate([X_test_cnn, X_test_rnn], y_test)

# Output only the accuracy and loss
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
