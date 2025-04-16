import tensorflow as tf
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def create_cnn(input_shape):
    """Improved CNN model with deeper layers and Batch Normalization."""
    inputs = Input(shape=input_shape)
    x = Conv1D(128, kernel_size=1, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.3)(x)
    x = Flatten()(x)
    cnn_output = Dense(128, activation='relu')(x)
    return Model(inputs, cnn_output)

def create_rnn(input_shape):
    """Improved RNN model with increased LSTM units and Batch Normalization."""
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    rnn_output = Dense(128, activation='relu')(x)
    return Model(inputs, rnn_output)

if __name__ == "__main__":
    # Load processed features
    data = pd.read_csv('/content/Trr1_preprocessed_data.csv')
    X = data.drop(columns=['Task'])
    y = data['Task']

  

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape for CNN and RNN
    X_train_cnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    X_train_rnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_rnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define models
    cnn_model = create_cnn((X_train_cnn.shape[1], 1))
    rnn_model = create_rnn((X_train_rnn.shape[1], 1))

    # Fusion model (CNN + RNN)
    combined = Concatenate()([cnn_model.output, rnn_model.output])
    final_output = Dense(1, activation='sigmoid')(combined)
    fusion_model = Model(inputs=[cnn_model.input, rnn_model.input], outputs=final_output)

    fusion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                         loss='binary_crossentropy', metrics=['accuracy'])

    # # Early stopping callback
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model
    train_history = fusion_model.fit([X_train_cnn, X_train_rnn], y_train, epochs=100, batch_size=64)
    fusion_model.save('fusion_model.h5')

    # Evaluate model
    train_accuracy = train_history.history['accuracy'][-1]
    print(f"Training Accuracy: {train_accuracy:.4f}")
  