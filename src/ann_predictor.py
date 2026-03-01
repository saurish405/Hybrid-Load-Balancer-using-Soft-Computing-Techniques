import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import os

class LoadPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        # Changed extension to .keras for better compatibility
        self.model_path = 'models/predictor.keras' 

    def train_from_csv(self, file_path):
        if not os.path.exists('models'): os.makedirs('models')
        df = pd.read_csv(file_path)
        data = self.scaler.fit_transform(df[['requests']])
        
        X, y = [], []
        for i in range(60, len(data)):
            X.append(data[i-60:i, 0])
            y.append(data[i, 0])
        
        # Updated to the new Input(shape) style to avoid warnings
        self.model = Sequential([
            Input(shape=(60,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Explicitly using the MeanSquaredError object fixes the 'mse' error
        self.model.compile(optimizer='adam', loss=MeanSquaredError())
        self.model.fit(np.array(X), np.array(y), epochs=5, verbose=0)
        self.model.save(self.model_path)
        print(f"Model saved successfully at {self.model_path}")

    def predict_next(self, history):
        if self.model is None:
            # We pass custom_objects just in case, but .keras usually handles this
            self.model = load_model(self.model_path)
        
        hist_scaled = self.scaler.transform(np.array(history).reshape(-1, 1))
        pred = self.model.predict(hist_scaled.reshape(1, 60), verbose=0)
        return self.scaler.inverse_transform(pred)[0][0]