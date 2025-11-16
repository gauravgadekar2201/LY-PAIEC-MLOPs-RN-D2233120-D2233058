import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib
import os

class MLModel:
    def __init__(self):
        self.model = None
        self.load_or_train_model()
    
    def load_or_train_model(self):
        model_path = "model.joblib"
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print("Model loaded from disk")
        else:
            print("Training new model...")
            X, y = make_classification(
                n_samples=1000, 
                n_features=20, 
                n_informative=15, 
                n_redundant=5,
                random_state=42
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            joblib.dump(self.model, model_path)
            print("Model trained and saved")
    
    def predict(self, features):
        if self.model is None:
            raise Exception("Model not loaded")
        
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)
        probability = self.model.predict_proba(features_array)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(np.max(probability[0])),
            "class_probabilities": probability[0].tolist()
        }