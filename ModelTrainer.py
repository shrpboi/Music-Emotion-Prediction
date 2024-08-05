import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.model = RandomForestClassifier(n_estimators=100)

    def train_and_evaluate(self):
        # Normalize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        # Feature Importance
        plt.barh(range(len(self.model.feature_importances_)), self.model.feature_importances_)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.show()
        
        # Save model to file
        joblib.dump(self.model, 'Emotion_Audio_Detection_Model.pkl')
        print("Model saved to Emotion_Audio_Detection_Model.pkl")
