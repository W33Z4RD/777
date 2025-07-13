
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional

class SimpleMLModel:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced')
        self.scaler = StandardScaler()

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Train the logistic regression model."""
        if len(features) == 0 or len(labels) == 0:
            logging.warning("Cannot train model with empty features or labels.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        accuracy = self.model.score(X_test_scaled, y_test)
        logging.info(f"Model trained with accuracy: {accuracy:.2f}")

    def predict(self, features: pd.DataFrame) -> Optional[float]:
        """Predict the probability of a positive outcome."""
        if len(features) == 0:
            return None

        # Check if the scaler is fitted by checking for the presence of its attributes
        if not hasattr(self.scaler, 'scale_'):
            logging.warning("Model is not trained yet (scaler is not fitted). Cannot predict.")
            return None
            
        features_scaled = self.scaler.transform(features)
        # Predict probability of the positive class (1)
        probability = self.model.predict_proba(features_scaled)[:, 1]
        return probability[0] if probability is not None else None
