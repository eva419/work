"""
Improved Disease Prediction Model for MediXpert
Uses the training_data.csv with binary symptom features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

class ImprovedDiseasePredictor:
    def __init__(self, data_path="../data"):
        self.data_path = data_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.model_type = "RandomForest"
        self.accuracy = 0.0
        
    def load_data(self):
        """Load the binary symptom training data"""
        try:
            df = pd.read_csv(os.path.join(self.data_path, "training_data.csv"))
            print(f"Loaded {len(df)} training records")
            print(f"Features: {df.shape[1] - 1}")  # -1 for the target column
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_data(self, df):
        """Prepare features and target from the dataset"""
        # The target column is 'prognosis'
        target_col = 'prognosis'
        
        # All other columns except the target are features (symptoms)
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Remove any rows with missing target values
        df_clean = df.dropna(subset=[target_col])
        
        # Fill missing feature values with 0 (assuming 0 means symptom not present)
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        # Store feature names for later use
        self.feature_names = list(feature_cols)
        
        print(f"Features shape: {X.shape}")
        print(f"Unique diseases: {len(np.unique(y))}")
        print(f"Disease distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for disease, count in zip(unique[:10], counts[:10]):  # Show first 10
            print(f"  {disease}: {count} samples")
        if len(unique) > 10:
            print(f"  ... and {len(unique) - 10} more diseases")
        
        return X, y
    
    def train_model(self, model_type="RandomForest"):
        """Train the disease prediction model"""
        # Load data
        df = self.load_data()
        if df is None:
            return False
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Initialize model based on type
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            )
        elif model_type == "LogisticRegression":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        elif model_type == "SVM":
            self.model = SVC(
                kernel='rbf',
                random_state=42,
                probability=True,
                C=1.0
            )
        
        # Train the model
        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train_encoded)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        self.accuracy = accuracy_score(y_test_encoded, y_pred)
        self.model_type = model_type
        
        print(f"Model Accuracy: {self.accuracy:.4f}")
        
        # Print classification report
        try:
            print("\nClassification Report:")
            print(classification_report(y_test_encoded, y_pred, 
                                      target_names=self.label_encoder.classes_,
                                      zero_division=0))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
        
        return True
    
    def predict_disease(self, symptoms_dict, top_n=3):
        """
        Predict disease based on symptoms dictionary
        symptoms_dict: dict with symptom names as keys and 1/0 as values
        """
        if self.model is None:
            return None
        
        # Create feature vector
        feature_vector = np.zeros(len(self.feature_names))
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in symptoms_dict:
                feature_vector[i] = symptoms_dict[feature_name]
        
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:top_n]
        
        predictions = []
        for idx in top_indices:
            disease_name = self.label_encoder.classes_[idx]
            confidence = probabilities[idx]
            predictions.append({
                'disease': disease_name,
                'confidence': float(confidence),
                'confidence_percentage': float(confidence * 100)
            })
        
        return predictions
    
    def predict_from_symptom_list(self, symptom_names, top_n=3):
        """
        Predict disease from a list of symptom names
        symptom_names: list of symptom names
        """
        # Create symptoms dictionary
        symptoms_dict = {}
        for symptom in symptom_names:
            # Clean symptom name to match feature names
            clean_symptom = symptom.lower().replace(' ', '_').replace('-', '_')
            if clean_symptom in self.feature_names:
                symptoms_dict[clean_symptom] = 1
        
        return self.predict_disease(symptoms_dict, top_n)
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance for RandomForest model"""
        if self.model_type != "RandomForest" or self.model is None:
            return None
        
        importances = self.model.feature_importances_
        
        # Get top features
        indices = np.argsort(importances)[::-1][:top_n]
        
        top_features = []
        for idx in indices:
            top_features.append({
                'feature': self.feature_names[idx],
                'importance': float(importances[idx])
            })
        
        return top_features
    
    def save_model(self, model_dir="../ml/models"):
        """Save the trained model and components"""
        if self.model is None:
            print("No model to save")
            return False
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(model_dir, f"disease_predictor_{self.model_type.lower()}_v2.pkl")
        encoder_path = os.path.join(model_dir, "disease_label_encoder_v2.pkl")
        features_path = os.path.join(model_dir, "feature_names_v2.pkl")
        
        # Save using joblib
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        joblib.dump(self.feature_names, features_path)
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'accuracy': self.accuracy,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'disease_count': len(self.label_encoder.classes_),
            'diseases': list(self.label_encoder.classes_)
        }
        
        metadata_path = os.path.join(model_dir, "model_metadata_v2.pkl")
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved to {model_dir}")
        return True
    
    def load_model(self, model_dir="../ml/models", model_type="randomforest"):
        """Load a pre-trained model"""
        try:
            model_path = os.path.join(model_dir, f"disease_predictor_{model_type}_v2.pkl")
            encoder_path = os.path.join(model_dir, "disease_label_encoder_v2.pkl")
            features_path = os.path.join(model_dir, "feature_names_v2.pkl")
            metadata_path = os.path.join(model_dir, "model_metadata_v2.pkl")
            
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.feature_names = joblib.load(features_path)
            
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.model_type = metadata.get('model_type', model_type)
                self.accuracy = metadata.get('accuracy', 0.0)
            
            print(f"Model loaded successfully from {model_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main function to train and test the improved disease prediction model"""
    print("Training Improved Disease Prediction Model for MediXpert")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ImprovedDiseasePredictor()
    
    # Train different models and compare
    models_to_train = ["RandomForest", "LogisticRegression", "SVM"]
    best_model = None
    best_accuracy = 0.0
    
    for model_type in models_to_train:
        print(f"\nTraining {model_type} model...")
        success = predictor.train_model(model_type)
        
        if success and predictor.accuracy > best_accuracy:
            best_accuracy = predictor.accuracy
            best_model = model_type
            # Save the best model
            predictor.save_model()
    
    print(f"\nBest model: {best_model} with accuracy: {best_accuracy:.4f}")
    
    # Test prediction with symptom list
    test_symptoms = ["itching", "skin_rash", "fatigue"]
    predictions = predictor.predict_from_symptom_list(test_symptoms)
    
    print(f"\nTest prediction for symptoms: {test_symptoms}")
    if predictions:
        for pred in predictions:
            print(f"- {pred['disease']}: {pred['confidence_percentage']:.2f}%")
    else:
        print("No predictions available")
    
    # Test prediction with symptoms dictionary
    symptoms_dict = {
        'itching': 1,
        'skin_rash': 1,
        'high_fever': 1,
        'headache': 1
    }
    
    predictions = predictor.predict_disease(symptoms_dict)
    print(f"\nTest prediction for symptoms dict: {symptoms_dict}")
    if predictions:
        for pred in predictions:
            print(f"- {pred['disease']}: {pred['confidence_percentage']:.2f}%")
    
    # Show feature importance for RandomForest
    if best_model == "RandomForest":
        predictor.load_model(model_type="randomforest")
        top_features = predictor.get_feature_importance()
        if top_features:
            print("\nTop 15 Important Features:")
            for feature in top_features[:15]:
                print(f"- {feature['feature']}: {feature['importance']:.4f}")

if __name__ == "__main__":
    main()

