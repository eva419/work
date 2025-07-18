"""
Disease Prediction Model for MediXpert
Uses machine learning to predict diseases based on symptoms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import re
import joblib
from datetime import datetime

class DiseasePredictor:
    def __init__(self, data_path="../data"):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.model = None
        self.model_type = "RandomForest"
        self.accuracy = 0.0
        self.feature_names = []
        
    def load_data(self):
        """Load and preprocess the disease-symptoms data"""
        try:
            # Load the main disease-symptoms dataset
            df1 = pd.read_csv(os.path.join(self.data_path, "Diseases_Symptoms.csv"))
            df2 = pd.read_csv(os.path.join(self.data_path, "Diseases_Symptoms2.csv"))
            
            # Combine datasets
            df = pd.concat([df1, df2], ignore_index=True)
            df = df.drop_duplicates()
            
            print(f"Loaded {len(df)} disease records")
            print(f"Unique diseases: {df['Name'].nunique()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_symptoms(self, symptoms_text):
        """Clean and preprocess symptoms text"""
        if pd.isna(symptoms_text):
            return ""
        
        # Convert to lowercase
        symptoms_text = str(symptoms_text).lower()
        
        # Remove special characters and extra spaces
        symptoms_text = re.sub(r'[^\w\s,]', '', symptoms_text)
        symptoms_text = re.sub(r'\s+', ' ', symptoms_text)
        
        # Split by comma and clean each symptom
        symptoms = [symptom.strip() for symptom in symptoms_text.split(',')]
        symptoms = [symptom for symptom in symptoms if symptom]
        
        return ' '.join(symptoms)
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        # Preprocess symptoms
        df['processed_symptoms'] = df['Symptoms'].apply(self.preprocess_symptoms)
        
        # Remove rows with empty symptoms
        df = df[df['processed_symptoms'] != '']
        
        # Prepare features (symptoms) and labels (diseases)
        X = df['processed_symptoms'].values
        y = df['Name'].values
        
        print(f"Training data shape: {X.shape}")
        print(f"Unique diseases in training: {len(np.unique(y))}")
        
        return X, y
    
    def train_model(self, model_type="RandomForest"):
        """Train the disease prediction model"""
        # Load data
        df = self.load_data()
        if df is None:
            return False
        
        # Prepare training data
        X, y = self.prepare_training_data(df)
        
        # Filter out diseases with only one sample for stratified split
        unique, counts = np.unique(y, return_counts=True)
        diseases_to_keep = unique[counts >= 2]
        
        # Filter data to keep only diseases with multiple samples
        mask = np.isin(y, diseases_to_keep)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        print(f"Filtered data: {len(X_filtered)} samples, {len(diseases_to_keep)} diseases")
        
        # Split data with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )
        except ValueError:
            # Fallback to random split if stratification still fails
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42
            )
        
        # Vectorize symptoms
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model based on type
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
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
        self.model.fit(X_train_vec, y_train_encoded)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vec)
        
        # Calculate accuracy
        self.accuracy = accuracy_score(y_test_encoded, y_pred)
        self.model_type = model_type
        
        print(f"Model Accuracy: {self.accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return True
    
    def predict_disease(self, symptoms_input, top_n=3):
        """Predict disease based on symptoms"""
        if self.model is None:
            return None
        
        # Preprocess input symptoms
        processed_symptoms = self.preprocess_symptoms(symptoms_input)
        
        if not processed_symptoms:
            return None
        
        # Vectorize input
        symptoms_vec = self.vectorizer.transform([processed_symptoms])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(symptoms_vec)[0]
        
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
    
    def save_model(self, model_dir="../ml/models"):
        """Save the trained model and components"""
        if self.model is None:
            print("No model to save")
            return False
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(model_dir, f"disease_predictor_{self.model_type.lower()}.pkl")
        vectorizer_path = os.path.join(model_dir, "symptoms_vectorizer.pkl")
        encoder_path = os.path.join(model_dir, "disease_label_encoder.pkl")
        
        # Save using joblib for better performance
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'accuracy': self.accuracy,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.vectorizer.get_feature_names_out()),
            'disease_count': len(self.label_encoder.classes_)
        }
        
        metadata_path = os.path.join(model_dir, "model_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved to {model_dir}")
        return True
    
    def load_model(self, model_dir="../ml/models", model_type="randomforest"):
        """Load a pre-trained model"""
        try:
            model_path = os.path.join(model_dir, f"disease_predictor_{model_type}.pkl")
            vectorizer_path = os.path.join(model_dir, "symptoms_vectorizer.pkl")
            encoder_path = os.path.join(model_dir, "disease_label_encoder.pkl")
            metadata_path = os.path.join(model_dir, "model_metadata.pkl")
            
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(encoder_path)
            
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.model_type = metadata.get('model_type', model_type)
                self.accuracy = metadata.get('accuracy', 0.0)
            
            print(f"Model loaded successfully from {model_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance for RandomForest model"""
        if self.model_type != "RandomForest" or self.model is None:
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_
        
        # Get top features
        indices = np.argsort(importances)[::-1][:top_n]
        
        top_features = []
        for idx in indices:
            top_features.append({
                'feature': feature_names[idx],
                'importance': float(importances[idx])
            })
        
        return top_features

def main():
    """Main function to train and save the disease prediction model"""
    print("Training Disease Prediction Model for MediXpert")
    print("=" * 50)
    
    # Initialize predictor
    predictor = DiseasePredictor()
    
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
    
    # Test prediction
    test_symptoms = "fever, headache, cough, fatigue"
    predictions = predictor.predict_disease(test_symptoms)
    
    print(f"\nTest prediction for symptoms: '{test_symptoms}'")
    for pred in predictions:
        print(f"- {pred['disease']}: {pred['confidence_percentage']:.2f}%")
    
    # Show feature importance for RandomForest
    if best_model == "RandomForest":
        predictor.load_model(model_type="randomforest")
        top_features = predictor.get_feature_importance()
        print("\nTop 10 Important Features:")
        for feature in top_features[:10]:
            print(f"- {feature['feature']}: {feature['importance']:.4f}")

if __name__ == "__main__":
    main()

