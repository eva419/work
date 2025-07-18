"""
Medical Chatbot Model for MediXpert
Provides intelligent responses to medical queries
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib
import os
import re
import json
from datetime import datetime

class MedicalChatbot:
    def __init__(self, data_path="../data"):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.intent_classifier = None
        self.qa_data = None
        self.responses = {}
        self.intents = {}
        self.similarity_threshold = 0.3
        
    def load_training_data(self):
        """Load chatbot training data"""
        try:
            # Load training data
            train_df = pd.read_csv(os.path.join(self.data_path, "train_data_chatbot.csv"))
            validation_df = pd.read_csv(os.path.join(self.data_path, "validation_data_chatbot.csv"))
            
            # Combine training and validation data
            df = pd.concat([train_df, validation_df], ignore_index=True)
            
            print(f"Loaded {len(df)} chatbot training records")
            print(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading chatbot data: {e}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def create_intent_patterns(self):
        """Create predefined intent patterns for medical queries"""
        self.intents = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
                'responses': [
                    "Hello! I'm your medical assistant. How can I help you today?",
                    "Hi there! I'm here to help with your medical questions.",
                    "Hello! What medical information can I assist you with?"
                ]
            },
            'symptoms': {
                'patterns': ['symptoms', 'symptom', 'feeling', 'pain', 'ache', 'hurt', 'sick'],
                'responses': [
                    "I understand you're experiencing symptoms. Can you describe them in detail?",
                    "Please tell me more about your symptoms so I can better assist you.",
                    "What specific symptoms are you experiencing?"
                ]
            },
            'disease_info': {
                'patterns': ['what is', 'tell me about', 'information about', 'explain', 'disease', 'condition'],
                'responses': [
                    "I can provide information about various medical conditions. What would you like to know?",
                    "I'm here to help explain medical conditions. Which one interests you?",
                    "What medical condition would you like me to explain?"
                ]
            },
            'treatment': {
                'patterns': ['treatment', 'cure', 'medicine', 'medication', 'therapy', 'how to treat'],
                'responses': [
                    "For treatment information, I recommend consulting with a healthcare professional.",
                    "Treatment options vary by condition. Please consult a doctor for personalized advice.",
                    "I can provide general treatment information, but please see a doctor for specific recommendations."
                ]
            },
            'appointment': {
                'patterns': ['appointment', 'book', 'schedule', 'doctor', 'consultation'],
                'responses': [
                    "I can help you find doctors and schedule appointments. What type of specialist do you need?",
                    "Would you like to book an appointment? I can help you find available doctors.",
                    "Let me help you schedule an appointment. What's your preferred specialization?"
                ]
            },
            'emergency': {
                'patterns': ['emergency', 'urgent', 'serious', 'critical', 'help', 'ambulance'],
                'responses': [
                    "If this is a medical emergency, please call emergency services immediately!",
                    "For urgent medical situations, please contact emergency services or go to the nearest hospital.",
                    "This sounds urgent. Please seek immediate medical attention or call emergency services."
                ]
            },
            'goodbye': {
                'patterns': ['bye', 'goodbye', 'see you', 'thanks', 'thank you'],
                'responses': [
                    "Take care! Remember to consult healthcare professionals for medical advice.",
                    "Goodbye! Stay healthy and don't hesitate to seek professional medical help when needed.",
                    "Thank you for using MediXpert. Take care of your health!"
                ]
            }
        }
    
    def classify_intent(self, user_input):
        """Classify user intent based on input"""
        user_input_lower = user_input.lower()
        
        for intent, data in self.intents.items():
            for pattern in data['patterns']:
                if pattern in user_input_lower:
                    return intent
        
        return 'general'
    
    def prepare_qa_data(self, df):
        """Prepare question-answer data from training dataset"""
        qa_pairs = []
        
        # Assuming the CSV has columns like 'question', 'answer', 'intent', etc.
        # Adapt based on actual column structure
        if 'question' in df.columns and 'answer' in df.columns:
            for _, row in df.iterrows():
                question = self.preprocess_text(row['question'])
                answer = str(row['answer'])
                
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'intent': row.get('intent', 'general')
                    })
        
        return qa_pairs
    
    def train_similarity_model(self):
        """Train similarity-based response model"""
        df = self.load_training_data()
        if df is None:
            return False
        
        # Create intent patterns
        self.create_intent_patterns()
        
        # Prepare Q&A data
        self.qa_data = self.prepare_qa_data(df)
        
        if not self.qa_data:
            print("No valid Q&A data found. Using predefined responses only.")
            return True
        
        # Extract questions for vectorization
        questions = [qa['question'] for qa in self.qa_data]
        
        # Fit vectorizer on questions
        self.vectorizer.fit(questions)
        
        print(f"Trained on {len(self.qa_data)} Q&A pairs")
        return True
    
    def get_response(self, user_input):
        """Generate response for user input"""
        user_input_processed = self.preprocess_text(user_input)
        
        # Classify intent
        intent = self.classify_intent(user_input)
        
        # Handle specific intents
        if intent in self.intents:
            responses = self.intents[intent]['responses']
            return np.random.choice(responses), intent, 1.0
        
        # If we have Q&A data, find similar questions
        if self.qa_data:
            return self.find_similar_response(user_input_processed)
        
        # Default response
        return self.get_default_response(), 'general', 0.5
    
    def find_similar_response(self, user_input):
        """Find most similar question and return corresponding answer"""
        if not self.qa_data:
            return self.get_default_response(), 'general', 0.0
        
        # Vectorize user input
        user_vector = self.vectorizer.transform([user_input])
        
        # Vectorize all questions
        questions = [qa['question'] for qa in self.qa_data]
        question_vectors = self.vectorizer.transform(questions)
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, question_vectors)[0]
        
        # Find best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > self.similarity_threshold:
            best_qa = self.qa_data[best_match_idx]
            return best_qa['answer'], best_qa['intent'], float(best_similarity)
        
        # No good match found
        return self.get_default_response(), 'general', float(best_similarity)
    
    def get_default_response(self):
        """Get default response when no good match is found"""
        default_responses = [
            "I understand your concern. For specific medical advice, I recommend consulting with a healthcare professional.",
            "That's an interesting question. For detailed medical information, please consult a doctor.",
            "I'd be happy to help, but for medical concerns, it's best to speak with a healthcare provider.",
            "For accurate medical advice tailored to your situation, please consult with a medical professional."
        ]
        return np.random.choice(default_responses)
    
    def save_model(self, model_dir="../ml/models"):
        """Save the chatbot model"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, "chatbot_vectorizer.pkl")
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save Q&A data
        qa_data_path = os.path.join(model_dir, "chatbot_qa_data.pkl")
        joblib.dump(self.qa_data, qa_data_path)
        
        # Save intents
        intents_path = os.path.join(model_dir, "chatbot_intents.pkl")
        joblib.dump(self.intents, intents_path)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'qa_pairs_count': len(self.qa_data) if self.qa_data else 0,
            'intents_count': len(self.intents),
            'similarity_threshold': self.similarity_threshold
        }
        
        metadata_path = os.path.join(model_dir, "chatbot_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        print(f"Chatbot model saved to {model_dir}")
        return True
    
    def load_model(self, model_dir="../ml/models"):
        """Load pre-trained chatbot model"""
        try:
            vectorizer_path = os.path.join(model_dir, "chatbot_vectorizer.pkl")
            qa_data_path = os.path.join(model_dir, "chatbot_qa_data.pkl")
            intents_path = os.path.join(model_dir, "chatbot_intents.pkl")
            
            self.vectorizer = joblib.load(vectorizer_path)
            self.qa_data = joblib.load(qa_data_path)
            self.intents = joblib.load(intents_path)
            
            print(f"Chatbot model loaded successfully from {model_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading chatbot model: {e}")
            return False
    
    def chat_session(self):
        """Interactive chat session for testing"""
        print("MediXpert Chatbot - Type 'quit' to exit")
        print("=" * 40)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Chatbot: Take care! Remember to consult healthcare professionals for medical advice.")
                break
            
            if not user_input:
                continue
            
            response, intent, confidence = self.get_response(user_input)
            print(f"Chatbot: {response}")
            print(f"[Intent: {intent}, Confidence: {confidence:.2f}]")

def main():
    """Main function to train and test the chatbot"""
    print("Training Medical Chatbot for MediXpert")
    print("=" * 40)
    
    # Initialize chatbot
    chatbot = MedicalChatbot()
    
    # Train the model
    success = chatbot.train_similarity_model()
    
    if success:
        # Save the model
        chatbot.save_model()
        
        # Test the chatbot
        test_queries = [
            "Hello, I need help",
            "I have a fever and headache",
            "What is diabetes?",
            "How to treat high blood pressure?",
            "I need to book an appointment",
            "This is an emergency!",
            "Thank you for your help"
        ]
        
        print("\nTesting chatbot responses:")
        print("-" * 30)
        
        for query in test_queries:
            response, intent, confidence = chatbot.get_response(query)
            print(f"\nUser: {query}")
            print(f"Bot: {response}")
            print(f"Intent: {intent}, Confidence: {confidence:.2f}")
        
        # Start interactive session
        print("\nStarting interactive chat session...")
        chatbot.chat_session()
    
    else:
        print("Failed to train chatbot model")

if __name__ == "__main__":
    main()

