"""
Medical API routes for MediXpert
Handles disease prediction, chatbot, doctors, appointments, and reports
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import os
import sys
import traceback
from datetime import datetime
import json

# Add the src directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from src.disease_prediction_v2 import ImprovedDiseasePredictor
    from src.chatbot_model import MedicalChatbot
except ImportError as e:
    print(f"Warning: Could not import ML models: {e}")
    ImprovedDiseasePredictor = None
    MedicalChatbot = None

medical_bp = Blueprint('medical', __name__)

# Initialize ML models
disease_predictor = None
chatbot = None

def init_ml_models():
    """Initialize ML models on first use"""
    global disease_predictor, chatbot
    
    if disease_predictor is None and ImprovedDiseasePredictor:
        try:
            disease_predictor = ImprovedDiseasePredictor()
            models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
            if os.path.exists(models_path):
                disease_predictor.load_model(models_path)
                print("Disease prediction model loaded successfully")
            else:
                print("Models directory not found, using default model")
        except Exception as e:
            print(f"Error loading disease prediction model: {e}")
    
    if chatbot is None and MedicalChatbot:
        try:
            chatbot = MedicalChatbot()
            models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
            if os.path.exists(models_path):
                chatbot.load_model(models_path)
                print("Chatbot model loaded successfully")
            else:
                print("Models directory not found, using default chatbot")
        except Exception as e:
            print(f"Error loading chatbot model: {e}")

# Mock data for demonstration
MOCK_DOCTORS = [
    {
        "id": 1,
        "name": "Dr. Sarah Johnson",
        "specialty": "Cardiology",
        "experience": "15 years",
        "rating": 4.9,
        "hospital": "Central Hospital",
        "location": "Downtown",
        "available": True,
        "phone": "+1-555-0101",
        "email": "sarah.johnson@centralhospital.com"
    },
    {
        "id": 2,
        "name": "Dr. Michael Chen",
        "specialty": "Dermatology",
        "experience": "12 years",
        "rating": 4.8,
        "hospital": "Westside Clinic",
        "location": "Westside",
        "available": False,
        "phone": "+1-555-0102",
        "email": "michael.chen@westsideclinic.com"
    },
    {
        "id": 3,
        "name": "Dr. Emily Davis",
        "specialty": "Pediatrics",
        "experience": "8 years",
        "rating": 4.7,
        "hospital": "Eastside Clinic",
        "location": "Eastside",
        "available": True,
        "phone": "+1-555-0103",
        "email": "emily.davis@eastsideclinic.com"
    },
    {
        "id": 4,
        "name": "Dr. Robert Wilson",
        "specialty": "Oncology",
        "experience": "20 years",
        "rating": 4.9,
        "hospital": "Cancer Treatment Center",
        "location": "Midtown",
        "available": True,
        "phone": "+1-555-0104",
        "email": "robert.wilson@ctc.com"
    },
    {
        "id": 5,
        "name": "Dr. Lisa Anderson",
        "specialty": "Neurology",
        "experience": "18 years",
        "rating": 4.8,
        "hospital": "Neurological Institute",
        "location": "Uptown",
        "available": False,
        "phone": "+1-555-0105",
        "email": "lisa.anderson@neuroinst.com"
    }
]

MOCK_APPOINTMENTS = [
    {
        "id": 1,
        "doctor_id": 1,
        "doctor_name": "Dr. Sarah Johnson",
        "specialty": "Cardiology",
        "date": "2024-07-20",
        "time": "10:00 AM",
        "status": "Confirmed",
        "type": "Consultation",
        "patient_name": "John Doe"
    },
    {
        "id": 2,
        "doctor_id": 3,
        "doctor_name": "Dr. Emily Davis",
        "specialty": "Pediatrics",
        "date": "2024-07-22",
        "time": "2:30 PM",
        "status": "Pending",
        "type": "Follow-up",
        "patient_name": "John Doe"
    }
]

MOCK_REPORTS = [
    {
        "id": 1,
        "name": "Blood Test Results",
        "date": "2024-07-15",
        "type": "Lab Report",
        "doctor": "Dr. Sarah Johnson",
        "status": "Reviewed",
        "file_url": "/api/reports/1/download"
    },
    {
        "id": 2,
        "name": "X-Ray Chest",
        "date": "2024-07-10",
        "type": "Radiology",
        "doctor": "Dr. Michael Chen",
        "status": "Pending Review",
        "file_url": "/api/reports/2/download"
    }
]

@medical_bp.route('/predict-disease', methods=['POST'])
@cross_origin()
def predict_disease():
    """Predict disease based on symptoms"""
    try:
        init_ml_models()
        
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        
        if not symptoms:
            return jsonify({'error': 'Symptoms are required'}), 400
        
        # If ML model is available, use it
        if disease_predictor:
            try:
                # Convert symptoms string to list
                symptom_list = [s.strip().lower() for s in symptoms.split(',')]
                predictions = disease_predictor.predict_from_symptom_list(symptom_list, top_n=3)
                
                if predictions:
                    # Convert to expected format
                    results = []
                    for pred in predictions:
                        severity = 'Low' if pred['confidence'] < 0.7 else 'Medium' if pred['confidence'] < 0.9 else 'High'
                        results.append({
                            'disease': pred['disease'],
                            'confidence': round(pred['confidence_percentage'], 1),
                            'severity': severity
                        })
                    
                    return jsonify({
                        'success': True,
                        'predictions': results,
                        'message': 'Disease prediction completed successfully'
                    })
            except Exception as e:
                print(f"ML prediction error: {e}")
                # Fall back to mock data
        
        # Mock predictions for demonstration
        mock_predictions = [
            {'disease': 'Common Cold', 'confidence': 85.2, 'severity': 'Low'},
            {'disease': 'Seasonal Allergy', 'confidence': 72.8, 'severity': 'Low'},
            {'disease': 'Viral Infection', 'confidence': 68.5, 'severity': 'Medium'}
        ]
        
        return jsonify({
            'success': True,
            'predictions': mock_predictions,
            'message': 'Disease prediction completed successfully (demo mode)'
        })
        
    except Exception as e:
        print(f"Error in predict_disease: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/chat', methods=['POST'])
@cross_origin()
def chat_with_bot():
    """Chat with medical AI assistant"""
    try:
        init_ml_models()
        
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # If chatbot model is available, use it
        if chatbot:
            try:
                response, intent, confidence = chatbot.get_response(message)
                return jsonify({
                    'success': True,
                    'response': response,
                    'intent': intent,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime('%I:%M %p')
                })
            except Exception as e:
                print(f"Chatbot error: {e}")
                # Fall back to mock response
        
        # Mock response for demonstration
        mock_responses = [
            "Thank you for sharing that information. For persistent symptoms, I recommend consulting with a healthcare professional for proper evaluation.",
            "I understand your concern. Based on what you've described, it would be best to schedule an appointment with a doctor.",
            "That's a good question. For specific medical advice, please consult with a qualified healthcare provider.",
            "I'm here to help with general health information. For personalized medical advice, please see a healthcare professional."
        ]
        
        import random
        response = random.choice(mock_responses)
        
        return jsonify({
            'success': True,
            'response': response,
            'intent': 'general',
            'confidence': 0.8,
            'timestamp': datetime.now().strftime('%I:%M %p')
        })
        
    except Exception as e:
        print(f"Error in chat_with_bot: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/doctors', methods=['GET'])
@cross_origin()
def get_doctors():
    """Get list of doctors with optional filtering"""
    try:
        specialty = request.args.get('specialty', '')
        search = request.args.get('search', '')
        
        doctors = MOCK_DOCTORS.copy()
        
        # Filter by specialty
        if specialty and specialty.lower() != 'all':
            doctors = [d for d in doctors if d['specialty'].lower() == specialty.lower()]
        
        # Filter by search term
        if search:
            search_lower = search.lower()
            doctors = [d for d in doctors if 
                      search_lower in d['name'].lower() or 
                      search_lower in d['hospital'].lower() or
                      search_lower in d['specialty'].lower()]
        
        return jsonify({
            'success': True,
            'doctors': doctors,
            'total': len(doctors)
        })
        
    except Exception as e:
        print(f"Error in get_doctors: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/doctors/<int:doctor_id>', methods=['GET'])
@cross_origin()
def get_doctor(doctor_id):
    """Get specific doctor details"""
    try:
        doctor = next((d for d in MOCK_DOCTORS if d['id'] == doctor_id), None)
        
        if not doctor:
            return jsonify({'error': 'Doctor not found'}), 404
        
        return jsonify({
            'success': True,
            'doctor': doctor
        })
        
    except Exception as e:
        print(f"Error in get_doctor: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/appointments', methods=['GET'])
@cross_origin()
def get_appointments():
    """Get user appointments"""
    try:
        status = request.args.get('status', '')
        
        appointments = MOCK_APPOINTMENTS.copy()
        
        # Filter by status if provided
        if status:
            appointments = [a for a in appointments if a['status'].lower() == status.lower()]
        
        return jsonify({
            'success': True,
            'appointments': appointments,
            'total': len(appointments)
        })
        
    except Exception as e:
        print(f"Error in get_appointments: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/appointments', methods=['POST'])
@cross_origin()
def book_appointment():
    """Book a new appointment"""
    try:
        data = request.get_json()
        
        required_fields = ['doctor_id', 'date', 'time', 'type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        # Find doctor
        doctor = next((d for d in MOCK_DOCTORS if d['id'] == data['doctor_id']), None)
        if not doctor:
            return jsonify({'error': 'Doctor not found'}), 404
        
        if not doctor['available']:
            return jsonify({'error': 'Doctor is not available'}), 400
        
        # Create new appointment
        new_appointment = {
            'id': len(MOCK_APPOINTMENTS) + 1,
            'doctor_id': data['doctor_id'],
            'doctor_name': doctor['name'],
            'specialty': doctor['specialty'],
            'date': data['date'],
            'time': data['time'],
            'status': 'Pending',
            'type': data['type'],
            'patient_name': data.get('patient_name', 'John Doe')
        }
        
        MOCK_APPOINTMENTS.append(new_appointment)
        
        return jsonify({
            'success': True,
            'appointment': new_appointment,
            'message': 'Appointment booked successfully'
        }), 201
        
    except Exception as e:
        print(f"Error in book_appointment: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/reports', methods=['GET'])
@cross_origin()
def get_reports():
    """Get user medical reports"""
    try:
        return jsonify({
            'success': True,
            'reports': MOCK_REPORTS,
            'total': len(MOCK_REPORTS)
        })
        
    except Exception as e:
        print(f"Error in get_reports: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/reports', methods=['POST'])
@cross_origin()
def upload_report():
    """Upload a new medical report"""
    try:
        # In a real implementation, this would handle file uploads
        data = request.get_json()
        
        new_report = {
            'id': len(MOCK_REPORTS) + 1,
            'name': data.get('name', 'New Report'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'type': data.get('type', 'General'),
            'doctor': data.get('doctor', 'Unknown'),
            'status': 'Pending Review',
            'file_url': f"/api/reports/{len(MOCK_REPORTS) + 1}/download"
        }
        
        MOCK_REPORTS.append(new_report)
        
        return jsonify({
            'success': True,
            'report': new_report,
            'message': 'Report uploaded successfully'
        }), 201
        
    except Exception as e:
        print(f"Error in upload_report: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/dashboard/stats', methods=['GET'])
@cross_origin()
def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        stats = {
            'total_appointments': len(MOCK_APPOINTMENTS),
            'predictions_made': 12,
            'reports_uploaded': len(MOCK_REPORTS),
            'messages': 15,
            'health_score': 85,
            'recent_activities': [
                {
                    'type': 'appointment',
                    'message': 'Appointment with Dr. Smith scheduled',
                    'time': '2 hours ago'
                },
                {
                    'type': 'prediction',
                    'message': 'AI diagnosis completed for symptoms',
                    'time': '4 hours ago'
                },
                {
                    'type': 'report',
                    'message': 'Blood test report uploaded',
                    'time': '1 day ago'
                },
                {
                    'type': 'chat',
                    'message': 'New message from Dr. Johnson',
                    'time': '2 days ago'
                }
            ]
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        print(f"Error in get_dashboard_stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@medical_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'MediXpert API is running',
        'timestamp': datetime.now().isoformat(),
        'ml_models': {
            'disease_predictor': disease_predictor is not None,
            'chatbot': chatbot is not None
        }
    })

