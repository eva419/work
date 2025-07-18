from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import json

class Disease(models.Model):
    """Disease information and details"""
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    code = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    symptoms = models.TextField(help_text="Comma-separated list of symptoms")
    treatments = models.TextField(help_text="Recommended treatments")
    severity_level = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='medium')
    
    # Additional medical information
    causes = models.TextField(blank=True, help_text="Common causes of the disease")
    prevention = models.TextField(blank=True, help_text="Prevention methods")
    complications = models.TextField(blank=True, help_text="Possible complications")
    prognosis = models.TextField(blank=True, help_text="Expected outcome and recovery")
    
    # Classification
    category = models.CharField(max_length=100, blank=True, help_text="Disease category (e.g., Infectious, Chronic)")
    icd_10_code = models.CharField(max_length=10, blank=True, help_text="ICD-10 classification code")
    
    # Metadata
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.code} - {self.name}"
    
    @property
    def symptoms_list(self):
        """Return symptoms as a list"""
        return [s.strip() for s in self.symptoms.split(',') if s.strip()]
    
    @property
    def treatments_list(self):
        """Return treatments as a list"""
        return [t.strip() for t in self.treatments.split(',') if t.strip()]
    
    class Meta:
        verbose_name = "Disease"
        verbose_name_plural = "Diseases"
        ordering = ['name']

class Symptom(models.Model):
    """Individual symptoms database"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    category = models.CharField(max_length=50, blank=True, help_text="e.g., Respiratory, Neurological")
    severity_indicator = models.CharField(max_length=20, blank=True, help_text="Indicates severity level")
    common_in_diseases = models.ManyToManyField(Disease, blank=True, related_name='common_symptoms')
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = "Symptom"
        verbose_name_plural = "Symptoms"
        ordering = ['name']

class DiseasePrediction(models.Model):
    """Disease prediction results from ML model"""
    patient = models.ForeignKey('patients.Patient', on_delete=models.CASCADE, related_name='predictions')
    symptoms_input = models.TextField(help_text="Original symptoms input by patient")
    processed_symptoms = models.TextField(help_text="Processed symptoms for ML model")
    
    # Prediction results
    predicted_disease = models.ForeignKey(Disease, on_delete=models.CASCADE)
    confidence_score = models.DecimalField(
        max_digits=5, 
        decimal_places=4,
        validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    
    # Alternative predictions
    alternative_predictions = models.JSONField(
        default=list,
        help_text="List of alternative predictions with confidence scores"
    )
    
    # Model information
    model_version = models.CharField(max_length=20, default="1.0")
    model_type = models.CharField(max_length=50, default="RandomForest")
    
    # Validation and feedback
    is_confirmed = models.BooleanField(default=False)
    confirmed_by = models.ForeignKey(
        'doctors.Doctor', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='confirmed_predictions'
    )
    doctor_notes = models.TextField(blank=True)
    actual_diagnosis = models.ForeignKey(
        Disease, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='actual_diagnoses'
    )
    
    # Timestamps
    prediction_date = models.DateTimeField(auto_now_add=True)
    confirmed_date = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.patient.patient_id} - {self.predicted_disease.name} ({self.confidence_score})"
    
    @property
    def symptoms_list(self):
        """Return processed symptoms as a list"""
        return [s.strip() for s in self.processed_symptoms.split(',') if s.strip()]
    
    @property
    def confidence_percentage(self):
        """Return confidence as percentage"""
        return float(self.confidence_score) * 100
    
    class Meta:
        verbose_name = "Disease Prediction"
        verbose_name_plural = "Disease Predictions"
        ordering = ['-prediction_date']

class MLModel(models.Model):
    """ML Model metadata and versioning"""
    MODEL_TYPE_CHOICES = [
        ('disease_prediction', 'Disease Prediction'),
        ('symptom_analysis', 'Symptom Analysis'),
        ('risk_assessment', 'Risk Assessment'),
        ('chatbot', 'Medical Chatbot'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=30, choices=MODEL_TYPE_CHOICES)
    version = models.CharField(max_length=20)
    description = models.TextField(blank=True)
    
    # Model performance metrics
    accuracy = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True)
    precision = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True)
    recall = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True)
    f1_score = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True)
    
    # Model files and configuration
    model_file_path = models.CharField(max_length=500, blank=True)
    config_file_path = models.CharField(max_length=500, blank=True)
    training_data_info = models.JSONField(default=dict)
    
    # Status and deployment
    is_active = models.BooleanField(default=False)
    is_deployed = models.BooleanField(default=False)
    trained_at = models.DateTimeField(null=True, blank=True)
    deployed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} v{self.version}"
    
    class Meta:
        verbose_name = "ML Model"
        verbose_name_plural = "ML Models"
        unique_together = ['name', 'version']
        ordering = ['-created_at']

class PredictionFeedback(models.Model):
    """Feedback on prediction accuracy for model improvement"""
    FEEDBACK_TYPE_CHOICES = [
        ('correct', 'Correct Prediction'),
        ('incorrect', 'Incorrect Prediction'),
        ('partially_correct', 'Partially Correct'),
        ('insufficient_info', 'Insufficient Information'),
    ]
    
    prediction = models.OneToOneField(DiseasePrediction, on_delete=models.CASCADE, related_name='feedback')
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPE_CHOICES)
    provided_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Detailed feedback
    comments = models.TextField(blank=True)
    suggested_disease = models.ForeignKey(Disease, on_delete=models.SET_NULL, null=True, blank=True)
    missing_symptoms = models.TextField(blank=True, help_text="Symptoms that should have been considered")
    additional_info = models.TextField(blank=True)
    
    # Ratings
    usefulness_rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="How useful was the prediction (1-5)"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Feedback for {self.prediction.patient.patient_id} - {self.feedback_type}"
    
    class Meta:
        verbose_name = "Prediction Feedback"
        verbose_name_plural = "Prediction Feedback"
        ordering = ['-created_at']

class ChatbotConversation(models.Model):
    """Chatbot conversation sessions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chatbot_conversations')
    session_id = models.CharField(max_length=100, unique=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    
    # Conversation metadata
    total_messages = models.IntegerField(default=0)
    user_satisfaction = models.IntegerField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    resolved_query = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Conversation {self.session_id} - {self.user.username}"
    
    class Meta:
        verbose_name = "Chatbot Conversation"
        verbose_name_plural = "Chatbot Conversations"
        ordering = ['-started_at']

class ChatbotMessage(models.Model):
    """Individual messages in chatbot conversations"""
    MESSAGE_TYPE_CHOICES = [
        ('user', 'User Message'),
        ('bot', 'Bot Response'),
        ('system', 'System Message'),
    ]
    
    conversation = models.ForeignKey(ChatbotConversation, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPE_CHOICES)
    content = models.TextField()
    
    # Bot response metadata
    intent_detected = models.CharField(max_length=100, blank=True)
    confidence_score = models.DecimalField(
        max_digits=5, 
        decimal_places=4, 
        null=True, 
        blank=True
    )
    response_time = models.FloatField(null=True, blank=True, help_text="Response time in seconds")
    
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.conversation.session_id} - {self.message_type}: {self.content[:50]}..."
    
    class Meta:
        verbose_name = "Chatbot Message"
        verbose_name_plural = "Chatbot Messages"
        ordering = ['timestamp']

