from django.db import models
from django.contrib.auth.models import User
from django.core.validators import RegexValidator
import uuid

class UserProfile(models.Model):
    """Extended user profile for all users"""
    ROLE_CHOICES = [
        ('patient', 'Patient'),
        ('doctor', 'Doctor'),
        ('admin', 'Admin'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='patient')
    phone_number = models.CharField(
        max_length=15,
        validators=[RegexValidator(regex=r'^\+?1?\d{9,15}$', message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")]
    )
    date_of_birth = models.DateField(null=True, blank=True)
    address = models.TextField(blank=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.get_full_name()} ({self.role})"
    
    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

class Patient(models.Model):
    """Patient specific information"""
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='patient_profile')
    patient_id = models.CharField(max_length=10, unique=True, editable=False)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    insurance_provider = models.CharField(max_length=100, blank=True)
    insurance_number = models.CharField(max_length=50, blank=True)
    emergency_contact_name = models.CharField(max_length=100, blank=True)
    emergency_contact_phone = models.CharField(max_length=15, blank=True)
    medical_history = models.TextField(blank=True, help_text="Previous medical conditions, surgeries, etc.")
    allergies = models.TextField(blank=True, help_text="Known allergies and reactions")
    current_medications = models.TextField(blank=True, help_text="Current medications and dosages")
    blood_type = models.CharField(max_length=5, blank=True, help_text="e.g., A+, B-, O+, AB-")
    height = models.FloatField(null=True, blank=True, help_text="Height in cm")
    weight = models.FloatField(null=True, blank=True, help_text="Weight in kg")
    is_active = models.BooleanField(default=True)
    registration_date = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        if not self.patient_id:
            # Generate unique patient ID
            last_patient = Patient.objects.order_by('-id').first()
            if last_patient:
                last_id = int(last_patient.patient_id[1:])  # Remove 'P' prefix
                new_id = last_id + 1
            else:
                new_id = 1
            self.patient_id = f"P{new_id:03d}"  # Format as P001, P002, etc.
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.patient_id} - {self.user.get_full_name()}"
    
    @property
    def age(self):
        """Calculate age from date of birth"""
        if self.user.profile.date_of_birth:
            from datetime import date
            today = date.today()
            dob = self.user.profile.date_of_birth
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return None
    
    @property
    def bmi(self):
        """Calculate BMI if height and weight are available"""
        if self.height and self.weight:
            height_m = self.height / 100  # Convert cm to meters
            return round(self.weight / (height_m ** 2), 2)
        return None
    
    class Meta:
        verbose_name = "Patient"
        verbose_name_plural = "Patients"
        ordering = ['-registration_date']

class PatientVitals(models.Model):
    """Patient vital signs records"""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='vitals')
    recorded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Vital signs
    systolic_bp = models.IntegerField(null=True, blank=True, help_text="Systolic blood pressure (mmHg)")
    diastolic_bp = models.IntegerField(null=True, blank=True, help_text="Diastolic blood pressure (mmHg)")
    heart_rate = models.IntegerField(null=True, blank=True, help_text="Heart rate (bpm)")
    temperature = models.FloatField(null=True, blank=True, help_text="Body temperature (°C)")
    respiratory_rate = models.IntegerField(null=True, blank=True, help_text="Respiratory rate (breaths/min)")
    oxygen_saturation = models.FloatField(null=True, blank=True, help_text="Oxygen saturation (%)")
    
    # Additional measurements
    weight = models.FloatField(null=True, blank=True, help_text="Weight in kg")
    height = models.FloatField(null=True, blank=True, help_text="Height in cm")
    
    notes = models.TextField(blank=True)
    recorded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.patient.patient_id} - Vitals on {self.recorded_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        verbose_name = "Patient Vitals"
        verbose_name_plural = "Patient Vitals"
        ordering = ['-recorded_at']

