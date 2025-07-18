from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from decimal import Decimal

class Specialization(models.Model):
    """Medical specializations"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = "Specialization"
        verbose_name_plural = "Specializations"
        ordering = ['name']

class Hospital(models.Model):
    """Hospital/Clinic information"""
    name = models.CharField(max_length=200)
    address = models.TextField()
    phone_number = models.CharField(max_length=15)
    email = models.EmailField(blank=True)
    website = models.URLField(blank=True)
    established_year = models.IntegerField(null=True, blank=True)
    bed_capacity = models.IntegerField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = "Hospital"
        verbose_name_plural = "Hospitals"
        ordering = ['name']

class Doctor(models.Model):
    """Doctor specific information"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='doctor_profile')
    doctor_id = models.CharField(max_length=10, unique=True, editable=False)
    specialization = models.ForeignKey(Specialization, on_delete=models.CASCADE)
    hospital = models.ForeignKey(Hospital, on_delete=models.CASCADE, related_name='doctors')
    
    # Professional details
    license_number = models.CharField(max_length=50, unique=True)
    years_experience = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(60)])
    qualification = models.CharField(max_length=200, help_text="e.g., MBBS, MD, MS")
    consultation_fee = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('0.00'))
    
    # Availability
    is_available = models.BooleanField(default=True)
    consultation_hours_start = models.TimeField(null=True, blank=True)
    consultation_hours_end = models.TimeField(null=True, blank=True)
    
    # Ratings and reviews
    rating = models.DecimalField(
        max_digits=3, 
        decimal_places=2, 
        default=Decimal('0.00'),
        validators=[MinValueValidator(0), MaxValueValidator(5)]
    )
    total_reviews = models.IntegerField(default=0)
    total_patients_treated = models.IntegerField(default=0)
    
    # Additional information
    bio = models.TextField(blank=True, help_text="Doctor's biography and expertise")
    languages_spoken = models.CharField(max_length=200, blank=True, help_text="Comma-separated languages")
    awards_recognition = models.TextField(blank=True)
    
    # Status
    is_verified = models.BooleanField(default=False)
    verification_date = models.DateTimeField(null=True, blank=True)
    joined_date = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        if not self.doctor_id:
            # Generate unique doctor ID
            last_doctor = Doctor.objects.order_by('-id').first()
            if last_doctor:
                last_id = int(last_doctor.doctor_id[1:])  # Remove 'D' prefix
                new_id = last_id + 1
            else:
                new_id = 1
            self.doctor_id = f"D{new_id:03d}"  # Format as D001, D002, etc.
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Dr. {self.user.get_full_name()} ({self.specialization.name})"
    
    @property
    def average_rating(self):
        """Calculate average rating from reviews"""
        reviews = self.reviews.all()
        if reviews:
            total_rating = sum([review.rating for review in reviews])
            return round(total_rating / len(reviews), 2)
        return 0.0
    
    class Meta:
        verbose_name = "Doctor"
        verbose_name_plural = "Doctors"
        ordering = ['-rating', 'user__first_name']

class DoctorAvailability(models.Model):
    """Doctor availability schedule"""
    WEEKDAY_CHOICES = [
        (0, 'Monday'),
        (1, 'Tuesday'),
        (2, 'Wednesday'),
        (3, 'Thursday'),
        (4, 'Friday'),
        (5, 'Saturday'),
        (6, 'Sunday'),
    ]
    
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='availability')
    weekday = models.IntegerField(choices=WEEKDAY_CHOICES)
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_available = models.BooleanField(default=True)
    max_patients = models.IntegerField(default=20, help_text="Maximum patients per day")
    
    def __str__(self):
        return f"{self.doctor.user.get_full_name()} - {self.get_weekday_display()} ({self.start_time}-{self.end_time})"
    
    class Meta:
        verbose_name = "Doctor Availability"
        verbose_name_plural = "Doctor Availability"
        unique_together = ['doctor', 'weekday']
        ordering = ['doctor', 'weekday']

class DoctorReview(models.Model):
    """Patient reviews for doctors"""
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='reviews')
    patient = models.ForeignKey('patients.Patient', on_delete=models.CASCADE)
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="Rating from 1 to 5 stars"
    )
    review_text = models.TextField(blank=True)
    is_anonymous = models.BooleanField(default=False)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        patient_name = "Anonymous" if self.is_anonymous else self.patient.user.get_full_name()
        return f"{patient_name} - {self.doctor.user.get_full_name()} ({self.rating}/5)"
    
    class Meta:
        verbose_name = "Doctor Review"
        verbose_name_plural = "Doctor Reviews"
        unique_together = ['doctor', 'patient']
        ordering = ['-created_at']

class DoctorEducation(models.Model):
    """Doctor's educational background"""
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='education')
    degree = models.CharField(max_length=100)
    institution = models.CharField(max_length=200)
    year_completed = models.IntegerField()
    specialization = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"{self.doctor.user.get_full_name()} - {self.degree} from {self.institution}"
    
    class Meta:
        verbose_name = "Doctor Education"
        verbose_name_plural = "Doctor Education"
        ordering = ['-year_completed']

class DoctorExperience(models.Model):
    """Doctor's work experience"""
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, related_name='experience')
    hospital_name = models.CharField(max_length=200)
    position = models.CharField(max_length=100)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    is_current = models.BooleanField(default=False)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.doctor.user.get_full_name()} - {self.position} at {self.hospital_name}"
    
    class Meta:
        verbose_name = "Doctor Experience"
        verbose_name_plural = "Doctor Experience"
        ordering = ['-start_date']

