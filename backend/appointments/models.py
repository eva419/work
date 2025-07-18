from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator
from django.utils import timezone
from datetime import datetime, timedelta

class Appointment(models.Model):
    """Patient appointments with doctors"""
    STATUS_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('confirmed', 'Confirmed'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('no_show', 'No Show'),
        ('rescheduled', 'Rescheduled'),
    ]
    
    APPOINTMENT_TYPE_CHOICES = [
        ('consultation', 'Consultation'),
        ('follow_up', 'Follow-up'),
        ('checkup', 'Regular Checkup'),
        ('emergency', 'Emergency'),
        ('therapy', 'Therapy'),
        ('surgery', 'Surgery'),
        ('diagnostic', 'Diagnostic'),
    ]
    
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]
    
    appointment_id = models.CharField(max_length=10, unique=True, editable=False)
    patient = models.ForeignKey('patients.Patient', on_delete=models.CASCADE, related_name='appointments')
    doctor = models.ForeignKey('doctors.Doctor', on_delete=models.CASCADE, related_name='appointments')
    
    # Appointment details
    appointment_date = models.DateField()
    appointment_time = models.TimeField()
    duration = models.IntegerField(default=30, help_text="Duration in minutes")
    appointment_type = models.CharField(max_length=20, choices=APPOINTMENT_TYPE_CHOICES, default='consultation')
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES, default='medium')
    
    # Reason and notes
    reason_for_visit = models.CharField(max_length=200)
    symptoms = models.TextField(blank=True, help_text="Patient reported symptoms")
    patient_notes = models.TextField(blank=True, help_text="Additional notes from patient")
    
    # Status and tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='scheduled')
    is_online = models.BooleanField(default=False, help_text="Online/Telemedicine appointment")
    
    # Fees and payment
    consultation_fee = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    is_paid = models.BooleanField(default=False)
    payment_method = models.CharField(max_length=50, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    confirmed_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Reminders
    reminder_sent = models.BooleanField(default=False)
    reminder_sent_at = models.DateTimeField(null=True, blank=True)
    
    def save(self, *args, **kwargs):
        if not self.appointment_id:
            # Generate unique appointment ID
            last_appointment = Appointment.objects.order_by('-id').first()
            if last_appointment:
                last_id = int(last_appointment.appointment_id[1:])  # Remove 'A' prefix
                new_id = last_id + 1
            else:
                new_id = 1
            self.appointment_id = f"A{new_id:03d}"  # Format as A001, A002, etc.
        
        # Set consultation fee from doctor if not set
        if not self.consultation_fee and self.doctor:
            self.consultation_fee = self.doctor.consultation_fee
            
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.appointment_id} - {self.patient.user.get_full_name()} with Dr. {self.doctor.user.get_full_name()}"
    
    @property
    def appointment_datetime(self):
        """Combine date and time into datetime object"""
        return datetime.combine(self.appointment_date, self.appointment_time)
    
    @property
    def is_past(self):
        """Check if appointment is in the past"""
        return self.appointment_datetime < timezone.now()
    
    @property
    def is_today(self):
        """Check if appointment is today"""
        return self.appointment_date == timezone.now().date()
    
    @property
    def time_until_appointment(self):
        """Time remaining until appointment"""
        if self.is_past:
            return None
        return self.appointment_datetime - timezone.now()
    
    def can_be_cancelled(self):
        """Check if appointment can be cancelled (at least 24 hours before)"""
        if self.status in ['completed', 'cancelled', 'no_show']:
            return False
        time_diff = self.appointment_datetime - timezone.now()
        return time_diff > timedelta(hours=24)
    
    def can_be_rescheduled(self):
        """Check if appointment can be rescheduled"""
        return self.status in ['scheduled', 'confirmed'] and not self.is_past
    
    class Meta:
        verbose_name = "Appointment"
        verbose_name_plural = "Appointments"
        ordering = ['appointment_date', 'appointment_time']
        unique_together = ['doctor', 'appointment_date', 'appointment_time']

class AppointmentHistory(models.Model):
    """Track appointment status changes"""
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='history')
    previous_status = models.CharField(max_length=20)
    new_status = models.CharField(max_length=20)
    changed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    reason = models.TextField(blank=True)
    changed_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.appointment.appointment_id} - {self.previous_status} to {self.new_status}"
    
    class Meta:
        verbose_name = "Appointment History"
        verbose_name_plural = "Appointment History"
        ordering = ['-changed_at']

class AppointmentSlot(models.Model):
    """Available appointment slots for doctors"""
    doctor = models.ForeignKey('doctors.Doctor', on_delete=models.CASCADE, related_name='slots')
    date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_available = models.BooleanField(default=True)
    is_blocked = models.BooleanField(default=False, help_text="Blocked by doctor/admin")
    block_reason = models.CharField(max_length=200, blank=True)
    max_appointments = models.IntegerField(default=1)
    current_appointments = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Dr. {self.doctor.user.get_full_name()} - {self.date} ({self.start_time}-{self.end_time})"
    
    @property
    def is_full(self):
        """Check if slot is fully booked"""
        return self.current_appointments >= self.max_appointments
    
    @property
    def available_spots(self):
        """Number of available spots in this slot"""
        return max(0, self.max_appointments - self.current_appointments)
    
    class Meta:
        verbose_name = "Appointment Slot"
        verbose_name_plural = "Appointment Slots"
        unique_together = ['doctor', 'date', 'start_time']
        ordering = ['date', 'start_time']

class AppointmentReminder(models.Model):
    """Appointment reminders"""
    REMINDER_TYPE_CHOICES = [
        ('email', 'Email'),
        ('sms', 'SMS'),
        ('push', 'Push Notification'),
        ('call', 'Phone Call'),
    ]
    
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='reminders')
    reminder_type = models.CharField(max_length=10, choices=REMINDER_TYPE_CHOICES)
    scheduled_time = models.DateTimeField()
    sent_at = models.DateTimeField(null=True, blank=True)
    is_sent = models.BooleanField(default=False)
    message = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.appointment.appointment_id} - {self.reminder_type} reminder"
    
    class Meta:
        verbose_name = "Appointment Reminder"
        verbose_name_plural = "Appointment Reminders"
        ordering = ['scheduled_time']

class WaitingList(models.Model):
    """Waiting list for fully booked appointments"""
    patient = models.ForeignKey('patients.Patient', on_delete=models.CASCADE)
    doctor = models.ForeignKey('doctors.Doctor', on_delete=models.CASCADE)
    preferred_date = models.DateField()
    preferred_time = models.TimeField(null=True, blank=True)
    appointment_type = models.CharField(max_length=20, choices=Appointment.APPOINTMENT_TYPE_CHOICES)
    reason = models.CharField(max_length=200)
    priority = models.CharField(max_length=10, choices=Appointment.PRIORITY_CHOICES, default='medium')
    is_notified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.patient.user.get_full_name()} waiting for Dr. {self.doctor.user.get_full_name()}"
    
    class Meta:
        verbose_name = "Waiting List"
        verbose_name_plural = "Waiting List"
        ordering = ['priority', 'created_at']

