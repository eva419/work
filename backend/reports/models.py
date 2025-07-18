from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
import os

def report_upload_path(instance, filename):
    """Generate upload path for medical reports"""
    return f"reports/{instance.patient.patient_id}/{filename}"

class MedicalReport(models.Model):
    """Medical reports uploaded by patients or doctors"""
    REPORT_TYPE_CHOICES = [
        ('lab_test', 'Lab Test'),
        ('radiology', 'Radiology'),
        ('pathology', 'Pathology'),
        ('cardiology', 'Cardiology'),
        ('prescription', 'Prescription'),
        ('discharge_summary', 'Discharge Summary'),
        ('consultation_note', 'Consultation Note'),
        ('surgery_report', 'Surgery Report'),
        ('vaccination', 'Vaccination Record'),
        ('other', 'Other'),
    ]
    
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('processed', 'Processed'),
        ('reviewed', 'Reviewed'),
        ('archived', 'Archived'),
    ]
    
    patient = models.ForeignKey('patients.Patient', on_delete=models.CASCADE, related_name='medical_reports')
    doctor = models.ForeignKey('doctors.Doctor', on_delete=models.SET_NULL, null=True, blank=True, related_name='reviewed_reports')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_reports')
    
    # Report details
    title = models.CharField(max_length=200)
    report_type = models.CharField(max_length=20, choices=REPORT_TYPE_CHOICES)
    description = models.TextField(blank=True)
    
    # File information
    file = models.FileField(
        upload_to=report_upload_path,
        validators=[FileExtensionValidator(allowed_extensions=['pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx'])]
    )
    file_size = models.IntegerField(null=True, blank=True, help_text="File size in bytes")
    file_type = models.CharField(max_length=10, blank=True)
    
    # Processing and OCR
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    is_processed = models.BooleanField(default=False)
    extracted_text = models.TextField(blank=True, help_text="Text extracted from OCR")
    processing_notes = models.TextField(blank=True)
    
    # Medical information extracted
    test_date = models.DateField(null=True, blank=True)
    test_results = models.JSONField(default=dict, blank=True, help_text="Structured test results")
    abnormal_findings = models.TextField(blank=True)
    recommendations = models.TextField(blank=True)
    
    # Review and validation
    is_reviewed = models.BooleanField(default=False)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    review_notes = models.TextField(blank=True)
    
    # Privacy and sharing
    is_confidential = models.BooleanField(default=False)
    shared_with_doctors = models.ManyToManyField('doctors.Doctor', blank=True, related_name='accessible_reports')
    
    # Timestamps
    upload_date = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        if self.file:
            self.file_size = self.file.size
            self.file_type = os.path.splitext(self.file.name)[1][1:].lower()
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.patient.patient_id} - {self.title}"
    
    @property
    def file_size_mb(self):
        """Return file size in MB"""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return 0
    
    class Meta:
        verbose_name = "Medical Report"
        verbose_name_plural = "Medical Reports"
        ordering = ['-upload_date']

class ReportTemplate(models.Model):
    """Templates for different types of medical reports"""
    name = models.CharField(max_length=100, unique=True)
    report_type = models.CharField(max_length=20, choices=MedicalReport.REPORT_TYPE_CHOICES)
    description = models.TextField(blank=True)
    
    # Template structure
    required_fields = models.JSONField(default=list, help_text="List of required fields")
    optional_fields = models.JSONField(default=list, help_text="List of optional fields")
    field_validations = models.JSONField(default=dict, help_text="Validation rules for fields")
    
    # Template content
    template_content = models.TextField(blank=True, help_text="HTML template content")
    css_styles = models.TextField(blank=True, help_text="CSS styles for template")
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.report_type})"
    
    class Meta:
        verbose_name = "Report Template"
        verbose_name_plural = "Report Templates"
        ordering = ['name']

class ReportAnnotation(models.Model):
    """Annotations and highlights on medical reports"""
    ANNOTATION_TYPE_CHOICES = [
        ('highlight', 'Highlight'),
        ('note', 'Note'),
        ('arrow', 'Arrow'),
        ('circle', 'Circle'),
        ('rectangle', 'Rectangle'),
    ]
    
    report = models.ForeignKey(MedicalReport, on_delete=models.CASCADE, related_name='annotations')
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    annotation_type = models.CharField(max_length=20, choices=ANNOTATION_TYPE_CHOICES)
    content = models.TextField(blank=True, help_text="Annotation text or notes")
    
    # Position information (for PDF/image annotations)
    page_number = models.IntegerField(default=1)
    x_coordinate = models.FloatField(null=True, blank=True)
    y_coordinate = models.FloatField(null=True, blank=True)
    width = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    
    # Styling
    color = models.CharField(max_length=7, default='#FFFF00', help_text="Hex color code")
    opacity = models.FloatField(default=0.5)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Annotation on {self.report.title} by {self.created_by.username}"
    
    class Meta:
        verbose_name = "Report Annotation"
        verbose_name_plural = "Report Annotations"
        ordering = ['-created_at']

class ReportSharingLog(models.Model):
    """Log of report sharing activities"""
    ACTION_CHOICES = [
        ('shared', 'Shared'),
        ('unshared', 'Unshared'),
        ('viewed', 'Viewed'),
        ('downloaded', 'Downloaded'),
    ]
    
    report = models.ForeignKey(MedicalReport, on_delete=models.CASCADE, related_name='sharing_logs')
    shared_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='shared_reports')
    shared_with = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_reports')
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    
    # Additional information
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    notes = models.TextField(blank=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.report.title} - {self.action} by {self.shared_by.username}"
    
    class Meta:
        verbose_name = "Report Sharing Log"
        verbose_name_plural = "Report Sharing Logs"
        ordering = ['-timestamp']

class LabTest(models.Model):
    """Specific lab test information"""
    TEST_CATEGORY_CHOICES = [
        ('blood', 'Blood Test'),
        ('urine', 'Urine Test'),
        ('stool', 'Stool Test'),
        ('imaging', 'Imaging'),
        ('biopsy', 'Biopsy'),
        ('culture', 'Culture'),
        ('genetic', 'Genetic Test'),
        ('hormone', 'Hormone Test'),
        ('cardiac', 'Cardiac Test'),
        ('other', 'Other'),
    ]
    
    report = models.ForeignKey(MedicalReport, on_delete=models.CASCADE, related_name='lab_tests')
    test_name = models.CharField(max_length=200)
    test_category = models.CharField(max_length=20, choices=TEST_CATEGORY_CHOICES)
    
    # Test details
    test_code = models.CharField(max_length=20, blank=True)
    specimen_type = models.CharField(max_length=100, blank=True)
    collection_date = models.DateTimeField(null=True, blank=True)
    result_date = models.DateTimeField(null=True, blank=True)
    
    # Results
    result_value = models.CharField(max_length=200, blank=True)
    result_unit = models.CharField(max_length=50, blank=True)
    reference_range = models.CharField(max_length=200, blank=True)
    is_abnormal = models.BooleanField(default=False)
    
    # Additional information
    methodology = models.CharField(max_length=200, blank=True)
    lab_name = models.CharField(max_length=200, blank=True)
    technician_notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.test_name} - {self.result_value} {self.result_unit}"
    
    class Meta:
        verbose_name = "Lab Test"
        verbose_name_plural = "Lab Tests"
        ordering = ['-result_date']

class Prescription(models.Model):
    """Prescription information from reports"""
    report = models.ForeignKey(MedicalReport, on_delete=models.CASCADE, related_name='prescriptions')
    prescribed_by = models.ForeignKey('doctors.Doctor', on_delete=models.SET_NULL, null=True, blank=True)
    
    # Medication details
    medication_name = models.CharField(max_length=200)
    dosage = models.CharField(max_length=100)
    frequency = models.CharField(max_length=100)
    duration = models.CharField(max_length=100)
    instructions = models.TextField(blank=True)
    
    # Prescription details
    prescription_date = models.DateField(null=True, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_completed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.medication_name} - {self.dosage}"
    
    class Meta:
        verbose_name = "Prescription"
        verbose_name_plural = "Prescriptions"
        ordering = ['-prescription_date']

