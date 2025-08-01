# Generated by Django 5.2.4 on 2025-07-18 16:34

import django.core.validators
import django.db.models.deletion
import reports.models
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('doctors', '0001_initial'),
        ('patients', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ReportTemplate',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('report_type', models.CharField(choices=[('lab_test', 'Lab Test'), ('radiology', 'Radiology'), ('pathology', 'Pathology'), ('cardiology', 'Cardiology'), ('prescription', 'Prescription'), ('discharge_summary', 'Discharge Summary'), ('consultation_note', 'Consultation Note'), ('surgery_report', 'Surgery Report'), ('vaccination', 'Vaccination Record'), ('other', 'Other')], max_length=20)),
                ('description', models.TextField(blank=True)),
                ('required_fields', models.JSONField(default=list, help_text='List of required fields')),
                ('optional_fields', models.JSONField(default=list, help_text='List of optional fields')),
                ('field_validations', models.JSONField(default=dict, help_text='Validation rules for fields')),
                ('template_content', models.TextField(blank=True, help_text='HTML template content')),
                ('css_styles', models.TextField(blank=True, help_text='CSS styles for template')),
                ('is_active', models.BooleanField(default=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Report Template',
                'verbose_name_plural': 'Report Templates',
                'ordering': ['name'],
            },
        ),
        migrations.CreateModel(
            name='MedicalReport',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('report_type', models.CharField(choices=[('lab_test', 'Lab Test'), ('radiology', 'Radiology'), ('pathology', 'Pathology'), ('cardiology', 'Cardiology'), ('prescription', 'Prescription'), ('discharge_summary', 'Discharge Summary'), ('consultation_note', 'Consultation Note'), ('surgery_report', 'Surgery Report'), ('vaccination', 'Vaccination Record'), ('other', 'Other')], max_length=20)),
                ('description', models.TextField(blank=True)),
                ('file', models.FileField(upload_to=reports.models.report_upload_path, validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx'])])),
                ('file_size', models.IntegerField(blank=True, help_text='File size in bytes', null=True)),
                ('file_type', models.CharField(blank=True, max_length=10)),
                ('status', models.CharField(choices=[('uploaded', 'Uploaded'), ('processing', 'Processing'), ('processed', 'Processed'), ('reviewed', 'Reviewed'), ('archived', 'Archived')], default='uploaded', max_length=20)),
                ('is_processed', models.BooleanField(default=False)),
                ('extracted_text', models.TextField(blank=True, help_text='Text extracted from OCR')),
                ('processing_notes', models.TextField(blank=True)),
                ('test_date', models.DateField(blank=True, null=True)),
                ('test_results', models.JSONField(blank=True, default=dict, help_text='Structured test results')),
                ('abnormal_findings', models.TextField(blank=True)),
                ('recommendations', models.TextField(blank=True)),
                ('is_reviewed', models.BooleanField(default=False)),
                ('reviewed_at', models.DateTimeField(blank=True, null=True)),
                ('review_notes', models.TextField(blank=True)),
                ('is_confidential', models.BooleanField(default=False)),
                ('upload_date', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('doctor', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='reviewed_reports', to='doctors.doctor')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='medical_reports', to='patients.patient')),
                ('shared_with_doctors', models.ManyToManyField(blank=True, related_name='accessible_reports', to='doctors.doctor')),
                ('uploaded_by', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='uploaded_reports', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Medical Report',
                'verbose_name_plural': 'Medical Reports',
                'ordering': ['-upload_date'],
            },
        ),
        migrations.CreateModel(
            name='LabTest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('test_name', models.CharField(max_length=200)),
                ('test_category', models.CharField(choices=[('blood', 'Blood Test'), ('urine', 'Urine Test'), ('stool', 'Stool Test'), ('imaging', 'Imaging'), ('biopsy', 'Biopsy'), ('culture', 'Culture'), ('genetic', 'Genetic Test'), ('hormone', 'Hormone Test'), ('cardiac', 'Cardiac Test'), ('other', 'Other')], max_length=20)),
                ('test_code', models.CharField(blank=True, max_length=20)),
                ('specimen_type', models.CharField(blank=True, max_length=100)),
                ('collection_date', models.DateTimeField(blank=True, null=True)),
                ('result_date', models.DateTimeField(blank=True, null=True)),
                ('result_value', models.CharField(blank=True, max_length=200)),
                ('result_unit', models.CharField(blank=True, max_length=50)),
                ('reference_range', models.CharField(blank=True, max_length=200)),
                ('is_abnormal', models.BooleanField(default=False)),
                ('methodology', models.CharField(blank=True, max_length=200)),
                ('lab_name', models.CharField(blank=True, max_length=200)),
                ('technician_notes', models.TextField(blank=True)),
                ('report', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='lab_tests', to='reports.medicalreport')),
            ],
            options={
                'verbose_name': 'Lab Test',
                'verbose_name_plural': 'Lab Tests',
                'ordering': ['-result_date'],
            },
        ),
        migrations.CreateModel(
            name='Prescription',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('medication_name', models.CharField(max_length=200)),
                ('dosage', models.CharField(max_length=100)),
                ('frequency', models.CharField(max_length=100)),
                ('duration', models.CharField(max_length=100)),
                ('instructions', models.TextField(blank=True)),
                ('prescription_date', models.DateField(blank=True, null=True)),
                ('start_date', models.DateField(blank=True, null=True)),
                ('end_date', models.DateField(blank=True, null=True)),
                ('is_active', models.BooleanField(default=True)),
                ('is_completed', models.BooleanField(default=False)),
                ('prescribed_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='doctors.doctor')),
                ('report', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='prescriptions', to='reports.medicalreport')),
            ],
            options={
                'verbose_name': 'Prescription',
                'verbose_name_plural': 'Prescriptions',
                'ordering': ['-prescription_date'],
            },
        ),
        migrations.CreateModel(
            name='ReportAnnotation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('annotation_type', models.CharField(choices=[('highlight', 'Highlight'), ('note', 'Note'), ('arrow', 'Arrow'), ('circle', 'Circle'), ('rectangle', 'Rectangle')], max_length=20)),
                ('content', models.TextField(blank=True, help_text='Annotation text or notes')),
                ('page_number', models.IntegerField(default=1)),
                ('x_coordinate', models.FloatField(blank=True, null=True)),
                ('y_coordinate', models.FloatField(blank=True, null=True)),
                ('width', models.FloatField(blank=True, null=True)),
                ('height', models.FloatField(blank=True, null=True)),
                ('color', models.CharField(default='#FFFF00', help_text='Hex color code', max_length=7)),
                ('opacity', models.FloatField(default=0.5)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('report', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='annotations', to='reports.medicalreport')),
            ],
            options={
                'verbose_name': 'Report Annotation',
                'verbose_name_plural': 'Report Annotations',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='ReportSharingLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('action', models.CharField(choices=[('shared', 'Shared'), ('unshared', 'Unshared'), ('viewed', 'Viewed'), ('downloaded', 'Downloaded')], max_length=20)),
                ('ip_address', models.GenericIPAddressField(blank=True, null=True)),
                ('user_agent', models.TextField(blank=True)),
                ('notes', models.TextField(blank=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('report', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='sharing_logs', to='reports.medicalreport')),
                ('shared_by', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='shared_reports', to=settings.AUTH_USER_MODEL)),
                ('shared_with', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='received_reports', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Report Sharing Log',
                'verbose_name_plural': 'Report Sharing Logs',
                'ordering': ['-timestamp'],
            },
        ),
    ]
